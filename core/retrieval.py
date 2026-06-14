"""
MappingMind — Hybrid Search + Rerank
======================================
CONCEPTS:
  - BM25: keyword frequency search — catches exact field names
  - Bi-encoder: semantic search — catches conceptual matches
  - RRF: merges both ranking lists fairly
  - Cross-encoder: reranks top results by reading query+chunk together
  - Metadata filtering: filter by system, year, status before search

WHY HYBRID FOR MAPPINGMIND:
  User asks "balance field mapping" →
  BM25 catches: cust_acct_bal, acct_bal, bal_dt (exact matches)
  Semantic catches: monetary, currency, amount (conceptual matches)
  Together = complete recall
"""

import os
import pickle
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np

load_dotenv()

# ── LOAD RESOURCES ─────────────────────────────────────────────────
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    client = chromadb.PersistentClient(path="./data/chroma_store")
    vectorstore = Chroma(
        collection_name="mappingmind",
        embedding_function=embedding_model,
        client=client
    )
    return vectorstore


def load_chunks() -> list[Document]:
    with open("data/chunks_cache.pkl", "rb") as f:
        return pickle.load(f)


# ── BM25 SEARCH ────────────────────────────────────────────────────
# Keyword search — fast, catches exact field names
# Critical for MappingMind: "cust_acct_bal" exact match

def bm25_search(query: str, chunks: list[Document], top_k: int = 10) -> list[tuple[Document, float]]:
    corpus = [doc.page_content.lower().split() for doc in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]


# ── SEMANTIC SEARCH ────────────────────────────────────────────────
# Meaning-based search — catches conceptual matches
# "monetary normalization" matches "currency field divide by 100"

def semantic_search(query: str, vectorstore, top_k: int = 10) -> list[tuple[Document, float]]:
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    # Convert distance to similarity
    return [(doc, 1 - score) for doc, score in results]


# ── RRF FUSION ─────────────────────────────────────────────────────
# Reciprocal Rank Fusion — merges BM25 + semantic rankings
# Does not compare raw scores (incomparable scales)
# Instead combines ranks — position in each list matters

def rrf_fusion(
    bm25_results: list[tuple[Document, float]],
    semantic_results: list[tuple[Document, float]],
    k: int = 60
) -> list[tuple[Document, float]]:

    scores = {}
    doc_map = {}

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(semantic_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[key], scores[key]) for key in sorted_keys]


# ── CROSS-ENCODER RERANK ───────────────────────────────────────────
# Reads query + chunk TOGETHER — much more accurate than bi-encoder
# Bi-encoder: embeds separately → fast, approximate
# Cross-encoder: joint reading → slow, precise
# Production pattern: retrieve 20 → rerank to top 5

def rerank(query: str, results: list[tuple[Document, float]], top_k: int = 5) -> list[tuple[Document, float]]:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc.page_content] for doc, _ in results]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip([doc for doc, _ in results], scores),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked[:top_k]

# ── MMR DIVERSITY SELECTION ────────────────────────────────────────
# Maximal Marginal Relevance:
# keeps results relevant to query, but avoids near-duplicate chunks.
# Production pattern: rerank for precision, then MMR for diverse evidence.
_mmr_model = SentenceTransformer("all-MiniLM-L6-v2")


def apply_mmr_diversity(
    query: str,
    results: list[tuple[Document, float]],
    top_k: int = 5,
    lambda_mult: float = 0.7,
) -> list[tuple[Document, float]]:
    """
    Apply Maximal Marginal Relevance on reranked results.

    lambda_mult closer to 1.0 = prioritize relevance.
    lambda_mult closer to 0.0 = prioritize diversity.
    """

    if not results or len(results) <= top_k:
        return results

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]
    texts = [doc.page_content for doc in docs]

    query_embedding = _mmr_model.encode(query, convert_to_tensor=True)
    doc_embeddings = _mmr_model.encode(texts, convert_to_tensor=True)

    query_similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

    selected_indices = []
    candidate_indices = list(range(len(results)))

    first_index = int(query_similarities.argmax())
    selected_indices.append(first_index)
    candidate_indices.remove(first_index)

    while candidate_indices and len(selected_indices) < top_k:
        best_score = -999
        best_index = None

        for idx in candidate_indices:
            relevance_score = float(query_similarities[idx])

            selected_embeddings = doc_embeddings[selected_indices]
            diversity_penalty = float(
                util.cos_sim(doc_embeddings[idx], selected_embeddings).max()
            )

            mmr_score = (
                lambda_mult * relevance_score
                - (1 - lambda_mult) * diversity_penalty
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx

        selected_indices.append(best_index)
        candidate_indices.remove(best_index)

    diversified_results = []

    for mmr_rank, idx in enumerate(selected_indices, start=1):
        doc = docs[idx]
        original_score = scores[idx]

        doc.metadata = {
            **doc.metadata,
            "mmr_rank": mmr_rank,
            "mmr_enabled": True,
        }

        diversified_results.append((doc, original_score))

    return diversified_results
# ── METADATA FILTER ────────────────────────────────────────────────
# Production feature: filter by system, year, status BEFORE search
# Reduces noise — user only sees approved mappings from relevant system

def filter_by_metadata(
    chunks: list[Document],
    system: str = None,
    year: str = None,
    status: str = "approved"
) -> list[Document]:
    filtered = chunks
    if system:
        filtered = [d for d in filtered if d.metadata.get("system", "").lower() == system.lower()]
    if year:
        filtered = [d for d in filtered if d.metadata.get("year", "") == year]
    if status:
        filtered = [d for d in filtered if d.metadata.get("status", "").lower() == status.lower()]
    return filtered if filtered else chunks  # fallback to all if filter too strict


# ── MAIN RETRIEVAL PIPELINE ────────────────────────────────────────
def retrieve(
    query: str,
    system_filter: str = None,
    top_k: int = 5
) -> list[tuple[Document, float]]:

    print(f"\n🔍 Query: {query}")

    vectorstore = load_vectorstore()
    chunks = load_chunks()

    # Apply metadata filter
    if system_filter:
        chunks = filter_by_metadata(chunks, system=system_filter)
        print(f"✅ Filtered to {len(chunks)} chunks for system: {system_filter}")

    # Step 1: BM25
    bm25_results = bm25_search(query, chunks, top_k=10)
    print(f"✅ BM25: {len(bm25_results)} results")

    # Step 2: Semantic
    semantic_results = semantic_search(query, vectorstore, top_k=10)
    print(f"✅ Semantic: {len(semantic_results)} results")

    # Step 3: RRF fusion
    fused = rrf_fusion(bm25_results, semantic_results)
    print(f"✅ RRF fusion: {len(fused)} results")

        # Step 4: Rerank
    reranked = rerank(query, fused[:10], top_k=min(len(fused), 10))
    print(f"✅ Reranked: top {len(reranked)} results")

    # Step 5: MMR diversity selection
    diversified = apply_mmr_diversity(
        query=query,
        results=reranked,
        top_k=top_k,
        lambda_mult=0.7,
    )
    print(f"✅ MMR diversity: top {len(diversified)} results")

    for i, (doc, score) in enumerate(diversified):
        print(f"\n [{i+1}] Score: {score:.3f}")
        print(f" MMR Rank: {doc.metadata.get('mmr_rank')}")
        print(f" {doc.page_content[:100]}...")

    return diversified

if __name__ == "__main__":
    results = retrieve("How should we map a customer balance field from CoreBanking?")