"""
MappingMind — Data Ingestion Pipeline
=======================================
CONCEPTS:
  - LangChain document loaders: standardized way to load any source
  - Chunking strategy: how we split mapping records for retrieval
  - Embeddings: text → vectors for semantic search
  - Qdrant: production vector DB with hybrid search support

SOURCES:
  - CSV: past mapping records (structured)
  - PDF: data dictionaries (unstructured)
  - Text: ADRs and rationale docs
"""

import os
import csv
import pickle
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

load_dotenv()

# ── EMBEDDING MODEL ────────────────────────────────────────────────
# Local embeddings — free, private, no data egress
# 384 dimensions — good balance of accuracy vs speed

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


# ── LOADER 1: CSV MAPPINGS ─────────────────────────────────────────
# Each row = one past mapping decision
# We convert each row to a rich text document for retrieval
# Rich text = field names + transformation + rationale + context

def load_csv_mappings(csv_path: str) -> list[Document]:
    docs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert row to rich natural language for better retrieval
            content = f"""
Mapping Record:
Source Field: {row['source_field']}
Target Field: {row['target_field']}
Transformation: {row['transformation']}
Rationale: {row['rationale']}
System: {row['system']}
Year: {row['year']}
Status: {row['status']}
            """.strip()

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": csv_path,
                    "doc_type": "mapping",
                    "source_field": row['source_field'],
                    "target_field": row['target_field'],
                    "system": row['system'],
                    "year": row['year'],
                    "status": row['status']
                }
            ))
    print(f"✅ Loaded {len(docs)} mapping records from CSV")
    return docs


# ── LOADER 2: ADR DOCUMENTS ────────────────────────────────────────
# Architecture Decision Records — explain WHY decisions were made
# Critical for MappingMind — rationale is as important as the mapping

ADR_DOCUMENTS = [
    """
ADR-001: Currency Field Normalization Standard
Date: 2021-03-15 | Status: Approved

Context:
Multiple source systems store monetary values differently.
CoreBanking stores in minor currency units (pence/cents).
LoanSystem stores in major currency units (pounds/dollars).
Inconsistent handling caused $240K reconciliation error in Q1 2021.

Decision:
All currency fields must be normalized to major units (USD/GBP) at ingestion.
Transformation must be documented with source unit confirmation from SME.
Default assumption: if source system is CoreBanking — divide by 100.

Consequences:
+ Eliminates currency reconciliation errors
+ Single standard across all target schemas
- Requires SME sign-off on every new currency field mapping
- Additional validation step adds 2 hours per integration
    """,
    """
ADR-002: Date Format Standardization
Date: 2021-06-20 | Status: Approved

Context:
Source systems use 6 different date formats: YYYYMMDD, MM/DD/YYYY,
DD-MON-YYYY, YYYYMM, Unix timestamp, and ISO 8601.
Wrong format assumption caused data load failure in PaymentSystem migration.

Decision:
All date fields must be converted to ISO 8601 (YYYY-MM-DD) at target.
Source format must be explicitly documented in mapping record.
YYYYMM sources: use LAST_DAY() to get end-of-month date.

Consequences:
+ Consistent date handling across all downstream reports
+ Eliminates date parsing errors
- YYYYMM → full date conversion requires business rule (first day vs last day)
    """,
    """
ADR-003: String Normalization Policy
Date: 2022-01-10 | Status: Approved

Context:
Legacy systems produce inconsistent string formats: mixed case, 
leading/trailing spaces, special characters in numeric fields.
Downstream joins failing due to case mismatch on customer_id fields.

Decision:
ID fields: TRIM + UPPER always applied
Description fields: INITCAP + TRIM
Numeric strings: REGEXP_REPLACE to strip non-numeric characters
Code fields: TRIM only — case preserved for system codes

Consequences:
+ Eliminates join failures from case/space mismatches
+ Standardized transformation library reusable across projects
- INITCAP can incorrectly capitalize abbreviations (e.g., 'USA' → 'Usa')
    """,
    """
ADR-004: Rejected Mapping Patterns — Do Not Use
Date: 2023-03-05 | Status: Approved

Context:
Recurring mistakes across 3 integrations causing post-go-live defects.
Documented to prevent repetition.

Rejected Pattern 1: Implicit type casting
Never use implicit CAST — always explicit. Implicit casting behavior
differs across Oracle, SQL Server, and Snowflake.

Rejected Pattern 2: NULL handling assumption
Never assume NULL means zero for numeric fields.
Always confirm with source system SME. CoreBanking NULL balance ≠ zero balance.

Rejected Pattern 3: Hardcoded lookup values
Never hardcode CASE WHEN values without a reference table.
Source system code values change without notice.

Consequences:
+ Prevents recurrence of known defects
+ Mandatory review checklist item before mapping approval
    """
]

def load_adr_documents() -> list[Document]:
    docs = []
    for i, adr_text in enumerate(ADR_DOCUMENTS):
        docs.append(Document(
            page_content=adr_text.strip(),
            metadata={
                "source": "internal-adrs",
                "doc_type": "adr",
                "adr_number": f"ADR-00{i+1}"
            }
        ))
    print(f"✅ Loaded {len(docs)} ADR documents")
    return docs


# ── CHUNKING ───────────────────────────────────────────────────────
# CSV mappings: each row = one chunk (atomic, don't split)
# ADRs: recursive split — preserve section boundaries

def chunk_documents(docs: list[Document]) -> list[Document]:
    mapping_docs = [d for d in docs if d.metadata.get("doc_type") == "mapping"]
    adr_docs = [d for d in docs if d.metadata.get("doc_type") == "adr"]

    # ADRs need splitting — they're longer
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    adr_chunks = splitter.split_documents(adr_docs)

    all_chunks = mapping_docs + adr_chunks
    print(f"✅ Total chunks: {len(all_chunks)} ({len(mapping_docs)} mappings + {len(adr_chunks)} ADR chunks)")
    return all_chunks


# ── STORE IN CHROMADB ──────────────────────────────────────────────
# Using ChromaDB for now — Qdrant added in Phase 3 with Docker
# Same LangChain interface — swap is one line change

def store_documents(chunks: list[Document]):
    embedding_model = get_embedding_model()

    client = chromadb.PersistentClient(path="./data/chroma_store")

    # Fresh start
    try:
        client.delete_collection("mappingmind")
    except:
        pass

    vectorstore = Chroma(
        collection_name="mappingmind",
        embedding_function=embedding_model,
        client=client
    )

    vectorstore.add_documents(chunks)
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

    # Cache for fast loading
    with open("data/chunks_cache.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Cache saved")

    return vectorstore


# ── MAIN PIPELINE ──────────────────────────────────────────────────
def run_ingestion():
    print("\n🚀 MappingMind Ingestion Pipeline")
    print("=" * 40)

    # Load all sources
    csv_docs = load_csv_mappings("data/mappings.csv")
    adr_docs = load_adr_documents()
    all_docs = csv_docs + adr_docs

    # Chunk
    chunks = chunk_documents(all_docs)

    # Store
    vectorstore = store_documents(chunks)

    print(f"\n✅ Ingestion complete. {len(chunks)} chunks ready for retrieval.")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()