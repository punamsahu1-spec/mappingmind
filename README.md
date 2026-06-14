# The MappingMind AI Project

## Phase 1 RAG Systems: Enterprise Data Mapping Intelligence

### A Learner-Focused Journey into Production Agentic RAG Systems

Learn to build a modern AI system from the ground up through hands-on implementation.

Master the most in-demand AI engineering skills: RAG, hybrid search, embeddings, vector databases, reranking, MMR, prompt engineering, context engineering, guardrails, evaluation, audit logging, semantic cache, monitoring foundations, and Agentic RAG.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-purple)
![RAG](https://img.shields.io/badge/RAG-Production_Ready-orange)
![Agentic AI](https://img.shields.io/badge/Agentic_AI-LangGraph_Style-black)

---

## System Architecture

![MappingMind Architecture](docs/images/mappingmind_architecture.png)

---

## About This Course

This is a learner-focused project where you build a complete enterprise data mapping assistant that understands mapping sheets, transformation rules, architecture decision records, and past implementation decisions.

MappingMind teaches you to build a production-style RAG system using practical engineering patterns. Unlike basic tutorials that jump straight to “chat with documents,” this project follows the professional path: start with clean application structure, build ingestion, add keyword retrieval, add semantic retrieval, combine both with hybrid search, improve precision with reranking, improve diversity with MMR, and then wrap the full system with Agentic RAG, guardrails, evaluation, semantic cache, audit logging, and production-hardening patterns.

> The Professional Difference: We build RAG systems the way successful enterprise teams do — strong retrieval foundations enhanced with AI, not AI-first demos that ignore search quality, evidence control, safety, and governance.

By the end of this project, you will have your own AI-powered enterprise mapping assistant and the technical skills to explain production Agentic RAG systems for any enterprise domain.

---

## What You'll Build

* Week 1: Application foundation with FastAPI, Streamlit, virtual environment, safe configuration, and project structure
* Week 2: Data ingestion pipeline for mapping CSVs and architecture decision records
* Week 3: BM25 keyword search for exact field names, mapping rules, and transformation terms
* Week 4: Chunking, embeddings, ChromaDB, hybrid search, RRF, reranking, and MMR
* Week 5: Complete RAG pipeline with Gemini LLM, grounded answers, citations, and Streamlit interface
* Week 6: Production hardening with RAGAS evaluation, audit logging, semantic cache, and dashboard visibility
* Week 7: Agentic RAG with guardrails, evidence grading, query rewriting, answer validation, and SME review escalation

---

## System Architecture Evolution

### Week 7: Agentic RAG for Enterprise Data Mapping

![MappingMind Architecture](docs/images/mappingmind_architecture.png)

Complete Week 7 architecture showing how the enterprise data mapping assistant connects the dashboard, API layer, Agentic RAG workflow, hybrid retrieval, reranking, grounded generation, validation, audit logging, semantic cache, and production hardening roadmap.

---

### LangGraph Agentic RAG Workflow

![Agentic RAG Workflow](docs/images/agentic_rag_workflow.png)

Detailed agentic workflow showing decision nodes, document grading, adaptive retrieval, query rewriting, answer validation, and SME review escalation.

Key innovations:

* Intelligent decision-making: the agent evaluates whether to answer, retry, block, or escalate
* Document grading: retrieved context is checked before generation
* Query rewriting: weak queries are refined and retried
* Guardrails: prompt-injection and out-of-domain detection reduce unsafe behavior
* Grounding validation: answers are checked against retrieved evidence
* SME review: weak or unsupported answers are escalated instead of hallucinated
* Transparency: full agent trace is available for debugging and trust

---

## Quick Start

### Prerequisites

* Python 3.12+
* Git
* Google Gemini API key
* Optional LangSmith API key
* 4GB+ RAM recommended
* Windows, macOS, or Linux

---

### Get Started

```bash
# 1. Clone and setup
git clone https://github.com/punamsahu1-spec/mappingmind.git
cd mappingmind

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
```

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with your own local keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=mappingmind
LANGCHAIN_TRACING_V2=true
```

Important:

```text
Never commit .env to GitHub.
Only .env.example should be tracked.
```

Run ingestion:

```bash
python core/ingest.py
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Weekly Learning Path

| Week   | Topic                                                                                                     | Status     | Code Area                                                                 |
| ------ | --------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------- |
| Week 1 | [Infrastructure Foundation](#week-1-infrastructure-foundation-)                                           | ✅ Complete | `api/`, `dashboard/`, `core/`, `.env.example`, `requirements.txt`         |
| Week 2 | [Data Ingestion Pipeline](#week-2-data-ingestion-pipeline-)                                               | ✅ Complete | `core/ingest.py`, `data/`, ChromaDB                                       |
| Week 3 | [Keyword Search First - The Critical Foundation](#week-3-keyword-search-first---the-critical-foundation-) | ✅ Complete | `core/retrieval.py`                                                       |
| Week 4 | [Chunking & Hybrid Search - The Semantic Layer](#week-4-chunking--hybrid-search---the-semantic-layer-)    | ✅ Complete | `core/ingest.py`, `core/retrieval.py`, ChromaDB                           |
| Week 5 | [Complete RAG Pipeline with LLM Integration](#week-5-complete-rag-pipeline-with-llm-integration-)         | ✅ Complete | `core/generation.py`, `dashboard/app.py`                                  |
| Week 6 | [Production Monitoring and Caching](#week-6-production-monitoring-and-caching-)                           | ✅ Complete | `core/evaluation.py`, `core/audit.py`, `core/cache.py`, dashboard signals |
| Week 7 | [Agentic RAG and Guardrails](#week-7-agentic-rag--guardrails-)                                            | ✅ Complete | `core/agent.py`, `core/guardrails.py`                                     |

---

## Access Your Services

| Service              | URL / Command                  | Purpose                              |
| -------------------- | ------------------------------ | ------------------------------------ |
| API Documentation    | `http://127.0.0.1:8000/docs`   | Interactive API testing              |
| Health Check         | `http://127.0.0.1:8000/health` | Verify backend service is running    |
| Ask API              | `POST /ask`                    | Ask MappingMind a mapping question   |
| Streamlit Dashboard  | `http://localhost:8501`        | User-friendly MappingMind interface  |
| Knowledge Ingestion  | `python core/ingest.py`        | Load mapping knowledge into ChromaDB |
| Evaluation Module    | `python core/evaluation.py`    | Run RAGAS-style evaluation checks    |
| ChromaDB Local Store | `data/chroma_store/`           | Local vector store persistence       |

---

## Week 1: Infrastructure Foundation ✅

Start here. Master the application foundation that powers modern RAG systems.

### Learning Objectives

* FastAPI backend setup with automatic documentation and health checks
* Streamlit dashboard for interactive user experience
* Safe environment variable management with `.env.example`
* Local Python virtual environment setup
* Clean project structure with separate API, dashboard, and core AI logic
* Professional Git hygiene for secrets and generated files

### Architecture Overview

![MappingMind Architecture](docs/images/mappingmind_architecture.png)

Infrastructure components:

* FastAPI: REST API layer with root, health, and ask endpoints
* Streamlit: user-facing dashboard for asking mapping questions
* Core AI modules: agent, retrieval, generation, guardrails, cache, audit, and evaluation
* Environment layer: `.env.example` for safe configuration
* Local runtime: Python virtual environment and dependency management

### Setup Guide

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Completion Guide

* Confirm the repo has `api/`, `core/`, `dashboard/`, and `data/`
* Confirm FastAPI starts with `uvicorn api.main:app --reload`
* Confirm dashboard starts with `streamlit run dashboard/app.py`
* Confirm `.env` is local only and `.env.example` is tracked

### Deep Dive

The most important production design choice in Week 1 is separation of concerns. The UI should not contain retrieval logic. The API should not contain prompt engineering. The agent should orchestrate the workflow, while retrieval, generation, guardrails, audit, cache, and evaluation remain separate modules.

---

## Week 2: Data Ingestion Pipeline ✅

Building on Week 1 infrastructure: learn to process enterprise mapping knowledge into a searchable knowledge base.

### Learning Objectives

* Load structured mapping records from CSV files
* Convert mapping rows into rich natural-language documents
* Load architecture decision records as unstructured evidence
* Chunk decision documents for retrieval
* Attach metadata for source traceability
* Persist processed knowledge in ChromaDB
* Cache processed chunks for faster retrieval startup

### Architecture Overview

![Ingestion Pipeline](docs/images/ingestion_pipeline.png)

Data pipeline components:

| Component                | Current Implementation                                    |
| ------------------------ | --------------------------------------------------------- |
| Mapping CSV Loader       | Loads mapping rows from `data/mappings.csv`               |
| Mapping Document Builder | Converts each mapping row into rich natural-language text |
| ADR Loader               | Loads internal ADR-style decision documents               |
| Text Chunker             | Splits ADR documents using recursive chunking             |
| Embedding Model          | Uses local sentence-transformer embeddings                |
| Vector Store             | Stores chunks in ChromaDB under `data/chroma_store/`      |
| Chunk Cache              | Saves processed chunks to `data/chunks_cache.pkl`         |

### Implementation Guide

```bash
python core/ingest.py
```

### Completion Guide

* Mapping CSV rows load successfully
* ADR documents load successfully
* ADR documents are chunked
* ChromaDB store is created locally
* Chunk cache is created locally but ignored by Git

### Deep Dive

Better ingestion creates better retrieval. Poorly prepared documents lead to weak answers even when the LLM is strong. MappingMind treats ingestion as the foundation of trust because mapping answers depend on source quality, metadata, and decision history.

---

## Week 3: Keyword Search First - The Critical Foundation ✅

Building on Weeks 1-2 foundation: implement the keyword search foundation that professional RAG systems rely on.

### Learning Objectives

* Understand why keyword search is essential for RAG systems
* Use BM25 to retrieve exact field names and transformation terms
* Support technical identifiers such as `cust_acct_bal`
* Retrieve exact mapping records before using semantic search
* Establish a retrieval baseline before adding vectors
* Understand why enterprise systems often need keyword and vector search together

### Architecture Overview

![Retrieval Pipeline](docs/images/retrieval_pipeline.png)

Search infrastructure components:

* BM25 retriever: finds exact terms, field names, and transformation patterns
* Mapping corpus: searchable text from mapping records and ADR chunks
* Relevance scores: keyword-based ranking signal
* Retrieval baseline: foundation for hybrid retrieval

### Setup Guide

```bash
python core/retrieval.py
```

### Completion Guide

* Test a query with an exact source-field name
* Verify that keyword retrieval surfaces relevant mapping records
* Compare exact-field queries with business-language queries

### Deep Dive

Enterprise RAG cannot rely only on embeddings. Field names, abbreviations, table names, system codes, and transformation identifiers often require exact keyword search. BM25 protects precision when users ask about technical mapping terms.

---

## Week 4: Chunking & Hybrid Search - The Semantic Layer ✅

Building on Week 3 foundation: add the semantic layer that makes search understand business meaning.

### Learning Objectives

* Generate embeddings for mapping knowledge
* Store vectors in ChromaDB
* Retrieve conceptually similar mapping examples
* Compare keyword search with semantic search
* Combine BM25 and vector search using RRF fusion
* Rerank results using a cross-encoder
* Apply MMR to reduce duplicate context

### Architecture Overview

![Retrieval Pipeline](docs/images/retrieval_pipeline.png)

Hybrid search infrastructure components:

* Text chunker: splits ADRs and mapping documents into retrievable units
* Embedding model: converts mapping knowledge into vectors
* ChromaDB: stores vector embeddings for semantic retrieval
* BM25 search: retrieves exact terms and technical identifiers
* RRF fusion: combines keyword and semantic rankings
* Cross-encoder reranker: improves precision of final evidence
* MMR diversity selection: reduces near-duplicate chunks before generation

### Setup Guide

```bash
python core/ingest.py
python core/retrieval.py
```

### Completion Guide

* Compare keyword-only, vector-only, and hybrid retrieval results
* Confirm field-name and business-language queries both work
* Confirm fused results are reranked
* Confirm MMR adds diverse final context

### Deep Dive

Hybrid retrieval is often the practical production default. BM25 captures exact technical precision. Vector search captures meaning. RRF balances both result sets. Reranking improves precision, and MMR improves diversity so the LLM receives richer evidence instead of repeated chunks.

---

## Week 5: Complete RAG Pipeline with LLM Integration ✅

Building on Week 4 hybrid search: add the LLM layer that turns search into grounded enterprise answers.

### Learning Objectives

* Build retrieved context for the LLM
* Use a grounded generation prompt
* Require source-backed responses
* Generate mapping recommendations and transformation logic
* Return citations, risks, assumptions, and confidence signals
* Display the final response in the Streamlit dashboard

### Architecture Overview

![MappingMind Architecture](docs/images/mappingmind_architecture.png)

Complete RAG infrastructure components:

* Retrieval pipeline: BM25 + vector search + RRF + reranker + MMR
* Context builder: formats cited evidence for the LLM
* Gemini LLM: generates grounded answers
* System prompt: restricts the answer to retrieved context
* Guardrails: checks prompt injection and grounding risk
* Dashboard: shows answer, sources, metrics, trace, cache, and audit events

### Setup Guide

```bash
python core/ingest.py
streamlit run dashboard/app.py
```

### Completion Guide

* Ask a mapping question in the dashboard
* Verify that the answer includes source evidence
* Verify that retrieved sources are visible
* Verify that the system refuses, blocks, or escalates weak evidence

### Deep Dive

The LLM is not the knowledge source. The retrieved context is the knowledge source. The LLM’s job is to synthesize a clear answer from the evidence provided by the retrieval pipeline.

---

## Week 6: Production Monitoring and Caching ✅

Building on Week 5 complete RAG system: add production monitoring foundations, evaluation, audit logging, and performance optimization.

### Learning Objectives

* Understand why production RAG requires observability
* Track query behavior, answer quality, retrieval decisions, and agent trace
* Prepare for LangSmith or Langfuse tracing
* Run RAGAS-style evaluation for answer quality
* Use semantic caching for repeated questions
* Track audit events for governance and debugging

### Architecture Overview

![Production Hardening Roadmap](docs/images/production_hardening_roadmap.png)

Production monitoring and caching components:

| Component                | Current Status                                                |
| ------------------------ | ------------------------------------------------------------- |
| RAGAS Evaluation         | ✅ Implemented in `core/evaluation.py`                         |
| Faithfulness Metric      | ✅ Implemented                                                 |
| Answer Relevancy Metric  | ✅ Implemented                                                 |
| LangSmith Config         | ✅ Environment variables present                               |
| Persistent Audit Log     | ✅ Implemented with local JSONL                                |
| Semantic Cache           | ✅ Implemented with embedding similarity                       |
| Monitoring Foundation    | ✅ Dashboard shows cache, decision, trace, and audit events    |
| Cost / Latency Dashboard | 🟡 Basic signals implemented; detailed trends can be expanded |

### Setup Guide

```bash
python core/evaluation.py
```

### Completion Guide

* Run evaluation after ingestion
* Confirm evaluation produces faithfulness and answer relevancy scores
* Confirm audit events are written to `logs/audit.jsonl`
* Confirm semantic cache is written to `data/semantic_cache.json`
* Confirm generated outputs are ignored by Git

### Deep Dive

Production monitoring is not just logging the final answer. A production RAG system should trace the full path: query, retrieval mode, retrieved chunks, reranking, MMR, prompt, model response, validation result, cache status, and final decision.

---

## Week 7: Agentic RAG & Guardrails ✅

Building on Week 6 monitoring foundation: add agentic decisioning, retries, validation, and safety.

### Learning Objectives

* Implement an agentic workflow around RAG
* Add input guardrails before retrieval
* Grade retrieved evidence before generation
* Rewrite weak queries and retry retrieval
* Validate whether the final answer is grounded
* Escalate weak answers instead of hallucinating
* Explain how Agentic RAG differs from basic RAG

### Architecture Overview

![Agentic RAG Workflow](docs/images/agentic_rag_workflow.png)

Agentic RAG components:

* Guard node: detects prompt injection and domain violations
* Retrieve node: runs hybrid retrieval
* Grade documents node: checks whether retrieved evidence is strong enough
* Rewrite query node: improves weak queries and retries retrieval
* Generate answer node: produces grounded response from retrieved context
* Validate answer node: checks grounding and hallucination risk
* SME review path: escalates weak or unsupported answers

### Setup Guide

```bash
python core/agent.py
streamlit run dashboard/app.py
```

### Completion Guide

* Test a normal mapping question
* Test a weak-evidence question
* Test a prompt-injection attempt
* Confirm the agent trace shows guard, retrieve, grade, generate, validate, and review decisions

### Deep Dive

Agentic RAG is useful when the system needs control decisions. Basic RAG retrieves and answers. Agentic RAG decides whether to retrieve, rewrite, answer, block, or escalate.

---

## Reference and Development Guide

### Tech Stack

| Layer          | Technology                | Purpose                                      |
| -------------- | ------------------------- | -------------------------------------------- |
| Language       | Python 3.12+              | Core application language                    |
| API            | FastAPI                   | Backend API and health endpoints             |
| UI             | Streamlit                 | Interactive user dashboard                   |
| LLM            | Gemini                    | Answer generation                            |
| Embeddings     | Sentence Transformers     | Semantic representation of mapping knowledge |
| Vector Store   | ChromaDB                  | Local vector database                        |
| Keyword Search | BM25                      | Exact field and term matching                |
| Fusion         | Reciprocal Rank Fusion    | Combines keyword and vector results          |
| Reranking      | Cross-Encoder             | Improves final evidence precision            |
| Diversity      | MMR                       | Reduces duplicate context                    |
| Guardrails     | Custom Python checks      | Prompt injection and domain safety           |
| Cache          | Semantic cache            | Reduces latency and repeated LLM calls       |
| Audit          | JSONL audit log           | Governance and traceability                  |
| Evaluation     | RAGAS + custom test cases | Faithfulness and answer relevancy checks     |
| Observability  | LangSmith-ready config    | Future tracing and monitoring                |

---

## Project Structure

```text
mappingmind/
  api/
    main.py
  core/
    agent.py
    audit.py
    cache.py
    evaluation.py
    generation.py
    guardrails.py
    ingest.py
    retrieval.py
  dashboard/
    app.py
  data/
    mappings.csv
  docs/
    images/
      mappingmind_architecture.png
      agentic_rag_workflow.png
      retrieval_pipeline.png
      ingestion_pipeline.png
      production_hardening_roadmap.png
  .env.example
  .gitignore
  requirements.txt
  README.md
```

Generated local files are intentionally ignored by Git:

```text
data/chroma_store/
data/chunks_cache.pkl
data/eval_results/
data/semantic_cache.json
logs/audit.jsonl
```

---

## API Endpoint Reference

| Endpoint  | Method | Purpose                                 |
| --------- | ------ | --------------------------------------- |
| `/`       | GET    | Root service information                |
| `/health` | GET    | Health check                            |
| `/docs`   | GET    | FastAPI interactive documentation       |
| `/ask`    | POST   | Ask MappingMind a data-mapping question |

Example `/ask` request body:

```json
{
  "query": "How should we map customer account balance from CoreBanking?",
  "user_id": "demo_user",
  "system_filter": "CoreBanking"
}
```

Example `/ask` response body:

```json
{
  "answer": "Recommended mapping with transformation logic...",
  "sources": ["mappings.csv", "ADR-001"],
  "agent_trace": ["guard", "retrieve", "grade", "generate", "validate"],
  "cache_hit": false,
  "decision": "answered"
}
```

---

## Essential Commands

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create env file
copy .env.example .env

# Run ingestion
python core/ingest.py

# Run retrieval
python core/retrieval.py

# Run API
uvicorn api.main:app --reload

# Run dashboard
streamlit run dashboard/app.py

# Run evaluation
python core/evaluation.py

# Check Git status
git status
```

---

## Target Audience

This project is designed for:

* AI engineers learning production RAG
* Data engineers learning GenAI architecture
* Solution architects learning Agentic RAG
* Product engineers building enterprise AI assistants
* Technical leaders evaluating practical GenAI system design
* Learners who want a weekend-buildable but serious GenAI portfolio project

---

## Troubleshooting

| Problem                              | Likely Cause                                   | Fix                                                           |
| ------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------- |
| Gemini call fails                    | Missing or invalid `GOOGLE_API_KEY`            | Update `.env` with a valid key                                |
| `.env` appears in GitHub             | File was tracked before `.gitignore` was fixed | Run `git rm --cached --sparse .env` and push                  |
| ChromaDB has stale data              | Old local vector store                         | Delete `data/chroma_store/` and rerun ingestion               |
| Streamlit cannot import core modules | Wrong working directory or missing path setup  | Run from repo root and ensure dashboard adds root to path     |
| FastAPI does not start               | Missing dependencies or import error           | Run `pip install -r requirements.txt` and check console error |
| Evaluation gives weak results        | Knowledge base not ingested                    | Run `python core/ingest.py` first                             |
| Prompt-injection test is not blocked | Guardrail patterns need update                 | Extend `core/guardrails.py` patterns                          |
| Cache does not hit                   | Similarity threshold too high                  | Lower `SEMANTIC_CACHE_THRESHOLD` in `.env`                    |

---

## Cost Structure

MappingMind is designed to run with minimal cost.

| Component                        | Cost                      |
| -------------------------------- | ------------------------- |
| Python                           | Free                      |
| FastAPI                          | Free                      |
| Streamlit local app              | Free                      |
| ChromaDB local vector store      | Free                      |
| Sentence Transformers embeddings | Free locally              |
| BM25 retrieval                   | Free                      |
| Cross-Encoder reranking          | Free locally              |
| MMR                              | Free locally              |
| Semantic cache                   | Free locally              |
| JSONL audit log                  | Free locally              |
| Gemini API                       | Depends on API usage      |
| LangSmith                        | Optional; depends on plan |
| GitHub hosting                   | Free for public repo      |

Recommended low-cost usage:

* Use small sample documents during development
* Keep top-k retrieval small
* Use reranking only on shortlisted chunks
* Use semantic cache for repeated questions
* Track token usage when moving beyond demo mode

---

## Security Notes

* Do not commit `.env`
* Use `.env.example` for placeholders only
* Rotate API keys if they were ever committed
* Keep vector stores, cache files, logs, and generated files out of Git
* Add role-based filtering before using real enterprise data
* Upgrade local audit/cache stores to managed services for enterprise deployment

Recommended `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example

venv/
.venv/
__pycache__/
*.pyc

data/chroma_store/
data/chunks_cache.pkl
data/eval_results/
data/semantic_cache.json
*.pkl

logs/
```

---

## Known Limitations

MappingMind is a portfolio-grade production RAG prototype, not a full enterprise deployment.

Current limitations:

* Uses sample data, not a real enterprise mapping repository
* Prompt-injection defense is rule-based and can be strengthened
* Audit logging is local JSONL and can be upgraded to a managed database or observability platform
* Semantic cache is local-file based and can be upgraded to Redis or a managed cache
* Multi-tenant access control is not yet implemented
* Full cost and latency trend dashboard can be expanded
* MCP and multi-agent workflows are intentionally out of scope for this phase

---

## Ready to Start Your AI Journey?

MappingMind helps you learn the practical implementation patterns behind modern Production Agentic RAG systems:

* keyword search first
* semantic retrieval second
* hybrid search for production quality
* reranking for precision
* MMR for diverse context
* grounded generation for trustworthy answers
* guardrails for safety
* semantic cache for latency and cost reduction
* audit logs for governance
* evaluation for regression checks
* dashboard visibility for production-style debugging

Start with Week 1, build each layer step by step, and by Week 7 you will have a serious enterprise-grade GenAI portfolio project.

> Build the system. Understand the trade-offs. Explain the architecture.
