from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from core.agent import ask_agent
from core.audit import write_audit_event

load_dotenv()

app = FastAPI(
    title="MappingMind API",
    description="Enterprise Data Mapping Intelligence powered by Production Agentic RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str
    user_id: str = "demo_user"
    system_filter: str | None = None


@app.get("/")
def root():
    return {
        "service": "MappingMind",
        "version": "1.0.0",
        "description": "Enterprise Data Mapping Intelligence powered by Agentic RAG",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MappingMind",
    }


@app.post("/ask")
def ask(request: AskRequest):
    """
    Ask MappingMind a data-mapping question.

    Flow:
    query -> guardrails -> hybrid retrieval -> reranking -> agentic validation -> answer
    """
    result = ask_agent(request.query)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    agent_trace = result.get("agent_trace", [])
    hallucination_score = result.get("hallucination_score", None)

    decision = "answered"

    if "requires sme review" in answer.lower() or "sme review" in answer.lower():
        decision = "human_review"

    if "blocked" in answer.lower() or "prompt injection" in answer.lower():
        decision = "blocked"

    write_audit_event(
        {
            "user_id": request.user_id,
            "query": request.query,
            "system_filter": request.system_filter,
            "answer": answer,
            "sources": sources,
            "agent_trace": agent_trace,
            "hallucination_score": hallucination_score,
            "rewrites": result.get("rewrites", 0),
            "cache_hit": result.get("cache_hit", False),
            "decision": decision,
        }
    )

    return {
        **result,
        "decision": decision,
    }