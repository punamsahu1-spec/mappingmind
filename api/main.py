from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from core.agent import ask_agent
from core.audit import write_audit_event
from core.cache import get_cached_answer, save_answer_to_cache

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
    cached_result = get_cached_answer(request.query)

    if cached_result:
        decision = "cache_hit"

        write_audit_event(
            {
                "user_id": request.user_id,
                "query": request.query,
                "system_filter": request.system_filter,
                "answer": cached_result.get("answer", ""),
                "sources": cached_result.get("sources", []),
                "agent_trace": cached_result.get("agent_trace", []),
                "hallucination_score": cached_result.get("hallucination_score", None),
                "rewrites": cached_result.get("rewrites", 0),
                "cache_hit": True,
                "cache_similarity": cached_result.get("cache_similarity"),
                "cached_from_query": cached_result.get("cached_from_query"),
                "decision": decision,
            }
        )

        return {
            **cached_result,
            "decision": decision,
        }

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

    result = {
        **result,
        "cache_hit": False,
        "decision": decision,
    }

    save_answer_to_cache(request.query, result)

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
            "cache_hit": False,
            "decision": decision,
        }
    )

    return result