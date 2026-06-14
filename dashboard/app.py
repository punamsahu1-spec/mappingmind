import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st

from core.agent import ask_agent
from core.audit import read_recent_audit_events, write_audit_event
from core.cache import get_cached_answer, save_answer_to_cache

st.set_page_config(
    page_title="MappingMind",
    page_icon="🧠",
    layout="wide",
)


def decide_status(answer: str) -> str:
    answer_lower = answer.lower()

    if "requires sme review" in answer_lower or "sme review" in answer_lower:
        return "human_review"

    if "blocked" in answer_lower or "prompt injection" in answer_lower:
        return "blocked"

    return "answered"


def show_sources(sources):
    if not sources:
        st.info("No sources returned.")
        return

    for i, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            st.markdown(f"**Source {i}**")
            st.json(source)
        else:
            st.markdown(f"**Source {i}:** {source}")


def show_agent_trace(agent_trace):
    if not agent_trace:
        st.info("No agent trace returned.")
        return

    for step in agent_trace:
        st.write(f"✅ {step}")


def ask_mappingmind(query: str, user_id: str, system_filter: str | None):
    """
    Dashboard flow:
    semantic cache -> agentic RAG -> cache save -> audit log
    """

    cached_result = get_cached_answer(query)

    if cached_result:
        decision = "cache_hit"

        write_audit_event(
            {
                "user_id": user_id,
                "query": query,
                "system_filter": system_filter,
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

    result = ask_agent(query)

    answer = result.get("answer", "")
    decision = decide_status(answer)

    result = {
        **result,
        "cache_hit": False,
        "decision": decision,
    }

    save_answer_to_cache(query, result)

    write_audit_event(
        {
            "user_id": user_id,
            "query": query,
            "system_filter": system_filter,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "agent_trace": result.get("agent_trace", []),
            "hallucination_score": result.get("hallucination_score", None),
            "rewrites": result.get("rewrites", 0),
            "cache_hit": False,
            "decision": decision,
        }
    )

    return result

st.title("🧠 MappingMind")
st.caption("Enterprise Data Mapping Intelligence — Powered by Production Agentic RAG")

st.markdown(
    """
    MappingMind helps data teams answer enterprise mapping questions using hybrid search,
    reranking, grounded generation, guardrails, semantic cache, and audit logging.
    """
)

with st.sidebar:
    st.header("Demo Controls")

    user_id = st.text_input("User ID", value="demo_user")
    system_filter = st.text_input("System Filter", value="CoreBanking")

    st.divider()

    st.markdown("### Try these questions")

    st.code("How should we map customer account balance from CoreBanking?")
    st.code("What is the right mapping for customer account balance?")
    st.code("How do we standardize date fields?")
    st.code("What mapping patterns were rejected earlier?")

query = st.text_area(
    "Ask a mapping question",
    value="How should we map customer account balance from CoreBanking?",
    height=100,
)

if st.button("Ask MappingMind", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("MappingMind is thinking..."):
            result = ask_mappingmind(
                query=query,
                user_id=user_id,
                system_filter=system_filter,
            )

            st.subheader("Answer")
            st.write(result.get("answer", "No answer generated."))

            st.subheader("Production Signals")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Cache Hit",
                    value=str(result.get("cache_hit", False)),
                    help="Shows whether the answer came from semantic cache.",
                )

            with col2:
                st.metric(
                    label="Rewrites",
                    value=result.get("rewrites", 0),
                    help="Shows how many times the agent rewrote the query.",
                )

            with col3:
                st.metric(
                    label="Hallucination Score",
                    value=result.get("hallucination_score", "N/A"),
                    help="Lower risk means answer is more grounded in retrieved evidence.",
                )

            with col4:
                st.metric(
                    label="Decision",
                    value=result.get("decision", "answered"),
                    help="Final routing decision: answered, blocked, human_review, or cache_hit.",
                )

            if result.get("cache_similarity") is not None:
                st.info(
                    f"Semantic cache matched previous query with similarity "
                    f"{result.get('cache_similarity')}. "
                    f"Cached from: {result.get('cached_from_query')}"
                )

            st.subheader("Sources Used")
            show_sources(result.get("sources", []))

            st.subheader("Agent Trace")
            show_agent_trace(result.get("agent_trace", []))

            with st.expander("Raw Result JSON"):
                st.json(result)

st.divider()
st.subheader("Recent Audit Events")

recent_events = read_recent_audit_events(limit=5)

if recent_events:
    for event in reversed(recent_events):
        title = f"{event.get('decision', 'unknown')} | {event.get('query', '')[:80]}"
        with st.expander(title):
            st.json(event)
else:
    st.info("No audit events found yet. Ask a question first.")
