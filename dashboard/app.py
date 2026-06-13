"""
MappingMind — Leadership Dashboard
====================================
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="MappingMind",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MappingMind")
st.caption("Enterprise Data Mapping Intelligence — Powered by Agentic RAG")

tab1, tab2, tab3 = st.tabs([
    "💬 Ask MappingMind",
    "📊 Analytics",
    "📚 Knowledge Base"
])

# ── TAB 1: QUERY ──────────────────────────────────────────────────
with tab1:
    st.header("Ask MappingMind")
    st.markdown("*Ask about any data mapping decision, transformation pattern, or past integration.*")

    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_area(
            "Your question:",
            value="How should we map customer_account_balance from CoreBanking system?",
            height=100
        )
        system_filter = st.selectbox(
            "Filter by system (optional):",
            ["All Systems", "CoreBanking_v1", "CoreBanking_v2",
             "CRM_Oracle", "CRM_SAP", "PaymentSystem_v2",
             "LoanSystem_v1", "HRMS_Oracle"]
        )

    with col2:
        st.subheader("💡 Example Questions")
        st.markdown("""
        - How to map balance fields from CoreBanking?
        - What is standard date transformation?
        - How have we handled currency normalization?
        - What mapping patterns should we avoid?
        - How to map customer ID fields?
        """)

    if st.button("🔍 Get Mapping Recommendation", type="primary"):

        if not os.path.exists("data/chunks_cache.pkl"):
            st.error("Run ingestion first: python -m core.ingest")
            st.stop()

        from core.generation import ask_mappingmind

        filter_val = None if system_filter == "All Systems" else system_filter

        with st.spinner("Searching mapping knowledge base..."):
            result = ask_mappingmind(query, system_filter=filter_val)

        guards = result["guardrails"]

        # Blocked queries
        if guards.get("injection", {}).get("blocked"):
            st.error(f"⛔ {result['answer']}")
            st.stop()

        if not guards.get("domain", {}).get("in_scope", True):
            st.warning(f"❓ {result['answer']}")
            st.stop()

        # Answer
        st.divider()
        st.subheader("📋 Mapping Recommendation")
        st.markdown(result["answer"])

        # Sources
        if result["sources"]:
            st.subheader("📎 Sources")
            cols = st.columns(len(result["sources"]))
            for i, source in enumerate(result["sources"]):
                cols[i].info(source)

        # Metrics
        st.divider()
        st.subheader("📊 Query Metrics")
        metrics = result["metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Latency", f"{metrics.get('latency_sec', 0)}s")
        m2.metric("Tokens In", metrics.get('tokens_in', 0))
        m3.metric("Tokens Out", metrics.get('tokens_out', 0))
        m4.metric("Sources Found", metrics.get('chunks_retrieved', 0))
        m5.metric("Grounding", f"{metrics.get('hallucination_score', 0):.0%}")

        # Guardrail status
        st.divider()
        st.subheader("🛡️ Guardrail Status")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Injection Guard", "✅ Passed")
        g2.metric("Domain Guard", "✅ In Scope")
        g3.metric("Confidence", 
                  "✅ Passed" if guards.get("confidence", {}).get("passed") else "⚠️ Low")
        hall_score = guards.get("hallucination", {}).get("score", 0)
        g4.metric("Hallucination", 
                  f"{'✅' if hall_score > 0.5 else '⚠️'} {hall_score:.0%}")

# ── TAB 2: ANALYTICS ──────────────────────────────────────────────
with tab2:
    st.header("📊 Business Impact Dashboard")
    st.caption("Projected metrics based on MappingMind adoption")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 ROI Summary")
        r1, r2 = st.columns(2)
        r1.metric("Time per Integration", "30 min", delta="-2.5 days")
        r2.metric("Annual Cost Saved", "$82,000", delta="-97%")
        r1.metric("Support Tickets/yr", "<5", delta="-75 tickets")
        r2.metric("Mapping Accuracy", "96%", delta="+18%")

    with col2:
        st.subheader("📈 Knowledge Base Stats")
        import pandas as pd
        import plotly.express as px

        stats = pd.DataFrame({
            "Source": ["CSV Mappings", "ADR Documents", "Total Chunks"],
            "Count": [20, 8, 28]
        })
        fig = px.bar(stats, x="Source", y="Count",
                    title="Knowledge Base Composition",
                    color="Source")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🏢 Systems Covered")
    systems = pd.DataFrame({
        "System": ["CoreBanking_v1", "CoreBanking_v2", "CRM_Oracle",
                   "CRM_SAP", "PaymentSystem_v2", "LoanSystem_v1", "HRMS_Oracle"],
        "Mappings": [4, 2, 4, 3, 3, 3, 1],
        "Status": ["✅ Active"] * 7
    })
    st.dataframe(systems, use_container_width=True)

# ── TAB 3: KNOWLEDGE BASE ─────────────────────────────────────────
with tab3:
    st.header("📚 Mapping Knowledge Base")
    st.caption("All ingested mapping records and ADRs")

    if os.path.exists("data/chunks_cache.pkl"):
        import pickle
        with open("data/chunks_cache.pkl", "rb") as f:
            chunks = pickle.load(f)

        mapping_chunks = [c for c in chunks if c.metadata.get("doc_type") == "mapping"]
        adr_chunks = [c for c in chunks if c.metadata.get("doc_type") == "adr"]

        st.metric("Total Chunks", len(chunks))

        st.subheader("📋 Mapping Records")
        import pandas as pd
        mapping_df = pd.DataFrame([{
            "Source Field": c.metadata.get("source_field", ""),
            "Target Field": c.metadata.get("target_field", ""),
            "System": c.metadata.get("system", ""),
            "Year": c.metadata.get("year", ""),
            "Status": c.metadata.get("status", "")
        } for c in mapping_chunks])
        st.dataframe(mapping_df, use_container_width=True)

        st.subheader("📜 ADR Documents")
        for chunk in adr_chunks:
            with st.expander(chunk.page_content[:60] + "..."):
                st.text(chunk.page_content)
    else:
        st.warning("Run ingestion first: python -m core.ingest")