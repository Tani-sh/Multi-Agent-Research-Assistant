"""
Streamlit UI for the Multi-Agent Research Assistant.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
from datetime import datetime
from graph import run_research


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .agent-search { border-color: #89b4fa; }
    .agent-summarize { border-color: #a6e3a1; }
    .agent-factcheck { border-color: #fab387; }
    .agent-report { border-color: #cba6f7; }
    .status-confirmed { color: #a6e3a1; }
    .status-conflicting { color: #fab387; }
    .status-unverified { color: #9399b2; }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Multi-Agent Research Assistant</h1>
        <p>Autonomous web research powered by LangGraph + Groq</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get your free API key at console.groq.com",
        )

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        st.divider()
        st.markdown("""
        ### 🤖 Pipeline Agents
        1. **🔍 SearchAgent** — DuckDuckGo web search
        2. **📝 SummarizeAgent** — LLM summarisation
        3. **✅ FactCheckAgent** — Cross-validation
        4. **📄 ReportAgent** — Markdown report
        """)

        st.divider()
        st.caption("Built with LangGraph • LangChain • Groq • Streamlit")

    # ── Input ─────────────────────────────────────────────────────────────
    query = st.text_input(
        "🔍 Enter your research query",
        placeholder="e.g., What are the latest breakthroughs in quantum computing?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_clicked = st.button("🚀 Research", type="primary", use_container_width=True)

    # ── Run Pipeline ──────────────────────────────────────────────────────
    if run_clicked and query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ Please enter your Groq API key in the sidebar.")
            return

        # Progress tracking
        progress = st.progress(0, text="Initialising pipeline...")

        with st.spinner("Running multi-agent pipeline..."):
            # Run the LangGraph pipeline
            progress.progress(10, text="🔍 SearchAgent: Searching the web...")

            result = run_research(query)

            progress.progress(100, text="✅ Pipeline complete!")

        # ── Display Results ───────────────────────────────────────────────
        st.divider()

        # Agent Status Cards
        st.subheader("🤖 Agent Pipeline Results")

        col_s, col_su, col_f, col_r = st.columns(4)

        with col_s:
            n_sources = len(result.get("sources", []))
            st.metric("🔍 Search", f"{n_sources} sources")

        with col_su:
            n_claims = len(result.get("key_claims", []))
            st.metric("📝 Summary", f"{n_claims} claims")

        with col_f:
            verified = result.get("verified_claims", [])
            confirmed = sum(1 for c in verified if c.get("status") == "confirmed")
            st.metric("✅ FactCheck", f"{confirmed}/{len(verified)} confirmed")

        with col_r:
            report_len = len(result.get("report", ""))
            st.metric("📄 Report", f"{report_len} chars")

        st.divider()

        # Tabs for detailed results
        tab_report, tab_sources, tab_claims, tab_raw = st.tabs([
            "📄 Report", "🔍 Sources", "✅ Fact-Check", "🗂️ Raw Data",
        ])

        with tab_report:
            report = result.get("report", "No report generated.")
            st.markdown(report)

            # One-click Markdown export
            st.divider()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.md"
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True,
            )

        with tab_sources:
            for i, src in enumerate(result.get("sources", []), 1):
                with st.expander(f"Source {i}: {src.get('title', 'Untitled')}", expanded=False):
                    st.markdown(f"**URL:** [{src.get('url', '')}]({src.get('url', '')})")
                    st.markdown(f"**Snippet:** {src.get('snippet', 'N/A')}")

        with tab_claims:
            st.markdown(f"**Overall Assessment:** {result.get('fact_check_summary', 'N/A')}")
            st.divider()

            for claim in result.get("verified_claims", []):
                status = claim.get("status", "unverified")
                icon = {"confirmed": "✅", "conflicting": "⚠️", "unverified": "❓"}.get(status, "❓")
                color = {"confirmed": "green", "conflicting": "orange", "unverified": "gray"}.get(status, "gray")

                st.markdown(
                    f"{icon} **[{status.upper()}]** {claim.get('statement', '')}"
                )
                st.caption(f"Evidence: {claim.get('evidence', 'N/A')}")
                st.divider()

        with tab_raw:
            st.json({
                "query": result.get("query"),
                "search_status": result.get("search_status"),
                "n_sources": len(result.get("sources", [])),
                "n_claims": len(result.get("key_claims", [])),
                "n_verified": len(result.get("verified_claims", [])),
                "errors": result.get("errors", []),
            })

        # Show errors if any
        if result.get("errors"):
            with st.expander("⚠️ Pipeline Errors", expanded=False):
                for err in result["errors"]:
                    st.warning(err)

    elif run_clicked and not query:
        st.warning("Please enter a research query.")


if __name__ == "__main__":
    main()
