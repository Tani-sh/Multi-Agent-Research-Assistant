"""
LangGraph pipeline — orchestrates the 4-agent research workflow.

DAG:  Search → Summarize → FactCheck → Report
"""

from langgraph.graph import StateGraph, END
from state import ResearchState
from agents.search_agent import search_agent
from agents.summarize_agent import summarize_agent
from agents.fact_check_agent import fact_check_agent
from agents.report_agent import report_agent


def build_graph() -> StateGraph:
    """
    Build the LangGraph state graph with the 4-agent pipeline.

    Flow:
        START → SearchAgent → SummarizeAgent → FactCheckAgent → ReportAgent → END
    """
    graph = StateGraph(ResearchState)

    # ── Add agent nodes ───────────────────────────────────────────────────
    graph.add_node("search", search_agent)
    graph.add_node("summarize", summarize_agent)
    graph.add_node("fact_check", fact_check_agent)
    graph.add_node("report", report_agent)

    # ── Define edges (DAG) ────────────────────────────────────────────────
    graph.set_entry_point("search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "fact_check")
    graph.add_edge("fact_check", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_research(query: str) -> dict:
    """
    Run the full research pipeline for a given query.

    Args:
        query: The research question to investigate.

    Returns:
        Final ResearchState with all agent outputs.
    """
    graph = build_graph()

    initial_state: ResearchState = {
        "query": query,
        "sources": [],
        "search_status": "pending",
        "summary": "",
        "key_claims": [],
        "verified_claims": [],
        "fact_check_summary": "",
        "report": "",
        "report_title": "",
        "errors": [],
    }

    print(f"\n{'═' * 60}")
    print(f"  Multi-Agent Research Assistant")
    print(f"  Query: {query}")
    print(f"{'═' * 60}\n")

    result = graph.invoke(initial_state)

    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete!")
    print(f"  Sources: {len(result.get('sources', []))}")
    print(f"  Claims verified: {len(result.get('verified_claims', []))}")
    print(f"  Report length: {len(result.get('report', ''))} chars")
    if result.get("errors"):
        print(f"  Errors: {len(result['errors'])}")
    print(f"{'═' * 60}\n")

    return result


if __name__ == "__main__":
    import os

    if not os.getenv("GROQ_API_KEY"):
        print("[!] Set GROQ_API_KEY environment variable first.")
        print("    export GROQ_API_KEY=your_key_here")
        exit(1)

    query = input("Enter your research query: ").strip()
    if not query:
        query = "What are the latest breakthroughs in quantum computing in 2024?"

    result = run_research(query)
    print("\n" + result.get("report", "No report generated."))
