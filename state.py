"""
Typed state schema shared across all agents in the LangGraph DAG.
"""

from typing import TypedDict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Source:
    """A single web source retrieved during research."""
    title: str
    url: str
    snippet: str
    content: str = ""


@dataclass
class Claim:
    """A factual claim extracted during summarisation."""
    statement: str
    status: str = "unverified"  # confirmed | conflicting | unverified
    evidence: str = ""


class ResearchState(TypedDict):
    """
    Typed state shared across the 4-agent LangGraph pipeline.

    Flow:  Search → Summarize → FactCheck → Report
    """
    # ── User input ────────────────────────────────────────────────────────
    query: str

    # ── SearchAgent output ────────────────────────────────────────────────
    sources: List[dict]            # raw search results (title, url, snippet, content)
    search_status: str             # "pending" | "completed" | "error"

    # ── SummarizeAgent output ─────────────────────────────────────────────
    summary: str                   # consolidated summary of all sources
    key_claims: List[dict]         # extracted factual claims

    # ── FactCheckAgent output ─────────────────────────────────────────────
    verified_claims: List[dict]    # claims with status: confirmed/conflicting/unverified
    fact_check_summary: str        # overall fact-check assessment

    # ── ReportAgent output ────────────────────────────────────────────────
    report: str                    # final Markdown report
    report_title: str              # generated title for the report

    # ── Metadata ──────────────────────────────────────────────────────────
    errors: List[str]              # any errors encountered during pipeline
