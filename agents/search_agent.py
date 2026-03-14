"""
SearchAgent — retrieves live web sources via DuckDuckGo.
"""

from duckduckgo_search import DDGS
from state import ResearchState


MAX_RESULTS = 6


def search_agent(state: ResearchState) -> dict:
    """
    Search the web for the user's query using DuckDuckGo.

    Retrieves up to 6 live sources per query, extracting title, URL,
    snippet, and full body text.
    """
    query = state["query"]
    sources = []
    errors = list(state.get("errors", []))

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))

        for r in results:
            sources.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
                "content": r.get("body", ""),
            })

        search_status = "completed"
        print(f"[SearchAgent] Retrieved {len(sources)} sources for: '{query}'")

    except Exception as e:
        search_status = "error"
        errors.append(f"SearchAgent error: {str(e)}")
        print(f"[SearchAgent] Error: {e}")

    return {
        "sources": sources,
        "search_status": search_status,
        "errors": errors,
    }
