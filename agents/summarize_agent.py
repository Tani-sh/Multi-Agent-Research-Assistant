"""
SummarizeAgent — consolidates search results into a summary with key claims.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState


SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research summarisation expert. Given multiple web sources about a topic, 
create a clear, comprehensive summary and extract key factual claims.

Return your response as JSON with exactly this structure:
{{
    "summary": "A comprehensive 3-5 paragraph summary synthesising all sources...",
    "key_claims": [
        {{"statement": "Specific factual claim 1"}},
        {{"statement": "Specific factual claim 2"}},
        ...
    ]
}}

Extract 5-8 specific, verifiable factual claims from the sources. 
Each claim should be a concrete statement that can be fact-checked."""),
    ("human", """Research query: {query}

Sources:
{sources_text}

Provide your JSON summary and extracted claims."""),
])


def summarize_agent(state: ResearchState) -> dict:
    """
    Summarise all retrieved sources and extract key factual claims.

    Uses Groq LLM (Llama 3) to synthesise information from multiple
    sources into a coherent summary with verifiable claims.
    """
    errors = list(state.get("errors", []))

    # Format sources for the prompt
    sources_text = ""
    for i, src in enumerate(state["sources"], 1):
        sources_text += f"\n--- Source {i}: {src['title']} ---\n"
        sources_text += f"URL: {src['url']}\n"
        sources_text += f"Content: {src['snippet']}\n\n"

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2048,
        )

        chain = SUMMARIZE_PROMPT | llm
        response = chain.invoke({
            "query": state["query"],
            "sources_text": sources_text,
        })

        # Parse JSON response
        content = response.content.strip()
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        summary = parsed.get("summary", content)
        key_claims = parsed.get("key_claims", [])

        print(f"[SummarizeAgent] Generated summary ({len(summary)} chars), "
              f"extracted {len(key_claims)} claims")

    except json.JSONDecodeError:
        # If JSON parsing fails, use raw text as summary
        summary = response.content.strip()
        key_claims = []
        errors.append("SummarizeAgent: Failed to parse structured claims")
        print("[SummarizeAgent] Warning: using raw text (JSON parse failed)")

    except Exception as e:
        summary = "Error generating summary."
        key_claims = []
        errors.append(f"SummarizeAgent error: {str(e)}")
        print(f"[SummarizeAgent] Error: {e}")

    return {
        "summary": summary,
        "key_claims": key_claims,
        "errors": errors,
    }
