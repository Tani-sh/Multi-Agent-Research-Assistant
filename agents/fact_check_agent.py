"""
FactCheckAgent — cross-validates claims against sources.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState


FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a rigorous fact-checking agent. Given a set of factual claims and the original 
source material, cross-validate each claim against the sources.

For each claim, assign one of these statuses:
- "confirmed" — the claim is directly supported by multiple sources
- "conflicting" — sources provide contradictory information about this claim
- "unverified" — insufficient evidence in the sources to verify this claim

Return your response as JSON:
{{
    "verified_claims": [
        {{
            "statement": "The original claim text",
            "status": "confirmed|conflicting|unverified",
            "evidence": "Brief explanation citing which sources support or contradict this"
        }},
        ...
    ],
    "fact_check_summary": "Overall assessment of claim reliability (2-3 sentences)"
}}"""),
    ("human", """Claims to verify:
{claims_text}

Original sources:
{sources_text}

Cross-validate each claim and return the JSON result."""),
])


def fact_check_agent(state: ResearchState) -> dict:
    """
    Cross-validate extracted claims against the original sources.

    Each claim is tagged as confirmed, conflicting, or unverified
    with supporting evidence.
    """
    errors = list(state.get("errors", []))

    # Format claims
    claims_text = ""
    for i, claim in enumerate(state.get("key_claims", []), 1):
        stmt = claim.get("statement", str(claim))
        claims_text += f"{i}. {stmt}\n"

    # Format sources
    sources_text = ""
    for i, src in enumerate(state["sources"], 1):
        sources_text += f"\n[Source {i}] {src['title']}\n"
        sources_text += f"URL: {src['url']}\n"
        sources_text += f"{src['snippet']}\n"

    if not claims_text.strip():
        return {
            "verified_claims": [],
            "fact_check_summary": "No claims to verify.",
            "errors": errors,
        }

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2048,
        )

        chain = FACT_CHECK_PROMPT | llm
        response = chain.invoke({
            "claims_text": claims_text,
            "sources_text": sources_text,
        })

        # Parse JSON
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        verified_claims = parsed.get("verified_claims", [])
        fact_check_summary = parsed.get("fact_check_summary", "")

        # Count statuses
        confirmed = sum(1 for c in verified_claims if c.get("status") == "confirmed")
        conflicting = sum(1 for c in verified_claims if c.get("status") == "conflicting")
        unverified = sum(1 for c in verified_claims if c.get("status") == "unverified")

        print(f"[FactCheckAgent] Verified {len(verified_claims)} claims: "
              f"✓ {confirmed} confirmed, ⚠ {conflicting} conflicting, "
              f"? {unverified} unverified")

    except json.JSONDecodeError:
        verified_claims = [
            {"statement": c.get("statement", str(c)), "status": "unverified", "evidence": "Parse error"}
            for c in state.get("key_claims", [])
        ]
        fact_check_summary = "Fact-check parsing encountered an error."
        errors.append("FactCheckAgent: JSON parse error")
        print("[FactCheckAgent] Warning: JSON parse failed")

    except Exception as e:
        verified_claims = []
        fact_check_summary = f"Fact-check error: {str(e)}"
        errors.append(f"FactCheckAgent error: {str(e)}")
        print(f"[FactCheckAgent] Error: {e}")

    return {
        "verified_claims": verified_claims,
        "fact_check_summary": fact_check_summary,
        "errors": errors,
    }
