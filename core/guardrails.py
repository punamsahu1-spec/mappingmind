"""
MappingMind — Guardrails
=========================
CONCEPTS:
  - Prompt injection: user hijacks system prompt
  - Domain boundary: out of scope queries blocked
  - Confidence gate: low retrieval = don't generate
  - Hallucination check: answer grounded in context?
"""

import re


# ── INJECTION GUARD ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (previous|prior|all) instructions",
    r"forget (everything|all|previous)",
    r"you are now",
    r"act as (a|an|if)",
    r"pretend (you are|to be)",
    r"reveal (your|the) (system|prompt|instructions)",
    r"jailbreak",
]

def check_prompt_injection(query: str) -> tuple[bool, str]:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query.lower()):
            return True, f"Prompt injection detected."
    return False, ""


# ── DOMAIN GUARD ───────────────────────────────────────────────────
MAPPING_KEYWORDS = [
    "map", "mapping", "field", "column", "transform", "source",
    "target", "schema", "data", "integration", "migration", "etl",
    "pipeline", "format", "convert", "cast", "normalize", "adr",
    "decision", "rationale", "why", "how", "pattern", "system",
    "balance", "date", "currency", "code", "type", "id", "amount"
]

def check_domain_boundary(query: str) -> tuple[bool, str]:
    if any(kw in query.lower() for kw in MAPPING_KEYWORDS):
        return True, ""
    return False, "Query outside MappingMind scope. Ask about data mapping decisions."


# ── CONFIDENCE GATE ────────────────────────────────────────────────
def check_retrieval_confidence(results: list, threshold: float = -5.0) -> tuple[bool, str]:
    if not results:
        return False, "No results retrieved."
    top_score = max(score for _, score in results)
    if top_score < threshold:
        return False, f"Low retrieval confidence ({top_score:.3f}). Query may be out of scope."
    return True, f"Confidence: {top_score:.3f}"


# ── HALLUCINATION CHECK ────────────────────────────────────────────
def check_hallucination(answer: str, results: list) -> tuple[float, str]:
    if not results:
        return 0.0, "No context — answer ungrounded."

    context = " ".join([doc.page_content for doc, _ in results]).lower()
    words = set(answer.lower().split())
    stopwords = {"the", "a", "an", "is", "are", "was", "and", "or", "to", "for", "of", "in"}
    meaningful = words - stopwords

    if not meaningful:
        return 0.0, "Answer too short."

    grounded = sum(1 for w in meaningful if w in context)
    score = grounded / len(meaningful)

    if score < 0.3:
        return score, f"Low grounding ({score:.0%}) — verify answer."
    elif score < 0.6:
        return score, f"Medium grounding ({score:.0%}) — review sources."
    return score, f"High grounding ({score:.0%}) — well supported."