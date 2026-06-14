import json
import os
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer, util


CACHE_PATH = os.getenv("SEMANTIC_CACHE_PATH", "data/semantic_cache.json")
SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))

_model = SentenceTransformer("all-MiniLM-L6-v2")


def _load_cache() -> list[dict[str, Any]]:
    path = Path(CACHE_PATH)

    if not path.exists():
        return []

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_cache(cache: list[dict[str, Any]]) -> None:
    path = Path(CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def get_cached_answer(query: str) -> dict[str, Any] | None:
    """
    Return a cached answer if the new query is semantically similar
    to a previous query.
    """
    cache = _load_cache()

    if not cache:
        return None

    query_embedding = _model.encode(query, convert_to_tensor=True)

    best_match = None
    best_score = -1.0

    for item in cache:
        cached_query = item.get("query", "")
        cached_embedding = _model.encode(cached_query, convert_to_tensor=True)
        score = float(util.cos_sim(query_embedding, cached_embedding)[0][0])

        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= SIMILARITY_THRESHOLD:
        cached_result = best_match.get("result", {})
        return {
            **cached_result,
            "cache_hit": True,
            "cache_similarity": round(best_score, 4),
            "cached_from_query": best_match.get("query"),
        }

    return None


def save_answer_to_cache(query: str, result: dict[str, Any]) -> None:
    """
    Save answer to semantic cache.

    We avoid caching human-review or blocked responses.
    """
    answer = result.get("answer", "")

    if "sme review" in answer.lower():
        return

    if "blocked" in answer.lower() or "prompt injection" in answer.lower():
        return

    cache = _load_cache()

    cache.append(
        {
            "query": query,
            "result": {
                **result,
                "cache_hit": False,
            },
        }
    )

    _save_cache(cache)