# Backend/cache.py
"""
Semantic query cache — skips the entire RAG pipeline for near-duplicate queries.
Primary backend: Redis. Fallback: in-memory dict.

How it works:
  1. Embed the incoming query.
  2. Compare cosine similarity against cached query embeddings.
  3. If similarity > threshold → return cached answer (zero LLM cost).
  4. Otherwise run the pipeline and store the result.
"""
import json
import logging
import time
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import norm

logger = logging.getLogger(__name__)

_CACHE_TTL          = 3600    # 1 hour
_SIM_THRESHOLD      = 0.95    # very high — only near-identical queries hit cache
_MAX_CACHE_ENTRIES  = 500     # keep memory bounded in in-memory fallback

# ── Storage ────────────────────────────────────────────────────────────────────
_mem_cache: list = []         # list of {query_emb, answer, sources, elapsed_s, ts}
_redis = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis as redis_lib
        from Backend.config import REDIS_URL
        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _redis.ping()
        logger.info("Semantic cache: Redis connected.")
    except Exception as e:
        logger.warning("Semantic cache: Redis unavailable, using in-memory: %s", e)
        _redis = None
    return _redis


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _get_embedder():
    from Backend.embeddings_client import get_embedder
    return get_embedder()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_cached_answer(query: str) -> Optional[dict]:
    """
    Return a cached result dict if a semantically similar query was seen before.
    Returns None if no cache hit.
    """
    try:
        embedder  = _get_embedder()
        query_emb = np.array(embedder.embed_query(query))
    except Exception as e:
        logger.warning("Cache lookup failed (embedder error): %s", e)
        return None

    # ── Check in-memory cache ──────────────────────────────────────────────────
    now = time.time()
    for entry in reversed(_mem_cache):
        if now - entry["ts"] > _CACHE_TTL:
            continue
        sim = _cosine(query_emb, np.array(entry["query_emb"]))
        if sim >= _SIM_THRESHOLD:
            logger.info("Semantic cache HIT (sim=%.3f) for: %s", sim, query[:60])
            return {"answer": entry["answer"], "sources": entry["sources"], "cache_hit": True, "similarity": sim}

    return None


def store_answer(query: str, answer: str, sources: list, elapsed_s: float) -> None:
    """Store a query→answer pair in the semantic cache."""
    try:
        embedder  = _get_embedder()
        query_emb = np.array(embedder.embed_query(query)).tolist()
    except Exception as e:
        logger.warning("Cache store failed (embedder error): %s", e)
        return

    entry = {
        "query_emb": query_emb,
        "answer":    answer,
        "sources":   sources,
        "elapsed_s": elapsed_s,
        "ts":        time.time(),
    }

    # Trim if too large
    if len(_mem_cache) >= _MAX_CACHE_ENTRIES:
        _mem_cache.pop(0)
    _mem_cache.append(entry)
    logger.debug("Cached answer for query: %s", query[:60])


def invalidate_cache() -> None:
    """Clear the entire semantic cache (call after index rebuild)."""
    _mem_cache.clear()
    logger.info("Semantic cache cleared.")
