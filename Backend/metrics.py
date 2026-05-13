# Backend/metrics.py
"""
Prometheus metrics for the RAG API.
Usage: mount on FastAPI app in api.py startup.

Tracked metrics:
  - rag_query_total          (counter)  — total queries by status
  - rag_query_duration_seconds (histogram) — end-to-end latency
  - rag_cache_hits_total     (counter)  — semantic cache hits
  - rag_retrieval_score      (histogram) — distribution of top-1 retrieval scores
  - rag_llm_errors_total     (counter)  — LLM call failures
"""
import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

# ── Prometheus client (optional dependency) ────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics endpoint disabled.")


if _PROMETHEUS_AVAILABLE:
    # ── Metric definitions ─────────────────────────────────────────────────────
    QUERY_COUNTER = Counter(
        "rag_query_total",
        "Total RAG queries",
        ["status"],          # labels: success | error | cache_hit
    )
    QUERY_DURATION = Histogram(
        "rag_query_duration_seconds",
        "End-to-end query latency",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
    CACHE_HIT_COUNTER = Counter(
        "rag_cache_hits_total",
        "Semantic cache hits",
    )
    RETRIEVAL_SCORE = Histogram(
        "rag_retrieval_score",
        "Top-1 retrieval score distribution",
        buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    LLM_ERROR_COUNTER = Counter(
        "rag_llm_errors_total",
        "LLM call failures",
        ["provider"],        # labels: openrouter | ollama
    )


# ── Public helpers (no-op if prometheus not installed) ─────────────────────────

def record_query(status: str, duration_s: float) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    QUERY_COUNTER.labels(status=status).inc()
    QUERY_DURATION.observe(duration_s)


def record_cache_hit() -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    CACHE_HIT_COUNTER.inc()
    QUERY_COUNTER.labels(status="cache_hit").inc()


def record_retrieval_score(score: float) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    RETRIEVAL_SCORE.observe(score)


def record_llm_error(provider: str) -> None:
    if not _PROMETHEUS_AVAILABLE:
        return
    LLM_ERROR_COUNTER.labels(provider=provider).inc()


# ── FastAPI route handler ──────────────────────────────────────────────────────

def metrics_endpoint():
    """Returns Prometheus metrics as plain text. Mount at GET /metrics."""
    if not _PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed\n", 200
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
