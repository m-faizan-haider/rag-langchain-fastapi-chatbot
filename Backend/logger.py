# Backend/logger.py
"""
Query audit logger — appends one JSON line per query to results/results.txt.
All modules should use: from Backend.logger import log_query_answer
"""
import json
import logging
import time
from pathlib import Path

from Backend.config import RAG_LOG

# Ensure results directory exists at import time
RAG_LOG.parent.mkdir(parents=True, exist_ok=True)

_logger = logging.getLogger(__name__)


def log_query_answer(
    question: str,
    answer:   str,
    sources:  list | None = None,
    meta:     dict | None = None,
) -> None:
    """
    Append an auditable JSONL entry to results/results.txt.
    sources can be a list of Document objects OR plain dicts.
    Non-fatal: logs a warning if write fails rather than crashing the request.
    """
    def _serialize_source(s) -> dict:
        if isinstance(s, dict):
            return s
        # LangChain Document object
        if hasattr(s, "metadata"):
            return {
                "source": s.metadata.get("source"),
                "page":   s.metadata.get("page", "N/A"),
            }
        return {"source": str(s)}

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question":  question,
        "answer":    answer,
        "sources":   [_serialize_source(s) for s in (sources or [])],
        "meta":      meta or {},
    }

    try:
        with open(RAG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        _logger.warning("Failed to write query log entry: %s", e)
