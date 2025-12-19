# Backend/logger.py
from Backend.config import RAG_LOG
from pathlib import Path
import time
import json

# Ensure results directory exists
RAG_LOG.parent.mkdir(parents=True, exist_ok=True)

def log_query_answer(question: str, answer: str, sources: list = None, meta: dict = None):
    """
    Append an auditable entry to results/results.txt in newline-delimited JSON format.
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": [
            {"source": s.metadata.get("source"), "page": s.metadata.get("page", "N/A")} for s in sources
        ] if sources else [],
        "meta": meta or {}
    }
    with open(RAG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
