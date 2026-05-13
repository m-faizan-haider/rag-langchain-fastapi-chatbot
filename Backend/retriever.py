# Backend/retriever.py
"""
Vectorstore loader — wraps FAISS for backward compatibility during
the Phase 1 migration to Qdrant.  All embedder access goes through
the single embeddings_client singleton.
"""
import logging
from pathlib import Path
from typing import Tuple

from langchain_community.vectorstores import FAISS

from Backend.config import INDEX_DIR, EMBEDDING_MODEL
from Backend.embeddings_client import get_embedder          # ← single source of truth
from Backend.faiss_manager import build_or_update_faiss_index, check_index

logger = logging.getLogger(__name__)


def load_vectorstore_and_check() -> Tuple[FAISS, dict]:
    """
    Ensure FAISS index exists and load the vectorstore.
    Returns (vectorstore, meta) where meta contains 'embedding_model'.
    """
    if not check_index():
        logger.warning("FAISS index missing — rebuilding now...")
        build_or_update_faiss_index()

    if not INDEX_DIR.exists():
        raise RuntimeError(f"FAISS index directory missing after rebuild: {INDEX_DIR}")

    embedder = get_embedder()   # reuses the cached singleton — no second model load

    try:
        vs = FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS vectorstore from {INDEX_DIR}: {e}") from e

    meta = {"embedding_model": EMBEDDING_MODEL}
    logger.info("FAISS index loaded using model: %s", meta["embedding_model"])
    return vs, meta
