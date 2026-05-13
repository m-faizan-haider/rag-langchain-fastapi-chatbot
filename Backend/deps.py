# Backend/deps.py
"""
FastAPI dependency injection for vectorstore access.
Uses application lifespan state instead of the thread-unsafe lru_cache pattern.
During the FAISS → Qdrant migration (Phase 1), this module bridges both.
"""
import logging
from typing import Tuple

from fastapi import Request

logger = logging.getLogger(__name__)


def get_vectorstore_and_meta(request: Request) -> Tuple[object, dict]:
    """
    Retrieve the vectorstore + metadata from FastAPI app state.
    The state is populated in the lifespan context manager in api.py.
    This is thread-safe: app.state is set once at startup, read-only afterwards.
    """
    vs   = request.app.state.vectorstore
    meta = request.app.state.vectorstore_meta
    if vs is None:
        raise RuntimeError("Vectorstore not initialised — check startup logs.")
    return vs, meta


def get_vectorstore_only(request: Request) -> object:
    """Shortcut for endpoints that only need the vectorstore."""
    return get_vectorstore_and_meta(request)[0]
