# Backend/embeddings_client.py
"""
Single source of truth for the embedding model.
All modules import get_embedder() from here — no duplicate instances.
"""
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from Backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings singleton.
    Thread-safe for read-only use (embeddings are stateless after init).
    Only one copy of the model is ever loaded in memory.
    """
    global _embedder
    if _embedder is None:
        logger.info("Initializing HuggingFaceEmbeddings with model: %s", EMBEDDING_MODEL)
        try:
            _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info("Embedder initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize HuggingFaceEmbeddings: %s", e)
            raise
    return _embedder


def reset_embedder() -> None:
    """Force re-initialization on next call (useful for testing)."""
    global _embedder
    _embedder = None
