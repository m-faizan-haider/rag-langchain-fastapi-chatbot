# Backend/chunker.py
"""
Smart document chunker.
Strategy:
  1. Try SemanticChunker (embedding-based) — finds natural topic boundaries.
  2. Fall back to RecursiveCharacterTextSplitter — always works.
"""
import logging
from typing import List
from langchain.schema import Document
from Backend.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def split_documents(docs: List[Document], use_semantic: bool = True) -> List[Document]:
    """
    Split documents into chunks.
    - use_semantic=True: attempts SemanticChunker first (better quality)
    - Falls back to RecursiveCharacterTextSplitter automatically
    """
    if use_semantic:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_huggingface import HuggingFaceEmbeddings
            from Backend.embeddings_client import get_embedder

            logger.info("Using SemanticChunker with model: %s", EMBEDDING_MODEL)
            embedder = get_embedder()
            splitter = SemanticChunker(
                embeddings=embedder,
                breakpoint_threshold_type="percentile",   # split where similarity drops most
                breakpoint_threshold_amount=90,           # top 10% dissimilarity = split point
            )
            splits = splitter.split_documents(docs)
            logger.info("SemanticChunker created %d chunks from %d docs", len(splits), len(docs))
            return splits

        except ImportError:
            logger.warning("langchain-experimental not installed — falling back to RecursiveCharacterTextSplitter")
        except Exception as e:
            logger.warning("SemanticChunker failed (%s) — falling back to RecursiveCharacterTextSplitter", e)

    # ── Fallback: character-based splitter ────────────────────────────────────
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph > sentence > word splits
    )
    splits = splitter.split_documents(docs)
    logger.info("RecursiveCharacterTextSplitter created %d chunks from %d docs", len(splits), len(docs))
    return splits
