# Backend/faiss_manager.py
import logging
import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from Backend.config import INDEX_DIR, DATA_DIR, EMBEDDING_MODEL
from Backend.document_loader import load_documents
from Backend.chunker import split_documents
from Backend.embeddings_client import get_embedder

logger = logging.getLogger(__name__)


def _remove_index_if_exists():
    if INDEX_DIR.exists():
        logger.info("Removing existing FAISS index...")
        shutil.rmtree(INDEX_DIR)


def build_or_update_faiss_index(force_rebuild: bool = False):
    """
    Build a FAISS index from documents in DATA_DIR.
    Produces index.faiss + index.pkl.
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    if INDEX_DIR.exists() and not force_rebuild:
        logger.info("FAISS index already exists. Skipping rebuild.")
        return

    logger.info("Loading documents from: %s", DATA_DIR)
    docs = load_documents()
    if not docs:
        raise RuntimeError("No documents found in data/sample_docs. Put your files there.")

    logger.info("Loaded %d documents. Splitting into chunks...", len(docs))
    splits = split_documents(docs)
    logger.info("Created %d chunks.", len(splits))

    logger.info("Creating embeddings and building FAISS index...")
    embedder     = get_embedder()
    vectorstore  = FAISS.from_documents(splits, embedder)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    logger.info("FAISS index saved to %s", INDEX_DIR)


def check_index() -> bool:
    """Check if FAISS index files exist (index.faiss + index.pkl)."""
    if not INDEX_DIR.exists():
        return False
    return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",   action="store_true", help="Check FAISS index status")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS index")
    args = parser.parse_args()

    if args.check:
        status = "exists" if check_index() else "missing"
        logger.info("FAISS index: %s", status)
    if args.rebuild:
        build_or_update_faiss_index(force_rebuild=True)
    if not args.check and not args.rebuild:
        build_or_update_faiss_index()
