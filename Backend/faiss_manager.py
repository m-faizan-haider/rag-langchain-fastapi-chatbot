# Backend/faiss_manager.py
import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
#from config import INDEX_DIR, DATA_DIR, EMBEDDING_MODEL
from Backend.config import INDEX_DIR, DATA_DIR, EMBEDDING_MODEL
from Backend.document_loader import load_documents
from Backend.chunker import split_documents
from Backend.embeddings_client import get_embedder


def _remove_index_if_exists():
    if INDEX_DIR.exists():
        print("🧹 Removing existing FAISS index...")
        shutil.rmtree(INDEX_DIR)


def build_or_update_faiss_index(force_rebuild: bool = False):
    """
    Build a FAISS index from documents in DATA_DIR.
    Only produces index.faiss + index.pkl (no metadata).
    """
    if not DATA_DIR.exists():
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    if INDEX_DIR.exists() and not force_rebuild:
        print("✅ FAISS index already exists. Skipping rebuild.")
        return

    print("🔍 Loading documents from:", DATA_DIR)
    docs = load_documents()
    if not docs:
        raise RuntimeError("❌ No documents found in data/sample_docs. Put your files there.")

    print(f"📚 Loaded {len(docs)} documents. Splitting into chunks...")
    splits = split_documents(docs)
    print(f"✂️ Created {len(splits)} chunks.")

    print("⚡ Creating embeddings...")
    embedder = get_embedder()

    print("⚡ Building FAISS index...")
    vectorstore = FAISS.from_documents(splits, embedder)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print("✅ FAISS index built and saved (index.faiss + index.pkl only).")


def check_index() -> bool:
    """
    Check if FAISS index files exist (index.faiss + index.pkl).
    """
    if not INDEX_DIR.exists():
        return False

    idx_faiss = INDEX_DIR / "index.faiss"
    idx_pkl = INDEX_DIR / "index.pkl"

    return idx_faiss.exists() and idx_pkl.exists()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check FAISS index status")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS index")
    args = parser.parse_args()

    if args.check:
        if check_index():
            print("✅ FAISS index exists (index.faiss + index.pkl).")
        else:
            print("❌ FAISS index missing.")
    if args.rebuild:
        build_or_update_faiss_index(force_rebuild=True)
    if not args.check and not args.rebuild:
        build_or_update_faiss_index()
