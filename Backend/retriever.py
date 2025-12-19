# Backend/retriever.py
import json
from pathlib import Path
from typing import Tuple
from langchain_community.vectorstores import FAISS

from .config import INDEX_DIR, EMBEDDING_MODEL
from .embeddings_client import get_embedder
from .faiss_manager import build_or_update_faiss_index, check_index

def load_vectorstore_and_check() -> Tuple[FAISS, dict]:
    """
    Ensure FAISS index exists (based on index files) and load the vectorstore.
    This version does NOT require metadata.json; it uses the embedding model from config.
    Returns (vectorstore, meta) where meta contains at least 'embedding_model'.
    """
    # If index not present according to faiss_manager's check, attempt rebuild
    if not check_index():
        print("⚠️ FAISS index missing — rebuilding now...")
        build_or_update_faiss_index()

    # After rebuild (or if present), verify index files exist
    if not INDEX_DIR.exists():
        raise RuntimeError(f"FAISS index directory missing after rebuild: {INDEX_DIR}")

    # Initialize embedder from config (guarantees consistent embedding model)
    try:
        embedder = get_embedder()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedder: {e}")

    # Load the FAISS vectorstore
    try:
        vs = FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS vectorstore from {INDEX_DIR}: {e}")

    meta = {"embedding_model": EMBEDDING_MODEL}
    print(f"ℹ️ Using FAISS index with {meta['embedding_model']}")
    return vs, meta


import os
import pickle
import time
import re
from pathlib import Path
from typing import Dict, List
import numpy as np

from Backend.config import EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings

CACHE_DIR = Path(__file__).resolve().parent / "faiss_verify_cache"
CACHE_FILE = CACHE_DIR / "sentence_embeddings_cache.pkl"

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder

def _split_sentences(text: str) -> List[str]:
    # simple sentence splitter (works well enough for English)
    return [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]

def build_sentence_embedding_cache(picked_docs: List[object], force_rebuild: bool = False):
    """
    picked_docs: list of LangChain Document objects (or similar) with metadata['source'] and .page_content
    This function builds a cache mapping filename -> {'sents': [sent_texts], 'embs': np.array([...])}
    and saves it to CACHE_FILE for reuse.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    if CACHE_FILE.exists() and not force_rebuild:
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            return cache
        except Exception:
            pass

    embedder = _get_embedder()
    cache: Dict[str, Dict] = {}
    # group by filename
    grouped = {}
    for doc in picked_docs:
        src = doc.metadata.get("source") or ""
        fname = Path(src).name
        grouped.setdefault(fname, []).append(doc.page_content.replace("\n", " ").strip())

    for fname, chunks in grouped.items():
        joined = " ".join(chunks)
        sents = _split_sentences(joined)
        # embed sentences in batches
        embs = []
        batch_size = 32
        for i in range(0, len(sents), batch_size):
            batch = sents[i:i+batch_size]
            # use embedder.embed_documents (wrapper)
            try:
                batch_embs = embedder.embed_documents(batch)
            except Exception:
                # fall back to embed_query per sentence
                batch_embs = [embedder.embed_query(b) for b in batch]
            embs.extend(batch_embs)
        if embs:
            cache[fname] = {"sents": sents, "embs": np.array(embs)}
        else:
            cache[fname] = {"sents": sents, "embs": np.zeros((0, 1))}

    # save to disk
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    return cache

def load_sentence_embedding_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def clear_sentence_embedding_cache():
    if CACHE_FILE.exists():
        os.remove(CACHE_FILE)
