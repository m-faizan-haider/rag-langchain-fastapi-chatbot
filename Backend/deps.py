from typing import Tuple
from functools import lru_cache
import time
from Backend.retriever import load_vectorstore_and_check, build_or_update_faiss_index

@lru_cache(maxsize=1)
def get_vectorstore_and_meta() -> Tuple[object, dict]:
    """
    Loads and caches the vectorstore + metadata (loaded once per process).
    Callers should depend on this to get the FAISS vectorstore.
    """
    vs, meta = load_vectorstore_and_check()
    return vs, meta

def reload_vectorstore_and_meta(force_rebuild: bool = False):
    """
    Force rebuild (or reload) of FAISS index. Useful for admin endpoint.
    This invalidates the cached loader by clearing the LRU cache.
    """
    # If we want to rebuild indexes from documents:
    if force_rebuild:
        # call your existing function to rebuild index; it writes index files to disk
        build_or_update_faiss_index(force_rebuild=True)

    # clear cache and re-load
    get_vectorstore_and_meta.cache_clear()
    # warm the cache
    vs_meta = get_vectorstore_and_meta()
    return vs_meta
