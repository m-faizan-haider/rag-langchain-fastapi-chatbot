# check_faiss_contents.py
from Backend.config import INDEX_DIR, EMBEDDING_MODEL
from Backend.embeddings_client import get_embedder
from langchain_community.vectorstores import FAISS
from pathlib import Path

def main():
    print("INDEX_DIR =", INDEX_DIR)
    if not INDEX_DIR.exists():
        print("FAISS index folder missing:", INDEX_DIR)
        return
    try:
        embedder = get_embedder()
        vs = FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        print("Failed to load FAISS vectorstore:", e)
        return

    # Try to get number of vectors/chunks
    try:
        n = getattr(vs, "index").ntotal  # faiss index attribute
    except Exception:
        n = None

    print("FAISS loaded. embedding_model:", EMBEDDING_MODEL)
    print("Number of stored vectors (faiss index ntotal):", n)

    # Show a few sample docs
    try:
        res = vs.similarity_search("example test", k=3)
        print("Sample search (k=3) returned", len(res), "results.")
        for i, d in enumerate(res, 1):
            print("---- SAMPLE", i, "----")
            print("Source:", d.metadata.get("source"))
            print("Page:", d.metadata.get("page"))
            print(d.page_content[:400].replace("\n", " ") + "...")
    except Exception as e:
        print("Similarity search failed:", e)

if __name__ == "__main__":
    main()
