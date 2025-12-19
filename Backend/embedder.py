from Backend.config import EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"⚡ Initializing HuggingFaceEmbeddings with model: {EMBEDDING_MODEL}")
        _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder

# Test
if __name__ == "__main__":
    print("🔍 Testing embedder.py ...")
    emb = get_embedder()
    vec = emb.embed_query("Test")
    print(f"✅ Vector length = {len(vec)}")
