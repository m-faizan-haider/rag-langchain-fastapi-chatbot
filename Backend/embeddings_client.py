# Backend/embeddings_client.py
from langchain_huggingface import HuggingFaceEmbeddings
from Backend.config import EMBEDDING_MODEL

def get_embedder():
    """
    Returns an initialized HuggingFaceEmbeddings instance using EMBEDDING_MODEL.
    """
    try:
        embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        print("❌ Failed to initialize HuggingFaceEmbeddings:", e)
        raise
    return embedder
