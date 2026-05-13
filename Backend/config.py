# Backend/config.py
from pathlib import Path
import os

# Project root resolved relative to this file (robust regardless of cwd).
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent 

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ─── Token limits ─────────────────────────────────────────────────────────────
EXTRACT_MAX_TOKENS = int(os.getenv("EXTRACT_MAX_TOKENS", 768))
SYNTH_MAX_TOKENS   = int(os.getenv("SYNTH_MAX_TOKENS",   768))

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data" / "sample_docs"
INDEX_DIR     = MODULE_ROOT  / "faiss_index"
METADATA_FILE = INDEX_DIR    / "metadata.json"
RAG_LOG       = PROJECT_ROOT / "results" / "results.txt"

# ─── Retrieval ────────────────────────────────────────────────────────────────
CANDIDATE_K      = int(os.getenv("CANDIDATE_K",      20))   # was 50 — too slow
FINAL_K          = int(os.getenv("FINAL_K",           5))   # was 8
SCORE_LIMIT_CHARS = int(os.getenv("SCORE_LIMIT_CHARS", 2000))

# ─── Cross-encoder reranker ───────────────────────────────────────────────────
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ─── LLM: OpenRouter / DeepSeek ──────────────────────────────────────────────
ROUTERAI_API_KEY    = os.getenv("ROUTERAI_API_KEY")
DEEPSEEK_ENDPOINT   = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "google/gemini-flash-1.5")

# ─── LLM: Ollama (local fallback — replaces Flan-T5) ─────────────────────────
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

# ─── Auth (JWT) ───────────────────────────────────────────────────────────────
JWT_SECRET_KEY     = os.getenv("JWT_SECRET_KEY",     "CHANGE-ME-IN-PRODUCTION")
JWT_ALGORITHM      = os.getenv("JWT_ALGORITHM",      "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 1440))

# ─── Qdrant (Phase 1 migration target) ───────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT",   6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

# ─── Redis (sessions + semantic cache) ───────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ─── Fact verification ────────────────────────────────────────────────────────
STRICT_VERBATIM = os.getenv("STRICT_VERBATIM", "false").lower() == "true"
