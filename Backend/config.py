# Backend/config.py
from pathlib import Path
import os

# Project root is two levels up when this module is imported from main.py at repo root.
# But to be robust, resolve relative to this file and go up one level to repo root.
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent

# ----------------- IMMUTABLE PARAMETERS (DO NOT CHANGE) -----------------
# These are CEO-approved and must not be modified by adapters.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

EXTRACT_MAX_TOKENS = int(os.getenv("EXTRACT_MAX_TOKENS", 768))
SYNTH_MAX_TOKENS = int(os.getenv("SYNTH_MAX_TOKENS", 768))
# -----------------------------------------------------------------------

# Paths (expected top-level structure)
DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
INDEX_DIR = MODULE_ROOT / "faiss_index"
METADATA_FILE = INDEX_DIR / "metadata.json"

# Retrieval params (non-immutable tuning)
CANDIDATE_K = int(os.getenv("CANDIDATE_K", 50))
FINAL_K = int(os.getenv("FINAL_K", 8))
SCORE_LIMIT_CHARS = int(os.getenv("SCORE_LIMIT_CHARS", 2000))

# Cross-encoder model
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Logs folder (top-level 'results' as you specified)
RAG_LOG = PROJECT_ROOT / "results" / "results.txt"

# DeepSeek / router.ai
ROUTERAI_API_KEY = os.getenv("ROUTERAI_API_KEY")
DEEPSEEK_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek/deepseek-chat-v3.1:free")

# Fallback local HF model
FALLBACK_HF_MODEL = os.getenv("FALLBACK_HF_MODEL", "google/flan-t5-large")
