# Backend/api.py
"""
FastAPI application — production-ready.
Features: lifespan startup, JWT auth, semantic cache, session memory, rate limiting.
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from Backend.schemas import QueryRequest, QueryResponse, SourceItem, FactVerificationItem, TokenRequest, TokenResponse
from Backend.deps import get_vectorstore_and_meta
from Backend.rag_query_rerank import ask_question
from Backend.logger import log_query_answer
from Backend.logging_config import setup_logging
from Backend.auth import create_access_token, verify_api_key, verify_token
from Backend.session_manager import create_session, get_history, append_turn, format_history_for_prompt
from Backend.cache import get_cached_answer, store_answer, invalidate_cache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan — replaces thread-unsafe lru_cache
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup, before any requests are served.
    Stores shared objects in app.state for thread-safe access.
    """
    setup_logging()
    logger.info("RAG API starting up...")

    # Import here to avoid circular imports at module level
    from Backend.retriever import load_vectorstore_and_check
    from Backend.faiss_manager import check_index, build_or_update_faiss_index

    if not check_index():
        logger.warning("FAISS index missing — building now...")
        build_or_update_faiss_index()

    vs, meta = load_vectorstore_and_check()
    app.state.vectorstore      = vs
    app.state.vectorstore_meta = meta
    logger.info("Vectorstore loaded | model=%s", meta.get("embedding_model"))

    yield  # ← application runs here

    logger.info("RAG API shutting down...")
    app.state.vectorstore = None


# ─────────────────────────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG LangChain FastAPI Chatbot",
    version="0.2.0",
    description="Production-grade Retrieval-Augmented Generation API",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],   # was ["*"] — restricted to what we actually use
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health & status endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG LangChain FastAPI Chatbot is running!", "version": "0.2.0"}


@app.get("/health")
def health(request: Request):
    try:
        vs, meta = get_vectorstore_and_meta(request)
        return {
            "status":          "ok",
            "embedding_model": meta.get("embedding_model"),
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faiss_check")
def faiss_check(request: Request):
    vs, meta = get_vectorstore_and_meta(request)
    return {"ok": True, "embedding_model": meta.get("embedding_model")}


# ─────────────────────────────────────────────────────────────────────────────
# Auth endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse)
def get_token(req: TokenRequest):
    """Exchange a static API key for a JWT bearer token."""
    if not verify_api_key(req.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    token = create_access_token(subject="api_user")
    from Backend.config import JWT_EXPIRE_MINUTES
    return TokenResponse(access_token=token, expires_in=JWT_EXPIRE_MINUTES * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics endpoint (Prometheus)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
def metrics():
    """Prometheus metrics scraping endpoint."""
    from Backend.metrics import metrics_endpoint
    return metrics_endpoint()


# ─────────────────────────────────────────────────────────────────────────────
# Session history endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    """Return conversation history for a session (for debugging / UI restore)."""
    from Backend.session_manager import get_history
    history = get_history(session_id)
    return {"session_id": session_id, "turn_count": len(history) // 2, "history": history}


# ─────────────────────────────────────────────────────────────────────────────
# Admin: reload index  (protected)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reload_faiss")
def reload_faiss(
    request: Request,
    force_rebuild: bool = Body(False, embed=True),
    _user: str = Depends(verify_token),   # ← protected
):
    """Rebuild or reload the FAISS index. Requires valid JWT."""
    from Backend.faiss_manager import build_or_update_faiss_index
    from Backend.retriever import load_vectorstore_and_check

    try:
        start = time.time()
        if force_rebuild:
            logger.info("Force-rebuilding FAISS index...")
            build_or_update_faiss_index(force_rebuild=True)

        vs, meta = load_vectorstore_and_check()
        request.app.state.vectorstore      = vs
        request.app.state.vectorstore_meta = meta
        elapsed = time.time() - start
        logger.info("Index reload done in %.2fs", elapsed)
        return {"ok": True, "elapsed_s": elapsed, "embedding_model": meta.get("embedding_model")}
    except Exception as e:
        logger.error("reload_faiss failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest, request: Request):
    vs, meta = get_vectorstore_and_meta(request)
    start    = time.time()

    logger.info("Query received | question=%s | debug=%s | top_k=%s | session=%s",
                req.question[:80], req.debug, req.top_k, req.session_id)

    # ── Semantic cache check ──────────────────────────────────────────────────
    cached = get_cached_answer(req.question)
    if cached:
        elapsed = time.time() - start
        return QueryResponse(
            question=req.question,
            answer=cached["answer"],
            sources=cached.get("sources", []),
            elapsed_s=elapsed,
            session_id=req.session_id,
            cache_hit=True,
        )

    # ── Session history ───────────────────────────────────────────────────────
    session_id = req.session_id or create_session()
    history    = get_history(session_id)
    history_block = format_history_for_prompt(history) if history else ""

    try:
        # ── Call pipeline with top_k wired through (was previously ignored) ──
        # Prepend history block to query for multi-turn awareness
        augmented_query = f"{history_block}\n\nCurrent question: {req.question}".strip() if history_block else req.question

        result = ask_question(
            vs,
            augmented_query,
            show_sources=False,
            debug=req.debug,
            top_k=req.top_k,
        )

        if req.debug:
            final_text, picked, facts_text, verification = result
        else:
            final_text, picked = result
            facts_text   = ""
            verification = []

        # ── Build source list ─────────────────────────────────────────────────
        sources: List[SourceItem] = []
        for item in picked:
            if isinstance(item, tuple) and len(item) == 2:
                score, doc = item
            else:
                score, doc = None, item
            src   = doc.metadata.get("source") if hasattr(doc, "metadata") else None
            fname = Path(src).name if src else "unknown"
            page  = doc.metadata.get("page")  if hasattr(doc, "metadata") else None
            sources.append(SourceItem(
                filename=fname,
                page=int(page) if page not in (None, "N/A") else None,
                score=float(score) if score is not None else None,
            ))

        elapsed = time.time() - start

        # ── Verification items ────────────────────────────────────────────────
        verif_items: List[FactVerificationItem] = [
            FactVerificationItem(
                fact=v.get("fact", ""),
                tag_fname=v.get("tag_fname", ""),
                tag_page=v.get("tag_page"),
                similarity=float(v.get("similarity", 0.0)),
                verbatim_match=bool(v.get("verbatim_match", False)),
                matched_preview=(v.get("matched_snippet_preview", "") or "")[:400],
            )
            for v in (verification or [])
        ]

        # ── Save to session ───────────────────────────────────────────────────
        try:
            append_turn(session_id, req.question, final_text)
        except Exception as e:
            logger.warning("Session append failed (non-fatal): %s", e)

        # ── Store in semantic cache ───────────────────────────────────────────
        try:
            store_answer(req.question, final_text, [s.dict() for s in sources], elapsed)
        except Exception as e:
            logger.warning("Cache store failed (non-fatal): %s", e)

        # ── Log (non-critical) ────────────────────────────────────────────────
        try:
            log_query_answer(
                req.question,
                final_text,
                sources=[s.dict() for s in sources],
                meta={"elapsed_s": elapsed, "session_id": session_id},
            )
        except Exception as log_err:
            logger.warning("Logging failed (non-fatal): %s", log_err)

        logger.info("Query answered | elapsed=%.2fs | sources=%d | session=%s", elapsed, len(sources), session_id)

        return QueryResponse(
            question=req.question,
            answer=final_text or "",
            sources=sources,
            elapsed_s=elapsed,
            session_id=session_id,
            cache_hit=False,
            extracted_facts_preview=facts_text or None,
            verification=verif_items or None,
        )

    except Exception as e:
        logger.exception("Unhandled error in /query: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")
