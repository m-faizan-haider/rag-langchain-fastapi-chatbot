# Backend/api.py
import time
from pathlib import Path
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from Backend.schemas import QueryRequest, QueryResponse, SourceItem, FactVerificationItem
from Backend.deps import get_vectorstore_and_meta, reload_vectorstore_and_meta
from Backend.rag_query_rerank import ask_question
from Backend.logger import log_query_answer

app = FastAPI(title="RAG LangChain FastAPI Chatbot", version="0.1")

# CORS - allow local dev (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RAG LangChain FastAPI Chatbot is running!"}

@app.get("/health")
def health():
    try:
        vs, meta = get_vectorstore_and_meta()
        return {"status": "ok", "embedding_model": meta.get("embedding_model") if isinstance(meta, dict) else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faiss_check")
def faiss_check(vs_meta: tuple = Depends(get_vectorstore_and_meta)):
    vs, meta = vs_meta
    return {"ok": True, "embedding_model": meta.get("embedding_model") if isinstance(meta, dict) else None}

@app.post("/reload_faiss")
def reload_faiss(force_rebuild: bool = Body(False, embed=True)):
    try:
        start = time.time()
        vs_meta = reload_vectorstore_and_meta(force_rebuild=force_rebuild)
        elapsed = time.time() - start
        vs, meta = vs_meta
        return {"ok": True, "elapsed_s": elapsed, "embedding_model": meta.get("embedding_model") if isinstance(meta, dict) else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest, vs_meta: tuple = Depends(get_vectorstore_and_meta)):
    vs, meta = vs_meta
    start = time.time()
    try:
        # call ask_question with debug flag
        result = ask_question(vs, req.question, show_sources=False, debug=req.debug)
        if req.debug:
            # result = (final_text, picked, facts_text, verification)
            final_text, picked, facts_text, verification = result
        else:
            final_text, picked = result
            facts_text = ""
            verification = []

        # Build sources list for response
        sources = []
        for item in picked:
            if isinstance(item, tuple) and len(item) == 2:
                score, doc = item
            else:
                # fallback if picked gives doc only
                score = None
                doc = item
            src = doc.metadata.get("source") if hasattr(doc, "metadata") else None
            fname = Path(src).name if src else "unknown"
            page = doc.metadata.get("page") if hasattr(doc, "metadata") else None
            sources.append(SourceItem(filename=fname, page=int(page) if page not in (None, "N/A") else None, score=float(score) if score is not None else None))

        elapsed = time.time() - start

        # Convert verification dicts to FactVerificationItem Pydantic objects
        verif_items = []
        for v in verification or []:
            verif_items.append(FactVerificationItem(
                fact=v.get("fact",""),
                tag_fname=v.get("tag_fname",""),
                tag_page=v.get("tag_page", None),
                similarity=float(v.get("similarity", 0.0)),
                verbatim_match=bool(v.get("verbatim_match", False)),
                matched_preview=(v.get("matched_snippet_preview","") or "")[:400]
            ))

        # Log query (non-blocking)
        try:
            log_query_answer(req.question, final_text, sources=[s.dict() for s in sources], meta={"elapsed_s": elapsed})
        except Exception:
            pass

        return QueryResponse(
            question=req.question,
            answer=final_text or "",
            sources=sources,
            elapsed_s=elapsed,
            extracted_facts_preview=facts_text or None,
            verification=verif_items or None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")
