# Backend/rag_query_rerank.py
"""
Core RAG pipeline: retrieve → rerank → extract facts → verify → synthesize.
Embedder access via the single embeddings_client singleton (no duplicate instances).
"""
import time
import re
import os
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm
import difflib
import string

from Backend.config import (
    CANDIDATE_K, FINAL_K, SCORE_LIMIT_CHARS,
    EXTRACT_MAX_TOKENS, SYNTH_MAX_TOKENS,
)
from Backend.reranker import rerank_with_crossencoder, _use_cross_encoder
from Backend.llm_client import generate_text
from Backend.logger import log_query_answer
from Backend.embeddings_client import get_embedder   # ← single source of truth

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def remove_redundancy(text: str) -> str:
    seen: set = set()
    out_lines: list = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out_lines.append(line)
    return "\n".join(out_lines)


def dedupe_docs(scored_docs: list) -> list:
    selected, seen_keys, seen_texts = [], set(), set()
    for score, doc in scored_docs:
        src  = doc.metadata.get("source")
        page = doc.metadata.get("page")
        key  = (src, page)
        if key in seen_keys:
            continue
        short = doc.page_content[:120].strip()
        if short in seen_texts:
            continue
        seen_keys.add(key)
        seen_texts.add(short)
        selected.append((score, doc))
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Quick text-based fact verifier (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text_for_compare(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _verify_facts_against_picked_quick(
    facts_text: str,
    picked: List[Tuple[float, object]],
    fuzzy_threshold: float = 0.60,
) -> list:
    """Fast normalized fuzzy verification fallback."""
    strict_mode = os.getenv("STRICT_VERBATIM", "false").lower() == "true"
    out = []
    bullet_pattern = re.compile(
        r'^[\-\*\u2022]\s*(?P<fact>.+?)\s*\[(?P<fname>[^\|\]]+)\s*\|\s*page\s*(?P<page>[^\]]+)\]\s*$',
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Build map filename -> merged text
    picked_map: dict = {}
    for _, doc in picked:
        src   = doc.metadata.get("source") or ""
        fname = Path(src).name
        picked_map.setdefault(fname, []).append(doc.page_content.replace("\n", " ").strip())
    for k in list(picked_map.keys()):
        picked_map[k] = " ".join(picked_map[k])

    for m in bullet_pattern.finditer(facts_text):
        raw_fact = m.group("fact").strip()
        fname    = m.group("fname").strip()
        page     = m.group("page").strip()

        matched_doc = picked_map.get(fname)
        verbatim    = False
        best_ratio  = 0.0
        preview     = "<picked doc not found>"

        if matched_doc:
            nfact = _normalize_text_for_compare(raw_fact)
            ndoc  = _normalize_text_for_compare(matched_doc)

            if nfact and nfact in ndoc:
                verbatim   = True
                best_ratio = 1.0
                orig_idx   = matched_doc.lower().find(raw_fact.lower())
                start      = max(0, orig_idx - 80) if orig_idx >= 0 else 0
                preview    = matched_doc[start:start + 300].replace("\n", " ").strip()
            elif not strict_mode:
                candidates = re.split(r'(?<=[\.?!])\s+', matched_doc)
                best = 0.0
                best_cand = matched_doc[:200]
                for cand in candidates:
                    cand_n = _normalize_text_for_compare(cand)
                    if not cand_n:
                        continue
                    ratio = difflib.SequenceMatcher(None, nfact, cand_n).ratio()
                    if ratio > best:
                        best      = ratio
                        best_cand = cand
                best_ratio = float(best)
                preview    = best_cand[:300].replace("\n", " ").strip()
                verbatim   = best_ratio >= float(fuzzy_threshold)

        out.append({
            "fact":                    raw_fact,
            "tag_fname":               fname,
            "tag_page":                page,
            "verbatim_match":          bool(verbatim),
            "similarity":              float(best_ratio),
            "matched_snippet_preview": preview,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-based fact verifier (primary)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _verify_facts_against_picked(
    facts_text: str,
    picked: List[Tuple[float, object]],
    fuzzy_threshold: float = 0.72,
) -> list:
    """Semantic verification via the shared embedder singleton."""
    strict_mode = os.getenv("STRICT_VERBATIM", "false").lower() == "true"
    out: list   = []

    bullet_pattern = re.compile(
        r'^[\-\*\u2022]\s*(?P<fact>.+?)\s*\[(?P<fname>[^\|\]]+)\s*\|\s*page\s*(?P<page>[^\]]+)\]\s*$',
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Build filename → merged text map
    picked_map: dict = {}
    for _, doc in picked:
        src   = doc.metadata.get("source") or ""
        fname = Path(src).name
        picked_map.setdefault(fname, []).append(doc.page_content.replace("\n", " ").strip())
    for k in list(picked_map.keys()):
        picked_map[k] = " ".join(picked_map[k])

    try:
        embedder = get_embedder()   # cached singleton — no extra model load
    except Exception as e:
        logger.warning("Embedder unavailable, falling back to quick string verification: %s", e)
        return _verify_facts_against_picked_quick(facts_text, picked, fuzzy_threshold=0.6)

    for m in bullet_pattern.finditer(facts_text):
        raw_fact = m.group("fact").strip()
        fname    = m.group("fname").strip()
        page     = m.group("page").strip()
        doc_text = picked_map.get(fname)

        best_sim    = 0.0
        best_preview = "<picked doc not found>"
        verbatim    = False

        if doc_text:
            try:
                fact_emb = np.array(embedder.embed_query(raw_fact))
            except Exception:
                try:
                    fact_emb = np.array(embedder.embed_documents([raw_fact])[0])
                except Exception as e:
                    logger.warning("Could not embed fact: %s", e)
                    fact_emb = None

            if fact_emb is None:
                qres = _verify_facts_against_picked_quick(
                    f"- {raw_fact} [{fname} | page {page}]",
                    [(0.0, type("D", (), {"metadata": {"source": fname, "page": page}, "page_content": doc_text})())],
                )
                if qres:
                    out.append(qres[0])
                continue

            candidates = re.split(r'(?<=[\.?!])\s+', doc_text)
            cand_texts = [c for c in candidates if c.strip()] or [doc_text[:1000]]

            try:
                cand_embs = np.array(embedder.embed_documents(cand_texts))
            except Exception:
                cand_embs_list = []
                for c in cand_texts:
                    try:
                        ce = embedder.embed_query(c)
                    except Exception:
                        try:
                            ce = embedder.embed_documents([c])[0]
                        except Exception:
                            ce = None
                    cand_embs_list.append(np.array(ce) if ce is not None else np.zeros(1))
                cand_embs = np.array(cand_embs_list)

            for te, cand in zip(cand_embs, cand_texts):
                try:
                    sim = _cosine(fact_emb, te)
                except Exception:
                    sim = 0.0
                if sim > best_sim:
                    best_sim     = float(sim)
                    best_preview = cand[:400].replace("\n", " ").strip()
            verbatim = best_sim >= float(fuzzy_threshold)

        out.append({
            "fact":                    raw_fact,
            "tag_fname":               fname,
            "tag_page":                page,
            "verbatim_match":          bool(verbatim),
            "similarity":              float(best_sim),
            "matched_snippet_preview": best_preview,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Extract & synthesis helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_facts_from_context(question: str, context: str) -> Tuple[str, str]:
    prompt = (
        "From the provided context, extract ALL distinct facts, principles, or statements "
        "that directly help answer the question.\n"
        "- Output up to 15 concise bullet points (one per line).\n"
        "- For each bullet include a short source tag at the end: [filename | page N]\n"
        "- If you cannot find any fact that directly answers the question, respond with "
        "exactly: NO_FACTS_FOUND\n"
        "- Do not add any text outside the bullet list or the NO_FACTS_FOUND token.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nFacts:"
    )
    facts_text, model_used = generate_text(prompt, max_tokens=EXTRACT_MAX_TOKENS)
    return facts_text or "", model_used


def synthesize_answer_from_facts(question: str, facts: str, context: str) -> Tuple[str, str]:
    SYSTEM_PROMPT = (
        "You are a highly knowledgeable assistant. Provide clear, complete, and structured "
        "answers using ONLY the given context and extracted facts.\n\n"
        "Rules:\n"
        "1. Use all relevant facts to synthesize a complete answer.\n"
        "2. Write clear formal sentences and short paragraphs.\n"
        "3. Organize the answer into numbered points or bullets when multiple items exist.\n"
        "4. If the facts do not contain the answer, respond exactly: "
        "\"The provided documents do not contain this information.\"\n"
        "5. Aim for 4–10 sentences or a short paragraph plus bullets.\n"
    )
    prompt = (
        SYSTEM_PROMPT + "\n"
        + "Extracted facts (bullet list):\n" + facts + "\n\n"
        + "Context (for reference):\n" + context + "\n\n"
        + f"Question: {question}\n\nAnswer:"
    )
    ans_text, model_used = generate_text(prompt, max_tokens=SYNTH_MAX_TOKENS)
    return ans_text or "", model_used


# ─────────────────────────────────────────────────────────────────────────────
# Main ask function
# ─────────────────────────────────────────────────────────────────────────────

def ask_question(
    vectorstore,
    query: str,
    show_sources: bool = False,
    debug: bool = False,
    top_k: int | None = None,
):
    """
    Main ask function.
    - debug=False: returns (final_text, picked)
    - debug=True:  returns (final_text, picked, facts_text, verification)
    - top_k: optional override for CANDIDATE_K
    """
    t0          = time.time()
    candidate_k = top_k if top_k and top_k > 0 else CANDIDATE_K
    candidates  = vectorstore.similarity_search(query, k=candidate_k)

    if not candidates:
        final_text = "⚠️ No relevant documents found for this query."
        logger.warning("No candidates found for query: %s", query[:80])
        log_query_answer(query, final_text, sources=[], meta={"reason": "no_candidates", "elapsed_s": time.time() - t0})
        return (final_text, []) if not debug else (final_text, [], "", [])

    # ── Rerank ────────────────────────────────────────────────────────────────
    if _use_cross_encoder:
        scored = rerank_with_crossencoder(query, candidates)
        min_s, max_s = min(s for s, _ in scored), max(s for s, _ in scored)
        if max_s - min_s < 1e-6:
            scored = [(50.0, d) for s, d in scored]
        else:
            scored = [((s - min_s) / (max_s - min_s) * 100.0, d) for s, d in scored]
    else:
        scored = [(50.0, d) for d in candidates]

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = dedupe_docs(scored)[:FINAL_K]

    # ── Build context ─────────────────────────────────────────────────────────
    context_parts = [
        f"[{Path(doc.metadata.get('source')).name} | page {doc.metadata.get('page', 'N/A')}] "
        f"(score={round(score, 1)})\n{doc.page_content[:1200].strip()}"
        for score, doc in picked
    ]
    context = "\n\n".join(context_parts)

    # ── Extract facts ─────────────────────────────────────────────────────────
    facts_text, extractor_model = extract_facts_from_context(query, context)
    facts_text = (facts_text or "").strip()

    logger.debug("Extractor model: %s | Facts preview: %s", extractor_model, facts_text[:200])

    if facts_text.strip() == "NO_FACTS_FOUND" or not facts_text:
        final_text = "The provided documents do not contain this information."
        log_query_answer(query, final_text, sources=[d for _, d in picked],
                         meta={"reason": "no_facts_token_or_empty", "elapsed_s": time.time() - t0})
        return (final_text, picked) if not debug else (final_text, picked, facts_text, [])

    # ── Validate grounding ────────────────────────────────────────────────────
    has_bullet     = bool(re.search(r'^[\-\*\u2022]\s+', facts_text, flags=re.MULTILINE))
    has_source_tag = bool(re.search(r'\[[^\]]+\s*\|\s*page', facts_text, flags=re.IGNORECASE))
    picked_fnames  = {Path(d.metadata.get("source")).name for _, d in picked if d.metadata.get("source")}
    found_tags     = re.findall(r'\[([^\|\]]+)\s*\|\s*page\s*([^\]]+)\]', facts_text, flags=re.IGNORECASE)
    found_fnames   = {t[0].strip() for t in found_tags}

    # ── Verify facts ──────────────────────────────────────────────────────────
    verification = _verify_facts_against_picked(facts_text, picked, fuzzy_threshold=0.72)

    if verification:
        for v in verification:
            status = "VERBATIM" if v["verbatim_match"] else "PARAPHRASED"
            logger.debug("Verification [%s|p%s] → %s (sim=%.2f)", v["tag_fname"], v["tag_page"], status, v.get("similarity", 0))

    tags_map_to_picked = bool(found_fnames and (found_fnames & picked_fnames))

    if not (has_bullet and has_source_tag and tags_map_to_picked):
        final_text = "The provided documents do not contain this information."
        log_query_answer(query, final_text, sources=[d for _, d in picked], meta={
            "reason":           "facts_not_grounded",
            "picked_files":     list(picked_fnames)[:10],
            "found_tags":       list(found_fnames)[:10],
            "extractor_model":  extractor_model,
            "elapsed_s":        time.time() - t0,
        })
        return (final_text, picked) if not debug else (final_text, picked, facts_text, verification)

    # ── Synthesize ────────────────────────────────────────────────────────────
    final_out, synth_model = synthesize_answer_from_facts(query, facts_text, context)
    final_text = remove_redundancy(final_out) if final_out else "⚠️ Error generating answer."

    logger.info("Query answered | synth_model=%s | elapsed=%.2fs", synth_model, time.time() - t0)

    if show_sources:
        for score, doc in picked:
            print(f"  score={round(score,1)}  {Path(doc.metadata.get('source')).name} | page: {doc.metadata.get('page', 'N/A')}")

    verif_summary = [
        {"fname": v["tag_fname"], "page": v["tag_page"], "sim": round(v["similarity"], 2), "verbatim": bool(v["verbatim_match"])}
        for v in verification
    ] if verification else []

    log_query_answer(
        query,
        final_text.strip(),
        sources=[d for _, d in picked],
        meta={
            "elapsed_s":        time.time() - t0,
            "extractor_model":  extractor_model,
            "synth_model":      synth_model,
            "extracted_preview": facts_text[:1000],
            "verification":     verif_summary,
        },
    )

    if debug:
        return final_text, picked, facts_text, verification
    return final_text, picked
