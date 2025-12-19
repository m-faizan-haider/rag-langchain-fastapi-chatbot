# Backend/rag_query_rerank.py
import time
import re
import os
from pathlib import Path
from typing import List, Tuple

# numerical + embeddings
import numpy as np
from numpy.linalg import norm

# local project imports
from Backend.config import CANDIDATE_K, FINAL_K, SCORE_LIMIT_CHARS, EXTRACT_MAX_TOKENS, SYNTH_MAX_TOKENS
from Backend.retriever import load_vectorstore_and_check
from Backend.reranker import rerank_with_crossencoder, _use_cross_encoder
from Backend.llm_client import generate_text
from Backend.logger import log_query_answer

# third-party for embeddings (LangChain wrapper you used elsewhere)
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------- Utility functions -----------------
def remove_redundancy(text: str) -> str:
    seen = set()
    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out_lines.append(line)
    return "\n".join(out_lines)


def dedupe_docs(scored_docs):
    selected, seen_keys, seen_texts = [], set(), set()
    for score, doc in scored_docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        key = (src, page)
        if key in seen_keys:
            continue
        short = doc.page_content[:120].strip()
        if short in seen_texts:
            continue
        seen_keys.add(key)
        seen_texts.add(short)
        selected.append((score, doc))
    return selected


# ----------------- Quick text-based verifier (fallback) -----------------
import difflib
import string

def _normalize_text_for_compare(s: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - replace newlines with spaces
    - collapse whitespace
    - normalize smart quotes to plain quotes
    """
    if not s:
        return ""
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s.lower()

def _verify_facts_against_picked_quick(facts_text: str, picked: List[Tuple[float, object]], fuzzy_threshold: float = 0.60):
    """
    Fast normalized fuzzy verification fallback.
    Lower fuzzy_threshold for more permissive matching (0.6 = permissive, 0.8 = strict).
    Returns list of dicts with similarity in 0..1 and matched preview.
    """
    strict_mode = os.getenv("STRICT_VERBATIM", "false").lower() == "true"
    out = []
    bullet_pattern = re.compile(
        r'^[\-\*\u2022]\s*(?P<fact>.+?)\s*\[(?P<fname>[^\|\]]+)\s*\|\s*page\s*(?P<page>[^\]]+)\]\s*$',
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Build map filename -> merged text (all picked chunks for that file)
    picked_map = {}
    for _, doc in picked:
        src = doc.metadata.get("source") or ""
        fname = Path(src).name
        picked_map.setdefault(fname, []).append(doc.page_content.replace("\n", " ").strip())
    for k in list(picked_map.keys()):
        picked_map[k] = " ".join(picked_map[k])

    for m in bullet_pattern.finditer(facts_text):
        raw_fact = m.group("fact").strip()
        fname = m.group("fname").strip()
        page = m.group("page").strip()

        matched_doc = picked_map.get(fname)
        verbatim = False
        best_ratio = 0.0
        preview = "<picked doc not found>"
        if matched_doc:
            nfact = _normalize_text_for_compare(raw_fact)
            ndoc = _normalize_text_for_compare(matched_doc)

            # exact normalized substring
            if nfact and nfact in ndoc:
                verbatim = True
                best_ratio = 1.0
                orig_idx = matched_doc.lower().find(raw_fact.lower())
                if orig_idx >= 0:
                    start = max(0, orig_idx - 80)
                    preview = matched_doc[start:start + 300].replace("\n", " ").strip()
                else:
                    preview = matched_doc[:300].replace("\n", " ").strip()
            else:
                if strict_mode:
                    verbatim = False
                    best_ratio = 0.0
                    preview = matched_doc[:300].replace("\n", " ").strip()
                else:
                    # fuzzy: compare against sentences and take best ratio
                    candidates = re.split(r'(?<=[\.\?\!])\s+', matched_doc)
                    best = 0.0
                    best_cand = matched_doc[:200]
                    for cand in candidates:
                        cand_n = _normalize_text_for_compare(cand)
                        if not cand_n:
                            continue
                        ratio = difflib.SequenceMatcher(None, nfact, cand_n).ratio()
                        if ratio > best:
                            best = ratio
                            best_cand = cand
                    best_ratio = float(best)
                    preview = best_cand[:300].replace("\n", " ").strip()
                    verbatim = best_ratio >= float(fuzzy_threshold)

        out.append({
            "fact": raw_fact,
            "tag_fname": fname,
            "tag_page": page,
            "verbatim_match": bool(verbatim),
            "similarity": float(best_ratio),
            "matched_snippet_preview": preview
        })

    return out


# ----------------- Embedding-based verifier (recommended) -----------------
_embedder_cache = None

def _get_embedder():
    global _embedder_cache
    if _embedder_cache is None:
        # EMBEDDING_MODEL should be defined in Backend.config and already used in your project
        from Backend.config import EMBEDDING_MODEL
        _embedder_cache = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedder_cache

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _verify_facts_against_picked(facts_text: str, picked: List[Tuple[float, object]], fuzzy_threshold: float = 0.72):
    """
    Semantic verification via embeddings.
    Returns list of dicts:
      { fact, tag_fname, tag_page, verbatim_match (bool), similarity (0..1), matched_snippet_preview }
    If embedder init fails, falls back to quick verifier.
    """
    strict_mode = os.getenv("STRICT_VERBATIM", "false").lower() == "true"
    out = []

    bullet_pattern = re.compile(
        r'^[\-\*\u2022]\s*(?P<fact>.+?)\s*\[(?P<fname>[^\|\]]+)\s*\|\s*page\s*(?P<page>[^\]]+)\]\s*$',
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # build text map for each filename (join chunks)
    picked_map = {}
    for _, doc in picked:
        src = doc.metadata.get("source") or ""
        fname = Path(src).name
        picked_map.setdefault(fname, []).append(doc.page_content.replace("\n", " ").strip())
    for k in list(picked_map.keys()):
        picked_map[k] = " ".join(picked_map[k])

    try:
        embedder = _get_embedder()
    except Exception as e:
        # embedder failed -> fallback to quick text-based verification
        print("⚠️ Embedder init failed, falling back to quick string verification:", e)
        return _verify_facts_against_picked_quick(facts_text, picked, fuzzy_threshold=0.6)

    for m in bullet_pattern.finditer(facts_text):
        raw_fact = m.group("fact").strip()
        fname = m.group("fname").strip()
        page = m.group("page").strip()

        doc_text = picked_map.get(fname)
        best_sim = 0.0
        best_preview = "<picked doc not found>"
        verbatim = False

        if doc_text:
            # embed the fact (some wrappers expose embed_query or embed_documents)
            try:
                fact_emb = np.array(embedder.embed_query(raw_fact))
            except Exception:
                try:
                    fact_emb = np.array(embedder.embed_documents([raw_fact])[0])
                except Exception as e:
                    print("⚠️ Embedder failed to embed fact:", e)
                    fact_emb = None

            if fact_emb is None:
                # fallback: try text-based quick check for this fact
                qres = _verify_facts_against_picked_quick("- " + raw_fact + f" [{fname} | page {page}]", [(0.0, type("D", (), {"metadata": {"source": fname, "page": page}, "page_content": doc_text}))])
                if qres and isinstance(qres, list):
                    entry = qres[0]
                    out.append(entry)
                    continue

            # split doc into candidate sentences
            candidates = re.split(r'(?<=[\.\?\!])\s+', doc_text)
            cand_texts = [c for c in candidates if c.strip()]
            if not cand_texts:
                cand_texts = [doc_text[:1000]]

            # compute embeddings for candidates (attempt batch)
            try:
                cand_embs = np.array(embedder.embed_documents(cand_texts))
            except Exception:
                # fallback: compute per candidate
                cand_embs = []
                for c in cand_texts:
                    try:
                        ce = embedder.embed_query(c)
                    except Exception:
                        try:
                            ce = embedder.embed_documents([c])[0]
                        except Exception:
                            ce = None
                    cand_embs.append(np.array(ce) if ce is not None else np.zeros(1))
                cand_embs = np.array(cand_embs)

            # find best cosine
            for te, cand in zip(cand_embs, cand_texts):
                try:
                    sim = _cosine(fact_emb, te)
                except Exception:
                    sim = 0.0
                if sim > best_sim:
                    best_sim = float(sim)
                    best_preview = cand[:400].replace("\n", " ").strip()
            verbatim = best_sim >= float(fuzzy_threshold)

        out.append({
            "fact": raw_fact,
            "tag_fname": fname,
            "tag_page": page,
            "verbatim_match": bool(verbatim),
            "similarity": float(best_sim),
            "matched_snippet_preview": best_preview
        })

    return out


# ----------------- Extract & Synthesis helpers -----------------
def extract_facts_from_context(question: str, context: str) -> Tuple[str, str]:
    prompt = (
        "From the provided context, extract ALL distinct facts, principles, or statements that directly help answer the question.\n"
        "- Output up to 15 concise bullet points (one per line).\n"
        "- For each bullet include a short source tag at the end in this format: [filename | page N]\n"
        "- If you cannot find any fact that directly answers the question, respond with exactly: NO_FACTS_FOUND\n"
        "- Do not add any text outside the bullet list or the exact NO_FACTS_FOUND token.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nFacts:"
    )
    facts_text, model_used = generate_text(prompt, max_tokens=EXTRACT_MAX_TOKENS)
    return facts_text or "", model_used


def synthesize_answer_from_facts(question: str, facts: str, context: str) -> Tuple[str, str]:
    SYSTEM_PROMPT = """
You are a highly knowledgeable assistant. Your goal is to provide clear, complete, and structured answers using ONLY the given context and extracted facts.

Follow these rules strictly:
1. Use all relevant facts to synthesize a complete answer.
2. Write clear formal sentences and short paragraphs.
3. Organize the answer into numbered points or bullets when multiple items exist.
4. If the facts do not contain the answer, respond exactly: "The provided documents do not contain this information."
5. Aim for 4–10 sentences or a short paragraph plus bullets.
"""
    prompt = SYSTEM_PROMPT + "\n"
    prompt += "Extracted facts (bullet list):\n" + facts + "\n\n"
    prompt += "Context (for reference):\n" + context + "\n\n"
    prompt += f"Question: {question}\n\nAnswer:"
    ans_text, model_used = generate_text(prompt, max_tokens=SYNTH_MAX_TOKENS)
    return ans_text or "", model_used


# ----------------- Main ask function -----------------
def ask_question(vectorstore, query: str, show_sources: bool = False, debug: bool = False):
    """
    Main ask function.
    - When debug=False: returns (final_text, picked)
    - When debug=True: returns (final_text, picked, facts_text, verification)
    """
    t0 = time.time()
    candidates = vectorstore.similarity_search(query, k=CANDIDATE_K)

    if not candidates:
        final_text = "⚠️ No relevant documents found for this query."
        print(final_text)
        log_query_answer(query, final_text, sources=[], meta={"reason": "no_candidates", "elapsed_s": time.time() - t0})
        return (final_text, []) if not debug else (final_text, [], "", [])

    # ---------- Rerank ----------
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

    # Build context
    context_parts = [
        f"[{Path(doc.metadata.get('source')).name} | page {doc.metadata.get('page','N/A')}] (score={round(score,1)})\n{doc.page_content[:1200].strip()}"
        for score, doc in picked
    ]
    context = "\n\n".join(context_parts)

    # ---------------- Extract & Verify Facts ----------------
    facts_text, extractor_model = extract_facts_from_context(query, context)
    facts_text = (facts_text or "").strip()

    # Debug print
    print("\n--- DEBUG: Extracted Facts (first 800 chars) ---")
    print(f"Model used for extraction: {extractor_model}")
    print(facts_text[:800])
    print("--- END DEBUG ---\n")

    # Early NO_FACTS_FOUND token
    if facts_text.strip() == "NO_FACTS_FOUND" or not facts_text:
        final_text = "The provided documents do not contain this information."
        print("\n=== ANSWER ===\n" + final_text + "\n")
        log_query_answer(query, final_text, sources=[d for _, d in picked], meta={"reason":"no_facts_token_or_empty", "elapsed_s": time.time()-t0})
        return (final_text, picked) if not debug else (final_text, picked, facts_text, [])

    # Validation checks
    has_bullet = bool(re.search(r'^[\-\*\u2022]\s+', facts_text, flags=re.MULTILINE))
    has_source_tag = bool(re.search(r'\[[^\]]+\s*\|\s*page', facts_text, flags=re.IGNORECASE))
    picked_fnames = { Path(d.metadata.get("source")).name for _, d in picked if d.metadata.get("source") }
    found_tags = re.findall(r'\[([^\|\]]+)\s*\|\s*page\s*([^\]]+)\]', facts_text, flags=re.IGNORECASE)
    found_fnames = { t[0].strip() for t in found_tags }

    # Perform verification (returns list of dicts)
    # embedding-based verification; adjust fuzzy_threshold as needed (0.72 default)
    verification = _verify_facts_against_picked(facts_text, picked, fuzzy_threshold=0.72)

    # Show verification debug
    if verification:
        print("\n--- FACT VERIFICATION ---")
        for v in verification:
            status = "VERBATIM" if v["verbatim_match"] else "PARAPHRASED/NO_MATCH"
            print(f"- [{v['tag_fname']} | page {v['tag_page']}] -> {status} (sim={v.get('similarity', 0):.2f})")
        print("--- END VERIFICATION ---\n")
    else:
        print("\n--- FACT VERIFICATION: no bullet-with-tag facts found ---\n")

    tags_map_to_picked = bool(found_fnames and (found_fnames & picked_fnames))

    # If not grounded -> refuse (audit & safe behavior)
    if not (has_bullet and has_source_tag and tags_map_to_picked):
        final_text = "The provided documents do not contain this information."
        print("\n=== ANSWER ===\n" + final_text + "\n")
        log_query_answer(query, final_text, sources=[d for _, d in picked], meta={
            "reason": "facts_not_grounded_or_mismatch",
            "picked_files": list(picked_fnames)[:10],
            "found_tags": list(found_fnames)[:10],
            "extracted_preview": facts_text[:1000],
            "extractor_model": extractor_model,
            "elapsed_s": time.time() - t0,
        })
        return (final_text, picked) if not debug else (final_text, picked, facts_text, verification)

    # ---------------- Synthesize ----------------
    final_out, synth_model = synthesize_answer_from_facts(query, facts_text, context)
    final_text = remove_redundancy(final_out) if final_out else "⚠️ Error generating answer."

    # Debug: show which model synthesized
    print(f"\n[DEBUG] synth_model_used: {synth_model}\n")

    # Display in terminal (interactive mode)
    print("\n=== QUESTION ===")
    print(query)
    print("\n=== ANSWER ===")
    print(final_text.strip() + "\n")

    if show_sources:
        print("=== SOURCES ===")
        for score, doc in picked:
            print(f"  score={round(score,1)}  {Path(doc.metadata.get('source')).name} | page: {doc.metadata.get('page','N/A')}")

    # Logging (includes extractor + synth model and a short preview of extracted facts + verification summary)
    verif_summary = [
        {"fname": v["tag_fname"], "page": v["tag_page"], "sim": round(v["similarity"], 2), "verbatim": bool(v["verbatim_match"])}
        for v in verification
    ] if verification else []

    log_query_answer(
        query,
        final_text.strip(),
        sources=[d for _, d in picked],
        meta={
            "elapsed_s": time.time() - t0,
            "extractor_model": extractor_model,
            "synth_model": synth_model,
            "extracted_preview": facts_text[:1000],
            "verification": verif_summary,
        },
    )

    # Return debug info if requested
    if debug:
        return final_text, picked, facts_text, verification

    return final_text, picked
