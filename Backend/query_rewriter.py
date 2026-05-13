# Backend/query_rewriter.py
"""
Query rewriting via HyDE (Hypothetical Document Embeddings).

How HyDE works:
  1. Ask the LLM to generate a short "hypothetical ideal answer".
  2. Embed THAT answer instead of the raw user query.
  3. The embedding of the hypothetical answer sits closer to real document
     chunks in vector space — improving retrieval precision significantly.

Fallback: if HyDE fails (LLM timeout / no API key), uses the original query.
"""
import logging
from typing import Tuple

from Backend.llm_client import generate_text

logger = logging.getLogger(__name__)

_HYDE_PROMPT_TEMPLATE = """\
Write a short, factual paragraph (3-5 sentences) that would be an ideal answer \
to the following question, as if it were extracted directly from a relevant document. \
Do not add any preamble, introduction, or "Here is..." phrase. Just the factual content.

Question: {question}

Ideal document excerpt:"""

_EXPAND_PROMPT_TEMPLATE = """\
Generate 3 alternative phrasings of the following question that mean the same thing \
but use different words. Output only the 3 questions, one per line, no numbering.

Question: {question}

Alternative phrasings:"""


def rewrite_with_hyde(question: str) -> Tuple[str, str]:
    """
    Generate a hypothetical document for the question via LLM.
    Returns (hypothetical_text, method_used).
    Falls back to original question on failure.
    """
    prompt = _HYDE_PROMPT_TEMPLATE.format(question=question)
    try:
        hyp_text, model = generate_text(prompt, max_tokens=256)
        if hyp_text and len(hyp_text.strip()) > 20:
            logger.debug("HyDE generated | model=%s | preview=%s", model, hyp_text[:80])
            return hyp_text.strip(), f"hyde/{model}"
    except Exception as e:
        logger.warning("HyDE rewriting failed: %s", e)
    return question, "original"


def expand_query(question: str) -> list[str]:
    """
    Generate up to 3 paraphrases of the question.
    Returns a list of query strings (including the original).
    Falls back to [original] on failure.
    """
    prompt = _EXPAND_PROMPT_TEMPLATE.format(question=question)
    try:
        text, _ = generate_text(prompt, max_tokens=150)
        if text:
            variants = [q.strip() for q in text.strip().splitlines() if q.strip()][:3]
            logger.debug("Query expansion: %d variants generated", len(variants))
            return [question] + variants
    except Exception as e:
        logger.warning("Query expansion failed: %s", e)
    return [question]
