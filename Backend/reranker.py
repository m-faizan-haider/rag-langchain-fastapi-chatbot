# Backend/reranker.py
import logging
from typing import List, Tuple
from Backend.config import CROSS_ENCODER_MODEL, SCORE_LIMIT_CHARS

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

_use_cross_encoder = False
_cross_encoder     = None


def try_init_cross_encoder():
    global _use_cross_encoder, _cross_encoder
    if CrossEncoder is None:
        _use_cross_encoder = False
        _cross_encoder     = None
        logger.warning("sentence-transformers not installed — CrossEncoder unavailable.")
        return
    try:
        _cross_encoder     = CrossEncoder(CROSS_ENCODER_MODEL)
        _use_cross_encoder = True
        logger.info("CrossEncoder reranker loaded: %s", CROSS_ENCODER_MODEL)
    except Exception as e:
        _use_cross_encoder = False
        _cross_encoder     = None
        logger.warning("CrossEncoder init failed, using uniform scores: %s", e)


def rerank_with_crossencoder(query: str, docs: List) -> List[Tuple]:
    if not _use_cross_encoder or _cross_encoder is None:
        raise RuntimeError("Cross-encoder not initialized")
    pairs  = [[query, d.page_content[:SCORE_LIMIT_CHARS]] for d in docs]
    scores = _cross_encoder.predict(pairs, batch_size=16)
    return list(zip(scores.tolist(), docs))


# Initialize on import
try_init_cross_encoder()
