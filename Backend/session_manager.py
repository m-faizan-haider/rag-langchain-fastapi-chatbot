# Backend/session_manager.py
"""
Conversation session manager — stores multi-turn chat history.
Primary backend: Redis (TTL-based, survives restarts).
Fallback: in-memory dict (single-process only, lost on restart).
"""
import json
import logging
import time
import uuid
from typing import List

logger = logging.getLogger(__name__)

_SESSION_TTL = 86400   # 24 hours in seconds
_MAX_HISTORY = 6       # keep last 6 turns (3 user + 3 assistant) to limit context size

# ── In-memory fallback ────────────────────────────────────────────────────────
_mem_store: dict = {}

# ── Redis client (lazy init) ──────────────────────────────────────────────────
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis as redis_lib
        from Backend.config import REDIS_URL
        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _redis.ping()
        logger.info("Session manager: Redis connected at %s", REDIS_URL)
    except Exception as e:
        logger.warning("Redis unavailable, using in-memory session store: %s", e)
        _redis = None
    return _redis


def _session_key(session_id: str) -> str:
    return f"rag:session:{session_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    r = _get_redis()
    if r:
        try:
            r.setex(_session_key(session_id), _SESSION_TTL, json.dumps([]))
        except Exception as e:
            logger.warning("Redis set failed: %s", e)
            _mem_store[session_id] = []
    else:
        _mem_store[session_id] = []
    logger.debug("Session created: %s", session_id)
    return session_id


def get_history(session_id: str) -> List[dict]:
    """Return the conversation history for a session (list of {role, content})."""
    r = _get_redis()
    if r:
        try:
            raw = r.get(_session_key(session_id))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
    return _mem_store.get(session_id, [])


def append_turn(session_id: str, question: str, answer: str) -> None:
    """Append a user→assistant turn to the session history."""
    history = get_history(session_id)
    history.append({"role": "user",      "content": question, "ts": time.time()})
    history.append({"role": "assistant", "content": answer,   "ts": time.time()})

    # Keep only the last N turns
    if len(history) > _MAX_HISTORY * 2:
        history = history[-(  _MAX_HISTORY * 2):]

    r = _get_redis()
    if r:
        try:
            r.setex(_session_key(session_id), _SESSION_TTL, json.dumps(history))
            return
        except Exception as e:
            logger.warning("Redis set failed: %s", e)
    _mem_store[session_id] = history


def delete_session(session_id: str) -> None:
    """Explicitly clear a session (e.g. user logs out)."""
    r = _get_redis()
    if r:
        try:
            r.delete(_session_key(session_id))
        except Exception:
            pass
    _mem_store.pop(session_id, None)


def format_history_for_prompt(history: List[dict]) -> str:
    """
    Format last N turns as a readable block for the LLM prompt.
    Only the last _MAX_HISTORY turns are included.
    """
    if not history:
        return ""
    lines = ["[CONVERSATION HISTORY]"]
    for turn in history[-(  _MAX_HISTORY * 2):]:
        role    = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"][:400]    # truncate very long turns
        lines.append(f"{role}: {content}")
    lines.append("[END HISTORY]")
    return "\n".join(lines)
