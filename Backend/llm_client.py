# Backend/llm_client.py
"""
LLM client with:
  Primary  — OpenRouter API (Gemini Flash / DeepSeek)
  Fallback — Ollama local (Mistral 7B) replaces the obsolete Flan-T5
"""
import logging
import os
import requests
from dotenv import load_dotenv

load_dotenv()

from Backend.config import (
    ROUTERAI_API_KEY,
    DEEPSEEK_ENDPOINT,
    DEEPSEEK_MODEL_NAME,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)

deepseek_available = bool(os.getenv("ROUTERAI_API_KEY")) and bool(
    os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME)
)
logger.debug(
    "LLM client init | deepseek_available=%s | model=%s",
    deepseek_available,
    os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME),
)


# ─────────────────────────────────────────────────────────────────────────────
# Primary: OpenRouter (Gemini Flash / DeepSeek)
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_routerai(prompt: str, max_tokens: int = 768):
    """Call OpenRouter API. Returns (text, model_name) or (None, model_name)."""
    model_name = os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME)
    if not deepseek_available:
        logger.warning("OpenRouter not configured (ROUTERAI_API_KEY missing).")
        return None, model_name

    headers = {
        "Authorization": f"Bearer {os.getenv('ROUTERAI_API_KEY')}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        logger.info("Calling OpenRouter | model=%s | max_tokens=%d", model_name, max_tokens)
        r = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            content = (choice.get("message") or {}).get("content") or choice.get("text")
            if content:
                logger.info("OpenRouter succeeded | model=%s", model_name)
                return content.strip(), model_name

        logger.error("OpenRouter: unexpected response format: %s", data)
        return None, model_name

    except Exception as e:
        logger.error("OpenRouter call failed: %s", e)
        return None, model_name


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: Ollama (local — replaces Flan-T5)
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_ollama(prompt: str, max_tokens: int = 768):
    """
    Call local Ollama server. Returns (text, model_name) or raises.
    Requires: `ollama pull mistral:7b` (or whichever OLLAMA_MODEL is set).
    """
    model_name = f"{OLLAMA_MODEL} (ollama-local)"
    try:
        payload = {
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }
        logger.info("Calling Ollama | model=%s | max_tokens=%d", OLLAMA_MODEL, max_tokens)
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        if text:
            logger.info("Ollama succeeded | model=%s", OLLAMA_MODEL)
            return text, model_name
        logger.error("Ollama returned empty response.")
        return None, model_name
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_text(prompt: str, max_tokens: int = 768):
    """
    Try OpenRouter first; fall back to local Ollama.
    Returns (text, model_used_str).
    """
    if deepseek_available:
        text, model_name = generate_with_routerai(prompt, max_tokens=max_tokens)
        if text:
            return text, model_name
        logger.warning("OpenRouter failed — falling back to Ollama...")

    # Ollama fallback
    try:
        text_local, local_model = generate_with_ollama(prompt, max_tokens=max_tokens)
        if text_local:
            return text_local, local_model
    except Exception as e:
        logger.error("Ollama fallback also failed: %s", e)

    return "", "none"


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    test_prompt = "Explain the Cost Optimization pillar of AWS Well-Architected Framework in 3 bullets."
    print("Testing LLM client...\n")
    text, model = generate_text(test_prompt, max_tokens=300)
    print(f"Model used: {model}\n\n{text}")
