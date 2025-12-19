# Backend/llm_client.py  (debugging / verbose)
import os
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv
load_dotenv()
from Backend.config import ROUTERAI_API_KEY, DEEPSEEK_ENDPOINT, DEEPSEEK_MODEL_NAME, FALLBACK_HF_MODEL

# Compute availability flag from environment
deepseek_available = bool(os.getenv("ROUTERAI_API_KEY")) and bool(os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME))

_gen_pipe = None

def _debug_env():
    print("DEBUG LLM CLIENT ENV:")
    print(f"  ROUTERAI_API_KEY set: {bool(os.getenv('ROUTERAI_API_KEY'))}")
    print(f"  DEEPSEEK_MODEL_NAME: {os.getenv('DEEPSEEK_MODEL_NAME', DEEPSEEK_MODEL_NAME)}")
    print(f"  deepseek_available flag: {deepseek_available}")

_debug_env()

def generate_with_routerai(prompt: str, max_tokens: int = 768):
    model_name = os.getenv("DEEPSEEK_MODEL_NAME", DEEPSEEK_MODEL_NAME)
    if not deepseek_available:
        print("⚠️ RouterAI not available (env missing). Skipping RouterAI call.")
        return None, model_name

    headers = {
        "Authorization": f"Bearer {os.getenv('ROUTERAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_output_tokens": max_tokens,
        "temperature": 0.0
    }

    try:
        print(f"ℹ️ Calling RouterAI / DeepSeek model: {model_name} (max_tokens={max_tokens})")
        r = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # debug print trimmed
        print("ℹ️ RouterAI response keys:", list(data.keys()))
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if content:
                    return content.strip(), model_name
            if "text" in choice:
                txt = choice.get("text")
                if txt:
                    return txt.strip(), model_name
        print("❌ RouterAI DeepSeek: unexpected response format (no usable text).", data)
        return None, model_name
    except Exception as e:
        print(f"❌ RouterAI DeepSeek API call failed: {e}")
        return None, model_name

def generate_with_local_flan(prompt: str, max_tokens: int = 768):
    global _gen_pipe
    model_name = f"{FALLBACK_HF_MODEL} (local)"
    if _gen_pipe is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_HF_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_HF_MODEL)
            _gen_pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                truncation=True,
                do_sample=False
            )
            print(f"ℹ️ Using local transformers generator: {FALLBACK_HF_MODEL}")
        except Exception as e:
            print("❌ Failed to initialize local transformers generator:", e)
            raise

    out = _gen_pipe(prompt, max_new_tokens=max_tokens, truncation=True)[0].get("generated_text", "").strip()
    return out, model_name

def generate_text(prompt: str, max_tokens: int = 768):
    """
    Unified API: try DeepSeek via OpenRouter first (if available), fallback to local HF.
    Returns (text, model_used)
    """
    # Try DeepSeek if available
    if deepseek_available:
        text, model_name = generate_with_routerai(prompt, max_tokens=max_tokens)
        if text:
            print(f"✅ RouterAI succeeded using model: {model_name}")
            return text, model_name
        print("⚠️ DeepSeek call failed or returned unexpected format. Falling back to local generator...")

    # Fallback to local generator
    text_local, local_model = generate_with_local_flan(prompt, max_tokens=max_tokens)
    return text_local, local_model

if __name__ == "__main__":
    test_prompt = "Explain the Cost Optimization pillar of AWS Well-Architected Framework."
    print("🔍 Testing LLM client with a sample query...\n")
    text, model = generate_text(test_prompt, max_tokens=300)
    print(f"✅ Model used: {model}\n")
    print(text)
