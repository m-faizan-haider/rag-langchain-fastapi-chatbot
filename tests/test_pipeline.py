"""
Full pipeline test — fires a real /query against the running backend.
Tests: retrieval → reranking → LLM generation → session memory → semantic cache
"""
import requests
import time
import sys
import json
from dotenv import load_dotenv

load_dotenv()
BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("  FULL PIPELINE DIAGNOSTIC")
print("=" * 60)

passed = 0
total  = 0

def check(name, condition, note=""):
    global passed, total
    total += 1
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f"  -- {note}" if note else ""))
    else:
        print(f"  [FAIL] {name}" + (f"  -- {note}" if note else ""))
    return condition

# ── Test 1: First query (cold) ─────────────────────────────────
print("\n[1] Cold query (no cache)...")
t0 = time.time()
r  = requests.post(f"{BASE}/query", json={"question": "What are the main topics covered in the documents?"}, timeout=120)
cold_elapsed = time.time() - t0

check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
if r.status_code == 200:
    data = r.json()
    check("Has answer",     bool(data.get("answer")),             f"{len(data.get('answer',''))} chars")
    check("Has sources",    len(data.get("sources", [])) > 0,     f"{len(data.get('sources',[]))} sources")
    check("Has session_id", bool(data.get("session_id")),         data.get("session_id","?")[:12])
    check("cache_hit=False",not data.get("cache_hit", True),     "correctly marked as not cached")
    check("elapsed_s OK",   data.get("elapsed_s", 99) < 60,      f"{data.get('elapsed_s',0):.2f}s")

    session_id = data.get("session_id")
    first_answer = data.get("answer","")

    print(f"\n  Answer preview: {first_answer[:200]}...")
    print(f"  Sources:")
    for s in data.get("sources", [])[:3]:
        print(f"    - {s['filename']} (page {s.get('page','?')}, score {s.get('score','?')})")
else:
    session_id = None
    first_answer = ""
    print(f"  Error: {r.text[:300]}")

# ── Test 2: Session memory (follow-up question) ─────────────────
if session_id:
    print(f"\n[2] Follow-up query (same session: {session_id[:12]}...)...")
    r2 = requests.post(f"{BASE}/query", json={
        "question":   "Can you give me more detail about the first point?",
        "session_id": session_id
    }, timeout=120)
    check("Follow-up HTTP 200", r2.status_code == 200)
    if r2.status_code == 200:
        d2 = r2.json()
        check("Same session echoed", d2.get("session_id") == session_id, "session preserved")
        check("Has answer", bool(d2.get("answer")), f"{len(d2.get('answer',''))} chars")

# ── Test 3: Semantic cache hit ─────────────────────────────────
print("\n[3] Repeat query (should be a semantic cache HIT)...")
time.sleep(1)
r3 = requests.post(f"{BASE}/query", json={"question": "What are the main topics covered in the documents?"}, timeout=30)
check("Cache query HTTP 200", r3.status_code == 200)
if r3.status_code == 200:
    d3 = r3.json()
    hit      = d3.get("cache_hit", False)
    fast     = d3.get("elapsed_s", 99) < cold_elapsed * 0.5
    check("cache_hit=True",   hit,  "correctly served from cache" if hit else "cache miss (embedder may still be loading)")
    check("Faster than cold", fast or hit, f"cold={cold_elapsed:.2f}s  cached={d3.get('elapsed_s',0):.2f}s")

# ── Test 4: top_k parameter ────────────────────────────────────
print("\n[4] top_k parameter override...")
r4 = requests.post(f"{BASE}/query", json={"question": "Summarize the document", "top_k": 3}, timeout=120)
check("top_k query HTTP 200", r4.status_code == 200)
if r4.status_code == 200:
    check("top_k respected", len(r4.json().get("sources", [])) <= 3, f"{len(r4.json().get('sources',[]))} sources returned")

# ── Final verdict ──────────────────────────────────────────────
print()
print("=" * 60)
pct = int(passed / total * 100) if total else 0
print(f"  RESULT: {passed}/{total} checks PASSED ({pct}%)")
if pct == 100:
    print("  STATUS: FULLY OPERATIONAL")
elif pct >= 75:
    print("  STATUS: MOSTLY WORKING — minor issues")
else:
    print("  STATUS: NEEDS FIXES")
print("=" * 60)
sys.exit(0 if passed == total else 1)
