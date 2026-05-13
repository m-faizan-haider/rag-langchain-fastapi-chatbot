import requests
import os
import sys
from dotenv import load_dotenv

load_dotenv()
BASE    = "http://127.0.0.1:8000"
passed  = 0
total   = 0

def check(name, got_status, expected, note=""):
    global passed, total
    total += 1
    ok = str(got_status).startswith(str(expected))
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"  [{status}] ({got_status}) {name}" + (f"  -- {note}" if note else ""))
    return ok

print("=" * 55)
print("  LIVE API DIAGNOSTIC")
print("=" * 55)

# 1. Root
r = requests.get(f"{BASE}/", timeout=5)
check("GET /", r.status_code, 200, r.json().get("version", ""))

# 2. Health
r = requests.get(f"{BASE}/health", timeout=5)
data = r.json()
check("GET /health", r.status_code, 200, "model: " + str(data.get("embedding_model", "?"))[-30:])

# 3. FAISS check
r = requests.get(f"{BASE}/faiss_check", timeout=5)
check("GET /faiss_check", r.status_code, 200)

# 4. Bad auth key
r = requests.post(f"{BASE}/auth/token", json={"api_key": "totally-wrong"}, timeout=5)
check("POST /auth/token (bad key -> 401)", r.status_code, 401)

# 5. Good auth key
api_key = os.getenv("RAG_API_KEYS", "dev-key-change-me").split(",")[0].strip()
r = requests.post(f"{BASE}/auth/token", json={"api_key": api_key}, timeout=5)
token = None
if r.status_code == 200:
    token = r.json().get("access_token", "")
    check("POST /auth/token (valid key -> JWT)", r.status_code, 200, f"{len(token)} char token")
else:
    check("POST /auth/token (valid key -> JWT)", r.status_code, 200, "key used: " + api_key)

# 6. Protected endpoint without token should be 401
r = requests.post(f"{BASE}/reload_faiss", json={"force_rebuild": False}, timeout=5)
check("POST /reload_faiss (no token -> 401)", r.status_code, 401)

# 7. Session endpoint
r = requests.get(f"{BASE}/session/nonexistent-session-id", timeout=5)
check("GET /session/:id", r.status_code, 200)

# 8. Metrics endpoint
r = requests.get(f"{BASE}/metrics", timeout=5)
check("GET /metrics", r.status_code, 200)

print()
print("=" * 55)
print(f"  RESULT: {passed}/{total} endpoints PASSED")
print("=" * 55)
sys.exit(0 if passed == total else 1)
