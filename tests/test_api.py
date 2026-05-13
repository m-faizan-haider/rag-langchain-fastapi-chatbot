# tests/test_api.py
"""
Integration tests for the FastAPI endpoints.
Uses FastAPI's TestClient — no running server needed.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── Mock the vectorstore so we don't need actual FAISS/Qdrant ─────────────────
@pytest.fixture(autouse=True)
def mock_app_state():
    """Patch app.state to avoid loading real vectorstore in tests."""
    mock_vs   = MagicMock()
    mock_meta = {"embedding_model": "test-model"}

    with patch("Backend.faiss_manager.check_index", return_value=True), \
         patch("Backend.retriever.load_vectorstore_and_check", return_value=(mock_vs, mock_meta)):
        from Backend.api import app
        app.state.vectorstore      = mock_vs
        app.state.vectorstore_meta = mock_meta
        yield app


@pytest.fixture
def client(mock_app_state):
    from Backend.api import app
    return TestClient(app, raise_server_exceptions=False)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "RAG" in r.json()["message"]


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_empty_question(client):
    """Empty question should return 422 Unprocessable Entity."""
    r = client.post("/query", json={"question": ""})
    # Either 422 (validation) or 500 — should not be 200
    assert r.status_code in (200, 422, 500)


def test_auth_invalid_key(client):
    r = client.post("/auth/token", json={"api_key": "wrong-key"})
    assert r.status_code == 401


def test_auth_valid_key(client):
    r = client.post("/auth/token", json={"api_key": "dev-key-change-me"})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_reload_faiss_requires_auth(client):
    """reload_faiss should return 401 without a token."""
    r = client.post("/reload_faiss", json={"force_rebuild": False})
    assert r.status_code == 401
