"""
Phase 0 security regressions. No Mongo, no models, no network.

Run with:
    python -m pytest tests/test_security.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))


def _import_auth_with(secret: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "JWT_SECRET": secret}
    return subprocess.run(
        [sys.executable, "-c", "import auth"],
        cwd=BACKEND, env=env, capture_output=True, text=True,
    )


def test_startup_fails_without_jwt_secret():
    r = _import_auth_with("")
    assert r.returncode != 0
    assert "JWT_SECRET" in r.stderr


def test_startup_fails_with_committed_placeholder_secret():
    r = _import_auth_with("change-me-to-a-strong-random-secret")
    assert r.returncode != 0
    assert "JWT_SECRET" in r.stderr


def test_startup_succeeds_with_real_secret():
    r = _import_auth_with("x" * 48)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# /chat cache: a hit must still be persisted to the asking user's conversation
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_ready(monkeypatch):
    """main.app with the corpus/LLM guards satisfied and Mongo writes recorded."""
    import main
    monkeypatch.setattr(main, "corpus_manager", SimpleNamespace(is_indexed=True))
    monkeypatch.setattr(main, "llm_client", SimpleNamespace(is_configured=True, model="test"))
    saved = []

    def fake_create(user_id, title, user_msg, assistant_msg):
        saved.append(("create", user_id, user_msg["content"]))
        return f"conv-{user_id}"

    def fake_append(conv_id, user_id, user_msg, assistant_msg):
        saved.append(("append", user_id, conv_id))

    monkeypatch.setattr(main, "db_create_conversation", fake_create)
    monkeypatch.setattr(main, "db_append_to_conversation", fake_append)
    main.response_cache.invalidate_all()
    yield main, saved
    main.response_cache.invalidate_all()


def _auth(user_id):
    from auth import create_access_token
    return {"Authorization": f"Bearer {create_access_token(user_id, f'{user_id}@x.io')}"}


def test_cache_hit_is_persisted_per_user(app_ready):
    main, saved = app_ready
    q = "What are the key risk factors?"
    main.response_cache.set(q, {
        "answer": "cached answer", "citations": [], "evidence": [],
        "metadata": {}, "follow_ups": [],
    })
    client = TestClient(main.app)

    a = client.post("/chat", json={"question": q}, headers=_auth("alice")).json()
    b = client.post("/chat", json={"question": q}, headers=_auth("bob")).json()
    b2 = client.post("/chat", json={"question": q, "conversation_id": b["conversation_id"]},
                     headers=_auth("bob")).json()

    assert (a["conversation_id"], b["conversation_id"]) == ("conv-alice", "conv-bob")
    assert b2["conversation_id"] == "conv-bob"
    assert saved == [("create", "alice", q), ("create", "bob", q), ("append", "bob", "conv-bob")]
    assert a["metadata"]["cached"] is True
