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


# ---------------------------------------------------------------------------
# /upload: auth, PDF-only, size cap, no client-controlled paths
# ---------------------------------------------------------------------------

import tempfile

PDF = b"%PDF-1.7\n" + b"0" * 100


@pytest.fixture
def upload_ready(monkeypatch):
    """main.app with ingestion faked out; records the path handed to it."""
    import main
    seen = {}

    class FakeCorpus:
        num_chunks = 3

        def __init__(self, *_):
            pass

        def add_document(self, pdf_path, **_):
            seen["pdf_path"] = pdf_path
            return 3

    monkeypatch.setattr(main, "RetrieverPipeline", lambda: None)
    monkeypatch.setattr(main, "CorpusManager", FakeCorpus)
    monkeypatch.setattr(main, "corpus_router", SimpleNamespace(register_session=lambda *a: None))
    main.limiter.reset()
    return TestClient(main.app), seen


def _upload(client, data, filename="report.pdf", headers=None):
    return client.post("/upload", files={"file": (filename, data, "application/pdf")},
                       data={"company_name": "ACME"}, headers=headers)


def test_upload_requires_login(upload_ready):
    client, _ = upload_ready
    assert _upload(client, PDF).status_code == 401


def test_upload_rejects_non_pdf_bytes(upload_ready):
    client, _ = upload_ready
    r = _upload(client, b"MZ\x90\x00 not a pdf", headers=_auth("alice"))
    assert r.status_code == 415


def test_upload_rejects_oversize(upload_ready, monkeypatch):
    import main
    client, _ = upload_ready
    monkeypatch.setattr(main.settings, "UPLOAD_MAX_MB", 1)
    r = _upload(client, PDF + b"0" * (1024 * 1024), headers=_auth("alice"))
    assert r.status_code == 413


def test_upload_ignores_client_filename_path(upload_ready):
    client, seen = upload_ready
    r = _upload(client, PDF, filename="../../escape.pdf", headers=_auth("alice"))
    assert r.status_code == 200
    tmp = os.path.realpath(tempfile.gettempdir())
    written = os.path.realpath(seen["pdf_path"])
    assert os.path.commonpath([tmp, written]) == tmp
    assert os.path.dirname(written) != tmp  # stays inside its own upload dir


# ---------------------------------------------------------------------------
# Rate limiting + anonymous trial
# ---------------------------------------------------------------------------

@pytest.fixture
def limited(app_ready, monkeypatch):
    """Cached answer for Q, anonymous cap of 2, fresh limiter counters."""
    main, saved = app_ready
    monkeypatch.setattr(main.settings, "ANON_FREE_QUESTIONS", 2)
    main.limiter.reset()
    main.response_cache.set("Revenue?", {"answer": "a", "citations": [], "evidence": [],
                                   "metadata": {}, "follow_ups": []})
    yield TestClient(main.app), saved
    main.limiter.reset()


def test_anonymous_can_ask_without_login_and_nothing_is_saved(limited):
    client, saved = limited
    r = client.post("/chat", json={"question": "Revenue?"})
    assert r.status_code == 200
    assert r.json()["conversation_id"] is None
    assert saved == []


def test_anonymous_trial_is_capped_with_sign_in_prompt(limited):
    client, _ = limited
    for _ in range(2):
        assert client.post("/chat", json={"question": "Revenue?"}).status_code == 200
    r = client.post("/chat", json={"question": "Revenue?"})
    assert r.status_code == 429
    assert "sign in" in r.json()["detail"].lower()


def test_signed_in_user_is_not_held_to_the_anonymous_cap(limited):
    client, _ = limited
    for _ in range(3):
        assert client.post("/chat", json={"question": "Revenue?"}, headers=_auth("alice")).status_code == 200


def test_login_is_rate_limited_per_ip(limited, monkeypatch):
    import main
    client, _ = limited
    monkeypatch.setattr(main, "users_collection", SimpleNamespace(find_one=lambda q: None))
    codes = [client.post("/login", json={"email": "a@x.io", "password": "p"}).status_code
             for _ in range(11)]
    assert codes[:10] == [401] * 10
    assert codes[10] == 429


# ---------------------------------------------------------------------------
# Index tarball bootstrap (ASSET_MODE=remote)
# ---------------------------------------------------------------------------

import io
import tarfile


def _tarball(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def remote_assets(monkeypatch):
    import asset_manager
    served = {}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield served["bytes"]

    monkeypatch.setattr(asset_manager.settings, "ASSET_MODE", "remote")
    monkeypatch.setattr(asset_manager.settings, "HF_CACHE_URL", "https://example.test/idx.tar.gz")
    monkeypatch.setattr(asset_manager.requests, "get", lambda *a, **k: FakeResponse())
    return asset_manager, served


def test_remote_index_lands_in_configured_dir(remote_assets, tmp_path):
    asset_manager, served = remote_assets
    served["bytes"] = _tarball({"index_cache/faiss.index": b"idx", "index_cache/chunks.pkl": b"c"})
    cache_dir = tmp_path / "data" / "index_cache"
    asset_manager.ensure_index_cache(str(cache_dir))
    assert (cache_dir / "faiss.index").read_bytes() == b"idx"


def test_tarball_cannot_write_outside_its_dir(remote_assets, tmp_path):
    asset_manager, served = remote_assets
    served["bytes"] = _tarball({"../../evil.txt": b"pwned", "faiss.index": b"idx"})
    cache_dir = tmp_path / "a" / "index_cache"
    with pytest.raises(tarfile.TarError):
        asset_manager.ensure_index_cache(str(cache_dir))
    assert not list(tmp_path.rglob("evil.txt"))
    assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# Generation wiring: intent prompts + citation verification
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_pipeline(monkeypatch):
    """answer_question() over one fake chunk; records the system prompt sent."""
    import main
    from core.metadata_schema import RetrievalResult
    sent = {}
    chunk = RetrievalResult(chunk_id="chunk_7", score=0.8, snippet="Revenue rose 10%.")

    def generate(system_prompt, user_message):
        sent["system"] = system_prompt
        return "Revenue rose 10% (chunk_7). Margins held at 20% (chunk_99)."

    def retrieve_context(raw_query, **_):
        intent = "comparison" if " vs " in raw_query else "single_entity"
        return [chunk], SimpleNamespace(intent=intent)

    monkeypatch.setattr(main, "corpus_manager", SimpleNamespace(is_indexed=True, chunk_metadata={}))
    monkeypatch.setattr(main, "pipeline", SimpleNamespace(chunks=[]))
    monkeypatch.setattr(main, "llm_client", SimpleNamespace(is_configured=True, model="t", generate=generate))
    monkeypatch.setattr(main, "retrieve_context", retrieve_context)
    monkeypatch.setattr(main, "refine_results", lambda results, **_: results)
    monkeypatch.setattr(main, "is_intelligent_parsing_enabled", lambda: False)
    main.response_cache.invalidate_all()
    yield main, sent
    main.response_cache.invalidate_all()


def test_comparison_question_gets_comparison_prompt(fake_pipeline):
    main, sent = fake_pipeline
    main.answer_question("TCS vs Infosys revenue")
    assert "COMPARISON" in sent["system"]
    main.answer_question("TCS revenue in 2024")
    assert "COMPARISON" not in sent["system"]


def test_hallucinated_citations_are_flagged(fake_pipeline):
    main, _ = fake_pipeline
    check = main.answer_question("TCS revenue in 2024")["metadata"]["citation_check"]
    assert check["valid_citations"] == ["chunk_7"]
    assert check["invalid_citations"] == ["chunk_99"]
