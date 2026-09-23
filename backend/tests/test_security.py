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
