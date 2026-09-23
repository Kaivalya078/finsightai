"""
test_auth.py
------------
Unit tests for password hashing and verification.

Self-contained: no Mongo, no network, no models required.

Run with:
    python -m pytest tests/test_auth.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auth import hash_password, verify_password


def test_password_roundtrip():
    """A correct password verifies against its own hash."""
    h = hash_password("correct horse battery")
    assert verify_password("correct horse battery", h) is True


def test_wrong_password_rejected():
    h = hash_password("correct horse battery")
    assert verify_password("wrong password", h) is False


def test_oauth_account_has_no_password():
    """
    Google-only accounts store password_hash=None. Logging into one with
    email+password must fail cleanly, not raise AttributeError on .encode().
    Regression guard for the /login 500.
    """
    assert verify_password("anything", None) is False
    assert verify_password("anything", "") is False


def test_hashes_are_salted():
    """Same password, different hashes."""
    assert hash_password("same") != hash_password("same")
