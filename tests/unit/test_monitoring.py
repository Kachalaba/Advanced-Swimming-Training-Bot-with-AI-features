"""Unit tests for video_analysis.monitoring — optional Sentry init."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_analysis import monitoring


def setup_function(_fn):
    monitoring._initialized = False


def test_no_dsn_returns_false(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert monitoring.init_sentry() is False


def test_blank_dsn_returns_false(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "   ")
    assert monitoring.init_sentry() is False


def test_missing_sdk_returns_false(monkeypatch):
    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.invalid/1")
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    assert monitoring.init_sentry() is False


def test_initializes_with_dsn_and_sdk(monkeypatch):
    fake_sdk = MagicMock()
    monkeypatch.setenv("SENTRY_DSN", "https://example@sentry.invalid/1")
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake_sdk)

    assert monitoring.init_sentry() is True
    fake_sdk.init.assert_called_once()
    assert fake_sdk.init.call_args.kwargs["dsn"] == "https://example@sentry.invalid/1"

    # Second call is an idempotent no-op.
    assert monitoring.init_sentry() is True
    fake_sdk.init.assert_called_once()
