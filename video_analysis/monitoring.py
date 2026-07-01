"""Optional Sentry error tracking shared by both app shells.

Usage (Streamlit entry point or FastAPI main):

    from video_analysis.monitoring import init_sentry
    init_sentry()

Initialization is a no-op unless SENTRY_DSN is set and sentry-sdk is
installed, so the app keeps working in minimal environments.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_initialized = False


def init_sentry() -> bool:
    """Initialize Sentry when SENTRY_DSN is configured.

    Returns True when Sentry is active after the call, False otherwise.
    Safe to call multiple times and in environments without sentry-sdk.
    """
    global _initialized
    if _initialized:
        return True

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("SPRINT_AI_ENV", "production"),
            traces_sample_rate=0.0,
        )
    except Exception as exc:
        logger.warning("Sentry init failed: %s", exc)
        return False

    _initialized = True
    logger.info("Sentry error tracking enabled")
    return True
