"""
Cached factory functions for heavy analyzer objects.

Inside a running Streamlit app the factories are wrapped with
``@st.cache_resource`` so the same instances are reused across reruns
instead of being recreated (and reloading model weights) on every user
interaction.

Outside Streamlit (FastAPI backend, CLI tools, tests) the module falls
back to a thread-safe in-process cache with identical semantics, so the
factories stay the single supported way to obtain analyzers in every
runtime.

Usage in page files and backend services:
    from video_analysis.analyzer_factory import get_stroke_analyzer
    analyzer = get_stroke_analyzer(fps=30.0)

Note: cached instances are shared across callers. Analyzers that keep
per-run mutable state (e.g. RunningAnalyzer tracking buffers) must not
be driven by several jobs concurrently — create a dedicated instance
for parallel workloads instead.
"""

from __future__ import annotations

import logging
import threading
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _fallback_cache_resource(func: F) -> F:
    """Thread-safe memoization keyed by call arguments (Streamlit-free)."""
    lock = threading.Lock()
    instances: dict[tuple, Any] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        with lock:
            if key not in instances:
                instances[key] = func(*args, **kwargs)
            return instances[key]

    wrapper.cache_clear = instances.clear  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def _resolve_cache_decorator() -> Callable[[F], F]:
    """Use ``st.cache_resource`` under a live Streamlit runtime, else fallback."""
    try:
        import streamlit as st
        from streamlit import runtime

        if runtime.exists():
            return st.cache_resource
        logger.debug("Streamlit installed but no runtime; using fallback cache")
    except Exception:
        logger.debug("Streamlit unavailable; using fallback cache")
    return _fallback_cache_resource


cache_resource = _resolve_cache_decorator()


@cache_resource
def get_stroke_analyzer(fps: float = 30.0):
    """Return a cached StrokeAnalyzer instance."""
    from video_analysis.stroke_analyzer import StrokeAnalyzer  # lazy import

    logger.info("Creating StrokeAnalyzer (fps=%.1f)", fps)
    return StrokeAnalyzer(fps=fps)


@cache_resource
def get_running_analyzer(fps: float = 30.0, multi_person: bool = True):
    """Return a cached RunningAnalyzer instance."""
    from video_analysis.running_analyzer import RunningAnalyzer

    logger.info("Creating RunningAnalyzer (fps=%.1f)", fps)
    return RunningAnalyzer(fps=fps, multi_person=multi_person)


@cache_resource
def get_cycling_analyzer(fps: float = 30.0):
    """Return a cached CyclingAnalyzer instance."""
    from video_analysis.cycling_analyzer import CyclingAnalyzer

    logger.info("Creating CyclingAnalyzer (fps=%.1f)", fps)
    return CyclingAnalyzer(fps=fps)


@cache_resource
def get_biomechanics_visualizer(trajectory_length: int = 30):
    """Return a cached BiomechanicsVisualizer instance."""
    from video_analysis.biomechanics_visualizer import BiomechanicsVisualizer

    logger.info("Creating BiomechanicsVisualizer")
    return BiomechanicsVisualizer(trajectory_length=trajectory_length)


@cache_resource
def get_exercise_analyzer(fps: float = 30.0):
    """Return a cached ExerciseAnalyzer instance."""
    from video_analysis.exercise_analyzer import ExerciseAnalyzer

    logger.info("Creating ExerciseAnalyzer (fps=%.1f)", fps)
    return ExerciseAnalyzer(fps=fps)


@cache_resource
def get_rehab_analyzer(fps: float = 30.0):
    """Return a cached RehabAnalyzer instance."""
    from video_analysis.rehab_analyzer import RehabAnalyzer

    logger.info("Creating RehabAnalyzer (fps=%.1f)", fps)
    return RehabAnalyzer(fps=fps)


@cache_resource
def get_biomechanics_analyzer():
    """Return a cached BiomechanicsAnalyzer instance."""
    from video_analysis.biomechanics_analyzer import BiomechanicsAnalyzer

    logger.info("Creating BiomechanicsAnalyzer")
    return BiomechanicsAnalyzer()
