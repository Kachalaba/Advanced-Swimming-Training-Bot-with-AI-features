"""Unit tests for the Streamlit-optional analyzer factory cache."""

import importlib
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_analysis import analyzer_factory
from video_analysis.analyzer_factory import _fallback_cache_resource


class TestFallbackCacheResource:
    """Tests for the Streamlit-free memoization decorator."""

    def test_same_args_return_same_instance(self):
        calls = []

        @_fallback_cache_resource
        def make(fps=30.0):
            calls.append(fps)
            return object()

        assert make(fps=30.0) is make(fps=30.0)
        assert calls == [30.0]

    def test_different_args_return_different_instances(self):
        @_fallback_cache_resource
        def make(fps=30.0):
            return object()

        assert make(fps=30.0) is not make(fps=60.0)

    def test_positional_and_keyword_args_are_distinct_keys(self):
        @_fallback_cache_resource
        def make(fps=30.0):
            return object()

        # Same logical call spelled differently — separate cache entries
        # are acceptable; the invariant is that repeats hit the cache.
        assert make(30.0) is make(30.0)
        assert make(fps=30.0) is make(fps=30.0)

    def test_cache_clear_resets_instances(self):
        @_fallback_cache_resource
        def make():
            return object()

        first = make()
        make.cache_clear()
        assert make() is not first

    def test_thread_safety_single_instance_under_contention(self):
        created = []

        @_fallback_cache_resource
        def make():
            created.append(1)
            return object()

        results = []
        threads = [threading.Thread(target=lambda: results.append(make())) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(created) == 1
        assert all(r is results[0] for r in results)

    def test_preserves_function_metadata(self):
        @_fallback_cache_resource
        def make_thing():
            """Docstring."""
            return object()

        assert make_thing.__name__ == "make_thing"
        assert make_thing.__doc__ == "Docstring."


class TestDecoratorResolution:
    """The module must import and cache without Streamlit installed."""

    def test_import_without_streamlit_uses_fallback(self, monkeypatch):
        # Simulate an environment where `import streamlit` fails
        # (e.g. the FastAPI backend container).
        monkeypatch.setitem(sys.modules, "streamlit", None)
        monkeypatch.setitem(sys.modules, "streamlit.runtime", None)
        module = importlib.reload(analyzer_factory)
        try:
            assert module.cache_resource is module._fallback_cache_resource
        finally:
            monkeypatch.undo()
            importlib.reload(analyzer_factory)

    def test_factories_are_wrapped(self):
        # Every public factory must go through the cache decorator.
        for name in (
            "get_stroke_analyzer",
            "get_running_analyzer",
            "get_cycling_analyzer",
            "get_biomechanics_visualizer",
            "get_exercise_analyzer",
            "get_rehab_analyzer",
            "get_biomechanics_analyzer",
        ):
            func = getattr(analyzer_factory, name)
            assert hasattr(func, "__wrapped__") or hasattr(func, "clear"), name
