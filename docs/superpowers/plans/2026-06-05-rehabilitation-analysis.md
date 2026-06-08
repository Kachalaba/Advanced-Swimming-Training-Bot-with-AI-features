# Rehabilitation Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete rehabilitation analysis workflow with bilateral ROM,
repetition counting, localized Streamlit UI, coaching, history persistence, and
unit coverage.

**Architecture:** A pure `RehabAnalyzer(BaseAnalyzer)` consumes named
MediaPipe-style keypoints. Protocol configuration is centralized in constants;
the Streamlit page reuses the current extraction/visualization pipeline and the
legacy database stores searchable aggregates plus the full JSON report.

**Tech Stack:** Python 3.11, Streamlit, NumPy, pandas/Plotly, SQLite, pytest.

---

### Task 1: Define The Analyzer Contract With Tests

**Files:**
- Create: `tests/unit/test_rehab_analyzer.py`
- Modify: `tests/fixtures/mock_keypoints.py`
- Modify: `video_analysis/base_analyzer.py`

- [ ] Add fixtures that generate bilateral shoulder and elbow cycles as named
  `{"x": ..., "y": ...}` MediaPipe keypoints.
- [ ] Add tests for empty input, invalid protocols, ROM/deficit metrics,
  asymmetry, and correct repetition counts.
- [ ] Run `pytest tests/unit/test_rehab_analyzer.py -v` and verify the import
  fails because `video_analysis.rehab_analyzer` does not exist.
- [ ] Extend `BaseAnalyzer._get_point()` to read mapping values with `x` and
  `y`, then keep the tests red until the analyzer exists.

### Task 2: Implement Rehabilitation Metrics

**Files:**
- Create: `video_analysis/rehab_analyzer.py`
- Modify: `video_analysis/constants.py`
- Modify: `video_analysis/analyzer_factory.py`

- [ ] Add `REHAB_PROTOCOLS`, joint definitions, smoothing, repetition-duration,
  completion, and symmetry thresholds to `constants.py`.
- [ ] Implement JSON-serializable bilateral ROM summaries and the two-threshold
  repetition state machine in `RehabAnalyzer`.
- [ ] Add deterministic feedback for ROM deficit, asymmetry, missing pose data,
  and successful target completion.
- [ ] Add cached `get_rehab_analyzer(fps=30.0)` to the analyzer factory.
- [ ] Run `pytest tests/unit/test_rehab_analyzer.py -v` until all tests pass.

### Task 3: Persist Rehabilitation Sessions

**Files:**
- Modify: `video_analysis/athlete_database.py`
- Modify: `tests/unit/test_athlete_database.py`

- [ ] Add a failing test that saves `session_type="rehab"` and checks protocol,
  repetitions, symmetry, completion score, and full JSON.
- [ ] Add the `rehab` extraction branch to `save_analysis_to_db()` without
  changing the database schema.
- [ ] Run the focused database test and then the full athlete database suite.

### Task 4: Add Localized Streamlit Workflow

**Files:**
- Create: `pages/rehabilitation.py`
- Modify: `i18n/translations.py`
- Modify: `app.py`
- Modify: `pages/history.py`

- [ ] Add all rehabilitation labels, statuses, metrics, feedback, errors, and
  protocol names in both Ukrainian and English.
- [ ] Implement upload, extraction, detection, pose collection, cached analyzer
  invocation, annotated-video generation, charts, feedback, AI coaching, and
  database saving.
- [ ] Register rehabilitation as tab 5 while preserving the order of all
  existing tabs.
- [ ] Add rehabilitation to history filters and icon selection.

### Task 5: Verify, Commit, And Publish

**Files:**
- Review all files changed by Tasks 1-4.

- [ ] Run `.venv/bin/python -m pytest tests/unit/ -v`.
- [ ] Run `make lint`.
- [ ] Run `git diff --check` and inspect `git diff --stat`.
- [ ] Confirm `data/athletes.db` remains unstaged.
- [ ] Commit implementation with a focused message.
- [ ] Push `claude/compassionate-rubin-gyhNU` to `origin`.

### Task 6: Add Live Webcam Analysis

**Files:**
- Create: `video_analysis/live_rehab.py`
- Create: `tests/unit/test_live_rehab.py`
- Modify: `pages/rehabilitation.py`
- Modify: `i18n/translations.py`
- Modify: `requirements.txt`
- Modify: `video_analysis/requirements.txt`

- [ ] Add a failing test for bounded frame collection and periodic rolling
  analysis without MediaPipe.
- [ ] Add a stateful `streamlit-webrtc` video processor that overlays current
  ROM, repetitions, and asymmetry.
- [ ] Add upload/live source selection, localized WebRTC controls, current
  report display, and live-session persistence.
- [ ] Re-run unit tests and the complete lint gate.
