# SPRINT AI — Codebase Guide for Claude Code

## Project overview

AI-powered triathlon video analysis platform. Analyzes swimming, running, cycling,
and dryland exercises via computer vision (YOLOv8 + MediaPipe). Built with Streamlit.

## Architecture

```
app.py                       # Entry point: Streamlit 7-tab UI
pages/
  swimming.py                # Swimming analysis page
  running.py                 # Running analysis page
  cycling.py                 # Cycling analysis page
  dryland.py                 # Dryland exercise page
  history.py                 # Athlete history + charts
  ai_assistant.py            # Claude API chat
  tools.py                   # Video utilities
video_analysis/
  base_analyzer.py           # Shared utilities (BaseAnalyzer mixin)
  analyzer_factory.py        # @st.cache_resource factory functions
  service_layer.py           # AnalysisService + AICoachProtocol
  models.py                  # SQLAlchemy ORM (AthleteModel, SessionModel)
  constants.py               # All magic numbers (SINGLE source of truth)
  stroke_analyzer.py         # Swimming: 40+ metrics
  running_analyzer.py        # Running: 30+ metrics
  cycling_analyzer.py        # Cycling: 35+ metrics
  exercise_analyzer.py       # Dryland exercise detection
  biomechanics_analyzer.py   # MediaPipe pose + hydrodynamics
  biomechanics_visualizer.py # Premium skeleton rendering (EMA + triple-guard)
  swimming_pose_analyzer.py  # Rotation-compensated pose analysis
  swimmer_detector.py        # YOLO + IoU tracking
  ai_coach.py                # Rule-based + Claude/OpenAI coaching
  ai_chat.py                 # Claude API chat with athlete context
  athlete_database.py        # Raw SQLite (legacy, active backend)
i18n/
  translations.py            # EN/UA translations via t("key")
tests/
  unit/                      # pytest unit tests (no cv2 required for base tests)
  integration/               # Full pipeline tests (requires cv2/mediapipe)
  fixtures/mock_keypoints.py # Reusable keypoint data generators
```

## Development commands

```bash
# Run app
python3 -m streamlit run app.py

# Run tests
pytest tests/unit/ -v
pytest tests/ -v --cov=video_analysis --cov-report=term-missing

# Code quality
make lint          # ruff + black + isort + mypy
black .
isort .
ruff check .
mypy video_analysis --ignore-missing-imports

# Docker
docker build -t sprint-ai .
docker compose up
```

## Key patterns

### Adding a new sport-specific metric

1. Add constants to `video_analysis/constants.py`
2. Add metric calculation to the relevant analyzer (inherit `BaseAnalyzer`)
3. Add translation keys to `i18n/translations.py` for both `"uk"` and `"en"`
4. Update the page to display the metric
5. Add a unit test in `tests/unit/`

### Shared utilities (BaseAnalyzer)

All sport analyzers inherit from `BaseAnalyzer` (`video_analysis/base_analyzer.py`):
- `_get_point(kps, name)` — unified keypoint extraction (list/tuple/object formats)
- `_calculate_angle(p1, p2, p3)` — 3-point joint angle (degrees)
- `_smooth(values, window)` — sliding-window mean filter
- `_ema(key, value)` — per-key exponential moving average

**Do NOT copy these methods into a new analyzer.** Inherit `BaseAnalyzer` instead.

### Model/analyzer caching

Never instantiate analyzers directly in page functions — use `analyzer_factory.py`:

```python
# Good
from video_analysis.analyzer_factory import get_stroke_analyzer
analyzer = get_stroke_analyzer(fps=30.0)

# Bad — creates a new object (and reloads models) on every Streamlit rerun
from video_analysis.stroke_analyzer import StrokeAnalyzer
analyzer = StrokeAnalyzer(fps=30.0)
```

### MediaPipe Pose caching

`get_pose_detector()` in `base_analyzer.py` returns a module-level cached instance.
**Do NOT** call `mp.solutions.pose.Pose(...)` directly inside `__init__` of new analyzers.

### i18n

All user-visible strings should go through `t("key")`:

```python
from i18n.translations import t
st.markdown(t("swim_title"))
status_text.text(t("status_extract_frames"))
```

Add new keys to **both** `"uk"` and `"en"` sections in `i18n/translations.py`.

### Database

- **Active backend**: `AthleteDatabase` in `athlete_database.py` (raw SQLite)
- **ORM models**: `AthleteModel` / `SessionModel` in `models.py` (SQLAlchemy)
- For new features, prefer the ORM via `get_engine()` + `Session`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude API key for AI coaching + chat |
| `OPENAI_API_KEY` | — | OpenAI fallback for AI coaching |
| `ATHLETE_DB_PATH` | `data/athletes_orm.db` | SQLAlchemy DB path |
| `SENTRY_DSN` | — | Sentry error tracking DSN |

## Testing

Unit tests in `tests/unit/` do **not** require cv2/mediapipe:
- `test_base_analyzer.py` — fully self-contained
- `test_stroke_analyzer.py` — uses mock keypoints
- `test_running_analyzer.py` — skipped if cv2 absent
- `test_cycling_analyzer.py` — skipped if cv2 absent

Run `pytest tests/unit/ -v` in any Python env after `pip install numpy pytest`.

## CI/CD

Three GitHub Actions workflows:
- `lint.yml`  — ruff, black, isort, mypy
- `tests.yml` — pytest + coverage
- `docker.yml`— Docker build + push (on tag)

Branches: `main`, `codex/**`, `claude/**` trigger lint on push.
