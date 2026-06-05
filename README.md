# SPRINT AI — Triathlon & Rehabilitation Video Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI: Lint](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/lint.yml/badge.svg)](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/lint.yml)
[![CI: Tests](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/tests.yml/badge.svg)](https://github.com/Kachalaba/Advanced-Swimming-Training-Bot-with-AI-features/actions/workflows/tests.yml)

AI-powered video analysis for coaches and physiotherapists. SPRINT AI turns an
ordinary phone or webcam video into clinical-grade biomechanics for **swimming,
running, cycling, dryland strength, and movement rehabilitation** — built on
YOLOv8 detection + MediaPipe pose estimation.

<p align="center">
  <img src="https://img.shields.io/badge/Swimming-40%2B_metrics-00D9FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Running-30%2B_metrics-10B981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Cycling-35%2B_metrics-F59E0B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Rehab-Live_ROM_%2B_symmetry-EF4444?style=for-the-badge" />
</p>

---

## Two ways to run it

The project ships **two front-ends that share the same analysis engine**
(`video_analysis/`):

| Stack | Command | Opens at | Status |
|-------|---------|----------|--------|
| **Web app** — Next.js + FastAPI | `docker compose up --build` | <http://localhost:3000> | Running & Rehabilitation wired end-to-end; other sports are landing pages being ported |
| **Streamlit prototype** | `python3 -m streamlit run app.py` | <http://localhost:8501> | All 8 disciplines, full feature parity |

### Web app (recommended)

```bash
docker compose up --build
# frontend → http://localhost:3000
# backend  → http://localhost:8000/api/health
```

Or run each part directly for development:

```bash
# backend (FastAPI)
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend (Next.js 16 / React 19)
cd frontend && npm install && npm run dev
```

### Streamlit prototype

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py        # → http://localhost:8501
```

### 🖥️ macOS desktop launcher

Get a double-clickable **Sprint AI** icon on your Desktop (starts the
docker-compose stack and opens the browser):

```bash
bash scripts/install-macos-launcher.sh
```

Details: [`docs/macos-launcher.md`](docs/macos-launcher.md).

---

## Disciplines

| Discipline | What it measures |
|------------|------------------|
| **Swimming** (40+ metrics) | Stroke phases, stroke rate, DPS, SWOLF, hand-entry angle, high-elbow catch, body roll, breathing pattern, kick symmetry |
| **Running** (30+ metrics) | Cadence, foot-strike type, overstriding, hip drop, arm symmetry, vertical oscillation, injury-risk score — with lock-on tracking |
| **Cycling** (35+ metrics) | Cadence, knee ROM, saddle-height/aero/stack fit, ankling, dead spots, pedal smoothness |
| **Dryland** | Exercise detection, joint angles, rep tempo, form scoring |
| **🩺 Rehabilitation** | Clinical ROM, left/right symmetry, target-ROM deficit, rep counting — **video upload _and_ live camera** |
| **History · AI Assistant · Tools** | Athlete progress + charts, Claude-powered chat & plans, side-by-side / zoom / highlights |

---

## 🩺 Rehabilitation / Kinesiotherapy

A markerless movement-rehab workspace for shoulder, elbow, hip and knee work.
Choose a protocol (shoulder flexion/abduction, elbow flexion, knee extension,
hip abduction); SPRINT measures each side independently and flags ROM deficits
against the clinical target.

**Two modes:**

- **Video upload** — `POST /api/analysis/rehabilitation` → live progress over
  Server-Sent Events → annotated video + ROM/symmetry report.
- **Live camera** — in-browser webcam streamed to the backend frame-by-frame,
  with a rolling-window analyzer and a real-time **postural map**:
  - shoulder / pelvis / trunk axes with severity colouring,
  - bilateral ROM, correct-rep count, and asymmetry index,
  - **relative camera-angle calibration** (ORB feature matching) so a tilted
    phone doesn't corrupt the geometry,
  - fullscreen mode (exit with `Esc`) and one-click save to athlete history,
  - frame scheduler tuned for smooth on-device capture (tested on MacBook Air M4).

> ⚕️ SPRINT AI is a training aid, not a medical device. With a single 2D camera,
> shoulder flexion and abduction are measured by the same joint angle — pick the
> protocol that matches your recording plane.

---

## Architecture

```
app.py                       Streamlit prototype (8 tabs)
pages/                       Streamlit pages (swimming … rehabilitation)
video_analysis/              Shared engine — analyzers, pose, DB, i18n
  ├─ base_analyzer.py        BaseAnalyzer mixin (angles, smoothing, EMA)
  ├─ stroke/running/cycling/exercise_analyzer.py
  ├─ rehab_analyzer.py       Bilateral ROM / symmetry / rep detection
  ├─ biomechanics_visualizer.py  MediaPipe skeleton + keypoints
  └─ athlete_database.py     Session persistence

backend/                     FastAPI REST API
  └─ app/
     ├─ api/                 analysis (running), rehabilitation (upload + live), athletes, health
     └─ services/            running, rehabilitation, camera_level (ORB roll), posture, jobs

frontend/                    Next.js 16 + React 19 + Tailwind (App Router)
  ├─ app/rehabilitation/     live workspace + upload
  └─ components/rehabilitation/  LiveRehabWorkspace, PostureOverlay, RehabUploader …

scripts/                     macOS desktop launcher
i18n/                        EN / UA translations
tests/unit/                  pytest (engine + backend services + API)
```

### Key REST endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/analysis/running` | Upload running video → `job_id` |
| `POST` | `/api/analysis/rehabilitation` | Upload rehab video → `job_id` |
| `GET`  | `/api/analysis/{kind}/{job_id}/events` | SSE progress + result |
| `GET`  | `/api/analysis/{kind}/{job_id}/video` | Annotated MP4 |
| `POST` | `/api/analysis/rehabilitation/live` | Open a live camera session |
| `POST` | `/api/analysis/rehabilitation/live/{id}/frame` | Analyze one webcam frame |
| `POST` | `/api/analysis/rehabilitation/live/{id}/save` | Save live session to history |

---

## Development

```bash
# Engine + backend tests (some skip without cv2 / fastapi installed)
pytest tests/unit/ -v

# Lint (ruff + black + isort + mypy)
make lint

# Frontend
cd frontend && npm run lint && npm run test     # eslint/tsc + vitest
```

CI (GitHub Actions): `lint.yml` (ruff/black/isort/mypy), `tests.yml` (pytest +
coverage), `frontend` (Next.js build), `docker.yml` (image build on tag).
Branches `main`, `claude/**`, `codex/**` trigger lint + tests.

### Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API for AI coaching + chat |
| `OPENAI_API_KEY` | OpenAI fallback for coaching |
| `ATHLETE_DB_PATH` | SQLAlchemy DB path (default `data/athletes_orm.db`) |
| `NEXT_PUBLIC_BACKEND_URL` | Backend URL for the frontend (default `http://localhost:8000`) |
| `SENTRY_DSN` | Error tracking |

---

## Tech stack

**Engine:** Python 3.11 · YOLOv8 (Ultralytics) · MediaPipe Pose (33 keypoints) ·
OpenCV · NumPy · EMA smoothing + lock-on tracking.
**Web app:** FastAPI · Server-Sent Events · WebRTC-style frame streaming ·
Next.js 16 · React 19 · TailwindCSS · Vitest.
**Data & AI:** SQLite / SQLAlchemy + Alembic · Claude API · Sentry.
**Ops:** Docker / docker-compose · GitHub Actions.

---

## Roadmap

- [x] Swimming, running, cycling, dryland analyzers
- [x] Athlete database, history, AI chat & plans, video tools
- [x] EN / UA localization, light / dark theme
- [x] FastAPI + Next.js web app (running pipeline)
- [x] **Rehabilitation: bilateral ROM, symmetry, video + live camera, posture map**
- [ ] Port swimming / cycling / dryland to the web pipeline
- [ ] Live-session lifecycle (idle-session reaper), multi-user hardening
- [ ] Garmin / Strava integration, training calendar

---

## Documentation

- [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Changelog](CHANGELOG.md)
- [macOS launcher](docs/macos-launcher.md)

## License

MIT — see [LICENSE](LICENSE).

<p align="center"><b>Built for coaches, physiotherapists, and athletes.</b><br>Swimming · Running · Cycling · Dryland · Rehabilitation</p>
