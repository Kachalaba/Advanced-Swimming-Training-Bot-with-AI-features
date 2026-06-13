# Sport Platform Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the FastAPI + Next.js sport workflow truthful and persistent by completing running save/history and introducing a reusable sport overview contract.

**Architecture:** Persist sport-specific results in the existing `TrainingSession.full_analysis` JSON while exposing a normalized read model from the athlete API. The frontend consumes that model through one shared client and renders real data or honest empty states. Streamlit remains a legacy shell and does not receive new product behavior.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, SQLite, pytest, Next.js, React, TypeScript, Vitest, Testing Library.

---

### Task 1: Normalize Persisted Sport Sessions

**Files:**
- Create: `video_analysis/sport_sessions.py`
- Create: `tests/unit/test_sport_sessions.py`

- [ ] Write failing tests for running, swimming, malformed JSON, and basic fallback sessions.
- [ ] Run `pytest tests/unit/test_sport_sessions.py -v` and confirm import/behavior failures.
- [ ] Implement `normalize_sport_session()` and `build_sport_overview()` with omitted unsupported metrics.
- [ ] Run `pytest tests/unit/test_sport_sessions.py -v` and confirm all tests pass.
- [ ] Commit with `feat: add normalized sport session read model`.

### Task 2: Expose Sport Overview API

**Files:**
- Modify: `backend/app/api/athletes.py`
- Create: `tests/unit/test_sport_overview_api.py`

- [ ] Write failing API tests for empty history, persisted running history, unknown athlete, and invalid sport.
- [ ] Run `pytest tests/unit/test_sport_overview_api.py -v` and confirm route failures.
- [ ] Add Pydantic response models and `GET /api/athletes/{athlete_id}/sports/{sport}/overview`.
- [ ] Run the focused API tests and existing athlete tests.
- [ ] Commit with `feat: expose athlete sport overviews`.

### Task 3: Save Running Analysis

**Files:**
- Modify: `video_analysis/athlete_database.py`
- Modify: `backend/app/api/analysis.py`
- Modify: `backend/app/services/jobs.py`
- Create: `tests/unit/test_running_api.py`
- Modify: `tests/unit/test_athlete_database.py`

- [ ] Write failing tests for running metric persistence, save by athlete ID, name fallback, idempotency, wrong-kind jobs, incomplete jobs, and video copying.
- [ ] Run focused tests and confirm expected failures.
- [ ] Add running extraction to `_save_analysis_for_athlete()`.
- [ ] Add typed running job lookup, annotated-video persistence, and the running save endpoint.
- [ ] Run focused tests and the complete backend unit suite.
- [ ] Commit with `feat: persist running analysis sessions`.

### Task 4: Connect Frontend Sport Data

**Files:**
- Modify: `frontend/lib/api.ts`
- Create: `frontend/lib/sportOverview.ts`
- Create: `frontend/lib/sportOverview.test.ts`
- Modify: `frontend/components/sports/SportLanding.tsx`
- Modify: `frontend/components/sports/SportLanding.test.tsx`
- Modify: `frontend/app/running/page.tsx`
- Create: `frontend/app/running/page.test.tsx`
- Modify: `frontend/app/swimming/page.tsx`
- Create: `frontend/app/swimming/page.test.tsx`

- [ ] Write failing tests for overview API calls, metric formatting, running empty state, real sessions, and swimming persisted history.
- [ ] Run focused Vitest files and confirm expected failures.
- [ ] Add frontend sport overview types/client and a reusable loading hook.
- [ ] Extend `SportLanding` to support loading, retryable errors, and honest empty insights.
- [ ] Connect running and swimming pages to persisted overviews.
- [ ] Run focused and complete frontend tests.
- [ ] Commit with `feat: show persisted sport history`.

### Task 5: Add Running Save UX And Remove Demo Claims

**Files:**
- Modify: `frontend/lib/analysis.ts`
- Create: `frontend/lib/analysis.test.ts`
- Modify: `frontend/app/running/[jobId]/page.tsx`
- Create: `frontend/app/running/[jobId]/page.test.tsx`
- Modify: `frontend/app/cycling/page.tsx`
- Modify: `frontend/app/dryland/page.tsx`

- [ ] Write failing tests for running save requests, athlete selection, successful idempotent save, save errors, and absence of demo claims.
- [ ] Run focused frontend tests and confirm expected failures.
- [ ] Add `saveRunningAnalysis()` and athlete loading to the result page.
- [ ] Add save controls and confirmation without losing analysis state on error.
- [ ] Replace cycling and dryland fictional values with capability/readiness content and empty session lists.
- [ ] Run complete frontend tests, typecheck, and production build.
- [ ] Commit with `feat: complete running handoff and honest sport pages`.

### Task 6: Documentation And Release Verification

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`

- [ ] Document Next.js + FastAPI as primary and Streamlit as legacy/demo.
- [ ] Document sport overview and running save endpoints.
- [ ] Run `make lint`.
- [ ] Run `pytest tests/unit/ -v`.
- [ ] Run `npm test`, `npm run typecheck`, and `npm run build` in `frontend/`.
- [ ] Run `docker compose build`.
- [ ] Start the stack and browser-check running, swimming, cycling, and dryland.
- [ ] Push the branch, create a ready PR, wait for CI, and merge into `main`.
