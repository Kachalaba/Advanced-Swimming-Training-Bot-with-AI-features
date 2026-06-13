# Rehabilitation Patient Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bilingual Patient Progress Dashboard from real persisted rehabilitation analyses.

**Architecture:** Add a read-only normalization endpoint to the existing athlete API, then consume it from a focused Next.js client page. Keep comparison math and localized observation generation in a pure frontend module, and render the trend with a small responsive SVG component.

**Tech Stack:** FastAPI, Pydantic, SQLite, pytest, Next.js 16, React 19, TypeScript, Vitest, Tailwind CSS.

---

### Task 1: Backend Progress Contract

**Files:**
- Modify: `backend/app/api/athletes.py`
- Modify: `tests/unit/test_athletes_api.py`

- [ ] **Step 1: Write failing API tests**

Add tests that provide valid, malformed, and incomplete rehabilitation
`TrainingSession` objects through the fake database. Assert that the endpoint:

```text
GET /api/athletes/3/rehabilitation/progress
```

returns the athlete, chronologically ordered normalized sessions, available
protocols, nullable quality metadata, and filters invalid stored reports.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest tests/unit/test_athletes_api.py -v
```

Expected: the new endpoint returns `404` because it does not exist.

- [ ] **Step 3: Implement normalization**

Add Pydantic response models and private helpers that parse
`session.full_analysis`, validate known rehab protocols and finite numeric
measurements, and sort observations by date.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
pytest tests/unit/test_athletes_api.py -v
```

Expected: all athlete API tests pass.

### Task 2: Frontend Data And Comparison Model

**Files:**
- Modify: `frontend/lib/api.ts`
- Create: `frontend/lib/rehabProgress.ts`
- Create: `frontend/lib/rehabProgress.test.ts`

- [ ] **Step 1: Write failing comparison tests**

Cover baseline/latest selection, metric deltas, one-session behavior, protocol
filtering, and neutral UK/EN observation copy.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -- lib/rehabProgress.test.ts
```

Expected: fail because the module does not exist.

- [ ] **Step 3: Implement types and pure selectors**

Add the progress response types and API method. Implement pure helpers for
protocol summaries, comparison metrics, and localized observations.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
npm test -- lib/rehabProgress.test.ts
```

Expected: all progress model tests pass.

### Task 3: Progress Visualization

**Files:**
- Create: `frontend/components/rehabilitation/RehabProgressChart.tsx`
- Create: `frontend/components/rehabilitation/RehabProgressChart.test.tsx`

- [ ] **Step 1: Write failing chart test**

Assert that the chart exposes left ROM, right ROM, and symmetry legends, point
values, and an accessible summary for multiple sessions.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -- components/rehabilitation/RehabProgressChart.test.tsx
```

Expected: fail because the component does not exist.

- [ ] **Step 3: Implement responsive SVG**

Render a dependency-free SVG chart with shared x positions, separate ROM and
percentage scaling, visible legends, and a text fallback for screen readers.

- [ ] **Step 4: Verify GREEN**

Run the focused component test and expect it to pass.

### Task 4: Patient Progress Dashboard

**Files:**
- Create: `frontend/app/rehabilitation/progress/page.tsx`
- Create: `frontend/app/rehabilitation/progress/page.test.tsx`
- Create: `frontend/lib/rehabProgressCopy.ts`

- [ ] **Step 1: Write failing page tests**

Cover loading, successful multi-session comparison, athlete switching, protocol
switching, UK/EN language switching, empty data, one-session state, and retry
after an API error.

- [ ] **Step 2: Verify RED**

Run:

```bash
npm test -- app/rehabilitation/progress/page.test.tsx
```

Expected: fail because the route is absent.

- [ ] **Step 3: Implement the page**

Fetch athletes and selected progress data in parallel where possible. Render
the clinical header, protocol selector, comparison metrics, chart, observation,
limitations, and session timeline using existing dark UI conventions.

- [ ] **Step 4: Verify GREEN**

Run the focused page test and expect all dashboard states to pass.

### Task 5: Rehabilitation Entry Point

**Files:**
- Modify: `frontend/app/rehabilitation/page.tsx`
- Modify: `frontend/app/rehabilitation/page.test.tsx`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] **Step 1: Write a failing navigation test**

Assert that the handoff action rail contains a localized link to
`/rehabilitation/progress`.

- [ ] **Step 2: Verify RED**

Run the rehabilitation page test and confirm the link is absent.

- [ ] **Step 3: Add the action and translations**

Add a compact `Patient progress` action that remains available independently
of the current analysis state.

- [ ] **Step 4: Verify GREEN**

Run the rehabilitation page test and expect all existing and new flows to pass.

### Task 6: Full Verification And Browser QA

**Files:**
- Modify only if verification identifies a defect.

- [ ] **Step 1: Run complete verification**

```bash
pytest tests/unit/test_athletes_api.py -v
ruff check backend/app/api/athletes.py tests/unit/test_athletes_api.py
cd frontend && npm test
cd frontend && npm run typecheck
cd frontend && npm run build
git diff --check
```

- [ ] **Step 2: Rebuild the frontend and backend containers**

```bash
docker compose build backend frontend
docker compose up -d backend frontend
```

- [ ] **Step 3: Browser QA**

Verify `/rehabilitation/progress` on desktop and 390x844 mobile. Exercise
athlete, protocol, and locale switching; check the chart and empty states; then
confirm no overflow or console errors.

### Task 7: Delivery

**Files:**
- Stage only Patient Progress files.

- [ ] **Step 1: Commit**

```bash
git add <explicit patient-progress files>
git commit -m "feat: add rehabilitation patient progress"
```

- [ ] **Step 2: Push and open a draft PR**

Push `codex/rehab-patient-progress` and open a draft PR into `main` with the
validation evidence.
