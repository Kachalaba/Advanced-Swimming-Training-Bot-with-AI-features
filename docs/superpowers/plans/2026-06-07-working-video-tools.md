# Working Video Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build working Trim and Frame Extractor tools with temporary download, idempotent History persistence, and post-restart artifact download.

**Architecture:** FastAPI starts background jobs in the existing `JobRegistry`; a focused tools service probes and processes media with FFprobe/FFmpeg. Next.js uses one workspace component for upload, operation controls, SSE progress, download, and History save. Saved artifacts live on the mounted `/data` volume and are exposed only through job/session ids.

**Tech Stack:** Python 3.11, FastAPI, FFmpeg/FFprobe, SQLite, React 19, Next.js 16, Vitest, pytest, Docker Compose.

---

### Task 1: Tool Processing Service

**Files:**
- Create: `backend/app/services/tools.py`
- Test: `tests/unit/test_tools_service.py`

- [ ] **Step 1: Write failing tests for source probing and timestamp generation**

Add tests asserting:

```python
assert interval_timestamps(5.0, 2.0) == [0.0, 2.0, 4.0]
assert count_timestamps(5.0, 1) == [0.0]
assert count_timestamps(5.0, 5) == [0.0, 1.0, 2.0, 3.0, 4.0]
```

Also mock `subprocess.run` and assert `probe_video()` parses duration, dimensions, and audio presence from FFprobe JSON.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_tools_service.py
```

Expected: collection failure because `backend.app.services.tools` does not exist.

- [ ] **Step 3: Implement probe and timestamp helpers**

Create:

```python
@dataclass(frozen=True)
class SourceVideoInfo:
    duration_sec: float
    width: int
    height: int
    has_audio: bool

def probe_video(path: Path) -> SourceVideoInfo: ...
def interval_timestamps(duration_sec: float, interval_sec: float) -> list[float]: ...
def count_timestamps(duration_sec: float, frame_count: int) -> list[float]: ...
```

Reject non-positive duration/values and any extraction producing more than
`MAX_EXTRACTED_FRAMES = 200`.

- [ ] **Step 4: Add failing processing tests**

Mock subprocess execution and assert:

- trim uses `libx264`, `yuv420p`, `+faststart`, and AAC when audio exists
- frame extraction invokes FFmpeg once per timestamp without shell execution
- the ZIP contains sequential JPEG names and `manifest.json`
- failed commands raise a stable `ToolProcessingError`

- [ ] **Step 5: Implement trim and frame archive processing**

Expose:

```python
def trim_video(source: Path, output: Path, start_sec: float, end_sec: float, info: SourceVideoInfo) -> dict: ...
def extract_frame_archive(source: Path, output: Path, mode: str, value: float, info: SourceVideoInfo) -> dict: ...
```

Return metadata used by API result events.

- [ ] **Step 6: Run service tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_tools_service.py
```

Expected: all tests pass.

### Task 2: Tools Job API

**Files:**
- Create: `backend/app/api/tools.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/services/jobs.py`
- Test: `tests/unit/test_tools_api.py`

- [ ] **Step 1: Write failing API contract tests**

Use `TestClient` with fake processor functions and assert:

- `POST /api/tools/trim` returns a job id
- `POST /api/tools/frames` validates mode-specific fields
- SSE uses shared `stream_job_events`
- completed job download returns its generated media type and filename
- unfinished and unknown jobs return `409` and `404`

- [ ] **Step 2: Run API tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_tools_api.py
```

Expected: import failure because the tools router does not exist.

- [ ] **Step 3: Implement uploads and background workers**

Create trim/frame endpoints that:

- reuse `validate_video_upload()` and `copy_upload_with_limit()`
- create `kind="tool"` jobs
- store operation parameters on the job
- push progress, result, and error events
- register the router at `/api/tools`

Extend `Job` with JSON-safe operation/artifact metadata rather than adding a
second registry.

- [ ] **Step 4: Implement status, SSE, and temporary download**

Add:

```text
GET /api/tools/{job_id}
GET /api/tools/{job_id}/events
GET /api/tools/{job_id}/download
```

Only resolve the artifact path stored on the known job.

- [ ] **Step 5: Run API and shared-job tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_tools_api.py tests/unit/test_jobs.py
```

Expected: all tests pass.

### Task 3: History Persistence And Saved Downloads

**Files:**
- Modify: `backend/app/api/tools.py`
- Modify: `backend/app/api/athletes.py`
- Modify: `video_analysis/athlete_database.py`
- Modify: `docker-compose.yml`
- Test: `tests/unit/test_tools_api.py`
- Test: `tests/unit/test_athletes_api.py`

- [ ] **Step 1: Write failing persistence tests**

Assert that:

- repeated save calls return one session id
- artifacts copy to `SESSION_ARTIFACT_DIR/tools/<job_id>/`
- saved records use `session_type="tool"`
- session download rejects non-tool sessions and missing files
- athlete session payload exposes `artifact_download_url`

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_tools_api.py tests/unit/test_athletes_api.py
```

Expected: failures for missing save/download behavior and payload field.

- [ ] **Step 3: Implement idempotent save**

Add:

```text
POST /api/tools/{job_id}/save
GET /api/tools/history/{session_id}/download
```

Persist operation metadata through `save_analysis_to_db()` and set
`SESSION_ARTIFACT_DIR=/data/session-artifacts` in Compose.

- [ ] **Step 4: Extend History session contract**

For tool sessions return:

```json
{
  "artifact_download_url": "/api/tools/history/123/download",
  "exercise_type": "trim",
  "summary": "Trim 1.0s–4.0s"
}
```

- [ ] **Step 5: Run persistence tests**

Run:

```bash
.venv/bin/pytest -q tests/unit/test_tools_api.py tests/unit/test_athletes_api.py tests/unit/test_athlete_database.py
```

Expected: all tests pass.

### Task 4: Frontend API And Workspace

**Files:**
- Create: `frontend/lib/tools.ts`
- Create: `frontend/components/tools/VideoToolsWorkspace.tsx`
- Create: `frontend/components/tools/VideoToolsWorkspace.test.tsx`
- Modify: `frontend/app/tools/page.tsx`

- [ ] **Step 1: Write failing workspace tests**

Mock `frontend/lib/tools.ts` and assert:

- Trim shows start/end controls and submits their values
- Frame Extractor switches between interval and exact-count controls
- progress event updates the progress bar
- result exposes Download and Save to History
- saved state disables duplicate saves

- [ ] **Step 2: Run the component test and verify RED**

Run:

```bash
cd frontend && npm test -- components/tools/VideoToolsWorkspace.test.tsx
```

Expected: failure because the workspace component does not exist.

- [ ] **Step 3: Implement frontend API client**

Define discriminated event/result types and functions:

```typescript
uploadTrimVideo(...)
uploadFrameExtraction(...)
subscribeToolJob(...)
toolDownloadUrl(...)
saveToolJob(...)
```

- [ ] **Step 4: Implement the workspace**

Build one file selector, operation tabs, mode-specific numeric fields, client
validation, SSE progress, error state, result metadata, Download link, and Save
button.

- [ ] **Step 5: Compose the Tools page**

Replace the disabled prototype upload panel with `VideoToolsWorkspace`. Mark
Trim and Frame Extractor as `Ready`; keep the remaining four cards disabled and
labelled `Prototype`.

- [ ] **Step 6: Run frontend tests and build**

Run:

```bash
cd frontend && npm test && npm run build
```

Expected: all tests and production build pass.

### Task 5: History UI

**Files:**
- Modify: `frontend/lib/api.ts`
- Modify: `frontend/app/history/page.tsx`
- Test: `frontend/components/tools/VideoToolsWorkspace.test.tsx`

- [ ] **Step 1: Add the tool session type**

Extend `SessionSummary` with:

```typescript
artifact_download_url: string | null;
```

Add `tool` to History filters, labels, and icon mapping.

- [ ] **Step 2: Render saved artifact details**

Tool rows display operation title, operation summary, and an enabled Download
link when `artifact_download_url` is present.

- [ ] **Step 3: Run typecheck through production build**

Run:

```bash
cd frontend && npm run build
```

Expected: TypeScript and Next.js build pass.

### Task 6: Integration And Browser QA

**Files:**
- No production files unless verification finds a defect.

- [ ] **Step 1: Run full repository quality gates**

Run:

```bash
.venv/bin/pytest -q tests/unit/
PATH="$PWD/.venv/bin:$PATH" make lint
cd frontend && npm test && npm run build
docker compose config --quiet
docker compose build backend frontend
docker compose up -d
```

Expected: zero failures and healthy containers.

- [ ] **Step 2: Verify real trim**

Upload `/tmp/sprint-running-demo.MOV`, trim `1.0` to `4.0`, then verify:

- result MP4 uses H.264/yuv420p
- browser video duration is approximately 3 seconds
- temporary download succeeds

- [ ] **Step 3: Verify both frame modes**

Verify:

- interval `2s` produces timestamps `0, 2, 4`
- exact count `5` produces five JPEGs
- both ZIP manifests match the files

- [ ] **Step 4: Verify persistence**

Save trim and one frame archive to History, restart backend, then verify both
saved-download endpoints still return the artifacts.

- [ ] **Step 5: Browser QA**

Using the Browser plugin, exercise:

```text
/tools -> choose operation -> upload -> process -> download/save -> /history
```

Check page identity, non-blank DOM, no framework overlay, no relevant console
errors, visual screenshot, and interaction state changes.

### Task 7: Git Publication

**Files:**
- Stage only source, tests, docs, and configuration.
- Exclude generated `data/athletes.db`, generated artifacts, and `.superpowers/`.

- [ ] **Step 1: Review the final diff**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

- [ ] **Step 2: Commit the accumulated presentation-readiness work**

Stage the existing verified source changes from the previous pass, excluding
generated local state, and commit:

```bash
git commit -m "feat: stabilize presentation workflows"
```

- [ ] **Step 3: Commit working video tools**

Stage tool implementation, tests, History integration, and this plan:

```bash
git commit -m "feat: add working video tools"
```

- [ ] **Step 4: Push current branch**

Run:

```bash
git push -u origin claude/compassionate-rubin-gyhNU
```

Expected: remote branch updates successfully.
