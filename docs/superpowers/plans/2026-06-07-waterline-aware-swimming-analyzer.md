# Waterline-Aware Swimming Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a working side-on freestyle analysis flow that detects the waterline, selects reliable stroke cycles, reports five confidence-aware technique zones, links the main issue to an exact video moment, prescribes a drill and mini-set, and saves the result to History.

**Architecture:** Add three focused `BaseAnalyzer`-compatible domain modules for waterline estimation, cycle selection, and technique scoring. Orchestrate them in a FastAPI swimming service that reuses the existing upload/job/SSE/video encoding and persistence patterns. Replace the demo swimming page with a native SPRINT AI upload and result experience while preserving the current visual system.

**Tech Stack:** Python 3.11, NumPy, OpenCV, MediaPipe, existing YOLO swimmer detector, FastAPI, pytest, Next.js 16, React 19, TypeScript, Tailwind CSS, Vitest, Testing Library, FFmpeg.

---

## File Structure

### New Backend Domain Modules

- `video_analysis/waterline_analyzer.py`
  - estimates and smooths a waterline and reports confidence
- `video_analysis/swimming_cycle_selector.py`
  - identifies complete stroke cycles and ranks the best non-overlapping cycles
- `video_analysis/swimming_technique_analyzer.py`
  - calculates confidence, five-zone summaries, primary issue, drill, and mini-set
- `backend/app/services/swimming.py`
  - orchestrates video quality, swimmer tracking, waterline, pose, cycles, technique,
    annotated video, and structured events

### New Frontend Modules

- `frontend/lib/swimming.ts`
  - typed swimming API client, SSE subscription, save, and video URL
- `frontend/components/swimming/SwimmingUploader.tsx`
  - native upload and filming guidance
- `frontend/components/swimming/SwimmingAnalysisWorkspace.tsx`
  - progress, result, evidence seeking, cycles, five zones, prescription, and save
- `frontend/app/swimming/[jobId]/page.tsx`
  - job result route

### Modified Modules

- `backend/app/api/analysis.py`
  - swimming upload, status, SSE, annotated video, and idempotent save endpoints
- `frontend/app/swimming/page.tsx`
  - replace demo-only uploader with the working swimming uploader and accurate copy
- `video_analysis/constants.py`
  - named waterline, confidence, and cycle thresholds

### New Tests

- `tests/unit/test_waterline_analyzer.py`
- `tests/unit/test_swimming_cycle_selector.py`
- `tests/unit/test_swimming_technique_analyzer.py`
- `tests/unit/test_swimming_service.py`
- `tests/unit/test_swimming_api.py`
- `frontend/lib/swimming.test.ts`
- `frontend/components/swimming/SwimmingAnalysisWorkspace.test.tsx`

## Task 1: Confidence-Aware Technique Contract

**Files:**
- Create: `video_analysis/swimming_technique_analyzer.py`
- Create: `tests/unit/test_swimming_technique_analyzer.py`
- Modify: `video_analysis/constants.py`

- [ ] **Step 1: Write failing confidence and diagnosis tests**

Create tests that define the public contract:

```python
from video_analysis.swimming_technique_analyzer import (
    ConfidenceInputs,
    SwimmingTechniqueAnalyzer,
)


def test_metric_confidence_uses_agreed_weights():
    analyzer = SwimmingTechniqueAnalyzer()
    confidence = analyzer.metric_confidence(
        ConfidenceInputs(
            landmark_visibility=1.0,
            temporal_continuity=0.8,
            waterline_clarity=0.6,
            identity_stability=1.0,
            cycle_coverage=0.5,
        )
    )
    assert confidence == 0.79
    assert analyzer.confidence_level(confidence) == "high"


def test_missing_zone_is_excluded_from_coverage_and_score():
    result = SwimmingTechniqueAnalyzer().build_result(
        cycle_metrics=[
            _cycle("cycle-1", body_issue=True, catch_available=False),
            _cycle("cycle-2", body_issue=True, catch_available=False),
        ]
    )
    assert result["coverage"] == {"available_zones": 4, "total_zones": 5}
    catch = next(zone for zone in result["zones"] if zone["id"] == "catch")
    assert catch["status"] == "insufficient_data"
    assert catch["score"] is None


def test_primary_issue_requires_two_confirming_cycles():
    result = SwimmingTechniqueAnalyzer().build_result(
        cycle_metrics=[
            _cycle("cycle-1", body_issue=True),
            _cycle("cycle-2", body_issue=False),
        ]
    )
    assert result["primary_issue"] is None


def test_primary_issue_includes_evidence_and_prescription():
    result = SwimmingTechniqueAnalyzer().build_result(
        cycle_metrics=[
            _cycle("cycle-1", body_issue=True, peak_sec=2.4),
            _cycle("cycle-2", body_issue=True, peak_sec=4.8),
        ]
    )
    assert result["primary_issue"]["zone_id"] == "body_position"
    assert result["primary_issue"]["evidence"][0]["peak_sec"] == 2.4
    assert result["prescription"]["drill"]["name"]
    assert result["prescription"]["mini_set"]["repetitions"] > 0
```

Test helpers must produce all five zones with deterministic confidence inputs,
metric values, and evidence intervals.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/unit/test_swimming_technique_analyzer.py -v
```

Expected: collection fails because `swimming_technique_analyzer` does not exist.

- [ ] **Step 3: Add named constants**

Add:

```python
SWIM_CONFIDENCE_HIGH = 0.78
SWIM_CONFIDENCE_MEDIUM = 0.55
SWIM_MIN_DIAGNOSIS_CYCLES = 2
SWIM_TARGET_CYCLE_COUNT = 5
SWIM_MIN_PARTIAL_CYCLE_COUNT = 2
```

- [ ] **Step 4: Implement the minimal technique analyzer**

Implement:

```python
@dataclass(frozen=True)
class ConfidenceInputs:
    landmark_visibility: float
    temporal_continuity: float
    waterline_clarity: float
    identity_stability: float
    cycle_coverage: float


class SwimmingTechniqueAnalyzer(BaseAnalyzer):
    ZONE_IDS = (
        "body_position",
        "rotation",
        "catch",
        "breathing",
        "kick",
    )

    def metric_confidence(self, inputs: ConfidenceInputs) -> float:
        value = (
            0.35 * inputs.landmark_visibility
            + 0.25 * inputs.temporal_continuity
            + 0.15 * inputs.waterline_clarity
            + 0.15 * inputs.identity_stability
            + 0.10 * inputs.cycle_coverage
        )
        return round(max(0.0, min(1.0, value)), 2)

    def confidence_level(self, value: float) -> str:
        if value >= SWIM_CONFIDENCE_HIGH:
            return "high"
        if value >= SWIM_CONFIDENCE_MEDIUM:
            return "medium"
        return "insufficient"
```

`build_result()` aggregates only zones with two sufficient cycle observations,
ranks repeated issues by impact and confidence, and maps issue codes to a
deterministic explanation, drill, and mini-set.

- [ ] **Step 5: Run focused and shared analyzer tests**

Run:

```bash
pytest tests/unit/test_swimming_technique_analyzer.py tests/unit/test_base_analyzer.py tests/unit/test_stroke_analyzer.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add video_analysis/constants.py video_analysis/swimming_technique_analyzer.py tests/unit/test_swimming_technique_analyzer.py
git commit -m "feat: add confidence-aware swimming technique contract"
```

## Task 2: Reliable Stroke Cycle Selection

**Files:**
- Create: `video_analysis/swimming_cycle_selector.py`
- Create: `tests/unit/test_swimming_cycle_selector.py`

- [ ] **Step 1: Write failing cycle-selection tests**

```python
from video_analysis.swimming_cycle_selector import SwimmingCycleSelector


def test_selects_best_non_overlapping_complete_cycles():
    frames = synthetic_freestyle_landmarks(cycles=6, fps=10.0)
    cycles = SwimmingCycleSelector(fps=10.0).select(frames, limit=5)
    assert len(cycles) == 5
    assert all(cycle.complete for cycle in cycles)
    assert all(cycles[i].end_frame <= cycles[i + 1].start_frame for i in range(4))


def test_cycle_quality_penalizes_tracking_and_waterline_gaps():
    clean = synthetic_freestyle_landmarks(cycles=3, fps=10.0)
    degraded = with_confidence_gap(clean, start=20, end=25)
    selector = SwimmingCycleSelector(fps=10.0)
    assert selector.select(clean)[0].quality > selector.select(degraded)[0].quality


def test_fewer_than_two_reliable_cycles_is_rejected():
    frames = synthetic_freestyle_landmarks(cycles=1, fps=10.0)
    result = SwimmingCycleSelector(fps=10.0).select(frames)
    assert len(result) < 2
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/unit/test_swimming_cycle_selector.py -v
```

Expected: missing module failure.

- [ ] **Step 3: Implement cycle data model and selector**

Use a focused contract:

```python
@dataclass(frozen=True)
class StrokeCycle:
    id: str
    start_frame: int
    peak_frame: int
    end_frame: int
    start_sec: float
    peak_sec: float
    end_sec: float
    quality: float
    complete: bool


class SwimmingCycleSelector(BaseAnalyzer):
    def select(
        self,
        frames: Sequence[Mapping[str, Any]],
        limit: int = SWIM_TARGET_CYCLE_COUNT,
    ) -> list[StrokeCycle]:
        signal = self._smooth(
            [self._phase_value(frame) for frame in frames],
            window=max(3, int(self.fps * 0.2)),
        )
        peaks = self._local_peaks(signal, min_gap=max(4, int(self.fps * 0.7)))
        candidates = [
            self._build_cycle(frames, signal, start, end, index)
            for index, (start, end) in enumerate(zip(peaks, peaks[1:]), start=1)
        ]
        reliable = [cycle for cycle in candidates if cycle.complete and cycle.quality >= 0.55]
        selected = sorted(reliable, key=lambda cycle: cycle.quality, reverse=True)[:limit]
        return sorted(selected, key=lambda cycle: cycle.start_frame)
```

Derive a smoothed phase signal from normalized wrist displacement relative to
the shoulder midpoint. Detect complete peak-to-peak windows, calculate quality
from phase coverage, landmark continuity, tracking confidence, waterline
confidence, and blur quality, then keep the highest-quality non-overlapping
cycles in chronological order.

- [ ] **Step 4: Run selector tests**

Run:

```bash
pytest tests/unit/test_swimming_cycle_selector.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add video_analysis/swimming_cycle_selector.py tests/unit/test_swimming_cycle_selector.py
git commit -m "feat: select reliable freestyle stroke cycles"
```

## Task 3: Waterline Estimation And Temporal Confidence

**Files:**
- Create: `video_analysis/waterline_analyzer.py`
- Create: `tests/unit/test_waterline_analyzer.py`

- [ ] **Step 1: Write failing synthetic-frame tests**

```python
import cv2
import numpy as np

from video_analysis.waterline_analyzer import WaterlineAnalyzer


def frame_with_waterline(y: int, width: int = 320, height: int = 180):
    frame = np.full((height, width, 3), (55, 45, 35), dtype=np.uint8)
    frame[y:, :] = (130, 85, 35)
    cv2.line(frame, (0, y), (width - 1, y), (230, 230, 230), 2)
    return frame


def test_detects_horizontal_waterline():
    estimate = WaterlineAnalyzer().analyze(frame_with_waterline(82))
    assert estimate.observed is True
    assert abs(estimate.y_at(160) - 82) <= 5
    assert estimate.confidence >= 0.55


def test_missing_frame_reuses_last_line_with_decaying_confidence():
    analyzer = WaterlineAnalyzer()
    first = analyzer.analyze(frame_with_waterline(82))
    missing = analyzer.analyze(np.zeros((180, 320, 3), dtype=np.uint8))
    assert missing.observed is False
    assert missing.confidence < first.confidence
    assert missing.y_at(160) == first.y_at(160)


def test_rejects_single_frame_jump():
    analyzer = WaterlineAnalyzer()
    analyzer.analyze(frame_with_waterline(80))
    jumped = analyzer.analyze(frame_with_waterline(145))
    assert jumped.y_at(160) < 120
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/unit/test_waterline_analyzer.py -v
```

Expected: missing module failure.

- [ ] **Step 3: Implement estimator**

Implement:

```python
@dataclass(frozen=True)
class WaterlineEstimate:
    slope: float
    intercept: float
    confidence: float
    observed: bool

    def y_at(self, x: float) -> float:
        return self.slope * x + self.intercept


class WaterlineAnalyzer(BaseAnalyzer):
    def analyze(
        self,
        frame: np.ndarray,
        swimmer_bbox: tuple[int, int, int, int] | None = None,
    ) -> WaterlineEstimate:
        candidates = self._candidate_lines(frame, swimmer_bbox)
        observed = self._best_candidate(candidates, frame.shape)
        if observed is None:
            return self._bridge_previous()
        stable = self._reject_jump(observed, frame.shape[0])
        smoothed = self._smooth_estimate(stable)
        self._previous = smoothed
        return smoothed
```

Use grayscale horizontal gradients, restrained Hough line candidates, swimmer
box proximity, line-length support, slope penalty, EMA smoothing, jump
rejection, and confidence decay for bridged estimates. Keep all OpenCV work
inside this module.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/unit/test_waterline_analyzer.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add video_analysis/waterline_analyzer.py tests/unit/test_waterline_analyzer.py
git commit -m "feat: add temporal waterline estimation"
```

## Task 4: Swimming Pipeline And Strict Overlay

**Files:**
- Create: `backend/app/services/swimming.py`
- Create: `tests/unit/test_swimming_service.py`
- Modify: `video_analysis/biomechanics_visualizer.py`

- [ ] **Step 1: Write failing pipeline tests with injected frame observations**

The service test must avoid model downloads by injecting a frame source and
observation provider:

```python
from backend.app.services.swimming import analyze_swimming_video


def test_pipeline_emits_stable_stages_and_structured_result(tmp_path):
    events = list(
        analyze_swimming_video(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path,
            observation_provider=fake_observations(cycles=4),
            render_video=fake_renderer,
        )
    )
    assert [event.stage for event in events if event.type == "progress"] == [
        "quality_gate",
        "tracking",
        "waterline",
        "pose",
        "cycles",
        "technique",
        "coaching",
        "rendering",
        "completed",
    ]
    result = events[-1].to_dict()
    assert result["type"] == "result"
    assert result["coverage"]["total_zones"] == 5
    assert len(result["cycles"]) == 4


def test_pipeline_returns_reshoot_error_for_one_cycle(tmp_path):
    events = list(
        analyze_swimming_video(
            video_path=tmp_path / "input.mp4",
            output_dir=tmp_path,
            observation_provider=fake_observations(cycles=1),
            render_video=fake_renderer,
        )
    )
    error = events[-1].to_dict()
    assert error["type"] == "error"
    assert error["code"] == "insufficient_cycles"
    assert error["reshoot_guidance"]
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/unit/test_swimming_service.py -v
```

Expected: missing service module failure.

- [ ] **Step 3: Implement typed service events and quality gate**

Define progress, result, and error events with `type`, stable `stage`, progress,
error code, and reshoot guidance. Probe source metadata and calculate frame
brightness, blur, swimmer size, tracking continuity, and body-in-frame coverage.

The service accepts optional injected providers in tests but defaults to real
frame extraction, existing YOLO detection, cached MediaPipe pose, waterline
analysis, cycle selection, technique analysis, and browser video encoding.

- [ ] **Step 4: Implement dual-zone pose merge**

For each tracked swimmer crop:

```python
above_mask = yy < waterline_y
below_mask = ~above_mask
above = normalize_above_water(crop, above_mask)
below = enhance_underwater(crop, below_mask)
```

Run pose on original and enhanced crops. For each landmark, select the candidate
with the highest adjusted confidence after waterline proximity, temporal
continuity, and identity penalties. Store:

```python
{
    "point": (x, y),
    "visibility": 0.84,
    "state": "observed",
    "water_zone": "below",
}
```

Bridged states must remain distinguishable from observed states.

- [ ] **Step 5: Add strict segment rendering**

Expose a helper that renders a connection only when both endpoints have
confidence `>= 0.55` and neither endpoint is bridge-only. Medium segments use
amber; high segments use cyan/side colors. Draw the waterline only at medium or
high confidence and mark selected evidence intervals on the output frames.

- [ ] **Step 6: Run service and visualization regression tests**

Run:

```bash
pytest tests/unit/test_swimming_service.py tests/unit/test_video_encoding.py tests/integration/test_video_pipeline.py -v
```

Expected: all available tests pass; environment-dependent integration skips are
acceptable only when their declared dependency is absent.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/swimming.py video_analysis/biomechanics_visualizer.py tests/unit/test_swimming_service.py
git commit -m "feat: orchestrate waterline-aware swimming analysis"
```

## Task 5: Swimming API, SSE, Video, And History Save

**Files:**
- Modify: `backend/app/api/analysis.py`
- Create: `tests/unit/test_swimming_api.py`

- [ ] **Step 1: Write failing API contract tests**

Follow the immediate-thread pattern from `test_tools_api.py`:

```python
def test_swimming_upload_starts_job(client):
    response = client.post(
        "/api/analysis/swimming",
        files={"video": ("freestyle.mp4", b"video", "video/mp4")},
    )
    assert response.status_code == 200
    assert response.json()["job_id"]


def test_swimming_status_rejects_other_job_kinds(client, registry):
    running = registry.create("running")
    assert client.get(f"/api/analysis/swimming/{running.id}").status_code == 404


def test_swimming_save_is_idempotent(client, completed_swimming_job):
    first = client.post(
        f"/api/analysis/swimming/{completed_swimming_job.id}/save",
        json={"athlete_name": "Nikita K."},
    )
    second = client.post(
        f"/api/analysis/swimming/{completed_swimming_job.id}/save",
        json={"athlete_name": "Nikita K."},
    )
    assert first.json()["session_id"] == second.json()["session_id"]
```

Also test SSE terminal events, annotated video `404` before readiness, and
structured error payloads.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/unit/test_swimming_api.py -v
```

Expected: swimming endpoints return `404`.

- [ ] **Step 3: Add swimming worker and endpoints**

Add:

```python
@router.post("/swimming")
async def upload_swimming(video: UploadFile = File(...), fps: float | None = Form(None)):
    suffix = validate_video_upload(
        filename=video.filename,
        content_type=video.content_type,
        declared_size=getattr(video, "size", None),
    )
    job = registry.create(kind="swimming")
    video_path = job.workspace / f"input{suffix}"
    with video_path.open("wb") as target:
        copy_upload_with_limit(video.file, target)
    Thread(
        target=_run_swimming_pipeline,
        args=(job, video_path, fps),
        daemon=True,
    ).start()
    return {"job_id": job.id}

@router.get("/swimming/{job_id}")
async def get_swimming_job(job_id: str):
    job = _get_job(job_id, kind="swimming")
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "event_count": len(job.events),
        "saved_session_id": job.saved_session_id,
    }

@router.get("/swimming/{job_id}/events")
async def stream_swimming_events(job_id: str):
    job = _get_job(job_id, kind="swimming")
    return StreamingResponse(
        stream_job_events(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/swimming/{job_id}/video")
async def stream_swimming_video(job_id: str):
    job = _get_job(job_id, kind="swimming")
    video_path = job.workspace / "annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not ready")
    return FileResponse(video_path, media_type="video/mp4")

@router.post("/swimming/{job_id}/save")
async def save_swimming_job(job_id: str, request: SaveAnalysisRequest):
    job = _get_job(job_id, kind="swimming")
    if job.status != "done" or not job.result:
        raise HTTPException(status_code=409, detail="Swimming analysis is not ready")
    if job.saved_session_id is None:
        job.saved_session_id = await asyncio.to_thread(
            save_analysis_to_db,
            athlete_name=request.athlete_name,
            session_type="swimming",
            analysis={"swimming_analysis": job.result},
            video_path=_persist_swimming_video(job),
        )
    return {"session_id": job.saved_session_id}
```

Only jobs with `kind == "swimming"` are accepted. Persist the annotated MP4
under `SESSION_VIDEO_DIR`, store the structured result under
`{"swimming_analysis": result}`, and set `session_type="swimming"`.

- [ ] **Step 4: Run API and job tests**

Run:

```bash
pytest tests/unit/test_swimming_api.py tests/unit/test_jobs.py tests/unit/test_upload_validation.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/analysis.py tests/unit/test_swimming_api.py
git commit -m "feat: expose swimming analysis API"
```

## Task 6: Typed Frontend Client

**Files:**
- Create: `frontend/lib/swimming.ts`
- Create: `frontend/lib/swimming.test.ts`

- [ ] **Step 1: Write failing client tests**

```typescript
import {
  swimmingVideoUrl,
  uploadSwimmingVideo,
} from "./swimming";

test("uploads a swimming video to the swimming endpoint", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ job_id: "swim-123" }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const result = await uploadSwimmingVideo(
    new File(["video"], "freestyle.mp4", { type: "video/mp4" }),
  );
  expect(result).toEqual({ jobId: "swim-123" });
  expect(fetchMock.mock.calls[0][0]).toContain("/api/analysis/swimming");
});

test("builds the annotated swimming video URL", () => {
  expect(swimmingVideoUrl("swim-123")).toContain(
    "/api/analysis/swimming/swim-123/video",
  );
});
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cd frontend && npm test -- lib/swimming.test.ts
```

Expected: missing module failure.

- [ ] **Step 3: Implement result types and API functions**

Define exact TypeScript types for:

- `SwimmingProgressEvent`
- `SwimmingErrorEvent`
- `SwimmingCycle`
- `SwimmingZone`
- `SwimmingPrimaryIssue`
- `SwimmingPrescription`
- `SwimmingResultEvent`

Implement upload, status fetch, SSE subscribe, annotated video URL, and save.
SSE `onerror` fetches current status once; it reports a connection error only
when the job is not already terminal.

- [ ] **Step 4: Run frontend client tests and typecheck**

Run:

```bash
cd frontend && npm test -- lib/swimming.test.ts && npm run typecheck
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/swimming.ts frontend/lib/swimming.test.ts
git commit -m "feat: add typed swimming analysis client"
```

## Task 7: Native Upload And Evidence-First Result UI

**Files:**
- Create: `frontend/components/swimming/SwimmingUploader.tsx`
- Create: `frontend/components/swimming/SwimmingAnalysisWorkspace.tsx`
- Create: `frontend/components/swimming/SwimmingAnalysisWorkspace.test.tsx`
- Create: `frontend/app/swimming/[jobId]/page.tsx`
- Modify: `frontend/app/swimming/page.tsx`

- [ ] **Step 1: Write failing workspace tests**

```tsx
test("renders the primary issue before zone details", () => {
  render(<SwimmingAnalysisWorkspace jobId="swim-123" initialResult={result} />);
  const mainIssue = screen.getByRole("heading", { name: /hips drop/i });
  const zones = screen.getByRole("heading", { name: /technique zones/i });
  expect(
    mainIssue.compareDocumentPosition(zones) & Node.DOCUMENT_POSITION_FOLLOWING,
  ).toBeTruthy();
});

test("seeks video to evidence peak when a zone moment is selected", async () => {
  render(<SwimmingAnalysisWorkspace jobId="swim-123" initialResult={result} />);
  const video = screen.getByTestId("swimming-video") as HTMLVideoElement;
  await userEvent.click(screen.getByRole("button", { name: /show at 12.4s/i }));
  expect(video.currentTime).toBe(12.4);
});

test("shows insufficient data without a zero score", () => {
  render(<SwimmingAnalysisWorkspace jobId="swim-123" initialResult={partialResult} />);
  expect(screen.getByText(/4 of 5 zones analyzed/i)).toBeInTheDocument();
  expect(screen.getByText(/kick.*insufficient data/i)).toBeInTheDocument();
  expect(screen.queryByText("Kick 0")).not.toBeInTheDocument();
});
```

Add tests for progress, reshoot guidance, cycle buttons, drill, mini-set, and
idempotent saved state.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cd frontend && npm test -- components/swimming/SwimmingAnalysisWorkspace.test.tsx
```

Expected: missing component failure.

- [ ] **Step 3: Implement upload page**

Keep `SportLanding` and use:

```tsx
<SportLanding
  uploader={<SwimmingUploader />}
  title="Swimming · Waterline-aware analysis"
  subtitle="Evidence-first freestyle analysis for side-on pool video."
  badges={[
    { icon: Waves, label: "Freestyle · Side view" },
    { variant: "success", label: "Confidence-aware" },
    { variant: "info", label: "Waterline tracking" },
  ]}
  hint="Record from the pool deck with the full swimmer visible for several complete cycles."
  accentRgb="34,211,238"
  metrics={[]}
  sessions={[]}
  insights={[]}
/>
```

Remove claims such as `40+ metrics`, `96% detection`, and multi-angle support
until the real implementation proves them. The uploader displays four concise
filming cues: side-on, stable phone, full body, several complete cycles.

- [ ] **Step 4: Implement result workspace in the existing style**

Use `ChartContainer`, `StatusBadge`, existing dark surfaces, cyan actions,
amber medium confidence, slate insufficient data, and rose failures.

Render in this order:

1. primary issue and why it matters
2. annotated video with evidence seeking and cycle navigation
3. five zones and explicit coverage
4. drill
5. mini-set
6. save action

Do not lead with an overall score.

- [ ] **Step 5: Run component tests, typecheck, and build**

Run:

```bash
cd frontend && npm test -- components/swimming/SwimmingAnalysisWorkspace.test.tsx lib/swimming.test.ts
cd frontend && npm run typecheck
cd frontend && npm run build
```

Expected: all commands pass.

- [ ] **Step 6: Commit**

```bash
git add frontend/app/swimming frontend/components/swimming frontend/lib/swimming.ts
git commit -m "feat: add evidence-first swimming analysis UI"
```

## Task 8: Full Verification And Presentation QA

**Files:**
- Modify only files required by failures found during verification

- [ ] **Step 1: Run backend unit suite**

Run:

```bash
pytest tests/unit/ -v
```

Expected: pass with only documented dependency skips.

- [ ] **Step 2: Run lint and formatting gates**

Run:

```bash
ruff check .
black --check .
isort --check-only .
```

Expected: pass.

- [ ] **Step 3: Run frontend gates**

Run:

```bash
cd frontend && npm test
cd frontend && npm run typecheck
cd frontend && npm run build
```

Expected: pass.

- [ ] **Step 4: Build and start Docker**

Run:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Expected: backend healthy and frontend running.

- [ ] **Step 5: Verify browser-visible workflow**

Use the in-app Browser at `http://127.0.0.1:3000/swimming` and verify:

- page matches Dashboard, Running, Rehabilitation, Tools, and History styling
- accurate upload guidance is visible
- upload starts a swimming job
- progress stages update
- partial and error states are readable
- evidence buttons seek the video
- cycles switch correctly
- five-zone coverage is explicit
- drill and mini-set are visible
- save reaches the completed state
- desktop and narrow viewport layouts remain usable

- [ ] **Step 6: Verify real footage when available**

Use one side-on freestyle clip containing at least four complete cycles. Record:

- swimmer lock continuity
- waterline stability
- selected cycle timestamps
- which landmarks are hidden rather than fabricated
- five-zone availability
- main issue repeatability
- evidence timestamp accuracy
- usefulness of drill and mini-set

If no representative clip is available, do not claim biomechanical accuracy.
Complete automated, API, Docker, and browser-shell verification and leave this
single acceptance item explicitly open.

- [ ] **Step 7: Commit verification fixes**

```bash
git add backend/app/api/analysis.py backend/app/services/swimming.py video_analysis frontend
git commit -m "fix: harden swimming analysis presentation flow"
```

Before staging, inspect `git diff --name-only` and remove any path from the
command that has no intentional verification fix. Skip this commit when
verification required no code changes.

## Task 9: Review And Publish

**Files:**
- Review all changes since `5b1098d`

- [ ] **Step 1: Inspect the final diff**

Run:

```bash
git diff --check 5b1098d..HEAD
git diff --stat 5b1098d..HEAD
git status --short
```

Expected: no whitespace errors; only the intentionally local
`data/athletes.db` and `.superpowers/` remain outside feature commits.

- [ ] **Step 2: Request code review**

Use `superpowers:requesting-code-review` against the complete feature range.
Address correctness, regression, data-contract, and missing-test findings before
publishing.

- [ ] **Step 3: Push the current branch**

```bash
git push origin claude/compassionate-rubin-gyhNU
```

Expected: push succeeds and updates the existing pull request.

- [ ] **Step 4: Verify GitHub checks**

Confirm lint, tests, and frontend checks are green. Fix and repush any failure
before declaring the implementation complete.
