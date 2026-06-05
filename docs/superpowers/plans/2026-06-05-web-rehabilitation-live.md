# Web Rehabilitation Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-quality Next.js rehabilitation section with uploaded-video analysis, smooth live webcam preview, a switchable postural coordinate overlay, relative optical camera-level calibration, compact live metrics, and fullscreen mode.

**Architecture:** FastAPI owns pose inference, rolling rehabilitation metrics, upload jobs, and relative camera-roll estimation. Next.js owns media capture, sampled frame transport with one-frame backpressure, responsive presentation, SVG overlays, and fullscreen state. Pure geometry and session-state modules isolate the calculations so a later macOS AVFoundation client can reuse the same contracts.

**Tech Stack:** Python 3.9/3.11-compatible FastAPI, OpenCV, MediaPipe-backed existing analyzers, pytest, Next.js 16, React 19, TypeScript, SVG, MediaDevices API, Fullscreen API, Vitest, Testing Library.

---

## File Structure

- `backend/app/services/posture.py`: pure landmark normalization and postural-axis calculations.
- `backend/app/services/camera_level.py`: ORB/RANSAC relative roll estimator with confidence and recalibration states.
- `backend/app/services/rehabilitation.py`: live-session registry, frame processing, rolling `RehabAnalyzer` integration, and upload pipeline.
- `backend/app/api/rehabilitation.py`: live-session, frame, upload-job, report, and annotated-video endpoints.
- `tests/unit/test_posture_service.py`: postural geometry and warning-direction tests.
- `tests/unit/test_camera_level.py`: calibration, angle, confidence, and invalidation tests.
- `tests/unit/test_rehabilitation_service.py`: bounded sessions, calibration, rolling reports, and invalid-frame tests.
- `tests/unit/test_rehabilitation_api.py`: request validation and API contract tests.
- `frontend/lib/rehabilitation.ts`: shared data types and backend client.
- `frontend/lib/frameScheduler.ts`: single-in-flight sampled-frame scheduler.
- `frontend/components/rehabilitation/useCameraSource.ts`: camera lifecycle and device errors.
- `frontend/components/rehabilitation/PostureOverlay.tsx`: scalable postural coordinate map.
- `frontend/components/rehabilitation/LiveMetricRail.tsx`: ROM, repetitions, asymmetry, and level display.
- `frontend/components/rehabilitation/LiveRehabWorkspace.tsx`: camera, frame transport, controls, calibration, and fullscreen.
- `frontend/components/rehabilitation/RehabUploader.tsx`: uploaded-video workflow.
- `frontend/app/rehabilitation/page.tsx`: route composition and protocol/input-mode state.
- `frontend/lib/*.test.ts` and `frontend/components/rehabilitation/*.test.tsx`: Vitest unit/component coverage.
- `frontend/lib/sections.ts`: rehabilitation navigation registration.
- `frontend/package.json`: Vitest and Testing Library scripts/dependencies.

### Task 1: Postural Geometry Contract

**Files:**
- Create: `backend/app/services/posture.py`
- Create: `tests/unit/test_posture_service.py`

- [ ] **Step 1: Write failing postural geometry tests**

```python
from backend.app.services.posture import calculate_posture


def test_calculate_posture_reports_axis_angles_and_trunk_offset():
    landmarks = {
        "left_shoulder": {"x": 0.30, "y": 0.30},
        "right_shoulder": {"x": 0.70, "y": 0.34},
        "left_hip": {"x": 0.36, "y": 0.62},
        "right_hip": {"x": 0.66, "y": 0.60},
    }

    result = calculate_posture(landmarks, frame_center_x=0.5)

    assert result["shoulder_angle_deg"] > 0
    assert result["pelvis_angle_deg"] < 0
    assert result["trunk_lean_deg"] != 0
    assert result["trunk_offset_pct"] != 0
    assert result["points"]["shoulder_mid"]["x"] == 0.5


def test_calculate_posture_returns_unavailable_for_missing_landmarks():
    result = calculate_posture({}, frame_center_x=0.5)
    assert result == {"available": False}
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest tests/unit/test_posture_service.py -v`

Expected: FAIL because `backend.app.services.posture` does not exist.

- [ ] **Step 3: Implement pure geometry**

Implement:

```python
def _axis_angle(left: Point, right: Point) -> float:
    return round(math.degrees(math.atan2(right["y"] - left["y"], right["x"] - left["x"])), 1)


def calculate_posture(landmarks: Mapping[str, Point], frame_center_x: float = 0.5) -> dict:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in landmarks for name in required):
        return {"available": False}
    # Return normalized points, shoulder/pelvis angles, trunk lean,
    # lateral offset, severity labels, and directional coaching text.
```

Use image coordinates consistently: positive shoulder/pelvis angle means the
right landmark is lower in the image. Include `available`, `points`,
`shoulder_angle_deg`, `pelvis_angle_deg`, `trunk_lean_deg`,
`trunk_offset_pct`, and per-axis severity.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/pytest tests/unit/test_posture_service.py -v`

Expected: all postural geometry tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/posture.py tests/unit/test_posture_service.py
git commit -m "feat: add rehabilitation posture geometry"
```

### Task 2: Relative Optical Camera Level

**Files:**
- Create: `backend/app/services/camera_level.py`
- Create: `tests/unit/test_camera_level.py`

- [ ] **Step 1: Write failing estimator tests**

Create synthetic high-contrast frames and verify:

```python
def test_calibrated_frame_reports_zero_roll():
    estimator = CameraLevelEstimator()
    frame = make_feature_frame()
    result = estimator.calibrate(frame)
    assert result["status"] == "level"
    assert result["angle_deg"] == 0.0


def test_rotated_frame_reports_relative_roll():
    estimator = CameraLevelEstimator()
    estimator.calibrate(make_feature_frame())
    result = estimator.measure(rotate(make_feature_frame(), 3.0))
    assert result["confidence"] >= 0.5
    assert 2.0 <= abs(result["angle_deg"]) <= 4.0


def test_featureless_frame_requests_recalibration():
    estimator = CameraLevelEstimator()
    estimator.calibrate(make_feature_frame())
    result = estimator.measure(np.zeros((240, 320, 3), dtype=np.uint8))
    assert result["status"] == "recalibrate"
    assert result["angle_deg"] is None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest tests/unit/test_camera_level.py -v`

Expected: FAIL because `CameraLevelEstimator` is missing.

- [ ] **Step 3: Implement ORB/RANSAC estimator**

`CameraLevelEstimator` must:

- detect ORB features in the calibration frame
- optionally mask the athlete bounding box
- match descriptors with Hamming distance
- estimate a partial affine transform with RANSAC
- derive roll from `atan2(matrix[1, 0], matrix[0, 0])`
- return `{angle_deg, confidence, status, direction}`
- classify `level` at `abs(angle) <= 0.7`, `adjust` above tolerance, and
  `recalibrate` when confidence is below `0.35`
- protect internal reference state with a lock

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/pytest tests/unit/test_camera_level.py -v`

Expected: all camera-level tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/camera_level.py tests/unit/test_camera_level.py
git commit -m "feat: add relative camera level estimator"
```

### Task 3: Stateful Rehabilitation Backend And API

**Files:**
- Create: `backend/app/services/rehabilitation.py`
- Create: `backend/app/api/rehabilitation.py`
- Modify: `backend/app/main.py`
- Create: `tests/unit/test_rehabilitation_service.py`
- Create: `tests/unit/test_rehabilitation_api.py`

- [ ] **Step 1: Write failing live-session service tests**

Use injected fake pose and analyzer functions:

```python
def test_live_session_keeps_bounded_keypoint_window():
    session = LiveRehabSession(
        protocol="shoulder_flexion",
        fps=5.0,
        pose_processor=fake_pose,
        analyzer=fake_analyzer,
        max_frames=3,
    )
    for _ in range(5):
        session.process_frame(sample_frame())
    assert session.keypoint_count == 3
    assert session.latest_update["report"]["protocol"] == "shoulder_flexion"


def test_calibration_is_session_scoped():
    session = make_session()
    update = session.process_frame(sample_frame(), calibrate=True)
    assert update["camera_level"]["angle_deg"] == 0.0
```

- [ ] **Step 2: Run service tests and verify RED**

Run: `.venv/bin/pytest tests/unit/test_rehabilitation_service.py -v`

Expected: FAIL because the service does not exist.

- [ ] **Step 3: Implement session processing**

`LiveRehabSession.process_frame()` must:

1. decode one BGR frame already validated by the API
2. run a session-owned biomechanics visualizer
3. normalize keypoints by frame width and height
4. append detected keypoints to a bounded deque
5. update `RehabAnalyzer` at the requested analysis interval
6. calculate the postural coordinate map
7. calibrate or measure camera roll
8. return one JSON-serializable update

Use a `LiveRehabRegistry` with `create`, `get`, and `delete`. Keep heavy model
construction behind injectable factories for tests. Do not share mutable
visualizer trajectory state across athlete sessions.

- [ ] **Step 4: Write failing API contract tests**

Test:

- valid protocol creates a session
- invalid protocol returns `400`
- JPEG frame returns landmarks, posture, metrics, and camera level
- oversized or undecodable image returns `400`
- deleted/unknown session returns `404`

Use FastAPI `TestClient` and monkeypatch the registry with fake sessions.

- [ ] **Step 5: Run API tests and verify RED**

Run: `.venv/bin/pytest tests/unit/test_rehabilitation_api.py -v`

Expected: FAIL because the router is not registered.

- [ ] **Step 6: Implement live endpoints**

Add:

```text
POST   /api/analysis/rehabilitation/live
POST   /api/analysis/rehabilitation/live/{session_id}/frame
GET    /api/analysis/rehabilitation/live/{session_id}
DELETE /api/analysis/rehabilitation/live/{session_id}
```

The frame endpoint accepts `image: UploadFile` and `calibrate: bool`. Limit live
frames to 2 MB and decode with `cv2.imdecode`.

- [ ] **Step 7: Implement uploaded-video endpoints**

Reuse the existing job registry pattern and add:

```text
POST /api/analysis/rehabilitation
GET  /api/analysis/rehabilitation/{job_id}
GET  /api/analysis/rehabilitation/{job_id}/events
GET  /api/analysis/rehabilitation/{job_id}/video
```

The worker reuses existing frame extraction, person detection,
`BiomechanicsVisualizer`, `RehabAnalyzer`, and annotated-video writing. Result
events include the full rehabilitation report.

- [ ] **Step 8: Verify backend GREEN**

Run:

```bash
.venv/bin/pytest tests/unit/test_posture_service.py tests/unit/test_camera_level.py \
  tests/unit/test_rehabilitation_service.py tests/unit/test_rehabilitation_api.py -v
.venv/bin/ruff check backend/app tests/unit
```

Expected: all selected tests PASS and Ruff reports no errors.

- [ ] **Step 9: Commit**

```bash
git add backend/app/api/rehabilitation.py backend/app/services/rehabilitation.py \
  backend/app/main.py tests/unit/test_rehabilitation_service.py \
  tests/unit/test_rehabilitation_api.py
git commit -m "feat: add web rehabilitation analysis api"
```

### Task 4: Frontend Contracts And Frame Backpressure

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/vitest.setup.ts`
- Create: `frontend/lib/rehabilitation.ts`
- Create: `frontend/lib/frameScheduler.ts`
- Create: `frontend/lib/frameScheduler.test.ts`

- [ ] **Step 1: Add the test runner**

Install:

```bash
cd frontend
npm install --save-dev vitest jsdom @testing-library/react @testing-library/jest-dom
```

Add scripts:

```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 2: Write a failing scheduler test**

```typescript
it("keeps only the newest queued frame while one request is in flight", async () => {
  const sent: string[] = [];
  const gates: Array<() => void> = [];
  const scheduler = createFrameScheduler(async (frame: string) => {
    sent.push(frame);
    await new Promise<void>((resolve) => gates.push(resolve));
  });

  scheduler.enqueue("frame-1");
  scheduler.enqueue("frame-2");
  scheduler.enqueue("frame-3");
  expect(sent).toEqual(["frame-1"]);
  gates.shift()?.();
  await scheduler.idle();
  expect(sent).toEqual(["frame-1", "frame-3"]);
});
```

- [ ] **Step 3: Run and verify RED**

Run: `cd frontend && npm test -- frameScheduler.test.ts`

Expected: FAIL because `createFrameScheduler` is missing.

- [ ] **Step 4: Implement scheduler and API types**

The scheduler exposes `enqueue`, `idle`, and `dispose`, allows one request in
flight, and replaces stale queued frames.

`rehabilitation.ts` defines:

- `RehabProtocol`
- `NormalizedPoint`
- `PostureUpdate`
- `CameraLevelUpdate`
- `LiveRehabUpdate`
- `createLiveRehabSession`
- `sendLiveRehabFrame`
- `deleteLiveRehabSession`
- upload/subscription helpers matching the backend routes

- [ ] **Step 5: Verify GREEN**

Run:

```bash
cd frontend
npm test -- frameScheduler.test.ts
npm run typecheck
```

Expected: scheduler tests PASS and TypeScript reports no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts \
  frontend/vitest.setup.ts frontend/lib/rehabilitation.ts \
  frontend/lib/frameScheduler.ts frontend/lib/frameScheduler.test.ts
git commit -m "feat: add rehabilitation frontend transport"
```

### Task 5: Camera, Postural Map, Fullscreen, And Premium Page

**Files:**
- Create: `frontend/components/rehabilitation/useCameraSource.ts`
- Create: `frontend/components/rehabilitation/PostureOverlay.tsx`
- Create: `frontend/components/rehabilitation/LiveMetricRail.tsx`
- Create: `frontend/components/rehabilitation/LiveRehabWorkspace.tsx`
- Create: `frontend/components/rehabilitation/RehabUploader.tsx`
- Create: `frontend/components/rehabilitation/PostureOverlay.test.tsx`
- Create: `frontend/components/rehabilitation/LiveRehabWorkspace.test.tsx`
- Create: `frontend/app/rehabilitation/page.tsx`
- Modify: `frontend/lib/sections.ts`
- Modify: `frontend/app/globals.css`

- [ ] **Step 1: Write failing overlay tests**

Verify that:

- shoulder and pelvic axes use normalized landmark coordinates
- labels display signed degrees
- hidden mode removes the complete SVG overlay
- missing posture returns no misleading angles

```tsx
render(<PostureOverlay visible posture={fixturePosture} />);
expect(screen.getByText("Плечи +2.7°")).toBeInTheDocument();
expect(screen.getByTestId("posture-plumb-axis")).toBeInTheDocument();
```

- [ ] **Step 2: Write failing workspace tests**

Mock `navigator.mediaDevices.getUserMedia`, frame transport, and fullscreen:

```tsx
it("toggles the map without reacquiring the camera", async () => {
  render(<LiveRehabWorkspace protocol="shoulder_flexion" />);
  await user.click(screen.getByRole("button", { name: "Start camera" }));
  await user.click(screen.getByRole("switch", { name: "Show postural map" }));
  expect(getUserMedia).toHaveBeenCalledTimes(1);
});
```

Also verify calibration sends `calibrate=true`, fullscreen targets the complete
workspace, and stopping the session releases all media tracks.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
cd frontend
npm test -- PostureOverlay.test.tsx LiveRehabWorkspace.test.tsx
```

Expected: FAIL because the components do not exist.

- [ ] **Step 4: Implement camera hook**

`useCameraSource` must:

- request only video after an explicit button click
- prefer 1280×720 and the user-facing camera
- expose `idle`, `requesting`, `live`, `denied`, `unavailable`, and `error`
- attach the stream to a supplied video ref
- stop all tracks on cleanup
- avoid reacquisition when UI overlay state changes

- [ ] **Step 5: Implement the postural SVG**

Use one responsive SVG with `viewBox="0 0 1000 1000"` above the video.
Render:

- calibrated vertical plumb axis
- horizontal reference
- shoulder and pelvic axes
- trunk centerline
- major-joint contour
- shoulder, pelvis, and trunk labels

Use cyan neutral lines, amber moderate warnings, rose larger deviations, and
subtle glow. Set `pointer-events: none`.

- [ ] **Step 6: Implement the live workspace**

Use:

- a visible `<video playsInline muted autoPlay>`
- an offscreen 640×360 canvas for JPEG capture at quality `0.72`
- a 200 ms sampling interval (5 FPS analysis)
- the single-in-flight scheduler
- CSS-only live metrics updates isolated from the page shell
- `requestFullscreen()` on the whole workspace
- `fullscreenchange` to keep button state accurate
- `Esc` through native browser behavior

The toolbar includes `Show postural map`, `Calibrate level`, `Fullscreen`, and
connection status. The fullscreen layout retains the metric rail and camera
level card.

- [ ] **Step 7: Implement upload mode and route composition**

Build `/rehabilitation` with:

- protocol selector for all five existing protocols
- `Live camera` / `Upload video` segmented control
- the approved dark cyan/amber/rose visual system
- concise training-aid disclaimer
- live workspace as the default mode
- upload progress and report state

Register `Rehabilitation` between `Dryland` and `History` using an appropriate
Lucide medical/activity icon.

- [ ] **Step 8: Verify frontend GREEN**

Run:

```bash
cd frontend
npm test
npm run typecheck
npm run build
```

Expected: all frontend tests PASS, TypeScript reports no errors, and Next.js
build completes successfully.

- [ ] **Step 9: Commit**

```bash
git add frontend/app/rehabilitation frontend/components/rehabilitation \
  frontend/lib/sections.ts frontend/app/globals.css
git commit -m "feat: add premium rehabilitation live workspace"
```

### Task 6: End-To-End Verification And Performance Pass

**Files:**
- Modify only files required by failures discovered during verification.

- [ ] **Step 1: Run complete local quality gates**

```bash
.venv/bin/pytest tests/unit/ -v
.venv/bin/ruff check .
cd frontend && npm test && npm run typecheck && npm run build
```

Expected: all commands exit `0`.

- [ ] **Step 2: Start the backend and frontend**

```bash
cd backend && ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

Verify `/api/health`, `/api/analysis/rehabilitation/live`, and
`http://localhost:3000/rehabilitation`.

- [ ] **Step 3: Verify the core browser workflow**

In the in-app browser:

1. Open `/rehabilitation`.
2. Start the camera and verify one permission request.
3. Confirm preview remains smooth while network analysis runs at about 5 FPS.
4. Toggle `Show postural map` off and on without stream reacquisition.
5. Calibrate the camera level and verify the relative label.
6. Enter fullscreen and verify metrics/controls remain visible.
7. Exit with `Esc`.
8. Stop the camera and verify tracks end.
9. Exercise upload mode with a small supported video.

- [ ] **Step 4: Perform MacBook Air M4 performance checks**

Use browser logs and request timing to verify:

- no more than one live frame request is in flight
- captured frames are at most 640×360 JPEG/WebP
- analysis frequency is approximately 5 FPS
- the preview video is not replaced or rerendered per analysis response
- no React update loop or unbounded landmark history exists
- stopping or navigating away deletes the backend session

- [ ] **Step 5: Capture final screenshots**

Capture:

- normal desktop live workspace
- postural map enabled
- fullscreen workspace
- camera permission/error fallback

Compare against the approved concept for typography, palette, overlay density,
edge controls, metric readability, and athlete visibility.

- [ ] **Step 6: Request code review**

Review all changes against:

- `docs/superpowers/specs/2026-06-05-web-rehabilitation-live-design.md`
- this implementation plan

Fix all critical and important findings.

- [ ] **Step 7: Final verification commit**

```bash
git add <only-files-changed-by-verification>
git commit -m "fix: polish rehabilitation live workflow"
```

Skip this commit when verification required no code changes.
