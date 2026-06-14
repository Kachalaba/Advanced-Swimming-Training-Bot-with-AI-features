# Dryland Product Workflow Design

## Goal

Complete the missing Dryland workflow in the primary Next.js and FastAPI
product. A coach or athlete selects one supported exercise, uploads a short
video, receives quality-gated repetition evidence and an annotated video, then
saves the result to a stable athlete profile and sees it in persisted history.

The first release supports:

- squat
- lunge
- push-up

Exercise selection is explicit. Automatic exercise classification is not used
to choose thresholds or produce a result.

## Product Principles

1. **Selection before inference.** The user selects the exercise so the
   analyzer can apply the correct landmarks, joint and repetition state
   machine.
2. **Evidence before score.** A clip without enough complete pose frames or a
   confirmed full repetition returns reshoot guidance, not zero-valued
   performance metrics.
3. **Comparable capture.** The first release uses a fixed side view for all
   three exercises. The full body and contact points must remain visible.
4. **Measured values only.** Repetition count, tempo, range of motion and
   consistency come from confirmed movement cycles. Missing evidence is
   omitted.
5. **One product contract.** Dryland follows the existing
   upload -> SSE -> result -> video -> athlete save -> overview flow used by
   running, swimming and cycling.
6. **Screening, not diagnosis.** The result describes visible movement in one
   clip. It does not diagnose injury, prescribe load or claim force and power
   measurements.

## Scope

### Included

- explicit squat, lunge or push-up selection before upload
- exercise-specific capture guidance
- validated video upload through the existing upload limits
- person lock-on and cached MediaPipe pose inference
- an exercise-specific metric-ready frame gate
- a full-cycle repetition state machine
- repetition count
- average repetition tempo
- average range of motion
- movement consistency score
- per-repetition duration and range-of-motion evidence
- annotated browser-compatible MP4 with skeleton, active joint angle and
  current repetition count
- result quality and pose-coverage evidence
- athlete selection and idempotent save
- persisted Dryland metrics, sessions and evidence-based insights
- honest empty, rejected, progress, result and save states
- desktop and mobile support in the existing dark SPRINT AI design system

### Deferred

- automatic exercise classification
- plank and isometric hold analysis
- loaded barbell velocity, force, power or weight estimation
- multi-person group analysis
- live camera analysis
- exercise programs and load prescription
- cross-exercise readiness or injury prediction
- cloud workers, authentication and multi-tenant access

## Exercise Contracts

All exercises require a fixed side view, one complete visible body side and a
stable camera. The service may use the clearer left or right side per frame but
must not combine incomplete sides into a synthetic pose.

### Squat

- required side: shoulder, hip, knee and ankle
- active angle: hip-knee-ankle knee angle
- ready phase: knee extended above the exercise-specific high threshold
- effort phase: knee flexed below the exercise-specific low threshold
- confirmed repetition: ready -> effort -> ready
- evidence: minimum angle, maximum angle, range of motion and cycle duration
- guidance: keep head, hip, knee and ankle visible; feet remain in frame

### Lunge

- required evidence: shoulder and hip plus both knees and both ankles
- active angle: the more flexed visible knee angle during the effort phase
- ready phase: both knees above the ready threshold
- effort phase: one knee below the effort threshold
- confirmed repetition: ready -> effort -> ready
- evidence: active-side label, minimum angle, maximum angle, range of motion
  and cycle duration
- guidance: record one repeated side per clip; keep both feet visible

### Push-Up

- required side: shoulder, elbow, wrist, hip and ankle
- active angle: shoulder-elbow-wrist elbow angle
- ready phase: elbow extended above the exercise-specific high threshold
- effort phase: elbow flexed below the exercise-specific low threshold
- confirmed repetition: ready -> effort -> ready
- evidence: minimum angle, maximum angle, range of motion and cycle duration
- guidance: camera at torso height; hands, shoulders, hips and feet remain
  visible

Thresholds and minimum durations live in `video_analysis/constants.py`. The
state machine rejects transitions that are too short to be a plausible
repetition and does not count a partial cycle at the beginning or end.

## Quality Contract

The pipeline reports:

```text
status
pose_coverage
metric_ready_frames
minimum_required_frames
warnings
```

A result is rejected when:

- no frames can be decoded
- the selected exercise is unsupported
- the required landmarks are absent in too many frames
- metric-ready coverage is below the minimum threshold
- fewer than the minimum number of metric-ready frames are available
- no complete repetition is confirmed

Usable but imperfect coverage may produce a result with warnings. Warnings are
shown beside the video and persisted with the analysis.

Interpolation may bridge short gaps only inside an otherwise continuous,
metric-ready movement. It must not manufacture a complete repetition across a
long pose loss.

## Analysis Core

`ExerciseAnalyzer` remains the domain analyzer and continues to inherit
`BaseAnalyzer`. It gains an explicit exercise profile instead of inferring the
tracked joint from whichever keys happen to appear first.

The analyzer API becomes conceptually:

```text
analyze(angle_frames, exercise_type, fps) -> ExerciseStats
```

`ExerciseStats` contains:

```text
exercise_type
tracked_joint
total_reps
avg_tempo
avg_range_of_motion
stability_score
min_angle
max_angle
reps
angle_history
```

Each repetition contains:

```text
rep_number
start_frame
effort_frame
end_frame
duration_sec
min_angle
max_angle
range_of_motion
active_side
```

Analyzer temporal state is reset at the start of every `analyze()` call.

## Backend Pipeline

The new service follows the existing sport pipeline boundaries:

```text
upload
  -> extract frames
  -> lock onto one person
  -> estimate pose under the shared MediaPipe processing lock
  -> keep exercise-specific metric-ready frames
  -> quality gate
  -> calculate active angles
  -> detect full repetitions
  -> render annotated MP4
  -> emit final result
```

The service uses:

- `extract_frames_from_video()`
- `detect_swimmer_in_frames()` for the existing person lock-on behavior
- `get_pose_detector(video_mode=True)`
- `get_pose_processing_lock()`
- `ExerciseAnalyzer`
- `open_intermediate_writer()` and `finalize_browser_video()`

Progress events use exercise-oriented labels: extraction, person lock, pose,
repetition detection, evidence rendering and completion.

## API Contract

The analysis router gains:

```text
POST /api/analysis/dryland
GET  /api/analysis/dryland/{job_id}
GET  /api/analysis/dryland/{job_id}/events
GET  /api/analysis/dryland/{job_id}/video
POST /api/analysis/dryland/{job_id}/save
```

Upload form fields:

```text
video
exercise_type: squat | lunge | push_up
fps: optional, default 15
```

The result payload contains:

```json
{
  "type": "result",
  "exercise_type": "squat",
  "analysis": {
    "tracked_joint": "knee",
    "total_reps": 6,
    "avg_tempo": 2.4,
    "avg_range_of_motion": 76.0,
    "stability_score": 91.0,
    "min_angle": 84.0,
    "max_angle": 166.0,
    "reps": []
  },
  "quality": {
    "status": "pass",
    "pose_coverage": 87.5,
    "metric_ready_frames": 105,
    "minimum_required_frames": 20,
    "warnings": []
  },
  "frames_total": 120,
  "frames_with_pose": 108,
  "video_path": "annotated.mp4"
}
```

Save follows the existing stable-athlete and idempotency rules. Wrong-kind,
missing and incomplete jobs are rejected.

## Persistence And History

Saved analyses use:

```text
session_type = "dryland"
exercise_type = selected exercise
reps = confirmed repetitions
avg_tempo = average repetition duration
stability_score = movement consistency
ai_score = stability score
full_analysis.dryland_analysis = complete result payload
video_path = durable copy of annotated.mp4
```

The normalized Dryland overview exposes:

- repetitions
- average tempo
- average range of motion
- movement consistency
- exercise type
- pose coverage

The latest valid session becomes the headline. Sessions from different
exercise types remain visible but are labelled clearly; the UI does not imply
that squat and push-up metrics are directly comparable.

Evidence-based insights are limited to:

- a saved baseline for the selected exercise
- high repetition-to-repetition tempo variation
- high range-of-motion variation
- quality warnings from the source clip

## Frontend Workflow

### Landing Page

The current Dryland readiness page becomes a real persisted workspace:

- real latest metrics and recent sessions
- honest empty history before the first save
- exercise selector with squat, lunge and push-up
- capture instructions that update with the selected exercise
- upload/drop zone
- no planned-workflow badge or disabled primary action

The selected exercise is included in the upload request. Changing selection
does not fabricate metrics or reset persisted history.

### Result Page

`/dryland/{jobId}` shows:

- selected exercise and analysis status
- progress or actionable capture rejection
- annotated video
- pose coverage and quality warnings
- repetitions, average tempo, average range of motion and consistency
- per-repetition table with duration and range of motion
- screening limitation
- athlete selector and save state

Zero confirmed repetitions is an error state, not a successful result with a
zero score.

### Visual System

The page reuses the existing dark SPRINT AI shell, `SportLanding`,
`ChartContainer`, `StatusBadge` and metric typography. Dryland keeps the violet
discipline accent. No separate light theme, new navigation model or marketing
surface is introduced.

## Error Handling

- invalid exercise type returns `422`
- invalid or oversized upload uses the shared upload validation responses
- insufficient pose evidence emits an SSE error with exercise-specific reshoot
  guidance
- no confirmed full repetition emits an SSE error explaining the required
  ready-effort-ready cycle
- video encoding failure preserves the actionable error and does not expose a
  broken result
- save returns `409` before completion
- save returns `404` for unknown athletes or wrong-kind jobs
- repeated save returns the original session ID
- frontend connection loss falls back to the job status endpoint before
  presenting an error

## Testing

Analyzer tests cover:

- squat ready-effort-ready counting
- lunge active-side selection
- push-up elbow-cycle counting
- partial cycles are not counted
- implausibly short transitions are rejected
- short missing-angle gaps may be bridged
- long missing-angle gaps do not create repetitions
- temporal state does not leak between analyses
- no repetitions return unavailable metrics rather than a fabricated score

Service and API tests cover:

- exercise-specific metric-ready landmarks
- quality pass and rejection
- shared MediaPipe lock serialization
- upload form validation
- progress, result, status, video and error events
- zero-repetition rejection
- stable-athlete save
- idempotent save
- wrong-kind and incomplete jobs

Persistence and overview tests cover:

- Dryland fields stored from the completed result
- normalized metrics omit unsupported values
- exercise-labelled summaries and sessions
- quality warnings become evidence-based insights

Frontend tests cover:

- exercise selection and capture guidance
- upload includes the selected exercise
- real empty and persisted history
- progress, result and rejection states
- per-repetition evidence
- athlete selection and idempotent save confirmation
- no planned-workflow or fabricated values
- mobile layout without horizontal overflow

## Release Gates

```text
PATH="<repo>/.venv/bin:$PATH" make lint
pytest tests/ -q
frontend npm test -- --run
frontend npm run typecheck
frontend npm run build
frontend npm audit --audit-level=high
docker compose build
browser QA: landing, result, rejection, save, history, desktop and 390px mobile
```

## Success Criteria

- a user can explicitly select squat, lunge or push-up and upload a video
- insufficient evidence produces actionable reshoot guidance
- a complete supported clip produces measured repetitions, tempo, ROM,
  consistency and annotated video
- partial movement is never counted as a full repetition
- the result can be saved exactly once to a stable athlete
- the saved session appears after restart in Dryland history
- the page contains no fictional athlete metrics or planned-workflow controls
- all automated, build, Docker and browser gates pass
