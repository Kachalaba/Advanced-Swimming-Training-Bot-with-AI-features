# Rehabilitation Analysis Design

## Goal

Add an eighth Streamlit tab for rehabilitation and kinesiotherapy video analysis.
The feature measures bilateral joint range of motion (ROM), counts controlled
repetitions, reports asymmetry and ROM deficits, provides coaching feedback, and
stores the session in athlete history.

## Architecture

`RehabAnalyzer` is a pure metrics component built on `BaseAnalyzer`. It accepts
per-frame MediaPipe-style keypoint dictionaries and a protocol identifier. It
does not load MediaPipe or OpenCV, so unit tests remain independent of those
packages. The Streamlit page reuses the existing frame extraction, person
detection, and biomechanics visualization pipeline to produce named keypoints
and an annotated video.

All protocol definitions and thresholds live in `video_analysis/constants.py`.
The analyzer reads those definitions to select the target joint, angle triplet,
movement direction, target ROM, repetition thresholds, and symmetry limits.

## Supported Protocols

- Shoulder flexion
- Shoulder abduction
- Elbow flexion
- Knee extension
- Hip abduction

Each protocol tracks both left and right sides. The report also includes
session-wide ROM summaries for shoulder, elbow, hip, and knee angle families.

## Metrics Contract

`RehabAnalyzer.analyze()` returns a JSON-serializable dictionary containing:

- protocol metadata and target ROM
- per-joint, per-side minimum, maximum, mean, and achieved ROM
- per-side total and correct repetitions
- per-repetition duration and ROM
- bilateral asymmetry index and symmetry score
- target ROM deficit and completion percentage
- angle histories for time-series charts
- concise deterministic feedback messages

The asymmetry index is `abs(left_rom - right_rom) / max(left_rom, right_rom) *
100`. A zero-ROM bilateral recording produces zero asymmetry rather than a
division error.

## Repetition Detection

Each joint uses a two-threshold state machine:

1. Detect the configured rest position.
2. Detect movement through the active threshold.
3. Count a repetition only after returning to rest.
4. Mark the repetition correct when its ROM reaches the configured completion
   ratio of the target ROM and its duration is within the allowed range.

This follows the state-machine concept from the MIT-licensed
`fitness-trainer-pose-estimation` project while using the project's own
`BaseAnalyzer._calculate_angle`.

## UI And Data Flow

The athlete can choose an uploaded video or a live browser webcam stream.

For uploaded video:

1. The athlete selects a rehabilitation protocol and FPS.
2. The page extracts frames and runs the existing person/pose pipeline.
3. Named keypoints are passed to the cached `get_rehab_analyzer()` instance.
4. The page renders summary metrics, bilateral joint tables, time-series and
   repetition ROM charts, deterministic feedback, and optional AI coaching.
5. The annotated video and report are displayed.
6. `save_analysis_to_db()` stores the session as `session_type="rehab"`.

For live webcam analysis, `streamlit-webrtc` sends browser frames to a stateful
processor. It keeps a bounded rolling window, updates the rehabilitation report
at a fixed frame interval, and overlays current ROM, repetitions, and asymmetry
on the returned stream. The athlete can inspect or save the current live report.

All visible page text uses `t()` with Ukrainian and English entries.

## Persistence

No schema migration is needed. Existing columns store searchable aggregates:

- `session_type`: `rehab`
- `exercise_type`: protocol identifier
- `reps`: correct target-joint repetitions
- `symmetry_score`: bilateral symmetry score
- `stability_score`: target ROM completion score
- `ai_score` and `ai_summary`: existing coach output
- `full_analysis`: complete rehabilitation report as JSON

History filters and icons are extended to include rehabilitation sessions.

## Error Handling

Missing or low-quality pose frames are skipped. Empty or too-short recordings
return a valid zero-valued report with actionable feedback. Invalid protocol
identifiers raise `ValueError` and are shown as localized UI errors.

## Testing

Unit tests use generated MediaPipe-style named keypoints and require only
NumPy/pytest. They cover bilateral ROM, asymmetry, repetition counting, deficit
calculation, empty input, invalid protocols, and raw dictionary keypoint
extraction. Database tests verify that rehab aggregates are stored without a
schema change.
