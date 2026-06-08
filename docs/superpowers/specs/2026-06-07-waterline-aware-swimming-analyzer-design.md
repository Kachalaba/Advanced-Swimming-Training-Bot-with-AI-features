# Waterline-Aware Swimming Analyzer Design

## Goal

Build a presentation-ready freestyle analyzer for side-on phone video recorded
from the pool deck. The analyzer is designed for triathletes and Masters
swimmers and addresses the core failure mode of generic pose estimation:
different parts of the swimmer repeatedly move above water, below water, and
through a reflective waterline.

The first result must answer four practical coaching questions:

1. What is the swimmer's main technical error?
2. At what exact moment can the swimmer see it?
3. Why does it reduce speed or increase energy cost?
4. What drill and mini-set should the swimmer do next?

The product differentiates from MySwimPro's plan and workout personalization by
creating an evidence-based bridge from a specific video moment to the next
training action.

## Product Principles

- Show less rather than inventing missing biomechanics.
- Treat confidence as part of the result, not an implementation detail.
- Diagnose repeated patterns, not one unusual frame.
- Connect every diagnosis to video evidence and a corrective action.
- Return useful partial analysis when only some technique zones are reliable.
- Keep the first version narrow: side-on freestyle recorded from the pool deck.

This system is a coaching aid. V1 does not claim laboratory-grade kinematics or
medical assessment.

## V1 Scope

### Included

- one uploaded video containing side-on freestyle
- one primary swimmer
- automatic waterline estimation
- separate above-water and underwater image treatment
- temporally tracked pose landmarks
- strict confidence-aware skeleton rendering
- automatic selection of three to five best complete stroke cycles
- analysis of five technique zones
- one primary issue with exact video evidence
- one corrective drill and one workout mini-set
- browser-compatible annotated video
- saving a structured result to athlete History

### Deferred

- other strokes
- overhead, front, rear, or underwater-only camera angles
- simultaneous multi-camera reconstruction
- exact 3D joint angles
- athlete identity recognition
- live poolside analysis
- model training or fine-tuning
- automatic pace measurement without a calibrated pool reference
- reliable kick-rate claims when the legs are occluded

Unsupported inputs must receive an actionable reshoot message instead of a
fabricated score.

## System Architecture

The implementation is a FastAPI and Next.js workflow built around the existing
job registry, SSE event stream, upload validation, video processing utilities,
analysis primitives, History storage, and `BaseAnalyzer` helpers.

### 1. Video Quality Gate

The gate probes duration, dimensions, frame rate, blur, brightness, visible
swimmer size, camera stability, and approximate side-on suitability.

It returns one of three outcomes:

- `pass`: analysis can proceed normally
- `partial`: analysis can proceed, but affected zones carry warnings
- `reject`: no trustworthy stroke analysis can be produced

The gate rejects videos when no swimmer can be locked, no complete cycle can be
found, or the image is unusable throughout. It does not reject an otherwise
useful video merely because one technique zone is occluded.

### 2. Swimmer Lock-On

YOLO detects candidate people. A temporal tracker selects and follows one
swimmer using bounding-box overlap, position continuity, size continuity, and
motion direction. A short detection gap is bridged using the last stable track;
a sustained or ambiguous loss lowers cycle quality or ends the usable segment.

The pipeline never switches silently to another swimmer.

### 3. Waterline Detector

For each usable frame, the detector estimates a normalized line
`y = slope * x + intercept`. Candidate lines come from horizontal edges,
brightness and color transitions, reflection structure, and their relationship
to the tracked swimmer box.

Temporal smoothing rejects abrupt line jumps. Each estimate includes a
confidence score. If a frame has weak evidence, the pipeline reuses a recent
stable estimate with decaying confidence rather than treating a guessed line as
fully observed.

### 4. Dual-Zone Pose

The swimmer crop is processed in two visual zones:

- above water: original color and contrast with mild normalization
- below water: color correction, local contrast enhancement, and restrained
  dehazing

Pose inference uses the best available evidence from the original crop and both
zone treatments. Landmark candidates are merged by confidence and temporal
consistency. The waterline itself is not a hard anatomical boundary: landmarks
near it receive an uncertainty penalty because reflections and refraction are
most disruptive there.

### 5. Temporal Joint Tracker

Landmarks are stabilized with existing EMA and movement guards. The tracker
maintains left/right identity, rejects impossible jumps, and permits short
gaps. It records observed, bridged, and missing states separately so downstream
metrics cannot mistake interpolation for direct evidence.

New analyzer classes use `BaseAnalyzer`; shared point, angle, smoothing, and EMA
logic is not duplicated. Existing swimming analyzers may be reused behind
adapters where their output is reliable, without broad unrelated refactoring.

### 6. Strict Confidence Layer

Every landmark, metric, cycle, zone, and diagnosis carries confidence. Skeleton
segments render only when both endpoints meet the display threshold. Missing
segments stay missing; the UI never draws a complete body merely for visual
continuity.

### 7. Stroke Cycle Selector

The selector identifies complete freestyle cycles from recurring wrist,
shoulder, and torso motion. It scores candidates for:

- phase completeness
- swimmer lock stability
- landmark continuity
- waterline stability
- useful body coverage
- low blur and low occlusion

The best three to five non-duplicate cycles are selected. If only two reliable
cycles exist, analysis may return a partial result but cannot promote a main
diagnosis unless the issue is confirmed in both. With fewer than two reliable
cycles, the analysis is rejected with reshoot guidance.

### 8. Technique Engine

The engine evaluates the five agreed technique zones. Each metric retains its
source cycles and exact evidence interval. It must be possible to trace every
displayed conclusion back to frames and confidence inputs.

### 9. Coaching Engine

Rule-based coaching ranks repeated issues by confidence, expected performance
impact, and actionability. It produces a deterministic baseline diagnosis,
drill, and mini-set. Optional external AI may rewrite the explanation for tone
but cannot invent metrics, evidence, or exercises outside the structured
analysis.

## Five-Zone Metric Contract

### Body Position

Primary observations:

- shoulder-to-hip-to-ankle body axis
- hip vertical offset and hip-drop angle
- head-to-spine alignment
- stability across the selected cycle

Output examples include `stable`, `hips dropping`, or `insufficient evidence`.
A body-position issue must persist through a meaningful portion of at least two
selected cycles.

### Torso Rotation

Primary observations:

- left and right shoulder-roll amplitude
- left and right hip-roll amplitude
- shoulder-to-hip rotation timing offset
- left/right asymmetry

V1 uses side-view 2D proxies and labels them as estimates. It does not present
the values as true 3D axial rotation.

### Catch And Arm Path

Primary observations:

- hand entry relative to the shoulder line
- entry width and crossover tendency
- elbow angle during early catch
- backward hand-path proxy
- left/right arm timing

Underwater catch metrics may be unavailable when bubbles, refraction, or the
pool edge hide the wrist or elbow. Unavailable catch evidence is not converted
into a negative score.

### Breathing And Head Return

Primary observations:

- detected breathing side
- peak head rotation
- head-return timing relative to hand entry
- body-line or hip-position disruption during breathing

The engine distinguishes a visible breathing event from ordinary head motion.
It does not diagnose breathing timing when no full breathing event appears in
the selected cycles.

### Kick

Primary observations:

- normalized vertical ankle amplitude
- left/right amplitude symmetry
- hip-driven versus knee-dominant motion proxy
- continuity of visible kicking

Exact kick rate is deferred when the feet are repeatedly outside the frame or
occluded. The UI describes the visible evidence rather than extrapolating it.

## Confidence Model

Confidence values are stored internally on a `0.0` to `1.0` scale.

### Display Levels

- `high`: `>= 0.78`
- `medium`: `>= 0.55` and `< 0.78`
- `insufficient`: `< 0.55`

High-confidence metrics and skeleton segments render normally. Medium-confidence
metrics render with a limited-precision label. Insufficient metrics are hidden
from scoring and shown only as unavailable.

### Metric Confidence

The initial deterministic score is:

```text
0.35 * landmark visibility
+ 0.25 * temporal continuity
+ 0.15 * waterline clarity
+ 0.15 * left/right identity stability
+ 0.10 * cycle coverage
```

All components are normalized to `0.0` through `1.0`. A metric may impose a
hard prerequisite, such as requiring both wrist and elbow evidence. Failing a
hard prerequisite makes the metric insufficient regardless of the weighted
score.

Direct observations and bridged observations are tracked separately. Bridged
landmarks reduce visibility and continuity; they cannot independently establish
a diagnosis.

### Zone And Diagnosis Rules

- A zone score requires sufficient evidence in at least two selected cycles.
- Missing zones are excluded, not scored as zero.
- The overall summary reports coverage, for example `4 of 5 zones analyzed`.
- A primary issue requires medium or high confidence.
- A primary issue must be reproduced in at least two selected cycles.
- Its evidence must include an interval and a peak timestamp.
- If no issue satisfies these rules, the result says that no repeatable primary
  issue was confirmed.

Any aggregate score is calculated only from available zones and is displayed
beside its coverage. V1 prioritizes zone statuses and evidence over a single
headline number.

## Data Flow

1. The user uploads a video and starts analysis.
2. FastAPI validates and stores the upload in a job workspace.
3. The quality gate produces warnings or an early actionable rejection.
4. Swimmer tracking and waterline estimation produce frame context.
5. Dual-zone pose and temporal tracking produce confidence-aware landmarks.
6. The cycle selector chooses three to five best complete cycles.
7. The technique engine evaluates all available zones.
8. The coaching engine chooses the main issue and prescription.
9. The renderer produces a browser-compatible annotated MP4.
10. FastAPI sends the structured result through SSE.
11. The user may save the result and annotated video to History.

## Backend Boundaries

The intended boundaries are:

- `backend/app/api/analysis.py`
  - swimming upload, status, events, video, and save endpoints
- `backend/app/services/swimming.py`
  - pipeline orchestration and progress events
- `video_analysis/waterline_analyzer.py`
  - frame waterline candidates, smoothing, and confidence
- `video_analysis/swimming_cycle_selector.py`
  - cycle detection, quality scoring, and selection
- `video_analysis/swimming_technique_analyzer.py`
  - five-zone metric and confidence contract
- existing detector, pose, biomechanics, visualization, coaching, and database
  modules
  - reused where compatible through small adapters

These module names are design targets. Implementation may place small data
models beside the owning service, but orchestration, waterline estimation,
cycle selection, and technique scoring must remain independently testable.

## API Contract

### Start Analysis

`POST /api/analysis/swimming`

Multipart fields:

- `video`
- `fps`, optional override; source FPS is preferred when reliable
- `athlete_name`, optional until save

Response:

```json
{"job_id": "abc123"}
```

### Job Endpoints

- `GET /api/analysis/swimming/{job_id}`
- `GET /api/analysis/swimming/{job_id}/events`
- `GET /api/analysis/swimming/{job_id}/video`
- `POST /api/analysis/swimming/{job_id}/save`

Save is idempotent and returns the same History session id when repeated.

### SSE Events

Progress stages are stable UI identifiers:

```text
upload
quality_gate
tracking
waterline
pose
cycles
technique
coaching
rendering
completed
```

Progress event:

```json
{
  "type": "progress",
  "stage": "cycles",
  "pct": 58,
  "label": "Selecting the clearest stroke cycles"
}
```

Result event:

```json
{
  "type": "result",
  "analysis_type": "swimming_freestyle_side",
  "quality": {
    "status": "partial",
    "warnings": ["Feet leave the frame in two selected cycles"]
  },
  "coverage": {
    "available_zones": 4,
    "total_zones": 5
  },
  "cycles": [
    {
      "id": "cycle-1",
      "start_sec": 12.4,
      "end_sec": 14.1,
      "quality": 0.86
    }
  ],
  "zones": [],
  "primary_issue": {},
  "prescription": {},
  "video_url": "/api/analysis/swimming/abc123/video"
}
```

Each zone contains `status`, `confidence`, `confidence_level`, metrics, and
evidence. Each evidence item contains `cycle_id`, `start_sec`, `peak_sec`, and
`end_sec`. The primary issue references one or more of those evidence items
instead of copying unrelated timestamps.

Prescription structure:

```json
{
  "drill": {
    "name": "Side kick with rotation",
    "purpose": "Improve hip-led rotation",
    "execution": "Six kicks on each side before switching",
    "common_mistake": "Turning the head before the torso",
    "success_cue": "Head stays aligned while the hips initiate rotation"
  },
  "mini_set": {
    "title": "Rotation control",
    "repetitions": 6,
    "distance_m": 50,
    "rest_sec": 20,
    "intensity": "easy aerobic",
    "focus": "Match head return to torso rotation"
  }
}
```

Error events include a machine-readable code, a user-facing message, and
reshoot guidance when applicable.

## Result Experience

The `/swimming` page replaces demo-only metrics with a real upload and result
workspace.

## Visual Design Contract

The swimming analyzer remains part of the existing SPRINT AI product. It does
not introduce a separate visual theme, navigation model, or component language.
Its distinctiveness comes from the evidence-first analysis experience.

The implementation must preserve:

- the existing application shell, navigation, page width, and responsive
  spacing
- `bg`, `surface`, and `elevated` dark surfaces
- subtle `border-white/[0.06]` style boundaries and restrained shadows
- cyan as the swimming and primary-action accent
- emerald, amber, and rose only for semantic success, caution, and error states
- Inter-style sans typography and tabular monospace treatment for measurements
- the existing `rounded-lg`, `rounded-xl`, and `rounded-2xl` radius hierarchy
- short `fade-in` and `slide-up` transitions without decorative motion

Existing primitives such as `ChartContainer`, `StatusBadge`, `FileDropZone`,
`MetricCard`, `SegmentedControl`, and established button styles are reused or
extended before creating new equivalents. The swimming workspace may use a
custom composition where video evidence requires it, but that composition must
look native beside Running, Rehabilitation, Tools, Dashboard, and History.

The annotated player uses the existing black video surface and restrained
overlay controls. Confidence presentation follows the product's semantic color
system:

- high confidence: cyan or emerald
- medium confidence: amber
- insufficient evidence: neutral slate
- processing or input failure: rose

The primary-issue card may receive stronger hierarchy through size, border
emphasis, and a subtle cyan glow. It must not introduce neon-heavy effects,
large marketing typography, unrelated gradients, or a dashboard-within-a-
dashboard appearance.

### Before Analysis

- supported-input guidance
- concise side-on filming instructions
- upload surface
- validation feedback
- staged progress

### Result Priority

1. **Primary issue**
   - concise diagnosis
   - confidence level
   - number of confirming cycles
   - why it costs speed or energy
2. **Exact moment**
   - video seeks to the peak timestamp
   - slow replay around the evidence interval
   - waterline and only reliable skeleton segments
   - navigation across selected cycles
3. **Five technique zones**
   - `good`, `needs attention`, or `insufficient data`
   - key metric and confidence
   - evidence link that seeks the player
   - explicit analyzed-zone coverage
4. **Corrective drill**
   - purpose, execution, common mistake, and success cue
5. **Next-workout mini-set**
   - volume, rest, intensity, and focus
6. **Save and compare**
   - save result to History
   - later comparison by zone, metric, issue, and confidence

The page must not lead with a decorative overall score. Evidence and corrective
action are the primary information hierarchy.

## Rendering Rules

- Output MP4 uses H.264, `yuv420p`, and fast-start metadata.
- The waterline is rendered only when its confidence is sufficient.
- A skeleton segment renders only when both endpoint landmarks are medium or
  high confidence.
- Medium-confidence segments are visually distinct from high-confidence ones.
- Interpolated landmarks are never rendered as direct observations.
- Evidence intervals receive subtle timeline markers.
- The annotated video remains understandable without the overlay.

## Failure And Recovery

### Reject With Reshoot Guidance

- no stable swimmer lock
- fewer than two reliable complete cycles
- sustained severe blur or darkness
- unsupported camera angle
- swimmer too small for meaningful pose evidence

Guidance states what failed and how to record the next clip, such as moving
closer, stabilizing the phone, keeping the full body in frame, or recording
several complete cycles.

### Return Partial Analysis

- one or more zones are occluded
- feet leave the frame
- breathing is not visible
- underwater wrist evidence is weak
- waterline confidence drops for a limited interval

The result lists unavailable zones and preserves trustworthy zones.

### Runtime Recovery

- SSE reconnects by fetching current job status and resubscribing while the job
  still exists.
- A processing exception marks the job as failed and emits one terminal error.
- Temporary artifacts remain job-scoped.
- Saving copies the structured result and annotated video to persistent storage.

## Persistence

Saved sessions use the existing athlete History backend with a swimming session
type and freestyle exercise type. The full structured result is stored in
`full_analysis`; the annotated artifact path is stored with the session.

History comparison must retain:

- analysis contract version
- quality status and warnings
- zone availability
- metric values and confidence
- selected cycle intervals
- primary issue
- drill and mini-set

Contract versioning prevents future algorithm changes from making old and new
results appear directly equivalent without context.

## Testing Strategy

### Unit Tests

- waterline temporal smoothing and confidence decay
- landmark merge and waterline penalty
- left/right identity continuity
- cycle completeness and ranking
- confidence thresholds and hard prerequisites
- missing zones excluded from aggregate calculations
- primary issue requires two confirming cycles
- deterministic drill and mini-set mapping
- JSON-safe result serialization

New test fixtures use synthetic keypoint sequences and generated frames so core
tests do not require a downloaded model or external API.

### API Tests

- upload validation
- job creation and status
- stable SSE stage sequence
- partial result contract
- reject result with reshoot guidance
- video availability
- idempotent History save

### Frontend Tests

- upload and progress states
- quality rejection
- partial coverage
- primary issue hierarchy
- evidence click seeks the video
- cycle navigation
- insufficient zones do not display scores
- save state and repeated save

### End-To-End Acceptance

At least one representative side-on freestyle clip must demonstrate:

- stable primary-swimmer tracking
- visible waterline overlay
- three to five selected cycles when available
- strict segment rendering without fabricated full skeleton
- all available zones with confidence
- one repeated main issue or an honest no-confirmed-issue result
- exact evidence seeking
- drill and mini-set
- successful History save

The flow must pass backend unit and API tests, frontend lint/build/tests, Docker
startup, and browser-visible verification at `/swimming`.

## Definition Of Done

V1 is complete when it:

- accepts supported side-on freestyle video
- evaluates source quality before diagnosis
- tracks one swimmer without silent identity switching
- estimates and uses the waterline
- selects the best three to five complete cycles when available
- analyzes the five agreed zones with confidence
- suppresses unreliable metrics and skeleton segments
- promotes only a repeated, evidence-backed primary issue
- opens the exact issue moment in the video
- provides one corrective drill and a concrete mini-set
- reports partial coverage honestly
- saves the versioned result to History
- passes automated and browser-visible verification
