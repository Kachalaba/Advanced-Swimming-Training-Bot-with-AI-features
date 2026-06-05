# Web Rehabilitation Live Design

## Goal

Move rehabilitation into the Next.js interface without losing the current
SPRINT AI visual system. The new section supports uploaded recordings and a
browser-webcam live mode. Live mode adds a switchable postural coordinate map,
relative camera-level calibration, compact rehabilitation metrics, and a
fullscreen training view.

The implementation must preserve boundaries that allow the camera and sensor
adapters to be replaced by native macOS implementations later without rewriting
the rehabilitation screen.

## Product Scope

The first web release includes:

- a `Rehabilitation` item in the existing top navigation
- the five rehabilitation protocols already supported by `RehabAnalyzer`
- uploaded-video analysis through the existing backend analysis pipeline
- live webcam capture with explicit camera permission
- a `Show postural map` switch that does not restart the camera
- relative camera-roll calibration
- fullscreen live mode with exit through `Esc`
- live ROM, repetition, asymmetry, and camera-level indicators
- the existing rehabilitation report and session persistence contract

The first release does not claim access to a Mac accelerometer. It does not
provide a medically validated diagnosis or an absolute gravity reference.

## Visual Direction

The page follows the existing dark SPRINT AI design system:

- `#0A0E14` page background
- cyan as the primary analysis accent
- amber for shoulder-axis warnings
- rose for pelvic-axis warnings
- emerald for calibrated and acceptable states
- translucent dark overlays with restrained borders and blur
- compact tabular numerals for live measurements

The primary live surface is one large camera canvas. Controls and metrics sit at
the edges so that the athlete remains visible. The postural map is a light,
semi-transparent overlay rather than a separate panel.

## Page Structure

The rehabilitation route is `/rehabilitation`.

The page contains:

1. A discipline header with protocol selection and input-mode control.
2. A live camera or upload workspace.
3. Live metrics for left/right ROM, correct repetitions, asymmetry, and camera
   level.
4. A results area using the existing rehabilitation report fields.
5. A concise training-aid disclaimer.

On narrow screens, controls wrap above the camera and metrics collapse into a
two-column rail. The camera remains the dominant surface.

## Live Camera Controls

The camera toolbar contains:

- `Show postural map`
- `Calibrate level`
- `Fullscreen`
- live connection status

Toggling the postural map changes only the overlay visibility. It must not
replace the video element, reacquire the camera, reset metrics, or restart the
analysis session.

Fullscreen uses the browser Fullscreen API on the complete live workspace, not
only the raw video element. The compact metrics, camera-level indicator, and
postural-map control remain visible. `Esc` exits fullscreen using the browser's
standard behavior.

## Postural Coordinate Map

The postural map is generated from detected pose landmarks and contains:

- a vertical plumb axis through the calibrated frame center
- a horizontal reference axis
- a shoulder axis between left and right shoulder landmarks
- a pelvic axis between left and right hip landmarks
- a trunk centerline between shoulder and hip midpoints
- a restrained body contour connecting major joints
- numeric shoulder, pelvic, and trunk deviations in degrees

The overlay communicates:

- which shoulder is higher
- which side of the pelvis is higher
- lateral trunk displacement from the plumb axis
- trunk lean relative to the vertical reference

Colors indicate state without hiding the camera image. Cyan is neutral, amber is
a moderate shoulder or trunk deviation, rose is a larger pelvic or asymmetry
warning, and emerald indicates an acceptable calibrated state.

Postural measurements are coaching signals derived from a two-dimensional
camera view. They must be labelled as training feedback, not diagnosis.

## Camera-Level Calibration

The web application cannot read a MacBook accelerometer through a supported
browser or macOS public API. Camera level is therefore explicitly relative to a
user-created calibration.

When the athlete selects `Calibrate level`:

1. The current camera frame becomes the zero-roll reference.
2. Static background features outside the detected athlete region are sampled.
3. Subsequent frames estimate camera rotation from robust background-feature
   tracking.
4. The displayed value is the estimated roll delta from the reference frame.

The estimator must exclude the athlete bounding region when enough background
features are available. It must use robust outlier rejection so body movement
does not dominate the estimate.

The UI labels the result `Relative to last calibration`. It shows:

- emerald `LEVEL` within the configured tolerance
- amber directional guidance outside tolerance
- `Recalibrate` when confidence is insufficient or the scene changes

If the environment has too few stable features, the application keeps the
visual grid available but does not invent an angle.

## Architecture

The frontend uses focused adapters and presentation components:

- `CameraSource` owns browser media acquisition and stream lifecycle.
- `LiveRehabTransport` sends sampled frames and receives analysis updates.
- `PostureOverlay` renders axes, contour, landmarks, and deviation labels.
- `CameraLevelProvider` exposes calibration, angle, confidence, and status.
- `LiveRehabWorkspace` composes video, overlays, controls, metrics, and
  fullscreen state.
- `RehabResults` renders the existing report contract.

The first `CameraSource` uses `navigator.mediaDevices.getUserMedia`. The first
`CameraLevelProvider` uses relative optical calibration. Future macOS adapters
can implement the same interfaces using AVFoundation or an external iPhone
motion source.

The backend owns pose inference and rehabilitation metrics. A live endpoint
accepts sampled JPEG or WebP frames, feeds named keypoints into the existing
cached `RehabAnalyzer`, and returns bounded rolling updates. The frontend
renders the postural map from returned landmarks and measurements.

Frame transport is rate-limited independently from video playback so the local
preview remains smooth when analysis is slower. Only one frame may be awaiting
analysis at a time; newer frames replace stale queued frames.

## Live Data Contract

Each live update contains:

- timestamp and frame sequence
- pose-detection confidence
- normalized major-joint landmarks
- left and right target-joint angles and ROM
- correct repetition count
- asymmetry index and symmetry score
- shoulder-axis angle
- pelvic-axis angle
- trunk lean and lateral displacement
- camera roll delta and camera-level confidence
- analysis warnings

Coordinates are normalized to the source frame so the overlay scales correctly
in normal and fullscreen layouts.

## State And Persistence

Starting a session creates one live-session state object containing:

- athlete and protocol
- camera device
- calibration reference
- postural-map visibility
- rolling keypoint window
- latest report
- start time and connection status

The postural-map visibility may be retained in local storage. Calibration data
is session-only because it is invalid after camera or device movement.

Saving a live session reuses the existing database representation:

- `session_type="rehab"`
- protocol in `exercise_type`
- correct reps in `reps`
- symmetry in `symmetry_score`
- completion in `stability_score`
- full rehabilitation report in `full_analysis`

## Error Handling

The live workspace provides specific states for:

- camera permission required
- permission denied
- no camera found
- camera disconnected
- backend unavailable
- pose not detected
- insufficient background features for level estimation
- calibration invalidated by a large scene change
- analysis delayed

Camera failure must not crash the route. Upload mode remains available as a
fallback. Fullscreen exits cleanly if the stream ends.

## Privacy And Performance

Camera access begins only after an explicit user action. The interface states
whether frames are processed locally or sent to the local backend.

The web release processes frames through the local SPRINT AI backend and does
not upload video to a third-party service. Live transport uses bounded frame
sizes, a configurable analysis rate, and backpressure. Components displaying
high-frequency measurements use refs or isolated state so the entire page does
not rerender for every camera frame.

## Testing

Unit tests cover:

- postural-axis calculations from normalized landmarks
- shoulder, pelvis, and trunk deviation direction
- camera-level status thresholds and low-confidence behavior
- overlay visibility without camera restart
- fullscreen state transitions
- frame backpressure and stale-frame replacement

Backend tests cover:

- live frame validation
- rolling report updates
- empty or low-confidence pose frames
- protocol validation
- compatibility with the existing `RehabAnalyzer` report

Browser verification covers:

- camera permission and denial states
- live preview start and stop
- postural-map toggle
- calibration and recalibration
- fullscreen entry and `Esc` exit
- responsive layout
- saving a completed live session

## Future macOS Path

The later macOS application can reuse the same page concepts and data contracts
while replacing:

- browser media capture with AVFoundation
- web transport with an in-process analysis service
- relative optical level with the best available native or paired-device
  provider

Because Mac accelerometer data is not available through a supported public API,
an absolute gravity reference would use a paired iPhone or another supported
external sensor rather than an undocumented Mac hardware interface.
