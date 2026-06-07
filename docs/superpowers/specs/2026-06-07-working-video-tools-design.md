# Working Video Tools Design

## Goal

Turn the existing `/tools` prototype into a working local video-processing
surface. The first release provides two reliable operations:

- trim a selected time range into a browser-compatible MP4
- extract JPEG frames either at a time interval or as an exact evenly spaced
  count

Each completed operation offers two independent actions:

- download the generated artifact immediately
- save the generated artifact and operation metadata to the current athlete's
  History

The tools run on the local SPRINT AI backend and do not send source videos to a
third-party service.

## Product Scope

### Trim And Cut

The user uploads one supported video and enters:

- start time in seconds
- end time in seconds

The backend validates the requested range against the real source duration and
creates an H.264 MP4 with `yuv420p` pixel format and fast-start metadata. The
result keeps the source dimensions and includes source audio when audio is
present.

### Frame Extractor

The user uploads one supported video and chooses one extraction mode:

1. `Every N seconds`
   - accepts a positive interval in seconds
   - extracts frames from the beginning of the video at that interval
2. `Exact frame count`
   - accepts the desired number of frames
   - distributes timestamps evenly across the source duration

The result is a ZIP archive containing:

- sequential JPEG files
- `manifest.json` with source filename, source duration, extraction mode,
  requested value, actual frame count, and timestamp for every image

The extractor caps one job at 200 JPEG files. Requests that would exceed the
cap fail before processing with an actionable validation message.

### Deferred Tools

`Stabilise & crop`, `Multi-angle merger`, `Slow-motion remaster`, and `Bulk
export` remain visible but disabled and labelled as prototypes. Their status
must not imply that they are executable.

## User Experience

The `/tools` page contains:

1. A header that states that two tools are available locally.
2. A tool selector for `Trim & cut` and `Frame extractor`.
3. One shared upload surface.
4. Operation-specific controls.
5. Processing progress and errors.
6. A result card with artifact metadata and:
   - `Download`
   - `Save to History`

Changing the selected tool preserves the selected source file but resets
operation progress and the previous result. Starting a new operation
replaces the current temporary result in the UI without deleting previously
saved History artifacts.

The UI does not claim background processing after the browser tab is closed.
The current in-memory job model remains explicitly single-server.

## Architecture

### Backend Modules

- `backend/app/api/tools.py`
  - validates request forms
  - stores uploads in the job workspace
  - starts processing threads
  - exposes status, SSE, download, and save endpoints
- `backend/app/services/tools.py`
  - probes source metadata with FFprobe
  - runs FFmpeg commands
  - creates frame manifests and ZIP archives
  - returns JSON-safe artifact metadata
- `backend/app/services/jobs.py`
  - remains the shared in-memory registry and event stream

The implementation reuses the existing upload validation helpers, job registry,
SSE event format, mounted `/data` volume, and `AthleteDatabase`.

### Frontend Modules

- `frontend/lib/tools.ts`
  - upload functions
  - event subscription
  - download URL
  - save-to-History function
- `frontend/components/tools/VideoToolsWorkspace.tsx`
  - source-file selection
  - tool and mode controls
  - validation, progress, result, download, and save states
- `frontend/app/tools/page.tsx`
  - page composition and deferred-tool cards

The workspace owns one active job subscription at a time and closes the
previous `EventSource` when a new job starts or the component unmounts.

## API Contract

### Start Trim

`POST /api/tools/trim`

Multipart fields:

- `video`
- `start_sec`
- `end_sec`

Response:

```json
{"job_id": "abc123"}
```

### Start Frame Extraction

`POST /api/tools/frames`

Multipart fields:

- `video`
- `mode`: `interval` or `count`
- `interval_sec` when mode is `interval`
- `frame_count` when mode is `count`

Response:

```json
{"job_id": "abc123"}
```

### Shared Job Endpoints

- `GET /api/tools/{job_id}`
- `GET /api/tools/{job_id}/events`
- `GET /api/tools/{job_id}/download`
- `POST /api/tools/{job_id}/save`
- `GET /api/tools/history/{session_id}/download`

The save request accepts `athlete_name` and is idempotent. Repeated calls return
the same History session id.

### Events

Progress:

```json
{"type": "progress", "pct": 40, "label": "Encoding clip"}
```

Result:

```json
{
  "type": "result",
  "operation": "trim",
  "artifact_name": "session-trim-10.0-20.0.mp4",
  "media_type": "video/mp4",
  "size_bytes": 123456,
  "metadata": {}
}
```

Error:

```json
{"type": "error", "message": "End time must be greater than start time"}
```

## Processing Rules

### Source Validation

The existing upload rules apply:

- MP4, MOV, AVI, or MKV
- maximum upload size from `MAX_VIDEO_UPLOAD_MB`
- hard byte limit while copying

FFprobe must successfully return positive duration, dimensions, and video
stream information before processing begins.

### Trim Encoding

FFmpeg produces:

- H.264 video through `libx264`
- `yuv420p`
- `+faststart`
- AAC audio when source audio exists

The requested range must satisfy:

- `start_sec >= 0`
- `end_sec > start_sec`
- `end_sec <= source duration`

### Frame Extraction

JPEG filenames use zero-padded sequence numbers:

```text
frame-0001.jpg
frame-0002.jpg
```

The manifest records the actual timestamp represented by each image. Exact
count mode returns the requested number unless source decoding fails. Interval
mode rejects requests that calculate to more than 200 frames instead of
silently truncating the result.

Interval timestamps are `0, interval, 2 * interval, ...` while the timestamp is
strictly below the source duration. Exact-count timestamps use
`index * duration / frame_count`, which returns one frame at timestamp zero
when the requested count is one and avoids requesting a frame exactly at the
end of the file.

## Temporary And Persistent Storage

Unarchived jobs remain under `/tmp/sprint-ai-jobs/<job_id>`.

Saving to History copies the generated artifact to:

```text
/data/session-artifacts/tools/<job_id>/<artifact_name>
```

The mounted `./data:/data` volume keeps saved artifacts across backend rebuilds
and restarts.

History uses:

- `session_type="tool"`
- `exercise_type="trim"` or `"frame_extractor"`
- operation metadata in `full_analysis`
- generated artifact path in `video_path`

The existing `video_path` column is treated as a generic session artifact path
for tool sessions. No database migration is required.

History adds a `Tools` discipline and displays:

- operation name
- processing date
- trim duration or extracted frame count
- `artifact saved`
- a download action for the persisted artifact

`GET /api/tools/history/{session_id}/download` serves saved tool artifacts. It
returns `404` when the session does not exist, is not a tool session, or its
artifact is missing. The frontend never receives arbitrary filesystem paths.

## Error Handling

The UI distinguishes:

- unsupported upload type
- upload too large
- unreadable or zero-duration video
- invalid trim range
- invalid extraction value
- extraction exceeding 200 frames
- FFmpeg or FFprobe failure
- backend unavailable
- expired temporary job
- artifact already removed
- History save failure

Failed processing never creates a History entry. Failed persistence does not
delete the temporary downloadable result, allowing the user to retry.

## Security And Resource Limits

- All output filenames are generated server-side.
- User filenames are used only as display metadata after basename
  normalization.
- Download endpoints resolve artifacts from the known job or session record;
  they do not accept filesystem paths.
- FFmpeg is called with argument arrays and never through a shell.
- One frame-extraction job is capped at 200 JPEG files.
- Existing upload byte limits remain enforced.

## Testing

### Backend Unit Tests

- trim range validation
- source probe parsing
- FFmpeg command construction without shell invocation
- interval timestamp calculation
- exact-count timestamp calculation
- 200-frame limit
- ZIP manifest contents
- result metadata
- idempotent History save
- persisted artifact download authorization by session record

### Frontend Tests

- selecting both available tools
- interval and exact-count mode controls
- request payload construction
- progress and error rendering
- download link after completion
- save button state and idempotent success state
- disabled prototype tool cards

### Integration And Browser QA

Use the existing short MOV fixture to verify:

1. trim upload produces a playable H.264 MP4
2. interval extraction produces the expected ZIP and manifest
3. exact-count extraction produces the requested JPEG count
4. both artifacts download from the result card
5. both artifacts save to History
6. saved artifacts still download after backend restart
7. `/tools` has no framework overlay or relevant console errors

## Presentation Definition Of Done

The feature is ready when a colleague can open `/tools`, process a real video
with either available tool, download the result, save it to History, restart the
backend, and download the saved artifact again without using the terminal.
