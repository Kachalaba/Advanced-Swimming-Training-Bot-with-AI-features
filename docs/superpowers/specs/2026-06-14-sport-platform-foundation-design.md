# Sport Platform Foundation Design

## Goal

Make the Next.js and FastAPI application the primary SPRINT AI product for
sport analysis, while retaining Streamlit as a frozen legacy/demo shell.

The first release establishes one reusable sport-session contract and completes
the running workflow from upload through saved history and progress. Swimming
uses the same persisted-session read model. Cycling and dryland remain visible,
but must stop presenting fictional results until their web analysis adapters
are implemented.

## Product Principles

1. **Evidence before polish.** The UI must distinguish persisted measurements
   from examples, unavailable features, and empty states.
2. **One athlete identity.** New sport sessions are saved against a stable
   athlete ID. Name-based creation remains only as a backward-compatible
   fallback.
3. **One session model.** Running, swimming, cycling, and dryland share a
   normalized API read model even when their underlying metrics differ.
4. **One primary application.** New product functionality is implemented in
   FastAPI and Next.js. Streamlit receives compatibility fixes only.
5. **Progress is discipline-specific.** Comparisons only use compatible
   sessions from the same sport and metric contract.
6. **No invented values.** Empty history displays an onboarding state instead
   of demo personal records, scores, sessions, or AI insights.

## Scope

### Included

- mark Next.js + FastAPI as the primary product in architecture documentation
- mark Streamlit as legacy/demo and freeze new feature development there
- introduce a normalized sport overview API
- normalize persisted running and swimming analysis payloads
- save completed running analyses by athlete ID
- persist annotated running videos beside other session videos
- expose running status with saved-session identity
- show real running metrics and history on the running landing page
- show real swimming history on the swimming landing page
- add a save action and athlete selection to the running result page
- replace cycling and dryland demo data with honest empty/readiness states
- add unit and frontend tests for normalization, persistence, and UI states

### Deferred

- cycling upload and analysis API
- dryland live/upload web workflow
- durable background queue
- authentication and multi-tenant clinics
- migration from raw SQLite to SQLAlchemy
- cloud video storage
- coach-created training plans
- cross-discipline readiness or injury predictions

These deferred items remain roadmap work. The shared contracts introduced here
must allow them to be added without changing existing frontend consumers.

## Architecture

```text
Next.js sport page
  -> GET /api/athletes/me
  -> GET /api/athletes/{athlete_id}/sports/{sport}/overview

Next.js result page
  -> POST /api/analysis/{sport}
  -> SSE analysis events
  -> POST /api/analysis/{sport}/{job_id}/save

FastAPI analysis API
  -> backend sport pipeline
  -> video_analysis analyzer
  -> normalized analysis payload
  -> AthleteDatabase session

FastAPI athlete API
  -> stored TrainingSession.full_analysis
  -> sport session normalizer
  -> SportOverview response
```

The existing SQLite `sessions` table remains the persistence source of truth.
No schema migration is required for this release. Sport-specific details remain
inside `full_analysis`; frequently used legacy columns are populated when a
reliable equivalent exists.

## Normalized Sport Read Model

The backend exposes:

```text
GET /api/athletes/{athlete_id}/sports/{sport}/overview
```

Supported sport values in the first release:

```text
swimming
running
cycling
dryland
```

The response contains:

- athlete identity
- sport
- total session count
- latest session date
- latest score when supported
- headline metrics from the latest valid session
- a chronological metric series for progress charts
- newest-first persisted session cards
- generated evidence-based insights from the latest valid session

Each metric uses:

```text
key
label
value
unit
higher_is_better
```

Each session card uses:

```text
id
date
score
summary
duration_sec
has_video
metrics
```

Unsupported or absent values are omitted, not returned as zero.

## Running Persistence

The running API gains:

```text
POST /api/analysis/running/{job_id}/save
```

The save request accepts:

```text
athlete_id: optional integer
athlete_name: optional fallback string
```

Validation rules:

- the job exists and has kind `running`
- analysis completed successfully
- a result payload exists
- athlete ID resolves when supplied
- repeated saves return the original session ID

The saved analysis contract is:

```json
{
  "running_analysis": {
    "type": "result",
    "analysis": {},
    "frames_total": 0,
    "frames_with_pose": 0,
    "video_path": "annotated.mp4"
  }
}
```

The database session stores:

- `session_type = "running"`
- `ai_score = efficiency_score`
- `symmetry_score = arm_symmetry`
- `full_analysis` with the complete result
- the copied annotated video path
- a concise summary derived from cadence and foot strike

## Swimming Read Compatibility

Existing swimming saves remain unchanged. The sport normalizer reads
`full_analysis.swimming_analysis` and exposes:

- overall technique score
- available-zone coverage
- main issue
- body position, rotation, catch, breathing, and kick scores when available

No swimming analysis behavior changes in this release.

## Frontend

### Running Landing

The running page loads the current athlete and their normalized running
overview. It shows:

- real latest metrics
- real saved sessions
- generated insights based on the latest result
- a clear empty state before the first saved analysis
- the existing uploader

The page never renders personal records or prior sessions that are not present
in the database.

### Running Result

After analysis, the user can:

- review annotated video and metrics
- select an athlete
- save the session
- see idempotent saved confirmation
- return to the running workspace

Save failures remain visible without discarding the completed analysis.

### Swimming Landing

The swimming page retains its evidence-first product explanation and uploader.
Its recent-session panel uses persisted swimming history. Static capability
cards remain because they describe the analyzer, not fabricated athlete data.

### Cycling And Dryland

Demo badges, fictional metrics, fictional sessions, and fictional insights are
removed. Their pages present current analyzer capabilities and an explicit
web-workflow readiness state. They do not claim that a user has completed
sessions.

## Error Handling

- overview returns `404` for an unknown athlete
- overview returns `422` for an unsupported sport
- malformed historic JSON is skipped and does not break the whole response
- completed sessions with unsupported payload versions remain visible with
  basic metadata but no fabricated metrics
- save returns `409` while analysis is incomplete
- save returns `404` for missing or wrong-kind jobs
- save returns `404` for an unknown athlete ID
- video-copy failure does not discard analysis; the session is saved with an
  empty video path and the error is logged
- frontend pages render retryable errors and retain upload controls

## Testing

Backend tests cover:

- running analysis normalization
- swimming analysis normalization
- malformed historic payloads
- empty sport overview
- athlete-not-found behavior
- running save by athlete ID
- running save by name fallback
- idempotent save
- wrong job kind and incomplete analysis
- annotated-video persistence

Frontend tests cover:

- loading and empty states
- real running metrics and sessions
- no demo data
- athlete selection and running save
- save success and failure
- persisted swimming sessions
- cycling and dryland honesty states

The release gate remains:

```text
make lint
pytest tests/unit/ -v
frontend npm test
frontend npm run typecheck
frontend npm run build
docker compose build
browser verification on running, swimming, cycling, and dryland
```

## Rollout

1. Merge the existing analyzer cleanup PR.
2. Add normalized sport read models and tests.
3. Add running persistence and tests.
4. Connect running and swimming landing pages to persisted data.
5. Remove fictional cycling and dryland data.
6. Run full CI and browser verification.
7. Merge through a focused PR.

## Success Criteria

- a running video can be analyzed, assigned to an athlete, saved, and displayed
  in that athlete's running workspace
- saved running metrics survive backend and frontend restarts
- saved swimming sessions appear in the swimming workspace
- running, cycling, and dryland pages contain no fictional athlete results
- the primary architecture is documented unambiguously
- all backend, frontend, build, Docker, and browser gates pass
