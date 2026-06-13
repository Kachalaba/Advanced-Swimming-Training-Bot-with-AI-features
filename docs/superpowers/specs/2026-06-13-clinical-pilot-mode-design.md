# Clinical Pilot Mode Design

## Goal

Turn the existing rehabilitation analysis, handoff report, and patient progress
surfaces into one complete workflow that a rehabilitation specialist can use
during a local pilot.

The first release is designed for one specialist working on one local computer.
Its domain model must also support two later scenarios without replacing the
core records:

1. multiple specialists working within a clinic
2. a patient completing assigned measurements at home

Clinical Pilot Mode remains a research and measurement-support tool. It does
not diagnose, prescribe treatment, or claim that a measured change represents
clinical recovery.

## Product Boundary

The pilot covers the workflow from selecting a patient to reviewing and
exporting a completed visit:

```text
Patient
  -> rehabilitation episode
  -> new clinical visit
  -> capture readiness
  -> live or uploaded analysis
  -> specialist review
  -> save
  -> comparison
  -> clinical report
```

The pilot does not include:

- authentication or user accounts
- clinic administration
- scheduling or billing
- diagnosis coding
- treatment prescription
- messaging
- cloud synchronization
- patient contact or payment details

## Architecture

The existing `Athlete` record remains the base person identity. Clinical Pilot
Mode adds three focused records around it.

### PatientProfile

`PatientProfile` stores the minimum rehabilitation context needed to conduct a
measurement session:

- linked athlete
- preferred display name
- affected side: left, right, bilateral, or unspecified
- concise clinical context
- precautions or contraindications
- active or archived status
- created and updated timestamps

The profile does not require a diagnosis, date of birth, address, phone number,
email, or insurance information.

### RehabEpisode

`RehabEpisode` represents one bounded course of rehabilitation:

- linked patient profile
- title
- primary rehabilitation protocol
- functional goal
- optional target ROM for each side
- status: active, completed, or archived
- start date
- optional completion date
- created and updated timestamps

A patient may have multiple episodes over time, but only one active episode per
protocol is allowed in the local pilot.

### ClinicalVisit

`ClinicalVisit` represents one professional measurement encounter:

- linked rehabilitation episode
- linked existing rehabilitation `TrainingSession` after analysis is saved
- visit timestamp
- capture source: live camera or uploaded video
- pre-session status note
- specialist observation
- capture quality status
- capture quality details
- review status: draft or finalized
- created and updated timestamps

The visit owns workflow context; the linked `TrainingSession` remains the source
of truth for measured biomechanics and the stored rehabilitation report.

### Existing TrainingSession

The current `sessions` table and `full_analysis.rehab_analysis` contract remain
unchanged. Clinical Pilot Mode references a saved rehabilitation session rather
than copying ROM, symmetry, repetition, or completion measurements into a
second table.

## Persistence

The local SQLite database remains the source of truth. The application creates
three additive tables:

```text
patient_profiles
rehab_episodes
clinical_visits
```

Initialization is idempotent and does not rewrite existing athlete or session
rows. Before the first schema update against an existing database, the
application creates one timestamped local backup beside the database.

Foreign keys connect:

```text
athletes.id
  -> patient_profiles.athlete_id
  -> rehab_episodes.patient_profile_id
  -> clinical_visits.rehab_episode_id

sessions.id
  -> clinical_visits.training_session_id
```

Patient profiles and episodes are archived instead of deleted in the first
release. No destructive delete action is exposed in the user interface.

## API

The backend exposes a clinical namespace while retaining the current athlete
and rehabilitation endpoints.

### Patient roster

```text
GET  /api/clinical/patients
POST /api/clinical/patients
GET  /api/clinical/patients/{patient_id}
PATCH /api/clinical/patients/{patient_id}
POST /api/clinical/patients/{patient_id}/archive
```

The patient detail response includes active episodes and a compact latest-visit
summary. It does not embed all historical session reports.

### Rehabilitation episodes

```text
POST  /api/clinical/patients/{patient_id}/episodes
PATCH /api/clinical/episodes/{episode_id}
POST  /api/clinical/episodes/{episode_id}/archive
GET   /api/clinical/episodes/{episode_id}
```

The episode detail response includes the visit timeline and normalized progress
observations for the episode protocol.

### Clinical visits

```text
POST  /api/clinical/episodes/{episode_id}/visits
GET   /api/clinical/visits/{visit_id}
PATCH /api/clinical/visits/{visit_id}
POST  /api/clinical/visits/{visit_id}/finalize
```

Creating a visit produces a draft before analysis starts. Finalization requires:

- a valid episode and patient
- a linked saved rehabilitation session
- a capture-quality decision
- a non-empty specialist review confirmation

An unacceptable capture can remain as a draft or be marked for repeat
measurement, but it cannot be finalized as the episode's current observation.

### Rehabilitation save integration

The live and upload save endpoints accept an athlete identifier in addition to
the existing name fallback. Clinical Pilot Mode always saves against the
selected patient's linked athlete and then attaches the returned session ID to
the draft visit.

Existing non-clinical callers remain compatible.

## User Experience

### Specialist workspace

The primary route is:

```text
/rehabilitation/clinical
```

It presents a compact local patient roster with:

- patient name and affected side
- active episode and protocol
- latest visit date
- latest bilateral ROM and symmetry when available
- a clear `New visit` action
- filters for active and archived patients

The workspace keeps the current premium dark SPRINT AI visual language and the
UK/EN locale behavior.

### Patient detail

The patient route is:

```text
/rehabilitation/clinical/patients/{patient_id}
```

It contains:

- profile and precautions
- active rehabilitation episode
- functional goal and professional-configured targets
- progress chart and baseline/current comparison
- visit timeline
- actions for a new visit, editing context, archiving, and opening a report

The current `/rehabilitation/progress` dashboard remains available as a compact
cross-athlete view. The patient detail reuses its comparison and chart logic
rather than implementing a separate interpretation of progress.

### New visit wizard

The new visit is a guided route:

```text
/rehabilitation/clinical/episodes/{episode_id}/visits/new
```

It has five explicit stages.

#### 1. Visit context

The specialist confirms:

- selected patient and episode
- current protocol
- short pre-session status note
- live camera or uploaded video

#### 2. Capture readiness

The application checks:

- camera permission when live capture is selected
- required body landmarks are visible
- target joint visibility
- camera orientation and level
- approximate subject distance and framing
- lighting and confidence warnings when available

The specialist can proceed only when readiness is acceptable or explicitly
acknowledge a warning. A hard failure such as missing target landmarks blocks
analysis.

#### 3. Analysis

The wizard embeds the existing `LiveRehabWorkspace` or `RehabUploader` rather
than building a second analysis implementation. The selected patient identity,
episode protocol, and draft visit remain fixed while analysis runs.

#### 4. Specialist review

After analysis, the specialist reviews:

- bilateral ROM
- symmetry
- repetitions and completion
- valid-frame or confidence information
- comparison with episode baseline and previous finalized visit
- capture-quality decision
- specialist observation

The application uses neutral wording. It never converts metric change into a
diagnosis or treatment recommendation.

#### 5. Save and handoff

Finalization:

1. saves or confirms the rehabilitation `TrainingSession`
2. links it to the clinical visit
3. records capture source and quality
4. records the specialist observation
5. marks the visit finalized
6. opens the visit summary

The summary provides actions to:

- open the bilingual clinical report
- export through the existing local print/PDF flow
- return to the patient record
- begin a repeat measurement

## Capture Quality

Capture quality is a first-class clinical context field, not a hidden technical
detail.

The normalized status is:

```text
acceptable
accepted_with_warning
repeat_required
```

Quality details may include:

- target landmarks intermittently missing
- insufficient valid frames
- unstable camera angle
- low inference confidence
- incomplete movement cycle
- specialist-entered note

Only `acceptable` and `accepted_with_warning` visits can be finalized.
`accepted_with_warning` requires explicit specialist acknowledgement and the
warning appears in the report.

Initial thresholds reuse existing report fields and camera readiness signals.
The product labels these thresholds as measurement quality checks, not medical
validity certification.

## Progress And Comparison

Episode progress uses finalized visits with a linked valid rehabilitation
session and the same episode protocol.

- episode baseline is the earliest finalized compatible visit
- previous is the finalized compatible visit immediately before the current
- current is the latest finalized compatible visit
- ROM deltas remain separate for left and right
- symmetry and completion deltas use percentage points
- capture warnings remain visible beside affected observations
- repeat-required drafts are excluded from the progress line

The patient detail and visit summary reuse the existing normalized progress
model wherever possible.

## Clinical Report

The existing `ClinicalReport` remains the rendering foundation. Clinical Pilot
Mode adds:

- patient display name
- rehabilitation episode and functional goal
- visit date and capture source
- comparison with baseline and previous visit
- capture quality and acknowledged warnings
- specialist observation
- existing research-prototype and method-limitation statements

The export remains local through browser print/PDF. No report data is sent to an
external document service.

## Localization

Every visible Clinical Pilot Mode string is provided in Ukrainian and English.
The existing rehabilitation locale storage and change event remain the single
locale mechanism.

Stored professional notes are not translated automatically. Measurements,
labels, generated neutral observations, validation messages, and reports follow
the active locale.

## Privacy And Safety

- all pilot data remains in local SQLite and local video storage
- clinical notes are never sent to an external AI service
- no analytics or telemetry containing patient context is added
- the UI avoids diagnosis, treatment, prognosis, and medical-device claims
- reports distinguish measurements, specialist observations, and generated
  neutral summaries
- archived records stay locally recoverable
- existing raw videos follow the current local retention behavior

The application must make local-only storage visible in the workspace and
reports.

## Future Expansion

The first release does not expose multi-user fields, but the boundaries support
them.

### Clinic scenario

Future clinic support adds:

- `clinic_id` ownership on patient profiles and episodes
- `therapist_id` attribution on clinical visits and observations
- authentication, role-based access, and audit history
- server-hosted database and encrypted object storage

### Patient home scenario

Future home support adds:

- professional-created assignments linked to an episode
- patient invitations and limited patient authentication
- capture instructions and device readiness
- asynchronous submission
- therapist review before a home measurement becomes finalized progress

Home submissions reuse `ClinicalVisit` as drafts with a patient capture source;
they do not bypass professional review.

## Error States

- no patients: show a guided patient creation action
- no active episode: show episode setup before allowing a visit
- camera denied: explain permission recovery and offer upload
- readiness blocked: identify the missing requirement
- analysis interrupted: preserve the draft visit and allow retry
- save failure: keep the reviewed result in memory and offer retry
- malformed historical report: omit its metrics and show the visit as
  unavailable rather than breaking the patient timeline
- database backup or initialization failure: stop clinical writes and display a
  clear local recovery message

## Testing

### Backend

- additive schema initialization and backup behavior
- patient, episode, and visit CRUD rules
- archive behavior
- one active episode per protocol
- visit finalization guards
- linking live and upload sessions by athlete ID
- progress excludes drafts and repeat-required captures
- malformed historical session handling

### Frontend

- patient roster states and filters
- patient creation and episode setup
- five-stage wizard navigation
- capture readiness gates and warning acknowledgement
- live and upload integration
- review and finalization guards
- baseline, previous, and current comparisons
- report fields and print action
- complete Ukrainian and English copy

### Integration And Browser QA

- create patient -> create episode -> complete live visit -> finalize -> view
  progress -> export report
- equivalent uploaded-video workflow
- camera denial and upload fallback
- poor-quality capture and repeat measurement
- interrupted save and retry
- desktop and tablet layouts
- no horizontal overflow or console errors
- existing `/rehabilitation`, `/rehabilitation/progress`, and non-clinical save
  flows remain operational

## Delivery Sequence

Clinical Pilot Mode is implemented before further Waterline-Aware swimming
analyzer work.

The implementation should be divided into reviewable vertical increments:

1. persistence and clinical API
2. specialist workspace and patient/episode setup
3. visit wizard and capture readiness
4. analysis save integration and visit finalization
5. progress/report integration and full browser QA

Each increment must preserve the existing local database and keep the current
rehabilitation demo usable.
