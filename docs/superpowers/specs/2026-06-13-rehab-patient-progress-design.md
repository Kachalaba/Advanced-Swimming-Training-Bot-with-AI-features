# Rehabilitation Patient Progress Design

## Goal

Turn persisted rehabilitation analyses into a longitudinal clinical discussion
surface. A rehabilitation specialist should be able to select an athlete and
protocol, see change over time, compare baseline with the latest compatible
session, and export the existing single-session handoff when deeper review is
needed.

This remains a research prototype. It observes movement-analysis measurements
and does not diagnose, prescribe treatment, or claim clinical improvement.

## Product Shape

Patient Progress lives at `/rehabilitation/progress` as a dedicated dark
dashboard in the existing SPRINT AI visual system. The rehabilitation page
links to it from the handoff rail, and the dashboard links back to the capture
workflow.

The first version is intentionally read-only. It consumes real sessions already
stored in SQLite and does not add patient records, treatment plans, diagnoses,
or cloud synchronization.

## Data Source And API

The active `AthleteDatabase` remains the source of truth. A new endpoint:

```text
GET /api/athletes/{athlete_id}/rehabilitation/progress
```

returns normalized rehabilitation observations derived from each session's
`full_analysis.rehab_analysis` payload.

```ts
type RehabProgressSession = {
  id: number;
  date: string;
  protocol: RehabProtocol;
  leftRom: number;
  rightRom: number;
  symmetry: number;
  repetitions: number;
  completionScore: number;
  validFrames: number | null;
  hasVideo: boolean;
};

type RehabProgressResponse = {
  athlete: Athlete;
  sessions: RehabProgressSession[];
  protocols: RehabProtocol[];
};
```

Malformed JSON, non-rehabilitation sessions, unknown protocols, or reports
without finite bilateral ROM values are skipped. Missing optional values remain
`null`; the API never invents measurements.

The endpoint returns observations in ascending chronological order so the
frontend does not need to reinterpret chronology. An unknown athlete returns
`404`.

## Comparison Rules

The dashboard compares only sessions with the selected protocol:

- baseline is the earliest compatible session
- current is the latest compatible session
- at least two sessions are required for a change statement
- ROM deltas are displayed separately for left and right
- symmetry and completion deltas are displayed in percentage points
- zero or negative deltas are described neutrally, not as treatment failure
- one session renders a baseline-only state

The client derives the comparison from the normalized response. This keeps the
API stable and makes the clinical wording locale-specific.

## Dashboard

The dashboard contains:

1. **Clinical header**
   - athlete selector
   - UK/EN language selector
   - link back to the rehabilitation capture page
   - research-prototype and local-data badges

2. **Protocol navigation**
   - only protocols present in persisted data
   - session count per protocol
   - latest session date

3. **Baseline versus current**
   - bilateral ROM
   - symmetry
   - target completion
   - repetitions
   - compact delta indicators

4. **Progress chart**
   - SVG chart with left ROM, right ROM, and symmetry series
   - responsive layout without a chart dependency
   - visible legend, point values, and accessible text summary

5. **Clinical observation**
   - neutral generated summary based on numeric deltas
   - measurement limitations
   - no recommendation or diagnosis

6. **Session timeline**
   - chronological session cards
   - protocol, date, ROM, symmetry, completion, and quality metadata
   - clear empty and single-session states

## Localization

All dashboard strings live in a dedicated symmetric UK/EN copy object. Protocol
labels reuse the existing rehabilitation copy. Dates use the active locale.
Changing language updates the dashboard without altering stored data.

## Navigation

The rehabilitation page adds a `Patient progress` action beside the handoff
actions. It opens `/rehabilitation/progress`. The progress page provides a
`New analysis` action back to `/rehabilitation`.

No top-level navigation item is added in this iteration; the feature stays
inside the rehabilitation workflow.

## Privacy And Safety

- data stays in the existing local SQLite database
- the dashboard does not request diagnosis, contact details, or date of birth
- athlete names already stored by the application may be displayed
- no analytics or external requests are added
- all generated copy states that measurements support professional discussion
- quality metadata is shown only when present

## Error And Empty States

- no athletes: explain that a saved rehabilitation analysis is required
- selected athlete has no rehab sessions: show a capture call to action
- one compatible session: show baseline data and request another measurement
- malformed stored reports: omit them without breaking the response
- backend failure: show a retry action while preserving the page shell

## Testing

Backend tests cover:

- normalized extraction from valid stored reports
- chronological ordering
- malformed and incomplete report filtering
- unknown athlete `404`

Frontend tests cover:

- comparison math and neutral observation copy
- loading, error, empty, and single-session states
- athlete and protocol selection
- UK/EN switching
- link from rehabilitation to progress

Browser QA covers desktop and 390px mobile layouts, locale switching, protocol
switching, empty states, no horizontal overflow, and no console errors.

## Delivery

Work is isolated on `codex/rehab-patient-progress`. The existing uncommitted
`data/athletes.db` and `.superpowers/` paths remain untouched and untracked by
the feature commit.
