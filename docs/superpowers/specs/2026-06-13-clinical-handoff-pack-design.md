# Clinical Handoff Pack Design

## Goal

Turn a completed rehabilitation analysis into a presentation-ready and
clinician-shareable handoff without transmitting patient data to an external
service. The pack must work for the deterministic demo and for real live or
uploaded analyses while preserving the existing dark Premium Clinical Lab
visual system.

## Scope

The Clinical Handoff Pack contains:

- a focused presentation mode
- a bilingual clinical report
- local PDF export through the browser print dialog
- optional session context
- one normalized report contract for demo, live, and upload results

It does not add diagnosis, cloud document generation, medical-device claims,
longitudinal history comparison, or new computer-vision metrics.

## Entry Points

The rehabilitation page adds two secondary actions:

- `Presentation mode` opens the current completed analysis as a focused stage.
- `Clinical report` opens a printable report preview.

Both actions remain available for the completed simulated demo. For live and
upload modes, they become enabled after a valid `RehabReport` exists. The UI
must explain the disabled state rather than silently doing nothing.

## Normalized Handoff Contract

Presentation and report components consume one view model rather than the raw
transport payload:

```ts
type RehabHandoff = {
  source: "demo" | "live" | "upload";
  locale: "uk" | "en";
  protocol: RehabProtocol;
  recordedAt: string;
  leftRom: number;
  rightRom: number;
  symmetry: number;
  repetitions: number;
  completionScore: number;
  confidence: number | null;
  poseCoverage: number | null;
  evidenceTimestamp: string | null;
  findingTitle: string;
  findingBody: string;
  disclaimer: string;
};
```

Small adapter functions map deterministic demo data and real `RehabReport`
payloads into this contract. Clinical copy remains observational. The adapters
must not invent confidence, pose coverage, evidence timestamps, or findings
when the source does not provide them.

## Session Context

The report preview includes two optional fields:

- patient/session code
- clinician note

Both fields remain in page memory only. They are included in the printed
document but are not sent to the backend or stored in local storage. The UI
labels the patient code as an identifier intended for anonymized use and does
not request a full name, diagnosis, contact information, or date of birth.

## Presentation Mode

Presentation mode is a dark full-viewport overlay designed for screen sharing
or an in-person demonstration.

It contains:

- SPRINT AI identity and protocol
- source label: simulated demo, live session, or uploaded session
- the analysis stage or an evidence summary
- bilateral ROM, symmetry, repetitions, completion, and available quality data
- one primary finding
- a compact non-diagnostic disclaimer
- controls to replay the demo when the source is simulated
- controls to open the clinical report and exit

The regular top navigation, athlete menus, upload controls, and page footer are
hidden while presentation mode is active.

The browser Fullscreen API is attempted when available. A fixed CSS overlay is
the guaranteed fallback, and `Escape` exits presentation mode in both cases.
Reduced-motion preferences disable non-essential transitions.

## Clinical Report

The clinical report is an on-screen preview with a print-specific A4 layout.
The normal application remains dark, while the printed document uses a
high-contrast white paper surface because clinical handoff readability and
printer compatibility take priority over the application theme.

The report contains:

1. SPRINT AI report header and generation timestamp
2. protocol and source
3. optional patient/session code and clinician note
4. bilateral ROM
5. symmetry, repetitions, target completion, and available quality measures
6. the primary observational finding
7. the evidence timestamp when available
8. measurement limitations and research-prototype disclaimer
9. a footer identifying the report as support for professional discussion

The demo report is visibly watermarked `SIMULATED DEMO` / `СИМУЛЬОВАНЕ ДЕМО`.
Real sessions never receive that watermark.

## PDF Export

`Export PDF` invokes `window.print()` from the report preview. Print CSS:

- hides the app shell, controls, backdrop, and non-report UI
- renders only the report on an A4-compatible white page
- preserves borders and essential semantic colors
- avoids clipped sections and awkward page breaks
- uses tabular numerals for measurements
- includes all limitations in the exported document

No PDF library or external document API is added. This keeps the bundle small,
works offline, and prevents report data from leaving the browser.

## Live And Upload Integration

`LiveRehabWorkspace` and `RehabUploader` expose completed reports through
callbacks owned by the rehabilitation page:

- live emits the latest valid report plus pose/confidence information available
  in the current update
- upload emits the completed report plus frame coverage
- switching protocol or starting a new analysis clears stale handoff data

Saving a session remains a separate existing action. Opening or exporting a
report must not automatically save or mutate backend state.

## Error And Empty States

- Presentation/report actions are disabled until a real analysis has results.
- Missing optional quality fields render as `Not available`, never as zero.
- Print support failures show an inline message and leave the preview open.
- Closing presentation or report mode returns the user to the previous input
  mode and current analysis state.
- Browser permission failures for native fullscreen fall back silently to the
  CSS overlay.

## Localization

Every visible string is added symmetrically to Ukrainian and English
`rehabCopy`. The selected rehabilitation locale controls:

- action labels
- presentation mode
- report preview and print output
- field placeholders
- source labels
- watermark
- limitations and empty states

## Accessibility And Privacy

- overlays use dialog semantics and accessible names
- focus moves into an opened overlay and returns to the invoking control
- `Escape` closes the topmost overlay
- all icon buttons have localized accessible names
- print and fullscreen actions are keyboard reachable
- optional context is explicitly local-only and is cleared on page reload
- no personal data is logged, persisted, or transmitted

## Testing

Unit and component tests cover:

- demo, live, and upload adapters
- no invented values for unavailable real-session fields
- presentation open, close, and `Escape`
- CSS fullscreen fallback
- report preview in Ukrainian and English
- simulated watermark only for demo
- optional context inclusion
- `window.print()` invocation
- report/live/upload availability gates
- stale result clearing after protocol changes

Browser QA covers:

- desktop presentation mode and exit
- report preview and browser print invocation
- Ukrainian and English output
- 390px mobile report preview
- live/upload disabled and completed states
- no console errors or horizontal overflow

## Delivery

Implementation uses a dedicated `codex/clinical-handoff-pack` branch. The
existing uncommitted `data/athletes.db` and `.superpowers/` paths remain
untouched. Verification follows:

```bash
npm test
npm run typecheck
npm run build
git diff --check
```

The frontend Docker container is rebuilt for browser QA, then the branch is
pushed and opened as a pull request into `main`.
