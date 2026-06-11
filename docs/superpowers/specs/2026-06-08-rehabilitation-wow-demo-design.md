# Rehabilitation Wow Demo Design

## Goal

Turn the rehabilitation page into a dark, premium clinical-lab experience that
demonstrates the product's value before asking for camera access. The page must
feel credible to rehabilitation professionals, visually distinctive, and ready
for a live colleague presentation.

## Visual Direction

The product remains dark-only. The base is ink black and graphite rather than
medical white. Cyan identifies measurement and geometry, emerald identifies
reliable or improved results, amber is reserved for limitations, and rose is
reserved for errors.

The interface uses fewer, larger surfaces:

- one cinematic analysis stage
- one clinical insight rail
- one evidence-led conclusion
- one before/after comparison

The design avoids cyberpunk decoration, excessive glass effects, dense card
grids, fake diagnostics, and bright marketing sections. Light appears only as a
controlled edge glow around the analyzed body and high-confidence data.

## First Viewport

The first viewport leads with the usable product rather than a warning.

The left side contains:

- headline: movement analysis with evidence in real time
- concise product explanation
- `Запустити демо` / `Run demo` as the primary action
- `Камера наживо` / `Live camera` as the secondary action
- protocol and language controls

The right side contains a live-looking clinical analysis stage. Before camera
access, it shows a deterministic demo session with:

- an anatomical human silhouette
- a posture skeleton
- shoulder angle arcs
- a calibrated vertical axis
- subtle movement trails
- current repetition phase
- a compact timeline

The demo is explicitly labeled as a simulated product walkthrough. It does not
pretend to be patient data.

## Demo Narrative

Pressing the primary action starts a short deterministic animation:

1. The arm raises through a shoulder-flexion repetition.
2. Left and right ROM values increase.
3. The repetition counter advances.
4. The timeline moves through the captured cycle.
5. The main conclusion appears with an exact evidence timestamp.
6. A before/after comparison shows improved configured-target completion.

The narrative communicates one result rather than many generic metrics:

- observed limitation
- numerical evidence
- likely compensatory pattern
- suggested professional review

No diagnostic language is used.

## Clinical Insight Rail

The rail contains high-signal values:

- left ROM
- right ROM
- bilateral symmetry
- completed repetitions
- pose visibility
- capture confidence

Values use tabular numerals and strong hierarchy. Confidence is visible so the
system never presents uncertain measurements as equally reliable.

## Safety Positioning

The research-prototype disclosure remains visible but becomes a compact,
expandable line below the main interaction. Target angles are described as
professional-configured exercise targets rather than clinical norms.

## Live And Upload Modes

The existing camera and upload workflows remain functional.

- Demo is the default presentation mode.
- Live camera is one deliberate click away.
- Upload remains available from the mode control.
- Switching language updates the complete rehabilitation experience.

The live layout is restructured on narrow screens so controls, instructions,
metrics, and camera calibration remain in normal document flow. No metric card
may cover the camera CTA or instructional copy.

## Responsive Behavior

Desktop uses an asymmetric two-column command-center layout. Tablet collapses
the insight rail below the analysis canvas. Mobile uses a single column with:

1. headline and actions
2. analysis canvas
3. high-signal metric grid
4. primary conclusion
5. before/after comparison
6. compact disclosure

The top navigation keeps horizontal scrolling but uses localized labels on the
rehabilitation route.

## Testing

Frontend tests cover:

- demo-first default state
- starting and replaying the demo
- deterministic metric and conclusion progression
- Ukrainian and English copy
- mode switching without requesting camera access
- compact disclosure behavior
- mobile-safe live structure

Browser QA covers desktop and 390px mobile viewports, both languages, demo
animation, live standby, upload mode, console errors, and layout overlap.
