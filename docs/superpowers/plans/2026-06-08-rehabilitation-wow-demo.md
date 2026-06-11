# Rehabilitation Wow Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a dark-only, demo-first Premium Clinical Lab rehabilitation experience with an animated evidence narrative and responsive live workflow.

**Architecture:** Add a deterministic client-side demo model and focused presentation components, then compose them from the existing rehabilitation page. Preserve the existing backend, live session, upload transport, protocol contracts, and localized copy while restructuring only the presentation layer.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS, Lucide icons, Vitest, Testing Library

---

### Task 1: Demo Contract And Tests

**Files:**
- Create: `frontend/components/rehabilitation/demoSession.ts`
- Create: `frontend/components/rehabilitation/RehabDemoStage.test.tsx`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] Define typed deterministic demo frames for shoulder flexion, ROM,
  symmetry, confidence, repetitions, evidence timestamp, conclusion, and
  before/after values.
- [ ] Write failing tests for the initial, running, completed, and replay states.
- [ ] Run `npm test -- components/rehabilitation/RehabDemoStage.test.tsx` and
  verify the expected missing-component failure.
- [ ] Implement the smallest demo state API required by the tests.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Premium Analysis Stage

**Files:**
- Create: `frontend/components/rehabilitation/RehabDemoStage.tsx`
- Create: `frontend/components/rehabilitation/RehabBodyFigure.tsx`
- Create: `frontend/components/rehabilitation/RomArc.tsx`
- Modify: `frontend/app/globals.css`

- [ ] Render a dark cinematic analysis canvas with a code-native anatomical
  silhouette, posture skeleton, ROM arcs, vertical calibration axis, movement
  trails, and timeline.
- [ ] Add purposeful animation driven by the deterministic demo frames and
  respect `prefers-reduced-motion`.
- [ ] Render current bilateral measurements and evidence time without
  diagnostic claims.
- [ ] Verify the focused component tests pass.

### Task 3: Insight Rail And Evidence Story

**Files:**
- Create: `frontend/components/rehabilitation/RehabInsightRail.tsx`
- Create: `frontend/components/rehabilitation/RehabEvidencePanel.tsx`
- Create: `frontend/components/rehabilitation/RehabComparison.tsx`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] Add high-signal ROM, symmetry, repetition, pose, and confidence values.
- [ ] Add one key conclusion linked to a demo timestamp and supporting metric.
- [ ] Add a before/after configured-target comparison.
- [ ] Add Ukrainian and English copy for every visible state.
- [ ] Test that switching locale changes the complete evidence narrative.

### Task 4: Demo-First Page Composition

**Files:**
- Modify: `frontend/app/rehabilitation/page.tsx`
- Modify: `frontend/app/rehabilitation/page.test.tsx`

- [ ] Make demo mode the default and place the product stage in the first
  viewport.
- [ ] Add primary `Run demo` and secondary `Live camera` actions.
- [ ] Move the prototype disclosure below the interaction and make details
  expandable.
- [ ] Preserve protocol, language, live, and upload controls.
- [ ] Verify page tests cover default demo, locale switching, live transition,
  upload transition, and disclosure.

### Task 5: Responsive Live Workspace

**Files:**
- Modify: `frontend/components/rehabilitation/LiveRehabWorkspace.tsx`
- Modify: `frontend/components/rehabilitation/LiveMetricRail.tsx`
- Modify: `frontend/components/rehabilitation/LiveRehabWorkspace.test.tsx`

- [ ] Replace mobile absolute metric positioning with a responsive normal-flow
  panel below the camera stage.
- [ ] Keep desktop overlays compact and readable.
- [ ] Ensure disabled controls do not obscure instructions.
- [ ] Preserve camera, save, calibration, fullscreen, and stop behavior.
- [ ] Run focused live-workspace tests.

### Task 6: Localized Rehabilitation Shell

**Files:**
- Modify: `frontend/components/layout/TopNav.tsx`
- Modify: `frontend/lib/sections.ts`
- Create: `frontend/components/layout/TopNav.test.tsx`

- [ ] Add route-aware Ukrainian and English navigation labels for the
  rehabilitation page using the saved rehab locale.
- [ ] Keep all other routes and section IDs unchanged.
- [ ] Test Ukrainian labels, English labels, and active navigation state.

### Task 7: Verification And Delivery

**Files:**
- Modify only if QA finds defects.

- [ ] Run `npm test`.
- [ ] Run `npm run typecheck`.
- [ ] Run `npm run build`.
- [ ] Run `git diff --check`.
- [ ] Rebuild the frontend Docker container.
- [ ] Verify desktop and 390px mobile layouts in the in-app browser.
- [ ] Verify demo start/replay, UA/EN, live standby, upload mode, and no console
  errors.
- [ ] Compare the final browser screenshots against the Premium Clinical Lab
  design specification and repair every material mismatch.
- [ ] Commit and push `codex/rehab-wow-demo`, then open a pull request into
  `main`.
