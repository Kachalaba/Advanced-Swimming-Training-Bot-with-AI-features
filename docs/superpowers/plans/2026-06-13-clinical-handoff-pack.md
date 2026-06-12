# Clinical Handoff Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bilingual presentation mode and locally printable clinical report for completed demo, live, and uploaded rehabilitation analyses.

**Architecture:** Introduce a normalized `RehabHandoff` view model and pure adapters, then build focused presentation/report overlays around it. Existing live and upload components emit completed analysis snapshots to the page; the page owns overlay state and optional local-only session context.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS, Lucide icons, Vitest, Testing Library, browser print API

---

### Task 1: Normalized Handoff Contract

**Files:**
- Create: `frontend/components/rehabilitation/rehabHandoff.ts`
- Create: `frontend/components/rehabilitation/rehabHandoff.test.ts`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] **Step 1: Write failing adapter tests**

Test that `createDemoHandoff()` maps the final deterministic frame, and
`createReportHandoff()` maps live/upload reports without inventing unavailable
quality values.

```ts
expect(createDemoHandoff("uk", "shoulder_flexion")).toMatchObject({
  source: "demo",
  leftRom: 154,
  rightRom: 136,
  confidence: 96,
  evidenceTimestamp: "0:03.2",
});

expect(
  createReportHandoff({
    source: "upload",
    locale: "en",
    protocol: "shoulder_flexion",
    report,
  }),
).toMatchObject({
  confidence: null,
  poseCoverage: null,
  evidenceTimestamp: null,
});
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
npm test -- components/rehabilitation/rehabHandoff.test.ts
```

Expected: fail because `rehabHandoff.ts` does not exist.

- [ ] **Step 3: Implement the view model and pure adapters**

Define:

```ts
export type RehabHandoff = {
  source: "demo" | "live" | "upload";
  locale: RehabLocale;
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

export type RehabAnalysisSnapshot = {
  report: RehabReport;
  confidence?: number | null;
  poseCoverage?: number | null;
};
```

Use localized observational findings. Normalize percentage-like inputs to
`0..100`, but preserve `null` when a source did not report the value.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
npm test -- components/rehabilitation/rehabHandoff.test.ts
```

Expected: all adapter tests pass.

### Task 2: Printable Clinical Report

**Files:**
- Create: `frontend/components/rehabilitation/ClinicalReport.tsx`
- Create: `frontend/components/rehabilitation/ClinicalReport.test.tsx`
- Modify: `frontend/app/globals.css`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] **Step 1: Write failing report tests**

Cover Ukrainian/English headings, demo-only watermark, optional patient code and
note, unavailable values, close behavior, and `window.print()`.

```ts
await user.click(screen.getByRole("button", { name: "Експортувати PDF" }));
expect(window.print).toHaveBeenCalledOnce();
expect(screen.getByText("СИМУЛЬОВАНЕ ДЕМО")).toBeInTheDocument();
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
npm test -- components/rehabilitation/ClinicalReport.test.tsx
```

Expected: fail because the report component does not exist.

- [ ] **Step 3: Implement report preview**

Create a dialog with:

- local-only session code input
- clinician note textarea
- A4 report surface
- measurement summary
- observational finding
- limitations
- demo watermark
- close and export actions

Use `window.print()` only from the explicit export button.

- [ ] **Step 4: Add print CSS**

Add `@media print` rules that hide every body child except
`[data-print-root]`, render an A4 white document, preserve semantic borders, and
avoid page breaks inside report sections.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
npm test -- components/rehabilitation/ClinicalReport.test.tsx
```

Expected: all report tests pass.

### Task 3: Presentation Mode

**Files:**
- Create: `frontend/components/rehabilitation/RehabPresentationMode.tsx`
- Create: `frontend/components/rehabilitation/RehabPresentationMode.test.tsx`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] **Step 1: Write failing presentation tests**

Cover dialog semantics, source/protocol/metrics, demo replay, report action,
Fullscreen API attempt, CSS fallback, close action, and `Escape`.

```ts
await user.click(screen.getByRole("button", { name: "Закрити презентацію" }));
expect(onClose).toHaveBeenCalledOnce();
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
npm test -- components/rehabilitation/RehabPresentationMode.test.tsx
```

Expected: fail because the presentation component does not exist.

- [ ] **Step 3: Implement the focused overlay**

Render a fixed dark overlay with SPRINT AI identity, source/protocol, a
demo-stage visualization for simulated data or evidence summary for real data,
metrics, finding, disclaimer, replay/report actions, and exit.

Attempt `requestFullscreen()` on mount when available. Keep the fixed overlay as
the fallback and exit native fullscreen when closing.

- [ ] **Step 4: Implement focus and keyboard behavior**

Move focus to the close button when opened, close on `Escape`, and return focus
through the page-owned invoking button after unmount.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
npm test -- components/rehabilitation/RehabPresentationMode.test.tsx
```

Expected: all presentation tests pass.

### Task 4: Live And Upload Result Callbacks

**Files:**
- Modify: `frontend/components/rehabilitation/LiveRehabWorkspace.tsx`
- Modify: `frontend/components/rehabilitation/LiveRehabWorkspace.test.tsx`
- Modify: `frontend/components/rehabilitation/RehabUploader.tsx`
- Create: `frontend/components/rehabilitation/RehabUploader.test.tsx`

- [ ] **Step 1: Write failing callback tests**

Verify live emits the latest valid `report` with camera confidence, upload emits
the completed report with pose coverage, and both clear stale data when a new
analysis starts or the component unmounts.

```ts
expect(onAnalysisChange).toHaveBeenCalledWith({
  report,
  confidence: 92,
  poseCoverage: null,
});
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
npm test -- components/rehabilitation/LiveRehabWorkspace.test.tsx components/rehabilitation/RehabUploader.test.tsx
```

Expected: new callback assertions fail.

- [ ] **Step 3: Add optional callbacks**

Add:

```ts
onAnalysisChange?: (snapshot: RehabAnalysisSnapshot | null) => void;
```

Live maps `camera_level.confidence` to a percentage and keeps pose coverage
`null`. Upload calculates pose coverage from `frames_with_pose / frames_total`
and keeps confidence `null`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused command and confirm all tests pass.

### Task 5: Page Integration And Availability Gates

**Files:**
- Modify: `frontend/app/rehabilitation/page.tsx`
- Modify: `frontend/app/rehabilitation/page.test.tsx`
- Modify: `frontend/lib/rehabCopy.ts`

- [ ] **Step 1: Write failing page-flow tests**

Cover:

- demo report and presentation actions enabled immediately
- live/upload actions disabled before results
- live/upload callbacks enable actions
- protocol change clears real handoff data
- opening/closing report and presentation
- locale switch updates overlays

- [ ] **Step 2: Run page tests and verify RED**

```bash
npm test -- app/rehabilitation/page.test.tsx
```

Expected: fail because handoff actions and overlays are absent.

- [ ] **Step 3: Add page-owned handoff state**

Keep separate snapshots by active real mode, derive the current
`RehabHandoff`, and clear real snapshots on protocol changes.

- [ ] **Step 4: Add compact action rail**

Add `Presentation mode` and `Clinical report` buttons in the demo evidence area
and the real-mode header. Disabled buttons include localized explanatory text
or accessible descriptions.

- [ ] **Step 5: Mount the overlays**

Render `RehabPresentationMode` and `ClinicalReport` from page state. Report
opened from presentation mode must preserve the handoff and close only the
presentation layer needed to avoid stacked dialogs.

- [ ] **Step 6: Run page tests and verify GREEN**

```bash
npm test -- app/rehabilitation/page.test.tsx
```

Expected: all page-flow tests pass.

### Task 6: Full Verification And Browser QA

**Files:**
- Modify only if QA identifies defects.

- [ ] **Step 1: Run the complete frontend suite**

```bash
npm test
npm run typecheck
npm run build
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Rebuild and restart frontend**

```bash
docker compose build frontend
docker compose up -d --no-deps frontend
```

- [ ] **Step 3: Verify desktop workflow**

At `http://127.0.0.1:3000/rehabilitation`:

- open presentation mode
- replay demo
- open report
- enter local-only context
- invoke print
- switch UK/EN
- close with buttons and `Escape`

- [ ] **Step 4: Verify mobile workflow**

At 390x844 confirm no horizontal overflow, the report preview is readable, and
the presentation controls remain reachable.

- [ ] **Step 5: Verify live/upload gates**

Confirm actions remain disabled before a real result and that demo remains
fully usable without camera or upload.

- [ ] **Step 6: Inspect screenshots and console**

Use `view_image` on latest desktop/mobile screenshots, confirm no material
visual regressions, and verify browser console has no errors.

### Task 7: Delivery

**Files:**
- Stage only Clinical Handoff Pack files.

- [ ] **Step 1: Commit implementation**

```bash
git add <explicit handoff files>
git commit -m "feat: add clinical handoff pack"
```

- [ ] **Step 2: Push branch**

```bash
git push -u origin codex/clinical-handoff-pack
```

- [ ] **Step 3: Open draft pull request**

Create a draft PR into `main` with summary and validation commands. Do not stage
or modify `data/athletes.db` or `.superpowers/`.
