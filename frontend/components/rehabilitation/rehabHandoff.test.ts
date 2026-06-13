import { describe, expect, it } from "vitest";

import type { RehabReport } from "@/lib/rehabilitation";

import {
  createDemoHandoff,
  createReportHandoff,
} from "./rehabHandoff";

const report: RehabReport = {
  protocol: "shoulder_flexion",
  total_correct_reps: 3,
  completion_score: 84,
  target_metrics: {
    left: { rom: 150 },
    right: { rom: 132 },
  },
  symmetry: {
    asymmetry_index: 12,
    score: 88,
  },
};

describe("rehab handoff adapters", () => {
  it("creates a complete localized handoff for the deterministic demo", () => {
    expect(
      createDemoHandoff("uk", "shoulder_flexion", "2026-06-13T10:00:00.000Z"),
    ).toMatchObject({
      source: "demo",
      locale: "uk",
      protocol: "shoulder_flexion",
      recordedAt: "2026-06-13T10:00:00.000Z",
      leftRom: 154,
      rightRom: 136,
      symmetry: 88,
      repetitions: 2,
      completionScore: 86,
      confidence: 96,
      poseCoverage: 100,
      evidenceTimestamp: "0:03.2",
      findingTitle: "Праве плече не досягає амплітуди лівого",
    });
  });

  it("does not invent unavailable quality or evidence values", () => {
    expect(
      createReportHandoff({
        source: "upload",
        locale: "en",
        protocol: "shoulder_flexion",
        report,
        recordedAt: "2026-06-13T10:00:00.000Z",
      }),
    ).toMatchObject({
      source: "upload",
      leftRom: 150,
      rightRom: 132,
      symmetry: 88,
      repetitions: 3,
      completionScore: 84,
      confidence: null,
      poseCoverage: null,
      evidenceTimestamp: null,
    });
  });

  it("normalizes reported quality percentages and keeps localized findings", () => {
    const handoff = createReportHandoff({
      source: "live",
      locale: "uk",
      protocol: "shoulder_flexion",
      report,
      confidence: 0.92,
      poseCoverage: 104,
      recordedAt: "2026-06-13T10:00:00.000Z",
    });

    expect(handoff.confidence).toBe(92);
    expect(handoff.poseCoverage).toBe(100);
    expect(handoff.findingBody).toContain("18°");
  });

  it("preserves persisted clinical context for the handoff report", () => {
    const handoff = createReportHandoff({
      source: "live",
      locale: "en",
      protocol: "shoulder_flexion",
      report,
      clinical: {
        patientName: "Patient A",
        episodeTitle: "Shoulder recovery",
        functionalGoal: "Reach an overhead shelf",
        captureSource: "live",
        captureQuality: "accepted_with_warning",
        qualityDetails: "Camera tilt acknowledged.",
        specialistObservation: "ROM remains asymmetric.",
        baselineDelta: {
          leftRom: 18,
          rightRom: 18,
          symmetry: 6,
          repetitions: 1,
          completionScore: 12,
        },
        previousDelta: null,
      },
    });

    expect(handoff.clinical).toMatchObject({
      patientName: "Patient A",
      functionalGoal: "Reach an overhead shelf",
      captureQuality: "accepted_with_warning",
      specialistObservation: "ROM remains asymmetric.",
    });
  });
});
