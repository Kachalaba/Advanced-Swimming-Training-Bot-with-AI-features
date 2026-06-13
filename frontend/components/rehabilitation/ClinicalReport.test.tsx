import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RehabReport } from "@/lib/rehabilitation";

import { ClinicalReport } from "./ClinicalReport";
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
  symmetry: { asymmetry_index: 12, score: 88 },
};

describe("ClinicalReport", () => {
  beforeEach(() => {
    Object.defineProperty(window, "print", {
      configurable: true,
      value: vi.fn(),
    });
  });

  it("renders and prints a Ukrainian simulated report with local context", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <ClinicalReport
        handoff={createDemoHandoff(
          "uk",
          "shoulder_flexion",
          "2026-06-13T10:00:00.000Z",
        )}
        onClose={onClose}
      />,
    );

    expect(
      screen.getByRole("dialog", { name: "Клінічний звіт про рух" }),
    ).toBeInTheDocument();
    expect(screen.getByText("СИМУЛЬОВАНЕ ДЕМО")).toBeInTheDocument();

    await user.type(
      screen.getByLabelText("Код пацієнта або сесії"),
      "CASE-024",
    );
    await user.type(
      screen.getByLabelText("Нотатка фахівця"),
      "Повторити оцінку через два тижні.",
    );

    expect(screen.getByDisplayValue("CASE-024")).toBeInTheDocument();
    expect(
      screen.getByDisplayValue("Повторити оцінку через два тижні."),
    ).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Експортувати PDF" }),
    );
    expect(window.print).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "Закрити звіт" }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("renders unavailable real-session values without a demo watermark", () => {
    render(
      <ClinicalReport
        handoff={createReportHandoff({
          source: "upload",
          locale: "en",
          protocol: "shoulder_flexion",
          report,
          recordedAt: "2026-06-13T10:00:00.000Z",
        })}
        onClose={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("dialog", { name: "Clinical movement report" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("SIMULATED DEMO")).not.toBeInTheDocument();
    expect(screen.getAllByText("Not available")).toHaveLength(3);
    expect(screen.getByText("Uploaded session")).toBeInTheDocument();
  });

  it("renders persisted clinical identity, progress, quality, and observation", () => {
    render(
      <ClinicalReport
        handoff={createReportHandoff({
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
        })}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText("Patient A")).toBeInTheDocument();
    expect(screen.getByText("Reach an overhead shelf")).toBeInTheDocument();
    expect(screen.getByText(/Baseline.*\+18°/i)).toBeInTheDocument();
    expect(screen.getByText("Accepted with warning")).toBeInTheDocument();
    expect(screen.getByText("ROM remains asymmetric.")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Patient or session code"),
    ).not.toBeInTheDocument();
  });
});
