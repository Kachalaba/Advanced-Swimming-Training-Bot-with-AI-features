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
});
