import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RehabReport } from "@/lib/rehabilitation";

import { RehabPresentationMode } from "./RehabPresentationMode";
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

describe("RehabPresentationMode", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });
    Object.defineProperty(document, "exitFullscreen", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });
  });

  it("renders a CSS fullscreen fallback and exits with Escape", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <RehabPresentationMode
        handoff={createReportHandoff({
          source: "live",
          locale: "uk",
          protocol: "shoulder_flexion",
          report,
        })}
        onClose={onClose}
        onOpenReport={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("dialog", { name: "Режим презентації" }),
    ).toHaveClass("fixed", "inset-0");
    expect(screen.getByText("Сесія наживо")).toBeInTheDocument();
    expect(screen.getByText("150°")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("requests native fullscreen and exposes demo replay and report actions", async () => {
    const user = userEvent.setup();
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });
    const onReplayDemo = vi.fn();
    const onOpenReport = vi.fn();
    const onClose = vi.fn();

    render(
      <RehabPresentationMode
        handoff={createDemoHandoff("en", "shoulder_flexion")}
        onClose={onClose}
        onOpenReport={onOpenReport}
        onReplayDemo={onReplayDemo}
      />,
    );

    expect(requestFullscreen).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "Replay demo" }));
    expect(onReplayDemo).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "Open clinical report" }));
    expect(onOpenReport).toHaveBeenCalledOnce();

    await user.click(
      screen.getByRole("button", { name: "Close presentation" }),
    );
    expect(onClose).toHaveBeenCalledOnce();
  });
});
