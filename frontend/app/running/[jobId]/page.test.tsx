import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import RunningResultPage from "./page";

const mocks = vi.hoisted(() => ({
  listAthletes: vi.fn(),
  saveRunningAnalysis: vi.fn(),
  subscribeAnalysis: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ jobId: "run-123" }),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    api: { ...actual.api, listAthletes: mocks.listAthletes },
  };
});

vi.mock("@/lib/analysis", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/analysis")>();
  return {
    ...actual,
    saveRunningAnalysis: mocks.saveRunningAnalysis,
    subscribeAnalysis: mocks.subscribeAnalysis,
  };
});

describe("RunningResultPage", () => {
  beforeEach(() => {
    mocks.listAthletes.mockResolvedValue([
      { id: "7", name: "Runner A", initials: "RA", handle: null },
      { id: "8", name: "Runner B", initials: "RB", handle: null },
    ]);
    mocks.saveRunningAnalysis.mockResolvedValue({ sessionId: 91 });
    mocks.subscribeAnalysis.mockImplementation((_jobId, onEvent) => {
      onEvent({
        type: "result",
        analysis: {
          cadence: 176,
          foot_strike_type: "midfoot",
          arm_symmetry: 92,
        },
        frames_total: 100,
        frames_with_pose: 90,
        video_path: "annotated.mp4",
      });
      return vi.fn();
    });
  });

  it("selects an athlete and saves the completed analysis", async () => {
    const user = userEvent.setup();
    render(<RunningResultPage />);

    await screen.findByRole("option", { name: "Runner A" });
    await user.selectOptions(
      screen.getByLabelText("Save for athlete"),
      "8",
    );
    await user.click(screen.getByRole("button", { name: "Save session" }));

    await waitFor(() =>
      expect(mocks.saveRunningAnalysis).toHaveBeenCalledWith("run-123", {
        athleteId: 8,
      }),
    );
    expect(screen.getByText("Saved as session #91")).toBeInTheDocument();
  });

  it("keeps the analysis visible when saving fails", async () => {
    const user = userEvent.setup();
    mocks.saveRunningAnalysis.mockRejectedValue(new Error("Database offline"));
    render(<RunningResultPage />);

    await screen.findByRole("option", { name: "Runner A" });
    await user.click(screen.getByRole("button", { name: "Save session" }));

    expect(await screen.findByText("Database offline")).toBeInTheDocument();
    expect(screen.getByText("176")).toBeInTheDocument();
  });
});
