import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CyclingResultPage from "./page";

const mocks = vi.hoisted(() => ({
  listAthletes: vi.fn(),
  saveCyclingAnalysis: vi.fn(),
  subscribeCyclingAnalysis: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ jobId: "cycle-123" }),
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
    saveCyclingAnalysis: mocks.saveCyclingAnalysis,
    subscribeCyclingAnalysis: mocks.subscribeCyclingAnalysis,
  };
});

describe("CyclingResultPage", () => {
  beforeEach(() => {
    mocks.listAthletes.mockResolvedValue([
      { id: "7", name: "Cyclist A", initials: "CA", handle: null },
      { id: "8", name: "Cyclist B", initials: "CB", handle: null },
    ]);
    mocks.saveCyclingAnalysis.mockResolvedValue({ sessionId: 73 });
    mocks.subscribeCyclingAnalysis.mockImplementation((_jobId, onEvent) => {
      onEvent({
        type: "result",
        analysis: {
          cadence: 91,
          avg_knee_angle_top: 74,
          avg_knee_angle_bottom: 146,
          pedal_smoothness: 84,
          upper_body_stability: 88,
          bike_fit_score: 86,
        },
        frames_total: 120,
        frames_with_pose: 108,
        quality: { status: "pass", pose_coverage: 90 },
        video_path: "annotated.mp4",
      });
      return vi.fn();
    });
  });

  it("shows bike-fit evidence and saves it to the selected athlete", async () => {
    const user = userEvent.setup();
    render(<CyclingResultPage />);

    expect(await screen.findByText("91")).toBeInTheDocument();
    expect(screen.getByText("146")).toBeInTheDocument();
    await user.selectOptions(
      screen.getByLabelText("Save for athlete"),
      "8",
    );
    await user.click(screen.getByRole("button", { name: "Save session" }));

    await waitFor(() =>
      expect(mocks.saveCyclingAnalysis).toHaveBeenCalledWith("cycle-123", {
        athleteId: 8,
      }),
    );
    expect(screen.getByText("Saved as session #73")).toBeInTheDocument();
  });
});
