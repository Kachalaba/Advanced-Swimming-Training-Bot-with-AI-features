import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import DrylandResultPage from "./page";

const mocks = vi.hoisted(() => ({
  listAthletes: vi.fn(),
  saveDrylandAnalysis: vi.fn(),
  subscribeDrylandAnalysis: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ jobId: "dryland-123" }),
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
    saveDrylandAnalysis: mocks.saveDrylandAnalysis,
    subscribeDrylandAnalysis: mocks.subscribeDrylandAnalysis,
  };
});

describe("DrylandResultPage", () => {
  beforeEach(() => {
    mocks.listAthletes.mockResolvedValue([
      { id: "7", name: "Athlete A", initials: "AA", handle: null },
      { id: "8", name: "Athlete B", initials: "AB", handle: null },
    ]);
    mocks.saveDrylandAnalysis.mockResolvedValue({ sessionId: 81 });
    mocks.subscribeDrylandAnalysis.mockImplementation((_jobId, onEvent) => {
      onEvent({ type: "progress", pct: 42, label: "Detecting full repetitions" });
      onEvent({
        type: "result",
        exercise_type: "squat",
        analysis: {
          exercise_type: "squat",
          tracked_joint: "knee",
          total_reps: 4,
          avg_tempo: 2.1,
          avg_range_of_motion: 74,
          stability_score: 88,
          min_angle: 86,
          max_angle: 170,
          reps: [
            {
              rep_number: 1,
              start_frame: 3,
              effort_frame: 12,
              end_frame: 22,
              duration_sec: 2.1,
              min_angle: 88,
              max_angle: 170,
              range_of_motion: 82,
              active_side: "left",
            },
          ],
        },
        frames_total: 100,
        frames_with_pose: 92,
        quality: {
          status: "pass",
          pose_coverage: 92,
          metric_ready_frames: 84,
          minimum_required_frames: 20,
        },
        video_path: "annotated.mp4",
      });
      return vi.fn();
    });
  });

  it("shows dryland evidence and saves it to the selected athlete", async () => {
    const user = userEvent.setup();
    render(<DrylandResultPage />);

    expect(await screen.findByText("Dryland evidence")).toBeInTheDocument();
    expect(screen.getByText("Squat evidence")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("2.1")).toBeInTheDocument();
    expect(screen.getByText("#1")).toBeInTheDocument();

    await screen.findByRole("option", { name: "Athlete A" });
    await user.selectOptions(screen.getByLabelText("Save for athlete"), "8");
    await user.click(screen.getByRole("button", { name: "Save session" }));

    await waitFor(() =>
      expect(mocks.saveDrylandAnalysis).toHaveBeenCalledWith("dryland-123", {
        athleteId: 8,
      }),
    );
    expect(screen.getByText("Saved as session #81")).toBeInTheDocument();
  });

  it("shows reshoot guidance when the backend rejects the clip", async () => {
    mocks.subscribeDrylandAnalysis.mockImplementation((_jobId, onEvent) => {
      onEvent({
        type: "error",
        message: "Too few metric-ready frames for a reliable dryland result.",
      });
      return vi.fn();
    });

    render(<DrylandResultPage />);

    expect(
      await screen.findByText("Capture quality is not sufficient"),
    ).toBeInTheDocument();
    expect(screen.getByText(/too few metric-ready frames/i)).toBeInTheDocument();
    expect(screen.queryByText("Dryland evidence")).not.toBeInTheDocument();
  });
});
