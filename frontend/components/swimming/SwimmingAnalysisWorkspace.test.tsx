import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  SwimmingAnalysisEvent,
  SwimmingResultEvent,
} from "@/lib/swimming";

import { SwimmingAnalysisWorkspace } from "./SwimmingAnalysisWorkspace";

const mocks = vi.hoisted(() => ({
  subscribeSwimmingAnalysis: vi.fn(),
  saveSwimmingAnalysis: vi.fn(),
  swimmingVideoUrl: vi.fn((jobId: string) => `/video/${jobId}`),
}));

vi.mock("@/lib/swimming", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/swimming")>("@/lib/swimming");
  return { ...actual, ...mocks };
});

function _result(kickAvailable = true): SwimmingResultEvent {
  return {
    type: "result",
    analysis_type: "swimming_freestyle_side",
    contract_version: "1.0",
    quality: {
      status: kickAvailable ? "pass" : "partial",
      warnings: kickAvailable ? [] : ["Feet leave the frame"],
    },
    coverage: {
      available_zones: kickAvailable ? 5 : 4,
      total_zones: 5,
    },
    overall_score: 82,
    cycles: [
      {
        id: "cycle-1",
        start_frame: 10,
        peak_frame: 14,
        end_frame: 20,
        start_sec: 1,
        peak_sec: 1.4,
        end_sec: 2,
        quality: 0.88,
        complete: true,
      },
    ],
    zones: [
      {
        id: "body_position",
        status: "needs_attention",
        score: 62,
        confidence: 0.86,
        confidence_level: "high",
        metrics: { hip_drop_ratio: 0.07 },
        issues: [],
        evidence: [
          {
            cycle_id: "cycle-1",
            start_sec: 12,
            peak_sec: 12.4,
            end_sec: 12.8,
          },
        ],
      },
      ...(["rotation", "catch", "breathing"] as const).map((id) => ({
        id,
        status: "good" as const,
        score: 88,
        confidence: 0.8,
        confidence_level: "high" as const,
        metrics: {},
        issues: [],
        evidence: [],
      })),
      {
        id: "kick",
        status: kickAvailable ? "good" : "insufficient_data",
        score: kickAvailable ? 84 : null,
        confidence: kickAvailable ? 0.76 : 0,
        confidence_level: kickAvailable ? "medium" : "insufficient",
        metrics: {},
        issues: [],
        evidence: [],
      },
    ],
    primary_issue: {
      zone_id: "body_position",
      issue_code: "hips_drop",
      title: "Hips drop below the body line",
      why_it_matters:
        "A lower hip position increases frontal drag and costs speed.",
      confidence: 0.86,
      confidence_level: "high",
      confirming_cycles: 3,
      evidence: [
        {
          cycle_id: "cycle-1",
          start_sec: 12,
          peak_sec: 12.4,
          end_sec: 12.8,
        },
      ],
    },
    prescription: {
      drill: {
        name: "Side kick with rotation",
        purpose: "Build a stable horizontal line.",
        execution: "Kick six beats on each side.",
        common_mistake: "Lifting the head.",
        success_cue: "Keep the hips near the surface.",
      },
      mini_set: {
        title: "Body-line control",
        repetitions: 6,
        distance_m: 50,
        rest_sec: 20,
        intensity: "easy aerobic",
        focus: "Hold the hips near the surface.",
      },
    },
    frames_total: 80,
    frames_with_pose: 72,
    video_path: "annotated.mp4",
  };
}

describe("SwimmingAnalysisWorkspace", () => {
  let emit: (event: SwimmingAnalysisEvent) => void;

  beforeEach(() => {
    vi.clearAllMocks();
    mocks.subscribeSwimmingAnalysis.mockImplementation(
      (_jobId: string, callback: typeof emit) => {
        emit = callback;
        return vi.fn();
      },
    );
    mocks.saveSwimmingAnalysis.mockResolvedValue({ sessionId: 91 });
  });

  it("renders the primary issue before technique zones", () => {
    render(
      <SwimmingAnalysisWorkspace
        jobId="swim-123"
        initialResult={_result()}
      />,
    );

    const mainIssue = screen.getByRole("heading", {
      name: "Hips drop below the body line",
    });
    const zones = screen.getByRole("heading", { name: "Technique zones" });

    expect(
      mainIssue.compareDocumentPosition(zones) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("seeks the video to the evidence peak", async () => {
    const user = userEvent.setup();
    render(
      <SwimmingAnalysisWorkspace
        jobId="swim-123"
        initialResult={_result()}
      />,
    );
    const video = screen.getByTestId("swimming-video") as HTMLVideoElement;

    await user.click(
      screen.getByRole("button", {
        name: "Show Body position at 12.4 seconds",
      }),
    );

    expect(video.currentTime).toBe(12.4);
  });

  it("shows insufficient data without inventing a zero score", () => {
    render(
      <SwimmingAnalysisWorkspace
        jobId="swim-123"
        initialResult={_result(false)}
      />,
    );

    expect(
      screen.getAllByText("4 of 5 zones analyzed").length,
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Insufficient data")).toBeInTheDocument();
    expect(screen.queryByText("Kick 0")).not.toBeInTheDocument();
  });

  it("shows the corrective drill and concrete mini-set", () => {
    render(
      <SwimmingAnalysisWorkspace
        jobId="swim-123"
        initialResult={_result()}
      />,
    );

    expect(screen.getByText("Side kick with rotation")).toBeInTheDocument();
    expect(screen.getByText("6 × 50 m")).toBeInTheDocument();
    expect(screen.getByText("20s rest")).toBeInTheDocument();
  });

  it("saves the result once and exposes the history session", async () => {
    const user = userEvent.setup();
    render(
      <SwimmingAnalysisWorkspace
        jobId="swim-123"
        initialResult={_result()}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Save to history" }));

    expect(mocks.saveSwimmingAnalysis).toHaveBeenCalledWith("swim-123");
    expect(
      screen.getByRole("button", { name: "Saved to history #91" }),
    ).toBeDisabled();
  });

  it("shows actionable reshoot guidance on a quality error", () => {
    render(<SwimmingAnalysisWorkspace jobId="swim-123" />);

    act(() => {
      emit({
        type: "error",
        code: "insufficient_cycles",
        message: "Fewer than two reliable complete cycles were found.",
        reshoot_guidance: "Record at least four complete freestyle cycles.",
      });
    });

    expect(
      screen.getByText("Fewer than two reliable complete cycles were found."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Record at least four complete freestyle cycles."),
    ).toBeInTheDocument();
  });
});
