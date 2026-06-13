import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  saveUploadedRehabSession,
  subscribeRehabAnalysis,
  uploadRehabVideo,
} from "@/lib/rehabilitation";

import { RehabUploader } from "./RehabUploader";

const report = {
  protocol: "shoulder_flexion" as const,
  total_correct_reps: 3,
  completion_score: 84,
  target_metrics: {
    left: { rom: 150 },
    right: { rom: 132 },
  },
  symmetry: { asymmetry_index: 12, score: 88 },
};

vi.mock("@/lib/rehabilitation", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/rehabilitation")>();
  return {
    ...original,
    saveUploadedRehabSession: vi.fn().mockResolvedValue({ sessionId: 91 }),
    uploadRehabVideo: vi.fn().mockResolvedValue({ jobId: "job-1" }),
    subscribeRehabAnalysis: vi.fn(),
  };
});

describe("RehabUploader", () => {
  beforeEach(() => {
    vi.mocked(uploadRehabVideo).mockClear();
    vi.mocked(subscribeRehabAnalysis).mockReset();
    vi.mocked(saveUploadedRehabSession).mockClear();
  });

  it("automatically analyzes a file selected by the clinical readiness step", async () => {
    const initialFile = new File(["video"], "visit.mp4", {
      type: "video/mp4",
    });
    vi.mocked(subscribeRehabAnalysis).mockReturnValue(vi.fn());

    render(
      <RehabUploader
        protocol="shoulder_flexion"
        initialFile={initialFile}
      />,
    );

    await waitFor(() =>
      expect(uploadRehabVideo).toHaveBeenCalledWith(
        initialFile,
        "shoulder_flexion",
      ),
    );
  });

  it("clears stale data and emits completed upload coverage", async () => {
    const user = userEvent.setup();
    const onAnalysisChange = vi.fn();
    vi.mocked(subscribeRehabAnalysis).mockImplementation((_jobId, onEvent) => {
      onEvent({
        type: "result",
        report,
        frames_total: 40,
        frames_with_pose: 30,
        video_path: null,
      });
      return vi.fn();
    });

    const { container } = render(
      <RehabUploader
        protocol="shoulder_flexion"
        onAnalysisChange={onAnalysisChange}
      />,
    );
    const input = container.querySelector('input[type="file"]');
    expect(input).not.toBeNull();

    await user.upload(
      input as HTMLInputElement,
      new File(["video"], "session.mp4", { type: "video/mp4" }),
    );

    expect(uploadRehabVideo).toHaveBeenCalledOnce();
    expect(onAnalysisChange).toHaveBeenCalledWith(null);
    await waitFor(() =>
      expect(onAnalysisChange).toHaveBeenLastCalledWith({
        report,
        confidence: null,
        poseCoverage: 75,
      }),
    );
    expect(screen.getByText("Відео проаналізовано")).toBeInTheDocument();
  });

  it("saves the upload to the explicit athlete and emits the session id", async () => {
    const user = userEvent.setup();
    const onSessionSaved = vi.fn();
    vi.mocked(subscribeRehabAnalysis).mockImplementation((_jobId, onEvent) => {
      onEvent({
        type: "result",
        report,
        frames_total: 40,
        frames_with_pose: 30,
        video_path: null,
      });
      return vi.fn();
    });

    const { container } = render(
      <RehabUploader
        protocol="shoulder_flexion"
        saveTarget={{ athleteId: 7, athleteName: "Patient A" }}
        onSessionSaved={onSessionSaved}
      />,
    );
    const input = container.querySelector('input[type="file"]');
    await user.upload(
      input as HTMLInputElement,
      new File(["video"], "session.mp4", { type: "video/mp4" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Зберегти в історію" }),
    );

    await waitFor(() =>
      expect(saveUploadedRehabSession).toHaveBeenCalledWith("job-1", {
        athleteId: 7,
        athleteName: "Patient A",
      }),
    );
    expect(onSessionSaved).toHaveBeenCalledOnce();
    expect(onSessionSaved).toHaveBeenCalledWith(91);
  });
});
