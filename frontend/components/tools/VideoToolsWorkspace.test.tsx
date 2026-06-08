import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VideoToolsWorkspace } from "./VideoToolsWorkspace";

const mocks = vi.hoisted(() => ({
  uploadTrimVideo: vi.fn(),
  uploadFrameExtraction: vi.fn(),
  subscribeToolJob: vi.fn(),
  saveToolJob: vi.fn(),
  toolDownloadUrl: vi.fn((jobId: string) => `/download/${jobId}`),
}));

vi.mock("@/lib/tools", () => mocks);

describe("VideoToolsWorkspace", () => {
  let emit: (event: {
    type: "result";
    operation: "trim" | "frame_extractor";
    artifact_name: string;
    media_type: string;
    size_bytes: number;
    metadata: Record<string, unknown>;
  }) => void;

  beforeEach(() => {
    vi.clearAllMocks();
    mocks.uploadTrimVideo.mockResolvedValue({ jobId: "trim-job" });
    mocks.uploadFrameExtraction.mockResolvedValue({ jobId: "frames-job" });
    mocks.saveToolJob.mockResolvedValue({ sessionId: 42 });
    mocks.subscribeToolJob.mockImplementation(
      (_jobId: string, callback: typeof emit) => {
        emit = callback;
        return vi.fn();
      },
    );
  });

  it("uploads a selected trim range", async () => {
    const user = userEvent.setup();
    render(<VideoToolsWorkspace />);
    const file = new File(["video"], "training.mp4", { type: "video/mp4" });

    await user.upload(screen.getByLabelText("Choose source video"), file);
    await user.clear(screen.getByLabelText("Start time in seconds"));
    await user.type(screen.getByLabelText("Start time in seconds"), "1");
    await user.clear(screen.getByLabelText("End time in seconds"));
    await user.type(screen.getByLabelText("End time in seconds"), "4");
    await user.click(screen.getByRole("button", { name: "Process video" }));

    expect(mocks.uploadTrimVideo).toHaveBeenCalledWith(file, 1, 4);
    expect(mocks.subscribeToolJob).toHaveBeenCalledWith(
      "trim-job",
      expect.any(Function),
    );
  });

  it("supports exact-count frame extraction", async () => {
    const user = userEvent.setup();
    render(<VideoToolsWorkspace />);
    const file = new File(["video"], "training.mov", {
      type: "video/quicktime",
    });

    await user.click(
      screen.getByRole("button", { name: "Frame extractor" }),
    );
    await user.upload(screen.getByLabelText("Choose source video"), file);
    await user.click(screen.getByRole("radio", { name: "Exact count" }));
    await user.clear(screen.getByLabelText("Number of frames"));
    await user.type(screen.getByLabelText("Number of frames"), "5");
    await user.click(screen.getByRole("button", { name: "Process video" }));

    expect(mocks.uploadFrameExtraction).toHaveBeenCalledWith(file, {
      mode: "count",
      frameCount: 5,
    });
  });

  it("offers download and saves a completed artifact to history", async () => {
    const user = userEvent.setup();
    render(<VideoToolsWorkspace />);
    const file = new File(["video"], "training.mp4", { type: "video/mp4" });

    await user.upload(screen.getByLabelText("Choose source video"), file);
    await user.click(screen.getByRole("button", { name: "Process video" }));
    act(() => {
      emit({
        type: "result",
        operation: "trim",
        artifact_name: "trim-training.mp4",
        media_type: "video/mp4",
        size_bytes: 2048,
        metadata: { duration_sec: 5 },
      });
    });

    expect(screen.getByRole("link", { name: "Download result" })).toHaveAttribute(
      "href",
      "/download/trim-job",
    );
    await user.click(
      screen.getByRole("button", { name: "Save to history" }),
    );

    expect(mocks.saveToolJob).toHaveBeenCalledWith("trim-job");
    expect(
      screen.getByRole("button", { name: "Saved to history #42" }),
    ).toBeDisabled();
  });
});
