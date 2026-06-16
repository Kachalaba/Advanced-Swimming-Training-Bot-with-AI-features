import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { RunningUploader } from "./RunningUploader";

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  uploadRunningVideo: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mocks.push }),
}));

vi.mock("@/lib/analysis", () => ({
  uploadRunningVideo: mocks.uploadRunningVideo,
}));

describe("RunningUploader", () => {
  beforeEach(() => {
    mocks.push.mockReset();
    mocks.uploadRunningVideo.mockReset();
    mocks.uploadRunningVideo.mockResolvedValue({ jobId: "run-42" });
  });

  it("shows scenario guidance and readiness checks before upload", async () => {
    const user = userEvent.setup();
    render(<RunningUploader />);

    expect(screen.getByRole("button", { name: /track/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByText(/readiness score/i)).toBeInTheDocument();
    expect(screen.getAllByText(/camera side-on/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/10-20 meters/i).length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /treadmill/i }));

    expect(screen.getAllByText(/belt and shoes visible/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/steady speed/i).length).toBeGreaterThan(0);
  });

  it("uploads the selected video and opens the result page", async () => {
    const user = userEvent.setup();
    render(<RunningUploader />);
    const file = new File(["video"], "run.mp4", { type: "video/mp4" });

    await user.upload(screen.getByLabelText(/running video file/i), file);

    expect(mocks.uploadRunningVideo).toHaveBeenCalledWith(file);
    expect(mocks.push).toHaveBeenCalledWith("/running/run-42");
  });
});
