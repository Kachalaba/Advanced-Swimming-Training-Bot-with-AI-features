import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { sendLiveRehabFrame } from "@/lib/rehabilitation";

import { LiveRehabWorkspace } from "./LiveRehabWorkspace";

const getUserMedia = vi.fn();
const report = {
  protocol: "shoulder_flexion" as const,
  total_correct_reps: 2,
  completion_score: 86,
  target_metrics: {
    left: { rom: 154 },
    right: { rom: 136 },
  },
  symmetry: { asymmetry_index: 12, score: 88 },
};

vi.mock("@/lib/rehabilitation", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/rehabilitation")>();
  return {
    ...original,
    createLiveRehabSession: vi.fn().mockResolvedValue({
      sessionId: "session-1",
      analysisFps: 5,
    }),
    deleteLiveRehabSession: vi.fn().mockResolvedValue(undefined),
    sendLiveRehabFrame: vi.fn(),
  };
});

describe("LiveRehabWorkspace", () => {
  beforeEach(() => {
    getUserMedia.mockReset();
    getUserMedia.mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    });
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia },
    });
    Object.defineProperty(HTMLMediaElement.prototype, "readyState", {
      configurable: true,
      get: () => 2,
    });
    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      configurable: true,
      value: () => ({ drawImage: vi.fn() }),
    });
    Object.defineProperty(HTMLCanvasElement.prototype, "toBlob", {
      configurable: true,
      value: (callback: BlobCallback) =>
        callback(new Blob(["frame"], { type: "image/jpeg" })),
    });
    vi.mocked(sendLiveRehabFrame).mockReset();
  });

  it("toggles the postural map without reacquiring the camera", async () => {
    const user = userEvent.setup();
    render(<LiveRehabWorkspace protocol="shoulder_flexion" />);

    await user.click(screen.getByRole("button", { name: "Запустити камеру" }));
    await user.click(
      screen.getByRole("switch", { name: "Показувати постуральну карту" }),
    );

    expect(getUserMedia).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("switch", { name: "Показувати постуральну карту" }),
    ).toHaveAttribute("aria-checked", "false");
  });

  it("requests fullscreen for the complete workspace", async () => {
    const user = userEvent.setup();
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });
    render(<LiveRehabWorkspace protocol="shoulder_flexion" />);

    await user.click(screen.getByRole("button", { name: "На весь екран" }));

    expect(requestFullscreen).toHaveBeenCalledTimes(1);
  });

  it("falls back to an app-level fullscreen surface and exits with Escape", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <LiveRehabWorkspace protocol="shoulder_flexion" />,
    );

    await user.click(screen.getByRole("button", { name: "На весь екран" }));

    expect(container.firstElementChild).toHaveClass("fixed", "inset-0");
    expect(
      screen.getByRole("button", { name: "Вийти з повноекранного режиму" }),
    ).toBeInTheDocument();

    await user.keyboard("{Escape}");

    expect(container.firstElementChild).not.toHaveClass("fixed");
  });

  it("emits completed analysis data with available camera confidence", async () => {
    const user = userEvent.setup();
    const onAnalysisChange = vi.fn();
    vi.mocked(sendLiveRehabFrame).mockResolvedValue({
      session_id: "session-1",
      sequence: 1,
      pose_detected: true,
      landmarks: {},
      posture: { available: false },
      camera_level: {
        angle_deg: 0,
        confidence: 0.92,
        status: "level",
        relative: true,
      },
      report,
    });
    render(
      <LiveRehabWorkspace
        protocol="shoulder_flexion"
        onAnalysisChange={onAnalysisChange}
      />,
    );

    expect(onAnalysisChange).toHaveBeenCalledWith(null);
    await user.click(screen.getByRole("button", { name: "Запустити камеру" }));

    await waitFor(
      () =>
        expect(onAnalysisChange).toHaveBeenLastCalledWith({
          report,
          confidence: 0.92,
          poseCoverage: null,
        }),
      { timeout: 1500 },
    );
  });
});
