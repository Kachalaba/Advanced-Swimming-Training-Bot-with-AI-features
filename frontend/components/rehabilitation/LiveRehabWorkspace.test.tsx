import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LiveRehabWorkspace } from "./LiveRehabWorkspace";

const getUserMedia = vi.fn();

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
});
