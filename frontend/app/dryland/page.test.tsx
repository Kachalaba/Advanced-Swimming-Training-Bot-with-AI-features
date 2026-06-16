import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import DrylandPage from "./page";

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  useSportOverview: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mocks.push }),
}));
vi.mock("@/lib/sportOverview", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/sportOverview")>();
  return { ...actual, useSportOverview: mocks.useSportOverview };
});

describe("DrylandPage", () => {
  beforeEach(() => {
    mocks.push.mockReset();
    mocks.useSportOverview.mockReturnValue({
      overview: null,
      loading: false,
      error: null,
      retry: vi.fn(),
    });
  });

  it("renders exercise selection before upload", async () => {
    render(<DrylandPage />);

    expect(
      screen.getByRole("heading", { name: /dryland/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select squat/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getAllByText(/fixed side view/i).length).toBeGreaterThan(0);

    await userEvent.click(screen.getByRole("button", { name: /select push-up/i }));

    expect(screen.getAllByText(/camera at torso height/i).length).toBeGreaterThan(
      0,
    );
    expect(
      screen.getByRole("button", { name: /upload push-up dryland video/i }),
    ).toBeInTheDocument();
  });

  it("offers the real workflow without fictional athlete results", () => {
    render(<DrylandPage />);

    expect(screen.queryByText("1,240")).not.toBeInTheDocument();
    expect(screen.queryByText("Core + mobility")).not.toBeInTheDocument();
    expect(screen.queryByText("Demo metrics")).not.toBeInTheDocument();
    expect(screen.queryByText("Web workflow planned")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upload session" }),
    ).toBeEnabled();
    expect(screen.getByText("No sessions yet")).toBeInTheDocument();
  });
});
