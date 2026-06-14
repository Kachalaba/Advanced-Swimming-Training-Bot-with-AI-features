import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CyclingPage from "./page";

const mocks = vi.hoisted(() => ({
  useSportOverview: vi.fn(),
}));

vi.mock("@/lib/sportOverview", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/sportOverview")>();
  return { ...actual, useSportOverview: mocks.useSportOverview };
});
vi.mock("@/components/sports/CyclingUploader", () => ({
  CyclingUploader: () => <div>Cycling uploader</div>,
}));

describe("CyclingPage", () => {
  beforeEach(() => {
    mocks.useSportOverview.mockReturnValue({
      overview: null,
      loading: false,
      error: null,
      retry: vi.fn(),
    });
  });

  it("offers the real workflow without fictional athlete results", () => {
    render(<CyclingPage />);

    expect(screen.queryByText("312")).not.toBeInTheDocument();
    expect(screen.queryByText("Sweet spot intervals")).not.toBeInTheDocument();
    expect(screen.queryByText("Demo metrics")).not.toBeInTheDocument();
    expect(screen.getByText("Cycling uploader")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upload session" }),
    ).toBeEnabled();
    expect(screen.getByText("No sessions yet")).toBeInTheDocument();
    expect(screen.queryByText("Web analysis adapter planned")).not.toBeInTheDocument();
  });
});
