import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import RunningPage from "./page";

const mocks = vi.hoisted(() => ({
  useSportOverview: vi.fn(),
}));

vi.mock("@/lib/sportOverview", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/sportOverview")>();
  return { ...actual, useSportOverview: mocks.useSportOverview };
});
vi.mock("@/components/sports/RunningUploader", () => ({
  RunningUploader: () => <div>Running uploader</div>,
}));

describe("RunningPage", () => {
  beforeEach(() => {
    mocks.useSportOverview.mockReturnValue({
      overview: null,
      loading: false,
      error: null,
      retry: vi.fn(),
    });
  });

  it("does not render fictional personal records or sessions", () => {
    render(<RunningPage />);

    expect(screen.queryByText("18:42")).not.toBeInTheDocument();
    expect(screen.queryByText("Tempo run · Track")).not.toBeInTheDocument();
    expect(screen.queryByText("Coming soon")).not.toBeInTheDocument();
    expect(screen.getByText("No sessions yet")).toBeInTheDocument();
    expect(screen.getByText("Capture checklist")).toBeInTheDocument();
    expect(screen.getByText("Lock the camera")).toBeInTheDocument();
  });
});
