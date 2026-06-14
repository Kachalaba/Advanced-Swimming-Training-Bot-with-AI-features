import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import SwimmingPage from "./page";

const mocks = vi.hoisted(() => ({
  useSportOverview: vi.fn(),
}));

vi.mock("@/lib/sportOverview", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/sportOverview")>();
  return { ...actual, useSportOverview: mocks.useSportOverview };
});
vi.mock("@/components/swimming/SwimmingUploader", () => ({
  SwimmingUploader: () => <div>Swimming uploader</div>,
  SwimmingFilmingGuide: () => <div>Filming guide</div>,
}));

describe("SwimmingPage", () => {
  beforeEach(() => {
    mocks.useSportOverview.mockReturnValue({
      overview: {
        athlete: { id: "1", name: "Swimmer", initials: "S", handle: null },
        sport: "swimming",
        total_sessions: 1,
        latest_session_date: "2026-06-14T10:00:00",
        latest_score: 82,
        headline_metrics: {},
        insights: [],
        score_series: [],
        sessions: [
          {
            id: 9,
            date: "2026-06-14T10:00:00",
            duration_sec: 12,
            score: 82,
            summary: "Torso rotation is asymmetric",
            has_video: true,
            metrics: {},
            quality: {},
            insights: [],
          },
        ],
      },
      loading: false,
      error: null,
      retry: vi.fn(),
    });
  });

  it("shows persisted swimming history", () => {
    render(<SwimmingPage />);

    expect(screen.getByText("Torso rotation is asymmetric")).toBeInTheDocument();
    expect(screen.getByText("82")).toBeInTheDocument();
  });
});
