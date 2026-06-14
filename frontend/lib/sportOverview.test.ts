import { describe, expect, it } from "vitest";

import type { SportOverview } from "./api";
import { toSportLandingData } from "./sportOverview";

const overview: SportOverview = {
  athlete: { id: "7", name: "Runner", initials: "R", handle: null },
  sport: "running",
  total_sessions: 1,
  latest_session_date: "2026-06-14T10:00:00",
  latest_score: 88,
  headline_metrics: {
    cadence: {
      key: "cadence",
      label: "Cadence",
      value: 178,
      unit: "spm",
      higher_is_better: null,
    },
  },
  insights: [
    {
      code: "arm_crossover",
      level: "warning",
      title: "Arm crosses the body midline",
      detail: "Review the video.",
    },
  ],
  score_series: [{ date: "2026-06-14T10:00:00", value: 88 }],
  sessions: [
    {
      id: 12,
      date: "2026-06-14T10:00:00",
      duration_sec: 65,
      score: 88,
      summary: "178 spm · Midfoot strike",
      has_video: true,
      metrics: {},
      quality: { pose_coverage: 90 },
      insights: [],
    },
  ],
};

describe("toSportLandingData", () => {
  it("maps persisted metrics, sessions and evidence-based insights", () => {
    const data = toSportLandingData(overview);

    expect(data.metrics).toEqual([
      { label: "Cadence", value: "178", unit: "spm" },
    ]);
    expect(data.sessions[0]).toMatchObject({
      id: 12,
      title: "178 spm · Midfoot strike",
      duration: "1:05",
      date: "2026-06-14",
      score: 88,
    });
    expect(data.insights[0]).toEqual({
      tag: "Movement flag",
      variant: "warn",
      title: "Arm crosses the body midline",
      detail: "Review the video.",
    });
  });

  it("returns empty collections for an empty overview", () => {
    const data = toSportLandingData({
      ...overview,
      total_sessions: 0,
      latest_session_date: null,
      latest_score: null,
      headline_metrics: {},
      insights: [],
      score_series: [],
      sessions: [],
    });

    expect(data).toEqual({ metrics: [], sessions: [], insights: [] });
  });
});
