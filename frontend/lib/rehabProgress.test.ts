import { describe, expect, it } from "vitest";

import {
  buildProgressObservation,
  compareRehabProgress,
  summarizeRehabProtocols,
  type RehabProgressSession,
} from "./rehabProgress";

const sessions: RehabProgressSession[] = [
  {
    id: 1,
    date: "2026-06-01T10:00:00",
    protocol: "shoulder_flexion",
    leftRom: 120,
    rightRom: 108,
    symmetry: 84,
    repetitions: 2,
    completionScore: 72,
    validFrames: 20,
    hasVideo: false,
  },
  {
    id: 2,
    date: "2026-06-05T10:00:00",
    protocol: "knee_extension",
    leftRom: 92,
    rightRom: 94,
    symmetry: 96,
    repetitions: 4,
    completionScore: 88,
    validFrames: null,
    hasVideo: true,
  },
  {
    id: 3,
    date: "2026-06-12T10:00:00",
    protocol: "shoulder_flexion",
    leftRom: 146,
    rightRom: 137,
    symmetry: 94,
    repetitions: 5,
    completionScore: 92,
    validFrames: 32,
    hasVideo: true,
  },
];

describe("rehabilitation progress model", () => {
  it("compares the earliest and latest compatible sessions", () => {
    const result = compareRehabProgress(sessions, "shoulder_flexion");

    expect(result?.baseline.id).toBe(1);
    expect(result?.current.id).toBe(3);
    expect(result?.deltas).toEqual({
      leftRom: 26,
      rightRom: 29,
      symmetry: 10,
      repetitions: 3,
      completionScore: 20,
    });
  });

  it("returns no comparison when only one compatible session exists", () => {
    expect(compareRehabProgress(sessions, "knee_extension")).toBeNull();
  });

  it("summarizes protocols by count and latest date", () => {
    expect(summarizeRehabProtocols(sessions)).toEqual([
      {
        protocol: "shoulder_flexion",
        count: 2,
        latestDate: "2026-06-12T10:00:00",
      },
      {
        protocol: "knee_extension",
        count: 1,
        latestDate: "2026-06-05T10:00:00",
      },
    ]);
  });

  it("uses neutral localized observation language for mixed change", () => {
    const comparison = compareRehabProgress(
      [
        sessions[0],
        {
          ...sessions[2],
          leftRom: 118,
          rightRom: 120,
          symmetry: 80,
        },
      ],
      "shoulder_flexion",
    );

    expect(buildProgressObservation(comparison, "uk")).toContain(
      "Праворуч ROM збільшився на 12°",
    );
    expect(buildProgressObservation(comparison, "uk")).toContain(
      "Ліворуч ROM зменшився на 2°",
    );
    expect(buildProgressObservation(comparison, "en")).toContain(
      "Symmetry changed by -4 pp",
    );
  });
});
