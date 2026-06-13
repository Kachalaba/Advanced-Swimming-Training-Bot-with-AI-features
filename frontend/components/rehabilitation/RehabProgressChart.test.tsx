import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { RehabProgressSession } from "@/lib/rehabProgress";

import { RehabProgressChart } from "./RehabProgressChart";

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

describe("RehabProgressChart", () => {
  it("renders three series with point values and an accessible summary", () => {
    render(<RehabProgressChart sessions={sessions} locale="en" />);

    expect(screen.getByText("Left ROM")).toBeInTheDocument();
    expect(screen.getByText("Right ROM")).toBeInTheDocument();
    expect(screen.getByText("Symmetry")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /two rehabilitation sessions/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("120°")).toBeInTheDocument();
    expect(screen.getByText("137°")).toBeInTheDocument();
    expect(screen.getByText("94%")).toBeInTheDocument();
  });
});
