import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PostureOverlay } from "./PostureOverlay";

const posture = {
  available: true,
  points: {
    left_shoulder: { x: 0.3, y: 0.3 },
    right_shoulder: { x: 0.7, y: 0.34 },
    left_hip: { x: 0.36, y: 0.62 },
    right_hip: { x: 0.66, y: 0.6 },
    shoulder_mid: { x: 0.5, y: 0.32 },
    hip_mid: { x: 0.51, y: 0.61 },
  },
  shoulder_angle_deg: 2.7,
  pelvis_angle_deg: -1.9,
  trunk_lean_deg: 0.8,
  trunk_offset_pct: 1,
} as const;

describe("PostureOverlay", () => {
  it("renders the postural coordinate axes and signed labels", () => {
    render(<PostureOverlay visible posture={posture} />);

    expect(screen.getByTestId("posture-plumb-axis")).toBeInTheDocument();
    expect(screen.getByText("Плечі +2.7°")).toBeInTheDocument();
    expect(screen.getByText("Таз −1.9°")).toBeInTheDocument();
    expect(screen.getByText("Тулуб +0.8°")).toBeInTheDocument();
  });

  it("removes the complete overlay when hidden", () => {
    render(<PostureOverlay visible={false} posture={posture} />);

    expect(screen.queryByTestId("posture-overlay")).not.toBeInTheDocument();
  });
});
