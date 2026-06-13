import { describe, expect, it } from "vitest";

import {
  evaluateCaptureReadiness,
  type CaptureReadinessInput,
} from "./captureReadiness";

const point = () => ({ x: 0.5, y: 0.5 });

const levelCamera = () => ({
  angle_deg: 0.7,
  confidence: 0.92,
  status: "level" as const,
  relative: true,
});

const completeShoulderLandmarks = () => ({
  left_elbow: point(),
  left_shoulder: point(),
  left_hip: point(),
  right_elbow: point(),
  right_shoulder: point(),
  right_hip: point(),
});

const readyInput = (): CaptureReadinessInput => ({
  protocol: "shoulder_flexion",
  poseDetected: true,
  landmarks: completeShoulderLandmarks(),
  cameraLevel: levelCamera(),
  confidence: 0.92,
});

describe("evaluateCaptureReadiness", () => {
  it("blocks when no pose is detected", () => {
    expect(
      evaluateCaptureReadiness({
        ...readyInput(),
        poseDetected: false,
        landmarks: {},
      }),
    ).toMatchObject({ state: "blocked", code: "pose_missing" });
  });

  it("blocks when target landmarks are missing", () => {
    expect(
      evaluateCaptureReadiness({
        ...readyInput(),
        landmarks: { left_shoulder: point() },
      }),
    ).toMatchObject({
      state: "blocked",
      code: "target_landmarks_missing",
    });
  });

  it("blocks when confidence is below the clinical floor", () => {
    expect(
      evaluateCaptureReadiness({
        ...readyInput(),
        confidence: 0.44,
      }),
    ).toMatchObject({ state: "blocked", code: "confidence_low" });
  });

  it("allows explicit acknowledgement for a tilted camera warning", () => {
    expect(
      evaluateCaptureReadiness({
        ...readyInput(),
        cameraLevel: {
          ...levelCamera(),
          status: "adjust",
          angle_deg: 4.5,
        },
      }),
    ).toMatchObject({ state: "warning", code: "camera_adjust" });
  });

  it("warns when confidence is usable but unstable", () => {
    expect(
      evaluateCaptureReadiness({
        ...readyInput(),
        confidence: 0.69,
      }),
    ).toMatchObject({ state: "warning", code: "confidence_warning" });
  });

  it("warns when only one side has complete target landmarks", () => {
    expect(
      evaluateCaptureReadiness({
        ...readyInput(),
        landmarks: {
          left_elbow: point(),
          left_shoulder: point(),
          left_hip: point(),
        },
      }),
    ).toMatchObject({
      state: "warning",
      code: "contralateral_landmarks_missing",
    });
  });

  it("returns ready for complete stable input", () => {
    expect(evaluateCaptureReadiness(readyInput())).toEqual({
      state: "ready",
      code: "ready",
      issues: [],
    });
  });
});
