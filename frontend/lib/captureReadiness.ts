import type {
  CameraLevelUpdate,
  NormalizedPoint,
  RehabProtocol,
} from "./rehabilitation";

export const CAPTURE_CONFIDENCE_BLOCKED_BELOW = 0.45;
export const CAPTURE_CONFIDENCE_READY_FROM = 0.7;

export type CaptureReadinessState = "ready" | "warning" | "blocked";

export type CaptureReadinessCode =
  | "ready"
  | "pose_missing"
  | "target_landmarks_missing"
  | "confidence_low"
  | "confidence_warning"
  | "camera_adjust"
  | "contralateral_landmarks_missing"
  | "upload_ready";

export type CaptureReadinessInput = {
  protocol: RehabProtocol;
  poseDetected: boolean;
  landmarks: Record<string, NormalizedPoint>;
  cameraLevel: CameraLevelUpdate | null;
  confidence: number;
};

export type CaptureReadinessResult = {
  state: CaptureReadinessState;
  code: CaptureReadinessCode;
  issues: CaptureReadinessCode[];
};

const TARGET_POINTS: Record<RehabProtocol, readonly string[]> = {
  shoulder_flexion: ["elbow", "shoulder", "hip"],
  shoulder_abduction: ["elbow", "shoulder", "hip"],
  elbow_flexion: ["shoulder", "elbow", "wrist"],
  knee_extension: ["hip", "knee", "ankle"],
  hip_abduction: ["shoulder", "hip", "knee"],
};

function sideIsComplete(
  side: "left" | "right",
  targetPoints: readonly string[],
  landmarks: Record<string, NormalizedPoint>,
): boolean {
  return targetPoints.every((point) => Boolean(landmarks[`${side}_${point}`]));
}

export function evaluateCaptureReadiness(
  input: CaptureReadinessInput,
): CaptureReadinessResult {
  if (!input.poseDetected) {
    return {
      state: "blocked",
      code: "pose_missing",
      issues: ["pose_missing"],
    };
  }

  const targetPoints = TARGET_POINTS[input.protocol];
  const leftComplete = sideIsComplete("left", targetPoints, input.landmarks);
  const rightComplete = sideIsComplete("right", targetPoints, input.landmarks);
  if (!leftComplete && !rightComplete) {
    return {
      state: "blocked",
      code: "target_landmarks_missing",
      issues: ["target_landmarks_missing"],
    };
  }

  if (input.confidence < CAPTURE_CONFIDENCE_BLOCKED_BELOW) {
    return {
      state: "blocked",
      code: "confidence_low",
      issues: ["confidence_low"],
    };
  }

  const warnings: CaptureReadinessCode[] = [];
  if (!input.cameraLevel || input.cameraLevel.status !== "level") {
    warnings.push("camera_adjust");
  }
  if (input.confidence < CAPTURE_CONFIDENCE_READY_FROM) {
    warnings.push("confidence_warning");
  }
  if (!leftComplete || !rightComplete) {
    warnings.push("contralateral_landmarks_missing");
  }

  if (warnings.length > 0) {
    return {
      state: "warning",
      code: warnings[0],
      issues: warnings,
    };
  }

  return { state: "ready", code: "ready", issues: [] };
}
