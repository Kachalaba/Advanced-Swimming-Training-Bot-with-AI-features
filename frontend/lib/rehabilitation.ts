const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const rehabProtocols = [
  "shoulder_flexion",
  "shoulder_abduction",
  "elbow_flexion",
  "knee_extension",
  "hip_abduction",
] as const;

export type RehabProtocol = (typeof rehabProtocols)[number];

export type NormalizedPoint = {
  x: number;
  y: number;
};

export type PostureSeverity = "good" | "moderate" | "warning";

export type PostureUpdate = {
  available: boolean;
  points?: Record<string, NormalizedPoint>;
  shoulder_angle_deg?: number;
  pelvis_angle_deg?: number;
  trunk_lean_deg?: number;
  trunk_offset_pct?: number;
  shoulder?: { severity: PostureSeverity; higher_side: string };
  pelvis?: { severity: PostureSeverity; higher_side: string };
  trunk?: {
    severity: PostureSeverity;
    lean_direction: string;
    offset_direction: string;
  };
};

export type CameraLevelUpdate = {
  angle_deg: number | null;
  confidence: number;
  status: "uncalibrated" | "level" | "adjust" | "recalibrate";
  direction?: "level" | "right_edge_high" | "left_edge_high";
  relative: boolean;
};

export type RehabReport = {
  protocol: RehabProtocol;
  total_correct_reps: number;
  completion_score: number;
  target_metrics: {
    left: { rom: number; [key: string]: unknown };
    right: { rom: number; [key: string]: unknown };
  };
  symmetry: {
    asymmetry_index: number;
    score: number;
  };
  [key: string]: unknown;
};

export type LiveRehabUpdate = {
  session_id: string;
  sequence: number;
  pose_detected: boolean;
  landmarks: Record<string, NormalizedPoint>;
  posture: PostureUpdate;
  camera_level: CameraLevelUpdate;
  report: RehabReport | null;
};

export type RehabAnalysisEvent =
  | { type: "progress"; pct: number; label: string }
  | ({ type: "result"; report: RehabReport; frames_total: number; frames_with_pose: number; video_path: string | null })
  | { type: "error"; message: string };

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep the status when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function createLiveRehabSession(
  protocol: RehabProtocol,
  fps = 5,
): Promise<{ sessionId: string; analysisFps: number }> {
  const response = await fetch(`${BACKEND_URL}/api/analysis/rehabilitation/live`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ protocol, fps }),
  });
  const data = await readJson<{ session_id: string; analysis_fps: number }>(response);
  return { sessionId: data.session_id, analysisFps: data.analysis_fps };
}

export async function sendLiveRehabFrame(
  sessionId: string,
  image: Blob,
  calibrate = false,
): Promise<LiveRehabUpdate> {
  const body = new FormData();
  body.append("image", image, "frame.jpg");
  body.append("calibrate", String(calibrate));
  const response = await fetch(
    `${BACKEND_URL}/api/analysis/rehabilitation/live/${sessionId}/frame`,
    { method: "POST", body },
  );
  return readJson<LiveRehabUpdate>(response);
}

export async function deleteLiveRehabSession(sessionId: string): Promise<void> {
  await fetch(`${BACKEND_URL}/api/analysis/rehabilitation/live/${sessionId}`, {
    method: "DELETE",
  });
}

export async function uploadRehabVideo(
  file: File,
  protocol: RehabProtocol,
  fps = 15,
): Promise<{ jobId: string }> {
  const body = new FormData();
  body.append("video", file);
  body.append("protocol", protocol);
  body.append("fps", String(fps));
  const response = await fetch(`${BACKEND_URL}/api/analysis/rehabilitation`, {
    method: "POST",
    body,
  });
  const data = await readJson<{ job_id: string }>(response);
  return { jobId: data.job_id };
}

export function subscribeRehabAnalysis(
  jobId: string,
  onEvent: (event: RehabAnalysisEvent) => void,
): () => void {
  const source = new EventSource(
    `${BACKEND_URL}/api/analysis/rehabilitation/${jobId}/events`,
  );
  source.onmessage = (message) => {
    const event = JSON.parse(message.data) as RehabAnalysisEvent;
    onEvent(event);
    if (event.type === "result" || event.type === "error") source.close();
  };
  source.onerror = () => source.close();
  return () => source.close();
}

export function rehabAnnotatedVideoUrl(jobId: string): string {
  return `${BACKEND_URL}/api/analysis/rehabilitation/${jobId}/video`;
}
