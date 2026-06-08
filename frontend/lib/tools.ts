const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export type ToolOperation = "trim" | "frame_extractor";

export type ToolResultEvent = {
  type: "result";
  operation: ToolOperation;
  artifact_name: string;
  media_type: string;
  size_bytes: number;
  metadata: Record<string, unknown>;
};

export type ToolJobEvent =
  | { type: "progress"; pct: number; label: string }
  | ToolResultEvent
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

export async function uploadTrimVideo(
  file: File,
  startSec: number,
  endSec: number,
): Promise<{ jobId: string }> {
  const body = new FormData();
  body.append("video", file);
  body.append("start_sec", String(startSec));
  body.append("end_sec", String(endSec));
  const response = await fetch(`${BACKEND_URL}/api/tools/trim`, {
    method: "POST",
    body,
  });
  const data = await readJson<{ job_id: string }>(response);
  return { jobId: data.job_id };
}

export async function uploadFrameExtraction(
  file: File,
  options:
    | { mode: "interval"; intervalSec: number }
    | { mode: "count"; frameCount: number },
): Promise<{ jobId: string }> {
  const body = new FormData();
  body.append("video", file);
  body.append("mode", options.mode);
  if (options.mode === "interval") {
    body.append("interval_sec", String(options.intervalSec));
  } else {
    body.append("frame_count", String(options.frameCount));
  }
  const response = await fetch(`${BACKEND_URL}/api/tools/frames`, {
    method: "POST",
    body,
  });
  const data = await readJson<{ job_id: string }>(response);
  return { jobId: data.job_id };
}

export function subscribeToolJob(
  jobId: string,
  onEvent: (event: ToolJobEvent) => void,
): () => void {
  const source = new EventSource(`${BACKEND_URL}/api/tools/${jobId}/events`);
  source.onmessage = (message) => {
    let event: ToolJobEvent;
    try {
      event = JSON.parse(message.data) as ToolJobEvent;
    } catch {
      return;
    }
    onEvent(event);
    if (event.type === "result" || event.type === "error") source.close();
  };
  source.onerror = () => source.close();
  return () => source.close();
}

export function toolDownloadUrl(jobId: string): string {
  return `${BACKEND_URL}/api/tools/${jobId}/download`;
}

export async function saveToolJob(
  jobId: string,
  athleteName = "Nikita K.",
): Promise<{ sessionId: number }> {
  const response = await fetch(`${BACKEND_URL}/api/tools/${jobId}/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ athlete_name: athleteName }),
  });
  const data = await readJson<{ session_id: number }>(response);
  return { sessionId: data.session_id };
}
