const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export type ProgressEvent = {
  type: "progress";
  pct: number;
  label: string;
};

export type ResultEvent = {
  type: "result";
  analysis: Record<string, unknown>;
  frames_total: number;
  frames_with_pose: number;
  video_path: string;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type AnalysisEvent = ProgressEvent | ResultEvent | ErrorEvent;

export async function uploadRunningVideo(
  file: File,
  fps = 30,
): Promise<{ jobId: string }> {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("fps", String(fps));
  const res = await fetch(`${BACKEND_URL}/api/analysis/running`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  const data = (await res.json()) as { job_id: string };
  return { jobId: data.job_id };
}

export async function saveRunningAnalysis(
  jobId: string,
  input: { athleteId?: number; athleteName?: string },
): Promise<{ sessionId: number }> {
  const res = await fetch(
    `${BACKEND_URL}/api/analysis/running/${jobId}/save`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        input.athleteId
          ? { athlete_id: input.athleteId }
          : { athlete_name: input.athleteName ?? "Athlete" },
      ),
    },
  );
  if (!res.ok) {
    throw new Error(`Save failed: ${res.status}`);
  }
  const data = (await res.json()) as { session_id: number };
  return { sessionId: data.session_id };
}

export function subscribeAnalysis(
  jobId: string,
  onEvent: (event: AnalysisEvent) => void,
): () => void {
  const url = `${BACKEND_URL}/api/analysis/running/${jobId}/events`;
  const source = new EventSource(url);
  source.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data) as AnalysisEvent;
      onEvent(data);
      if (data.type === "result" || data.type === "error") {
        source.close();
      }
    } catch (err) {
      console.error("bad event payload", err, e.data);
    }
  };
  source.onerror = () => {
    source.close();
  };
  return () => source.close();
}

export function annotatedVideoUrl(jobId: string): string {
  return `${BACKEND_URL}/api/analysis/running/${jobId}/video`;
}
