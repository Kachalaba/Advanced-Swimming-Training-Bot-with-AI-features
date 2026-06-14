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
  quality?: {
    status: "pass" | "fail";
    pose_coverage: number;
    warnings?: string[];
  };
  video_path: string;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type AnalysisEvent = ProgressEvent | ResultEvent | ErrorEvent;

type AnalysisSport = "running" | "cycling";

async function uploadSportVideo(
  sport: AnalysisSport,
  file: File,
  fps = 30,
): Promise<{ jobId: string }> {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("fps", String(fps));
  const res = await fetch(`${BACKEND_URL}/api/analysis/${sport}`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  const data = (await res.json()) as { job_id: string };
  return { jobId: data.job_id };
}

async function saveSportAnalysis(
  sport: AnalysisSport,
  jobId: string,
  input: { athleteId?: number; athleteName?: string },
): Promise<{ sessionId: number }> {
  const res = await fetch(
    `${BACKEND_URL}/api/analysis/${sport}/${jobId}/save`,
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

function subscribeSportAnalysis(
  sport: AnalysisSport,
  jobId: string,
  onEvent: (event: AnalysisEvent) => void,
): () => void {
  const url = `${BACKEND_URL}/api/analysis/${sport}/${jobId}/events`;
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

function sportAnnotatedVideoUrl(
  sport: AnalysisSport,
  jobId: string,
): string {
  return `${BACKEND_URL}/api/analysis/${sport}/${jobId}/video`;
}

export function uploadRunningVideo(file: File, fps = 30) {
  return uploadSportVideo("running", file, fps);
}

export function saveRunningAnalysis(
  jobId: string,
  input: { athleteId?: number; athleteName?: string },
) {
  return saveSportAnalysis("running", jobId, input);
}

export function subscribeAnalysis(
  jobId: string,
  onEvent: (event: AnalysisEvent) => void,
) {
  return subscribeSportAnalysis("running", jobId, onEvent);
}

export function annotatedVideoUrl(jobId: string): string {
  return sportAnnotatedVideoUrl("running", jobId);
}

export function uploadCyclingVideo(file: File, fps = 30) {
  return uploadSportVideo("cycling", file, fps);
}

export function saveCyclingAnalysis(
  jobId: string,
  input: { athleteId?: number; athleteName?: string },
) {
  return saveSportAnalysis("cycling", jobId, input);
}

export function subscribeCyclingAnalysis(
  jobId: string,
  onEvent: (event: AnalysisEvent) => void,
) {
  return subscribeSportAnalysis("cycling", jobId, onEvent);
}

export function cyclingAnnotatedVideoUrl(jobId: string): string {
  return sportAnnotatedVideoUrl("cycling", jobId);
}
