const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export type SwimmingConfidenceLevel = "high" | "medium" | "insufficient";
export type SwimmingZoneStatus =
  | "good"
  | "needs_attention"
  | "insufficient_data";

export type SwimmingEvidence = {
  cycle_id: string;
  start_sec: number;
  peak_sec: number;
  end_sec: number;
};

export type SwimmingCycle = {
  id: string;
  start_frame: number;
  peak_frame: number;
  end_frame: number;
  start_sec: number;
  peak_sec: number;
  end_sec: number;
  quality: number;
  complete: boolean;
};

export type SwimmingZone = {
  id: "body_position" | "rotation" | "catch" | "breathing" | "kick";
  status: SwimmingZoneStatus;
  score: number | null;
  confidence: number;
  confidence_level: SwimmingConfidenceLevel;
  metrics: Record<string, number | string>;
  evidence: SwimmingEvidence[];
  issues: Array<{
    issue_code: string;
    confirming_cycles: number;
    confidence: number;
    impact: number;
    evidence: SwimmingEvidence[];
  }>;
};

export type SwimmingPrimaryIssue = {
  zone_id: SwimmingZone["id"];
  issue_code: string;
  title: string;
  why_it_matters: string;
  confidence: number;
  confidence_level: SwimmingConfidenceLevel;
  confirming_cycles: number;
  evidence: SwimmingEvidence[];
};

export type SwimmingPrescription = {
  drill: {
    name: string;
    purpose: string;
    execution: string;
    common_mistake: string;
    success_cue: string;
  };
  mini_set: {
    title: string;
    repetitions: number;
    distance_m: number;
    rest_sec: number;
    intensity: string;
    focus: string;
  };
};

export type SwimmingProgressEvent = {
  type: "progress";
  stage:
    | "quality_gate"
    | "tracking"
    | "waterline"
    | "pose"
    | "cycles"
    | "technique"
    | "coaching"
    | "rendering"
    | "completed";
  pct: number;
  label: string;
};

export type SwimmingResultEvent = {
  type: "result";
  analysis_type: "swimming_freestyle_side";
  contract_version: string;
  quality: {
    status: "pass" | "partial";
    warnings: string[];
  };
  coverage: {
    available_zones: number;
    total_zones: number;
  };
  overall_score: number | null;
  cycles: SwimmingCycle[];
  zones: SwimmingZone[];
  primary_issue: SwimmingPrimaryIssue | null;
  prescription: SwimmingPrescription | null;
  frames_total: number;
  frames_with_pose: number;
  video_path: string;
};

export type SwimmingErrorEvent = {
  type: "error";
  code: string;
  message: string;
  reshoot_guidance?: string;
};

export type SwimmingAnalysisEvent =
  | SwimmingProgressEvent
  | SwimmingResultEvent
  | SwimmingErrorEvent;

export type SwimmingJobStatus = {
  id: string;
  status: "queued" | "running" | "done" | "error";
  result: SwimmingResultEvent | SwimmingErrorEvent | null;
  error: string | null;
  event_count: number;
  saved_session_id: number | null;
};

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep the HTTP status for non-JSON responses.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function uploadSwimmingVideo(
  file: File,
  fps?: number,
): Promise<{ jobId: string }> {
  const body = new FormData();
  body.append("video", file);
  if (fps !== undefined) body.append("fps", String(fps));
  const response = await fetch(`${BACKEND_URL}/api/analysis/swimming`, {
    method: "POST",
    body,
  });
  const data = await readJson<{ job_id: string }>(response);
  return { jobId: data.job_id };
}

export async function getSwimmingJob(
  jobId: string,
): Promise<SwimmingJobStatus> {
  const response = await fetch(
    `${BACKEND_URL}/api/analysis/swimming/${jobId}`,
  );
  return readJson<SwimmingJobStatus>(response);
}

export function subscribeSwimmingAnalysis(
  jobId: string,
  onEvent: (event: SwimmingAnalysisEvent) => void,
): () => void {
  const source = new EventSource(
    `${BACKEND_URL}/api/analysis/swimming/${jobId}/events`,
  );
  let closed = false;

  const close = () => {
    if (closed) return;
    closed = true;
    source.close();
  };

  source.onmessage = (message) => {
    let event: SwimmingAnalysisEvent;
    try {
      event = JSON.parse(message.data) as SwimmingAnalysisEvent;
    } catch {
      return;
    }
    onEvent(event);
    if (event.type === "result" || event.type === "error") close();
  };

  source.onerror = async () => {
    close();
    try {
      const job = await getSwimmingJob(jobId);
      if (job.result?.type === "result" || job.result?.type === "error") {
        onEvent(job.result);
        return;
      }
    } catch {
      // The connection error below is the actionable result.
    }
    onEvent({
      type: "error",
      code: "connection_lost",
      message: "Connection to the analysis job was lost.",
      reshoot_guidance: "Reload this page to reconnect to the current job.",
    });
  };

  return close;
}

export function swimmingVideoUrl(jobId: string): string {
  return `${BACKEND_URL}/api/analysis/swimming/${jobId}/video`;
}

export async function saveSwimmingAnalysis(
  jobId: string,
  input: { athleteId?: number; athleteName?: string },
): Promise<{ sessionId: number }> {
  const response = await fetch(
    `${BACKEND_URL}/api/analysis/swimming/${jobId}/save`,
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
  const data = await readJson<{ session_id: number }>(response);
  return { sessionId: data.session_id };
}
