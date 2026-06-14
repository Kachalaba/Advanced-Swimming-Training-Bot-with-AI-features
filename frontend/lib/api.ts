import type { RehabProgressSession } from "./rehabProgress";

export type Athlete = {
  id: string;
  name: string;
  initials: string;
  handle?: string | null;
};

export type SessionSummary = {
  id: number;
  athlete_id: number;
  session_type: string;
  date: string;
  duration_sec: number;
  distance_m: number;
  reps: number;
  exercise_type: string;
  score: number;
  summary: string;
  has_video: boolean;
  artifact_download_url: string | null;
};

export type RehabProgressResponse = {
  athlete: Athlete;
  sessions: RehabProgressSession[];
  protocols: RehabProgressSession["protocol"][];
};

export type SportName = "swimming" | "running" | "cycling" | "dryland";

export type SportOverviewMetric = {
  key: string;
  label: string;
  value: number;
  unit: string;
  higher_is_better: boolean | null;
};

export type SportOverviewInsight = {
  code: string;
  level: string;
  title: string;
  detail: string;
};

export type SportOverviewSession = {
  id: number;
  date: string;
  duration_sec: number;
  score: number | null;
  summary: string;
  has_video: boolean;
  metrics: Record<string, SportOverviewMetric>;
  quality: Record<string, number>;
  insights: SportOverviewInsight[];
};

export type SportOverview = {
  athlete: Athlete;
  sport: SportName;
  total_sessions: number;
  latest_session_date: string | null;
  latest_score: number | null;
  headline_metrics: Record<string, SportOverviewMetric>;
  insights: SportOverviewInsight[];
  score_series: { date: string; value: number }[];
  sessions: SportOverviewSession[];
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export function backendAssetUrl(path: string): string {
  return path.startsWith("http") ? path : `${BACKEND_URL}${path}`;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  me: () => get<Athlete>("/api/athletes/me"),
  listAthletes: () => get<Athlete[]>("/api/athletes"),
  listSessions: (athleteId: string) =>
    get<SessionSummary[]>(`/api/athletes/${athleteId}/sessions`),
  sportOverview: (athleteId: string, sport: SportName) =>
    get<SportOverview>(
      `/api/athletes/${athleteId}/sports/${sport}/overview`,
    ),
  rehabilitationProgress: (athleteId: string) =>
    get<RehabProgressResponse>(
      `/api/athletes/${athleteId}/rehabilitation/progress`,
    ),
};
