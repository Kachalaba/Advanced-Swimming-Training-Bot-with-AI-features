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
};
