export type Athlete = {
  id: string;
  name: string;
  initials: string;
  handle?: string | null;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  me: () => get<Athlete>("/api/athletes/me"),
  listAthletes: () => get<Athlete[]>("/api/athletes"),
};
