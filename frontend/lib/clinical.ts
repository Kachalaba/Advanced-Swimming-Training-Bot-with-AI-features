import type { Athlete } from "./api";
import type { RehabProgressSession } from "./rehabProgress";
import type { RehabProtocol } from "./rehabilitation";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export type AffectedSide =
  | "left"
  | "right"
  | "bilateral"
  | "unspecified";
export type PatientStatus = "active" | "archived";
export type EpisodeStatus = "active" | "completed" | "archived";
export type CaptureSource = "live" | "upload";
export type CaptureQuality =
  | "acceptable"
  | "accepted_with_warning"
  | "repeat_required";
export type VisitStatus = "draft" | "finalized";

export type ClinicalAthlete = Pick<Athlete, "id" | "name">;

export type RehabEpisode = {
  id: number;
  patientProfileId: number;
  title: string;
  protocol: RehabProtocol;
  functionalGoal: string;
  targetLeftRom: number | null;
  targetRightRom: number | null;
  status: EpisodeStatus;
  startedAt: string;
  completedAt: string | null;
  createdAt: string;
  updatedAt: string;
};

export type ClinicalVisit = {
  id: number;
  rehabEpisodeId: number;
  trainingSessionId: number | null;
  visitedAt: string;
  captureSource: CaptureSource;
  preSessionNote: string;
  specialistObservation: string;
  captureQuality: CaptureQuality | null;
  captureQualityDetails: string;
  warningAcknowledged: boolean;
  status: VisitStatus;
  createdAt: string;
  updatedAt: string;
};

export type PatientProfile = {
  id: number;
  athleteId: number;
  displayName: string;
  affectedSide: AffectedSide;
  clinicalContext: string;
  precautions: string;
  status: PatientStatus;
  createdAt: string;
  updatedAt: string;
  athlete: ClinicalAthlete;
  activeEpisode: RehabEpisode | null;
  latestVisit: ClinicalVisit | null;
};

export type PatientDetail = PatientProfile & {
  episodes: RehabEpisode[];
};

export type ClinicalProgressObservation = RehabProgressSession & {
  visitId: number;
  trainingSessionId: number;
  captureQuality: Exclude<CaptureQuality, "repeat_required">;
  captureQualityDetails: string;
};

export type EpisodeDetail = RehabEpisode & {
  patient: PatientProfile;
  athlete: ClinicalAthlete;
  visits: ClinicalVisit[];
  progress: ClinicalProgressObservation[];
};

export type CreatePatientInput = {
  athleteId: number;
  displayName: string;
  affectedSide: AffectedSide;
  clinicalContext: string;
  precautions: string;
};

export type UpdatePatientInput = Partial<
  Omit<CreatePatientInput, "athleteId">
>;

export type CreateEpisodeInput = {
  title: string;
  protocol: RehabProtocol;
  functionalGoal: string;
  targetLeftRom: number | null;
  targetRightRom: number | null;
};

export type UpdateEpisodeInput = Partial<
  Omit<CreateEpisodeInput, "protocol">
> & {
  status?: EpisodeStatus;
};

export type CreateVisitInput = {
  captureSource: CaptureSource;
  preSessionNote: string;
};

export type UpdateVisitInput = {
  trainingSessionId?: number;
  specialistObservation?: string;
  captureQuality?: CaptureQuality;
  captureQualityDetails?: string;
  warningAcknowledged?: boolean;
};

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      // Keep the status when the backend did not provide JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

function json(method: string, body?: object): RequestInit {
  return {
    method,
    body: body === undefined ? undefined : JSON.stringify(body),
  };
}

export const clinicalApi = {
  listPatients: (includeArchived = false) =>
    request<PatientProfile[]>(
      `/api/clinical/patients?includeArchived=${includeArchived}`,
    ),
  createPatient: (input: CreatePatientInput) =>
    request<PatientProfile>(
      "/api/clinical/patients",
      json("POST", input),
    ),
  getPatient: (patientId: number | string) =>
    request<PatientDetail>(`/api/clinical/patients/${patientId}`),
  updatePatient: (
    patientId: number | string,
    input: UpdatePatientInput,
  ) =>
    request<PatientProfile>(
      `/api/clinical/patients/${patientId}`,
      json("PATCH", input),
    ),
  archivePatient: (patientId: number | string) =>
    request<PatientProfile>(
      `/api/clinical/patients/${patientId}/archive`,
      json("POST"),
    ),
  createEpisode: (
    patientId: number | string,
    input: CreateEpisodeInput,
  ) =>
    request<RehabEpisode>(
      `/api/clinical/patients/${patientId}/episodes`,
      json("POST", input),
    ),
  getEpisode: (episodeId: number | string) =>
    request<EpisodeDetail>(`/api/clinical/episodes/${episodeId}`),
  updateEpisode: (
    episodeId: number | string,
    input: UpdateEpisodeInput,
  ) =>
    request<RehabEpisode>(
      `/api/clinical/episodes/${episodeId}`,
      json("PATCH", input),
    ),
  archiveEpisode: (episodeId: number | string) =>
    request<RehabEpisode>(
      `/api/clinical/episodes/${episodeId}/archive`,
      json("POST"),
    ),
  createVisit: (
    episodeId: number | string,
    input: CreateVisitInput,
  ) =>
    request<ClinicalVisit>(
      `/api/clinical/episodes/${episodeId}/visits`,
      json("POST", input),
    ),
  getVisit: (visitId: number | string) =>
    request<ClinicalVisit>(`/api/clinical/visits/${visitId}`),
  updateVisit: (
    visitId: number | string,
    input: UpdateVisitInput,
  ) =>
    request<ClinicalVisit>(
      `/api/clinical/visits/${visitId}`,
      json("PATCH", input),
    ),
  finalizeVisit: (visitId: number | string) =>
    request<ClinicalVisit>(
      `/api/clinical/visits/${visitId}/finalize`,
      json("POST"),
    ),
};

export function finalizedProgressVisits(
  visits: ClinicalVisit[],
): ClinicalVisit[] {
  return visits.filter(
    (visit) =>
      visit.status === "finalized" &&
      (visit.captureQuality === "acceptable" ||
        visit.captureQuality === "accepted_with_warning"),
  );
}
