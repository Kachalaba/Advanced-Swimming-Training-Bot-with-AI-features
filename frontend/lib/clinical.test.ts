import { afterEach, describe, expect, it, vi } from "vitest";

import {
  clinicalApi,
  finalizedProgressVisits,
  type ClinicalVisit,
} from "./clinical";
import { saveLiveRehabSession } from "./rehabilitation";

const visit = (
  overrides: Partial<ClinicalVisit> = {},
): ClinicalVisit => ({
  id: 1,
  rehabEpisodeId: 21,
  trainingSessionId: 91,
  visitedAt: "2026-06-13T12:00:00+00:00",
  captureSource: "live",
  preSessionNote: "",
  specialistObservation: "Reviewed.",
  captureQuality: "acceptable",
  captureQualityDetails: "",
  warningAcknowledged: false,
  status: "finalized",
  createdAt: "2026-06-13T12:00:00+00:00",
  updatedAt: "2026-06-13T12:00:00+00:00",
  ...overrides,
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("clinical domain", () => {
  it("keeps only finalized visits with usable capture quality", () => {
    expect(
      finalizedProgressVisits([
        visit({ id: 1, status: "draft" }),
        visit({ id: 2, captureQuality: "repeat_required" }),
        visit({ id: 3, captureQuality: "accepted_with_warning" }),
        visit({ id: 4, captureQuality: "acceptable" }),
      ]).map((item) => item.id),
    ).toEqual([3, 4]);
  });

  it("creates a patient using the camel-case API contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 11,
        athleteId: 7,
        displayName: "Patient A",
        affectedSide: "right",
        clinicalContext: "Shoulder mobility",
        precautions: "",
        status: "active",
        createdAt: "2026-06-13T12:00:00+00:00",
        updatedAt: "2026-06-13T12:00:00+00:00",
        athlete: { id: 7, name: "Patient A" },
        activeEpisode: null,
        latestVisit: null,
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await clinicalApi.createPatient({
      athleteId: 7,
      displayName: "Patient A",
      affectedSide: "right",
      clinicalContext: "Shoulder mobility",
      precautions: "",
    });

    expect(fetchMock.mock.calls[0][0]).toContain("/api/clinical/patients");
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      athleteId: 7,
      displayName: "Patient A",
      affectedSide: "right",
      clinicalContext: "Shoulder mobility",
      precautions: "",
    });
  });

  it("sends athlete ID when a clinical live analysis is saved", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: 91 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await saveLiveRehabSession("live-1", {
      athleteId: 7,
      athleteName: "Patient A",
    });

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      athlete_id: 7,
      athlete_name: "Patient A",
    });
  });
});
