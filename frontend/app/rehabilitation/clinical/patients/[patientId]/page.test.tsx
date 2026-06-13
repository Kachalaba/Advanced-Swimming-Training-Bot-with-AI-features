import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EpisodeDetail, PatientDetail } from "@/lib/clinical";

import PatientDetailPage from "./page";

const mocks = vi.hoisted(() => ({
  getPatient: vi.fn(),
  getEpisode: vi.fn(),
  createEpisode: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ patientId: "11" }),
}));

vi.mock("@/lib/clinical", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/clinical")>();
  return {
    ...original,
    clinicalApi: {
      ...original.clinicalApi,
      getPatient: mocks.getPatient,
      getEpisode: mocks.getEpisode,
      createEpisode: mocks.createEpisode,
    },
  };
});

const episode = {
  id: 21,
  patientProfileId: 11,
  title: "Shoulder recovery",
  protocol: "shoulder_flexion" as const,
  functionalGoal: "Reach an overhead shelf",
  targetLeftRom: 150,
  targetRightRom: 140,
  status: "active" as const,
  startedAt: "2026-06-01T10:00:00+00:00",
  completedAt: null,
  createdAt: "2026-06-01T10:00:00+00:00",
  updatedAt: "2026-06-01T10:00:00+00:00",
};

const patient: PatientDetail = {
  id: 11,
  athleteId: 7,
  displayName: "Patient A",
  affectedSide: "right",
  clinicalContext: "Post-operative shoulder mobility",
  precautions: "Stop on sharp pain",
  status: "active",
  createdAt: "2026-06-01T10:00:00+00:00",
  updatedAt: "2026-06-01T10:00:00+00:00",
  athlete: { id: 7, name: "Patient A" },
  activeEpisode: episode,
  latestVisit: null,
  episodes: [episode],
};

const episodeDetail: EpisodeDetail = {
  ...episode,
  patient,
  athlete: patient.athlete,
  visits: [
    {
      id: 31,
      rehabEpisodeId: 21,
      trainingSessionId: 91,
      visitedAt: "2026-06-01T10:00:00+00:00",
      captureSource: "live",
      preSessionNote: "Stable",
      specialistObservation: "Baseline reviewed.",
      captureQuality: "acceptable",
      captureQualityDetails: "",
      warningAcknowledged: false,
      status: "finalized",
      createdAt: "2026-06-01T10:00:00+00:00",
      updatedAt: "2026-06-01T10:00:00+00:00",
    },
  ],
  progress: [
    {
      id: 91,
      visitId: 31,
      trainingSessionId: 91,
      date: "2026-06-01T10:00:00+00:00",
      protocol: "shoulder_flexion",
      leftRom: 120,
      rightRom: 108,
      symmetry: 84,
      repetitions: 2,
      completionScore: 72,
      validFrames: 20,
      hasVideo: false,
      captureQuality: "acceptable",
      captureQualityDetails: "",
    },
    {
      id: 92,
      visitId: 32,
      trainingSessionId: 92,
      date: "2026-06-12T10:00:00+00:00",
      protocol: "shoulder_flexion",
      leftRom: 142,
      rightRom: 130,
      symmetry: 91,
      repetitions: 4,
      completionScore: 88,
      validFrames: 30,
      hasVideo: true,
      captureQuality: "accepted_with_warning",
      captureQualityDetails: "Camera tilt",
    },
  ],
};

describe("PatientDetailPage", () => {
  beforeEach(() => {
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: () => null,
        setItem: vi.fn(),
      },
    });
    mocks.getPatient.mockReset();
    mocks.getEpisode.mockReset();
    mocks.createEpisode.mockReset();
    mocks.getPatient.mockResolvedValue(patient);
    mocks.getEpisode.mockResolvedValue(episodeDetail);
    mocks.createEpisode.mockResolvedValue(episode);
  });

  it("shows context, active episode, progress, and finalized visits", async () => {
    render(<PatientDetailPage />);

    expect(
      await screen.findByRole("heading", { name: "Patient A" }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Reach an overhead shelf").length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("142°").length).toBeGreaterThan(0);
    expect(screen.getByText("Baseline reviewed.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Новий візит" })).toHaveAttribute(
      "href",
      "/rehabilitation/clinical/episodes/21/visits/new",
    );
  });

  it("creates an episode when the patient has no active course", async () => {
    const user = userEvent.setup();
    mocks.getPatient.mockResolvedValue({
      ...patient,
      activeEpisode: null,
      episodes: [],
    });
    render(<PatientDetailPage />);

    const createButtons = await screen.findAllByRole("button", {
      name: "Створити курс",
    });
    await user.click(createButtons[0]);
    await user.type(
      screen.getByRole("textbox", { name: "Назва курсу" }),
      "Shoulder recovery",
    );
    await user.type(
      screen.getByRole("textbox", { name: "Функціональна ціль" }),
      "Reach overhead",
    );
    await user.click(
      screen.getByRole("button", { name: "Створити курс" }),
    );

    await waitFor(() =>
      expect(mocks.createEpisode).toHaveBeenCalledWith(
        11,
        expect.objectContaining({
          protocol: "shoulder_flexion",
          functionalGoal: "Reach overhead",
        }),
      ),
    );
  });
});
