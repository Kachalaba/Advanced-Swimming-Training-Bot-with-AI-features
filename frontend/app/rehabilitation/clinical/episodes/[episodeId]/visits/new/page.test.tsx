import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  ClinicalVisit,
  EpisodeDetail,
} from "@/lib/clinical";
import type { LiveRehabUpdate } from "@/lib/rehabilitation";

import NewClinicalVisitPage from "./page";

const mocks = vi.hoisted(() => ({
  getEpisode: vi.fn(),
  createVisit: vi.fn(),
  updateVisit: vi.fn(),
  finalizeVisit: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ episodeId: "21" }),
}));

vi.mock("@/lib/clinical", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/clinical")>();
  return {
    ...original,
    clinicalApi: {
      ...original.clinicalApi,
      getEpisode: mocks.getEpisode,
      createVisit: mocks.createVisit,
      updateVisit: mocks.updateVisit,
      finalizeVisit: mocks.finalizeVisit,
    },
  };
});

const point = { x: 0.5, y: 0.5 };
const report = {
  protocol: "shoulder_flexion" as const,
  total_correct_reps: 3,
  completion_score: 84,
  target_metrics: {
    left: { rom: 150 },
    right: { rom: 132 },
  },
  symmetry: { asymmetry_index: 12, score: 88 },
};

function liveUpdate(
  overrides: Partial<LiveRehabUpdate> = {},
): LiveRehabUpdate {
  return {
    session_id: "live-1",
    sequence: 1,
    pose_detected: true,
    landmarks: {
      left_elbow: point,
      left_shoulder: point,
      left_hip: point,
      right_elbow: point,
      right_shoulder: point,
      right_hip: point,
    },
    posture: { available: false },
    camera_level: {
      angle_deg: 0,
      confidence: 0.92,
      status: "level",
      relative: true,
    },
    report,
    ...overrides,
  };
}

vi.mock("@/components/rehabilitation/LiveRehabWorkspace", () => ({
  LiveRehabWorkspace: ({
    onLiveUpdate,
    onAnalysisChange,
    onSessionSaved,
  }: {
    onLiveUpdate?: (update: LiveRehabUpdate) => void;
    onAnalysisChange?: (snapshot: {
      report: typeof report;
      confidence: number;
      poseCoverage: null;
    }) => void;
    onSessionSaved?: (sessionId: number) => void;
  }) => (
    <div>
      <button type="button" onClick={() => onLiveUpdate?.(liveUpdate())}>
        Publish ready
      </button>
      <button
        type="button"
        onClick={() =>
          onLiveUpdate?.(
            liveUpdate({
              pose_detected: false,
              landmarks: {},
            }),
          )
        }
      >
        Publish blocked
      </button>
      <button
        type="button"
        onClick={() =>
          onLiveUpdate?.(
            liveUpdate({
              camera_level: {
                angle_deg: 4.5,
                confidence: 0.92,
                status: "adjust",
                relative: true,
              },
            }),
          )
        }
      >
        Publish warning
      </button>
      <button
        type="button"
        onClick={() =>
          onAnalysisChange?.({
            report,
            confidence: 0.92,
            poseCoverage: null,
          })
        }
      >
        Publish analysis
      </button>
      <button type="button" onClick={() => onSessionSaved?.(91)}>
        Publish saved
      </button>
    </div>
  ),
}));

vi.mock("@/components/rehabilitation/RehabUploader", () => ({
  RehabUploader: () => <div>Upload analyzer</div>,
}));

const draftVisit: ClinicalVisit = {
  id: 31,
  rehabEpisodeId: 21,
  trainingSessionId: null,
  visitedAt: "2026-06-13T10:00:00+00:00",
  captureSource: "live",
  preSessionNote: "No pain at rest",
  specialistObservation: "",
  captureQuality: null,
  captureQualityDetails: "",
  warningAcknowledged: false,
  status: "draft",
  createdAt: "2026-06-13T10:00:00+00:00",
  updatedAt: "2026-06-13T10:00:00+00:00",
};

const episodeDetail: EpisodeDetail = {
  id: 21,
  patientProfileId: 11,
  title: "Shoulder recovery",
  protocol: "shoulder_flexion",
  functionalGoal: "Reach an overhead shelf",
  targetLeftRom: 150,
  targetRightRom: 140,
  status: "active",
  startedAt: "2026-06-01T10:00:00+00:00",
  completedAt: null,
  createdAt: "2026-06-01T10:00:00+00:00",
  updatedAt: "2026-06-01T10:00:00+00:00",
  athlete: { id: 7, name: "Patient A" },
  patient: {
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
    activeEpisode: null,
    latestVisit: null,
  },
  visits: [],
  progress: [
    {
      id: 71,
      visitId: 30,
      trainingSessionId: 71,
      date: "2026-06-01T10:00:00+00:00",
      protocol: "shoulder_flexion",
      leftRom: 132,
      rightRom: 114,
      symmetry: 82,
      repetitions: 2,
      completionScore: 72,
      validFrames: 24,
      hasVideo: false,
      captureQuality: "acceptable",
      captureQualityDetails: "",
    },
  ],
};

describe("NewClinicalVisitPage", () => {
  beforeEach(() => {
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: () => "en",
        setItem: vi.fn(),
      },
    });
    mocks.getEpisode.mockReset();
    mocks.createVisit.mockReset();
    mocks.updateVisit.mockReset();
    mocks.finalizeVisit.mockReset();
    mocks.getEpisode.mockResolvedValue(episodeDetail);
    mocks.createVisit.mockResolvedValue(draftVisit);
    mocks.updateVisit.mockImplementation(
      async (_id: number, values: Partial<ClinicalVisit>) => ({
        ...draftVisit,
        ...values,
      }),
    );
    mocks.finalizeVisit.mockResolvedValue({
      ...draftVisit,
      trainingSessionId: 91,
      specialistObservation: "ROM remains asymmetric.",
      captureQuality: "acceptable",
      status: "finalized",
    });
  });

  it("creates one draft and advances through the complete visit", async () => {
    const user = userEvent.setup();
    render(<NewClinicalVisitPage />);

    await user.type(
      await screen.findByRole("textbox", { name: "Pre-session status" }),
      "No pain at rest",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(mocks.createVisit).toHaveBeenCalledWith(21, {
      captureSource: "live",
      preSessionNote: "No pain at rest",
    });

    await user.click(screen.getByRole("button", { name: "Publish ready" }));
    await user.click(screen.getByRole("button", { name: "Start analysis" }));
    await user.click(screen.getByRole("button", { name: "Publish analysis" }));
    await user.click(screen.getByRole("button", { name: "Publish saved" }));
    await user.click(screen.getByRole("button", { name: "Review result" }));

    expect(screen.getByText("Baseline comparison")).toBeInTheDocument();
    await user.type(
      screen.getByRole("textbox", { name: "Specialist observation" }),
      "ROM remains asymmetric.",
    );
    await user.click(screen.getByRole("button", { name: "Finalize visit" }));

    await waitFor(() => expect(mocks.finalizeVisit).toHaveBeenCalledWith(31));
    expect(mocks.createVisit).toHaveBeenCalledOnce();
    expect(await screen.findByText("Visit finalized")).toBeInTheDocument();
  });

  it("does not start analysis while readiness is blocked", async () => {
    const user = userEvent.setup();
    render(<NewClinicalVisitPage />);
    await user.click(
      await screen.findByRole("button", { name: "Continue" }),
    );
    await user.click(screen.getByRole("button", { name: "Publish blocked" }));

    expect(
      screen.getByRole("button", { name: "Start analysis" }),
    ).toBeDisabled();
  });

  it("requires acknowledgement before proceeding with a readiness warning", async () => {
    const user = userEvent.setup();
    render(<NewClinicalVisitPage />);
    await user.click(
      await screen.findByRole("button", { name: "Continue" }),
    );
    await user.click(screen.getByRole("button", { name: "Publish warning" }));
    expect(
      screen.getByRole("button", { name: "Start analysis" }),
    ).toBeDisabled();

    await user.click(
      screen.getByRole("checkbox", {
        name: "I acknowledge this measurement limitation",
      }),
    );
    expect(
      screen.getByRole("button", { name: "Start analysis" }),
    ).toBeEnabled();
  });
});
