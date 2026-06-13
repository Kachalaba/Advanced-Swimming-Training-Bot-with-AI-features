import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { PatientProfile } from "@/lib/clinical";

import ClinicalWorkspacePage from "./page";

const mocks = vi.hoisted(() => ({
  listAthletes: vi.fn(),
  listPatients: vi.fn(),
  createPatient: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    api: {
      ...original.api,
      listAthletes: mocks.listAthletes,
    },
  };
});

vi.mock("@/lib/clinical", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/clinical")>();
  return {
    ...original,
    clinicalApi: {
      ...original.clinicalApi,
      listPatients: mocks.listPatients,
      createPatient: mocks.createPatient,
    },
  };
});

const patient: PatientProfile = {
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
  activeEpisode: {
    id: 21,
    patientProfileId: 11,
    title: "Shoulder recovery",
    protocol: "shoulder_flexion",
    functionalGoal: "Reach overhead",
    targetLeftRom: 150,
    targetRightRom: 140,
    status: "active",
    startedAt: "2026-06-13T12:00:00+00:00",
    completedAt: null,
    createdAt: "2026-06-13T12:00:00+00:00",
    updatedAt: "2026-06-13T12:00:00+00:00",
  },
  latestVisit: null,
};

describe("ClinicalWorkspacePage", () => {
  beforeEach(() => {
    const storage = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => storage.get(key) ?? null,
        setItem: (key: string, value: string) => storage.set(key, value),
      },
    });
    mocks.listAthletes.mockReset();
    mocks.listPatients.mockReset();
    mocks.createPatient.mockReset();
    mocks.listAthletes.mockResolvedValue([
      { id: "7", name: "Patient A", initials: "PA" },
      { id: "8", name: "Patient B", initials: "PB" },
    ]);
    mocks.listPatients.mockResolvedValue([patient]);
    mocks.createPatient.mockResolvedValue({
      ...patient,
      id: 12,
      athleteId: 8,
      displayName: "Patient B",
      athlete: { id: 8, name: "Patient B" },
      activeEpisode: null,
    });
  });

  it("renders active patient cards from the clinical roster", async () => {
    render(<ClinicalWorkspacePage />);

    expect(
      await screen.findByRole("heading", { name: "Кабінет фахівця" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Patient A")).toBeInTheDocument();
    expect(screen.getByText("Shoulder recovery")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Відкрити картку" }),
    ).toHaveAttribute("href", "/rehabilitation/clinical/patients/11");
  });

  it("creates a clinical profile from an existing athlete", async () => {
    const user = userEvent.setup();
    render(<ClinicalWorkspacePage />);

    await screen.findByText("Patient A");
    await user.click(screen.getByRole("button", { name: "Новий пацієнт" }));
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Пов'язаний спортсмен" }),
      "8",
    );
    await user.clear(screen.getByRole("textbox", { name: "Ім'я для відображення" }));
    await user.type(
      screen.getByRole("textbox", { name: "Ім'я для відображення" }),
      "Patient B",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Сторона обмеження" }),
      "left",
    );
    await user.type(
      screen.getByRole("textbox", { name: "Клінічний контекст" }),
      "Knee mobility",
    );
    await user.click(screen.getByRole("button", { name: "Створити профіль" }));

    await waitFor(() =>
      expect(mocks.createPatient).toHaveBeenCalledWith({
        athleteId: 8,
        displayName: "Patient B",
        affectedSide: "left",
        clinicalContext: "Knee mobility",
        precautions: "",
      }),
    );
    expect(await screen.findByText("Patient B")).toBeInTheDocument();
  });

  it("loads archived patients only when the archive filter is selected", async () => {
    const user = userEvent.setup();
    render(<ClinicalWorkspacePage />);

    await screen.findByText("Patient A");
    await user.click(screen.getByRole("button", { name: "Архів" }));

    await waitFor(() =>
      expect(mocks.listPatients).toHaveBeenLastCalledWith(true),
    );
  });
});
