import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RehabProgressResponse } from "@/lib/api";

import RehabProgressPage from "./page";

const mocks = vi.hoisted(() => ({
  listAthletes: vi.fn(),
  rehabilitationProgress: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    api: {
      ...original.api,
      listAthletes: mocks.listAthletes,
      rehabilitationProgress: mocks.rehabilitationProgress,
    },
  };
});

const progress: RehabProgressResponse = {
  athlete: {
    id: "3",
    name: "CASE-024",
    initials: "C",
  },
  protocols: ["shoulder_flexion", "knee_extension"],
  sessions: [
    {
      id: 1,
      date: "2026-06-01T10:00:00",
      protocol: "shoulder_flexion",
      leftRom: 120,
      rightRom: 108,
      symmetry: 84,
      repetitions: 2,
      completionScore: 72,
      validFrames: 20,
      hasVideo: false,
    },
    {
      id: 2,
      date: "2026-06-05T10:00:00",
      protocol: "knee_extension",
      leftRom: 92,
      rightRom: 94,
      symmetry: 96,
      repetitions: 4,
      completionScore: 88,
      validFrames: null,
      hasVideo: true,
    },
    {
      id: 3,
      date: "2026-06-12T10:00:00",
      protocol: "shoulder_flexion",
      leftRom: 146,
      rightRom: 137,
      symmetry: 94,
      repetitions: 5,
      completionScore: 92,
      validFrames: 32,
      hasVideo: true,
    },
  ],
};

describe("RehabProgressPage", () => {
  beforeEach(() => {
    const storage = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        clear: () => storage.clear(),
        getItem: (key: string) => storage.get(key) ?? null,
        removeItem: (key: string) => storage.delete(key),
        setItem: (key: string, value: string) => storage.set(key, value),
      },
    });
    mocks.listAthletes.mockReset();
    mocks.rehabilitationProgress.mockReset();
    mocks.listAthletes.mockResolvedValue([
      { id: "3", name: "CASE-024", initials: "C" },
      { id: "4", name: "CASE-031", initials: "C" },
    ]);
    mocks.rehabilitationProgress.mockResolvedValue(progress);
  });

  it("renders baseline versus current progress from persisted sessions", async () => {
    render(<RehabProgressPage />);

    expect(
      await screen.findByRole("heading", { name: "Прогрес пацієнта" }),
    ).toBeInTheDocument();
    expect(await screen.findByText(/Базова сесія:/)).toBeInTheDocument();
    expect(screen.getByText(/Поточна сесія:/)).toBeInTheDocument();
    expect(screen.getByText("+26°")).toBeInTheDocument();
    expect(screen.getByText("+10 в.п.")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /реабілітаційні сесії/i }),
    ).toBeInTheDocument();
  });

  it("switches protocol, athlete, and locale without mixing sessions", async () => {
    const user = userEvent.setup();
    render(<RehabProgressPage />);

    await waitFor(() =>
      expect(mocks.rehabilitationProgress).toHaveBeenCalledWith("3"),
    );
    await screen.findByText(/Базова сесія:/);
    await user.click(
      await screen.findByRole("button", { name: /Розгинання коліна/ }),
    );
    expect(
      screen.getByText(/потрібна ще одна сумісна сесія/i),
    ).toBeInTheDocument();

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Пацієнт" }),
      "4",
    );
    await waitFor(() =>
      expect(mocks.rehabilitationProgress).toHaveBeenLastCalledWith("4"),
    );

    await user.click(screen.getByRole("button", { name: "English" }));
    expect(
      screen.getByRole("heading", { name: "Patient progress" }),
    ).toBeInTheDocument();
    expect(localStorage.getItem("sprint-ai-rehab-locale")).toBe("en");
  });

  it("shows an empty state and retries a failed request", async () => {
    const user = userEvent.setup();
    mocks.rehabilitationProgress
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({
        ...progress,
        protocols: [],
        sessions: [],
      });

    render(<RehabProgressPage />);

    expect(
      await screen.findByText("Не вдалося завантажити прогрес"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Спробувати ще раз" }));

    expect(
      await screen.findByText("Ще немає збережених реабілітаційних сесій"),
    ).toBeInTheDocument();
  });
});
