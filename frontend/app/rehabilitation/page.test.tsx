import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RehabAnalysisSnapshot } from "@/components/rehabilitation/rehabHandoff";

import RehabilitationPage from "./page";

vi.mock("@/components/rehabilitation/LiveRehabWorkspace", () => ({
  LiveRehabWorkspace: ({
    locale,
    onAnalysisChange,
  }: {
    locale: string;
    onAnalysisChange?: (snapshot: RehabAnalysisSnapshot | null) => void;
  }) => (
    <div data-testid="live-workspace">
      {locale}
      <button
        type="button"
        onClick={() =>
          onAnalysisChange?.({
            report: {
              protocol: "shoulder_flexion",
              total_correct_reps: 2,
              completion_score: 86,
              target_metrics: {
                left: { rom: 154 },
                right: { rom: 136 },
              },
              symmetry: { asymmetry_index: 12, score: 88 },
            },
            confidence: 0.92,
            poseCoverage: null,
          })
        }
      >
        emit live result
      </button>
    </div>
  ),
}));

vi.mock("@/components/rehabilitation/RehabUploader", () => ({
  RehabUploader: ({
    locale,
    onAnalysisChange,
  }: {
    locale: string;
    onAnalysisChange?: (snapshot: RehabAnalysisSnapshot | null) => void;
  }) => (
    <div data-testid="rehab-uploader">
      {locale}
      <button
        type="button"
        onClick={() =>
          onAnalysisChange?.({
            report: {
              protocol: "shoulder_flexion",
              total_correct_reps: 3,
              completion_score: 84,
              target_metrics: {
                left: { rom: 150 },
                right: { rom: 132 },
              },
              symmetry: { asymmetry_index: 12, score: 88 },
            },
            confidence: null,
            poseCoverage: 75,
          })
        }
      >
        emit upload result
      </button>
    </div>
  ),
}));

describe("RehabilitationPage", () => {
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
  });

  it("opens with a polished Ukrainian demo and keeps limitations available", () => {
    render(<RehabilitationPage />);

    expect(
      screen.getByRole("heading", {
        name: "Рух видно. Докази — у кожному кадрі.",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Дослідницький прототип")).toBeInTheDocument();
    expect(screen.getByTestId("rehab-body-figure")).toBeInTheDocument();
    expect(screen.queryByTestId("live-workspace")).not.toBeInTheDocument();
    expect(screen.getByText(/не є медичним виробом/i)).not.toBeVisible();
  });

  it("switches the complete demo experience to English", async () => {
    const user = userEvent.setup();
    render(<RehabilitationPage />);

    await user.click(screen.getByRole("button", { name: "English" }));

    expect(
      screen.getByRole("heading", {
        name: "See the movement. Prove it frame by frame.",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Research prototype")).toBeInTheDocument();
    expect(screen.getByText(/simulated demo session/i)).toBeInTheDocument();
    expect(localStorage.getItem("sprint-ai-rehab-locale")).toBe("en");
  });

  it("keeps live and upload workflows one click away", async () => {
    const user = userEvent.setup();
    render(<RehabilitationPage />);

    await user.click(screen.getByRole("button", { name: "Камера наживо" }));
    expect(screen.getByTestId("live-workspace")).toHaveTextContent("uk");

    await user.click(
      screen.getByRole("button", { name: "Завантажити відео" }),
    );
    expect(screen.getByTestId("rehab-uploader")).toHaveTextContent("uk");
  });

  it("opens presentation and report from the completed demo", async () => {
    const user = userEvent.setup();
    render(<RehabilitationPage />);

    await user.click(
      screen.getByRole("button", { name: "Режим презентації" }),
    );
    expect(
      screen.getByRole("dialog", { name: "Режим презентації" }),
    ).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Відкрити клінічний звіт" }),
    );
    expect(
      screen.getByRole("dialog", { name: "Клінічний звіт про рух" }),
    ).toBeInTheDocument();
  });

  it("links to longitudinal patient progress independently of analysis state", () => {
    render(<RehabilitationPage />);

    expect(
      screen.getByRole("link", { name: "Прогрес пацієнта" }),
    ).toHaveAttribute("href", "/rehabilitation/progress");
  });

  it("links to the clinical specialist workspace", () => {
    render(<RehabilitationPage />);

    expect(
      screen.getByRole("link", { name: "Кабінет фахівця" }),
    ).toHaveAttribute("href", "/rehabilitation/clinical");
  });

  it("gates real handoff actions until analysis results exist", async () => {
    const user = userEvent.setup();
    render(<RehabilitationPage />);

    await user.click(screen.getByRole("button", { name: "Камера наживо" }));
    expect(
      screen.getByRole("button", { name: "Режим презентації" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Клінічний звіт" }),
    ).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "emit live result" }));
    expect(
      screen.getByRole("button", { name: "Режим презентації" }),
    ).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Клінічний звіт" }),
    ).toBeEnabled();
  });

  it("clears a real handoff when the protocol changes", async () => {
    const user = userEvent.setup();
    render(<RehabilitationPage />);

    await user.click(screen.getByRole("button", { name: "Камера наживо" }));
    await user.click(screen.getByRole("button", { name: "emit live result" }));
    expect(
      screen.getByRole("button", { name: "Клінічний звіт" }),
    ).toBeEnabled();

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Протокол" }),
      "knee_extension",
    );
    expect(
      screen.getByRole("button", { name: "Клінічний звіт" }),
    ).toBeDisabled();
  });
});
