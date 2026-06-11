import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import RehabilitationPage from "./page";

vi.mock("@/components/rehabilitation/LiveRehabWorkspace", () => ({
  LiveRehabWorkspace: ({ locale }: { locale: string }) => (
    <div data-testid="live-workspace">{locale}</div>
  ),
}));

vi.mock("@/components/rehabilitation/RehabUploader", () => ({
  RehabUploader: ({ locale }: { locale: string }) => (
    <div data-testid="rehab-uploader">{locale}</div>
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
});
