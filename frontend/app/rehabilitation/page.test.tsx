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

  it("defaults to Ukrainian and presents the clinical limitations prominently", () => {
    render(<RehabilitationPage />);

    expect(
      screen.getByRole("heading", { name: "Аналіз руху, ROM і відновлення" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Дослідницький прототип")).toBeInTheDocument();
    expect(
      screen.getByText(/не є медичним виробом/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/цільові кути налаштовує фахівець/i),
    ).toBeInTheDocument();
    expect(screen.getByTestId("live-workspace")).toHaveTextContent("uk");
  });

  it("switches the complete rehabilitation experience to English", async () => {
    const user = userEvent.setup();
    render(<RehabilitationPage />);

    await user.click(screen.getByRole("button", { name: "English" }));

    expect(
      screen.getByRole("heading", {
        name: "Movement, ROM and recovery analysis",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Research prototype")).toBeInTheDocument();
    expect(screen.getByText(/not a medical device/i)).toBeInTheDocument();
    expect(screen.getByTestId("live-workspace")).toHaveTextContent("en");
    expect(localStorage.getItem("sprint-ai-rehab-locale")).toBe("en");
  });
});
