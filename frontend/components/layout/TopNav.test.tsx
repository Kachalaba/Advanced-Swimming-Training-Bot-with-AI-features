import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  APP_LOCALE_CHANGE_EVENT,
  APP_LOCALE_STORAGE_KEY,
} from "@/lib/appLocale";

import { TopNav } from "./TopNav";

const mocks = vi.hoisted(() => ({
  pathname: "/running",
  me: vi.fn(),
  listAthletes: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => mocks.pathname,
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      me: mocks.me,
      listAthletes: mocks.listAthletes,
    },
  };
});

describe("TopNav", () => {
  beforeEach(() => {
    const storage = new Map<string, string>();
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => storage.get(key) ?? null,
        setItem: (key: string, value: string) => storage.set(key, value),
        removeItem: (key: string) => storage.delete(key),
        clear: () => storage.clear(),
      },
    });
    mocks.pathname = "/running";
    mocks.me.mockResolvedValue({
      id: "7",
      name: "Nikita",
      initials: "NK",
      handle: "@coach",
    });
    mocks.listAthletes.mockResolvedValue([
      { id: "7", name: "Nikita", initials: "NK", handle: "@coach" },
    ]);
    document.documentElement.lang = "en";
  });

  it("uses the English product shell by default", async () => {
    render(<TopNav />);

    expect(screen.getByRole("link", { name: "Running" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Switch language to English" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Search sessions and drills…")).toBeInTheDocument();
    expect((await screen.findAllByText("Nikita")).length).toBeGreaterThan(0);
  });

  it("marks the overview route as the active page", () => {
    mocks.pathname = "/";
    render(<TopNav />);

    expect(screen.getByRole("link", { name: "Overview" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("switches navigation chrome to Ukrainian and persists the choice", async () => {
    const user = userEvent.setup();
    render(<TopNav />);

    await user.click(
      screen.getByRole("button", {
        name: "Перемкнути мову на українську",
      }),
    );

    expect(window.localStorage.getItem(APP_LOCALE_STORAGE_KEY)).toBe("uk");
    expect(document.documentElement.lang).toBe("uk");
    expect(screen.getByRole("link", { name: "Біг" })).toBeInTheDocument();
    expect(screen.getByText("Пошук сесій і вправ…")).toBeInTheDocument();
  });

  it("syncs with the existing rehabilitation locale event", async () => {
    window.localStorage.setItem(APP_LOCALE_STORAGE_KEY, "uk");
    render(<TopNav />);

    expect(await screen.findByRole("link", { name: "Біг" })).toBeInTheDocument();

    window.dispatchEvent(
      new CustomEvent(APP_LOCALE_CHANGE_EVENT, { detail: "en" }),
    );

    await waitFor(() =>
      expect(
        screen.getByRole("link", { name: "Running" }),
      ).toBeInTheDocument(),
    );
    expect(document.documentElement.lang).toBe("en");
  });
});
