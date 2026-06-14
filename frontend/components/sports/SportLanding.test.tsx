import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SportLanding } from "./SportLanding";

describe("SportLanding", () => {
  const scrollIntoView = vi.fn();

  beforeEach(() => {
    scrollIntoView.mockClear();
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
    });
  });

  it("moves both upload actions to the uploader and supports a truthful subtitle", async () => {
    const user = userEvent.setup();
    render(
      <SportLanding
        title="Swimming"
        subtitle="Side-view analysis"
        badges={[]}
        hint="Confidence-aware"
        accentRgb="34,211,238"
        metrics={[]}
        sessions={[]}
        insights={[]}
        uploader={<div>Uploader target</div>}
        uploadSubtitle="Side-on freestyle only"
      />,
    );

    expect(screen.getByText("Side-on freestyle only")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Upload session" }));
    await user.click(
      screen.getByRole("button", { name: "Upload first session" }),
    );

    expect(scrollIntoView).toHaveBeenCalledTimes(2);
    expect(scrollIntoView).toHaveBeenLastCalledWith({
      behavior: "smooth",
      block: "start",
    });
  });

  it("renders loading and retryable error states without fake sessions", async () => {
    const user = userEvent.setup();
    const retry = vi.fn();
    const { rerender } = render(
      <SportLanding
        title="Running"
        subtitle="Analysis"
        badges={[]}
        hint="Measured data only"
        accentRgb="34,211,238"
        metrics={[]}
        sessions={[]}
        insights={[]}
        dataState="loading"
      />,
    );

    expect(screen.getByText("Loading saved sessions…")).toBeInTheDocument();

    rerender(
      <SportLanding
        title="Running"
        subtitle="Analysis"
        badges={[]}
        hint="Measured data only"
        accentRgb="34,211,238"
        metrics={[]}
        sessions={[]}
        insights={[]}
        dataState="error"
        onRetry={retry}
      />,
    );

    expect(screen.getByText("Could not load saved sessions")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalledOnce();
  });

  it("shows an honest empty insight state", () => {
    render(
      <SportLanding
        title="Running"
        subtitle="Analysis"
        badges={[]}
        hint="Measured data only"
        accentRgb="34,211,238"
        metrics={[]}
        sessions={[]}
        insights={[]}
      />,
    );

    expect(
      screen.getByText("Insights appear after a saved analysis"),
    ).toBeInTheDocument();
  });

  it("does not offer upload actions before a web workflow is connected", () => {
    render(
      <SportLanding
        title="Cycling"
        subtitle="Analysis"
        badges={[]}
        hint="Measured data only"
        accentRgb="16,185,129"
        metrics={[]}
        sessions={[]}
        insights={[]}
        uploadAvailable={false}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Workflow planned" }),
    ).toBeDisabled();
    expect(screen.queryByText("Upload first session")).not.toBeInTheDocument();
    expect(
      screen.getByText(
        "No saved web sessions yet. This workflow opens after the analysis adapter is connected.",
      ),
    ).toBeInTheDocument();
  });
});
