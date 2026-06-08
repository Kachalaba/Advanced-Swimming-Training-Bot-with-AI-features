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
});
