import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RehabDemoStage } from "./RehabDemoStage";
import { demoFrames } from "./demoSession";

describe("RehabDemoStage", () => {
  it("renders an evidence-led clinical analysis scene", () => {
    render(
      <RehabDemoStage
        frame={demoFrames[demoFrames.length - 1]}
        locale="uk"
        running={false}
      />,
    );

    expect(screen.getByTestId("rehab-body-figure")).toBeInTheDocument();
    expect(screen.getByTestId("left-rom-arc")).toBeInTheDocument();
    expect(screen.getByTestId("right-rom-arc")).toBeInTheDocument();
    expect(screen.getByText("154°")).toBeInTheDocument();
    expect(screen.getByText("136°")).toBeInTheDocument();
    expect(screen.getByText("0:04.8")).toBeInTheDocument();
    expect(screen.getByText(/демонстраційна сесія/i)).toBeInTheDocument();
  });

  it("renders the English simulated-session disclosure", () => {
    render(
      <RehabDemoStage frame={demoFrames[0]} locale="en" running />,
    );

    expect(screen.getByText(/simulated demo session/i)).toBeInTheDocument();
    expect(screen.getByText("Analyzing")).toBeInTheDocument();
  });
});
