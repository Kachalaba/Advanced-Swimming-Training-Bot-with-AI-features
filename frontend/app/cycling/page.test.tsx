import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import CyclingPage from "./page";

describe("CyclingPage", () => {
  it("does not claim fictional athlete results", () => {
    render(<CyclingPage />);

    expect(screen.queryByText("312")).not.toBeInTheDocument();
    expect(screen.queryByText("Sweet spot intervals")).not.toBeInTheDocument();
    expect(screen.queryByText("Demo metrics")).not.toBeInTheDocument();
    expect(screen.getByText("Web analysis adapter planned")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Workflow planned" }),
    ).toBeDisabled();
    expect(screen.queryByText("Upload first session")).not.toBeInTheDocument();
  });
});
