import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DrylandPage from "./page";

describe("DrylandPage", () => {
  it("does not claim fictional athlete results", () => {
    render(<DrylandPage />);

    expect(screen.queryByText("1,240")).not.toBeInTheDocument();
    expect(screen.queryByText("Core + mobility")).not.toBeInTheDocument();
    expect(screen.queryByText("Demo metrics")).not.toBeInTheDocument();
    expect(screen.getByText("Web workflow planned")).toBeInTheDocument();
  });
});
