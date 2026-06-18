import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DashboardView } from "./DashboardView";

describe("DashboardView", () => {
  it("opens real analysis workflows from the upload action", async () => {
    const user = userEvent.setup();
    render(<DashboardView athleteName="Nikita K." />);

    await user.click(screen.getByRole("button", { name: "Upload session" }));

    const menu = screen.getByRole("menu", {
      name: "Choose analysis workflow",
    });
    expect(menu).toBeInTheDocument();
    expect(
      screen.getByRole("menuitem", { name: /Swimming/ }),
    ).toHaveAttribute("href", "/swimming");
    expect(
      screen.getByRole("menuitem", { name: /Rehabilitation/ }),
    ).toHaveAttribute("href", "/rehabilitation");
  });

  it("links overview actions to persisted history and sport workflows", () => {
    render(<DashboardView athleteName="Nikita K." />);

    expect(screen.getByRole("link", { name: "View all" })).toHaveAttribute(
      "href",
      "/history",
    );
    expect(
      screen.getByRole("link", {
        name: /Running.*Cadence and gait/,
      }),
    ).toHaveAttribute("href", "/running");
  });
});
