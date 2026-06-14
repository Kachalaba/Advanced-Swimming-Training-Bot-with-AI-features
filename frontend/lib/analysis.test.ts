import { afterEach, describe, expect, it, vi } from "vitest";

import { saveRunningAnalysis } from "./analysis";

describe("saveRunningAnalysis", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("saves a completed job against a stable athlete id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: 91 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const saved = await saveRunningAnalysis("run-123", { athleteId: 7 });

    expect(saved).toEqual({ sessionId: 91 });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/analysis/running/run-123/save"),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ athlete_id: 7 }),
      },
    );
  });

  it("surfaces save failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 409 }),
    );

    await expect(
      saveRunningAnalysis("run-123", { athleteId: 7 }),
    ).rejects.toThrow("Save failed: 409");
  });
});
