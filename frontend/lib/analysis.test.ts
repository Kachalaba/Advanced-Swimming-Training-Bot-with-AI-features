import { afterEach, describe, expect, it, vi } from "vitest";

import {
  drylandAnnotatedVideoUrl,
  saveCyclingAnalysis,
  saveDrylandAnalysis,
  saveRunningAnalysis,
  uploadCyclingVideo,
  uploadDrylandVideo,
} from "./analysis";

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

describe("cycling analysis API", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uploads cycling video to the cycling workflow", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "cycle-123" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["video"], "trainer.mp4", { type: "video/mp4" });
    const uploaded = await uploadCyclingVideo(file, 30);

    expect(uploaded).toEqual({ jobId: "cycle-123" });
    expect(fetchMock.mock.calls[0][0]).toContain("/api/analysis/cycling");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
  });

  it("saves cycling evidence against a stable athlete id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: 73 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const saved = await saveCyclingAnalysis("cycle-123", { athleteId: 7 });

    expect(saved).toEqual({ sessionId: 73 });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/analysis/cycling/cycle-123/save"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ athlete_id: 7 }),
      }),
    );
  });
});

describe("dryland analysis API", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uploads dryland video with the selected exercise profile", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "dryland-123" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["video"], "squat.mp4", { type: "video/mp4" });
    const uploaded = await uploadDrylandVideo(file, "squat", 15);

    expect(uploaded).toEqual({ jobId: "dryland-123" });
    expect(fetchMock.mock.calls[0][0]).toContain("/api/analysis/dryland");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });

    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(body.get("video")).toBe(file);
    expect(body.get("exercise_type")).toBe("squat");
    expect(body.get("fps")).toBe("15");
  });

  it("builds the dryland annotated video url", () => {
    expect(drylandAnnotatedVideoUrl("dryland-123")).toContain(
      "/api/analysis/dryland/dryland-123/video",
    );
  });

  it("saves dryland evidence against a stable athlete id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: 81 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const saved = await saveDrylandAnalysis("dryland-123", { athleteId: 7 });

    expect(saved).toEqual({ sessionId: 81 });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/analysis/dryland/dryland-123/save"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ athlete_id: 7 }),
      }),
    );
  });
});
