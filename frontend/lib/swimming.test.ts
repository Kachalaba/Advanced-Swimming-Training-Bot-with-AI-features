import { afterEach, describe, expect, it, vi } from "vitest";

import {
  saveSwimmingAnalysis,
  swimmingVideoUrl,
  uploadSwimmingVideo,
} from "./swimming";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("swimming API client", () => {
  it("uploads a swimming video to the swimming endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "swim-123" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["video"], "freestyle.mp4", {
      type: "video/mp4",
    });

    const result = await uploadSwimmingVideo(file);

    expect(result).toEqual({ jobId: "swim-123" });
    expect(fetchMock.mock.calls[0][0]).toContain("/api/analysis/swimming");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
    expect(fetchMock.mock.calls[0][1].body).toBeInstanceOf(FormData);
  });

  it("surfaces the backend validation detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({ detail: "Unsupported video format" }),
      }),
    );

    await expect(
      uploadSwimmingVideo(
        new File(["video"], "freestyle.txt", { type: "text/plain" }),
      ),
    ).rejects.toThrow("Unsupported video format");
  });

  it("builds the annotated swimming video URL", () => {
    expect(swimmingVideoUrl("swim-123")).toContain(
      "/api/analysis/swimming/swim-123/video",
    );
  });

  it("saves the completed analysis for the selected athlete", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ session_id: 91 }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await saveSwimmingAnalysis("swim-123", { athleteId: 7 });

    expect(result).toEqual({ sessionId: 91 });
    expect(fetchMock.mock.calls[0][0]).toContain(
      "/api/analysis/swimming/swim-123/save",
    );
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      athlete_id: 7,
    });
  });
});
