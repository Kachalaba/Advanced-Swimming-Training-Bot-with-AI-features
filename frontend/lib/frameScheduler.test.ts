import { describe, expect, it } from "vitest";

import { createFrameScheduler } from "./frameScheduler";

describe("createFrameScheduler", () => {
  it("keeps only the newest queued frame while one request is in flight", async () => {
    const sent: string[] = [];
    const gates: Array<() => void> = [];
    const scheduler = createFrameScheduler<string>(async (frame) => {
      sent.push(frame);
      await new Promise<void>((resolve) => gates.push(resolve));
    });

    scheduler.enqueue("frame-1");
    scheduler.enqueue("frame-2");
    scheduler.enqueue("frame-3");
    await Promise.resolve();

    expect(sent).toEqual(["frame-1"]);
    gates.shift()?.();
    await Promise.resolve();
    await Promise.resolve();

    expect(sent).toEqual(["frame-1", "frame-3"]);
    gates.shift()?.();
    await scheduler.idle();
  });

  it("does not send frames after disposal", async () => {
    const sent: string[] = [];
    const scheduler = createFrameScheduler<string>(async (frame) => {
      sent.push(frame);
    });

    scheduler.dispose();
    scheduler.enqueue("late-frame");
    await scheduler.idle();

    expect(sent).toEqual([]);
  });
});
