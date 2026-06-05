export type FrameScheduler<T> = {
  enqueue: (frame: T) => void;
  idle: () => Promise<void>;
  dispose: () => void;
};

export function createFrameScheduler<T>(
  send: (frame: T) => Promise<void>,
): FrameScheduler<T> {
  let pending: T | undefined;
  let inFlight = false;
  let disposed = false;
  let idleResolvers: Array<() => void> = [];

  function resolveIdle() {
    if (inFlight || pending !== undefined) return;
    const resolvers = idleResolvers;
    idleResolvers = [];
    resolvers.forEach((resolve) => resolve());
  }

  async function drain() {
    if (inFlight || disposed) return;
    inFlight = true;
    while (!disposed && pending !== undefined) {
      const current = pending;
      pending = undefined;
      try {
        await send(current);
      } catch {
        // Transport errors are surfaced by the caller-owned send function.
      }
    }
    inFlight = false;
    resolveIdle();
  }

  return {
    enqueue(frame) {
      if (disposed) return;
      pending = frame;
      void drain();
    },
    idle() {
      if (!inFlight && pending === undefined) return Promise.resolve();
      return new Promise<void>((resolve) => idleResolvers.push(resolve));
    },
    dispose() {
      disposed = true;
      pending = undefined;
      resolveIdle();
    },
  };
}
