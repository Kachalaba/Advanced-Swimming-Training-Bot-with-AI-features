"use client";

import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus =
  | "idle"
  | "requesting"
  | "live"
  | "denied"
  | "unavailable"
  | "error";

type CameraMessages = {
  unavailable: string;
  denied: string;
  startError: string;
};

export function useCameraSource(
  videoRef: RefObject<HTMLVideoElement | null>,
  messages: CameraMessages,
) {
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, [videoRef]);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("unavailable");
      setError(messages.unavailable);
      throw new Error("Camera API unavailable");
    }
    if (streamRef.current) return streamRef.current;

    setStatus("requesting");
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 60 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setStatus("live");
      return stream;
    } catch (cause) {
      const denied =
        cause instanceof DOMException &&
        (cause.name === "NotAllowedError" || cause.name === "SecurityError");
      setStatus(denied ? "denied" : "error");
      setError(
        denied
          ? messages.denied
          : messages.startError,
      );
      throw cause;
    }
  }, [messages.denied, messages.startError, messages.unavailable, videoRef]);

  useEffect(() => stop, [stop]);

  return { status, error, start, stop };
}
