"use client";

import {
  Camera,
  Crosshair,
  Maximize2,
  Minimize2,
  Power,
  Save,
  ScanLine,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { createFrameScheduler, type FrameScheduler } from "@/lib/frameScheduler";
import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";
import {
  createLiveRehabSession,
  deleteLiveRehabSession,
  saveLiveRehabSession,
  sendLiveRehabFrame,
  type LiveRehabUpdate,
  type RehabProtocol,
} from "@/lib/rehabilitation";

import { LiveMetricRail } from "./LiveMetricRail";
import { PostureOverlay } from "./PostureOverlay";
import { useCameraSource } from "./useCameraSource";

const ANALYSIS_INTERVAL_MS = 200;
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 360;

export function LiveRehabWorkspace({
  protocol,
  locale = "uk",
}: {
  protocol: RehabProtocol;
  locale?: RehabLocale;
}) {
  const copy = rehabCopy[locale].live;
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workspaceRef = useRef<HTMLDivElement>(null);
  const sessionIdRef = useRef<string | null>(null);
  const timerRef = useRef<number | null>(null);
  const schedulerRef = useRef<FrameScheduler<Blob> | null>(null);
  const calibrateNextRef = useRef(false);
  const [showPosture, setShowPosture] = useState(true);
  const [update, setUpdate] = useState<LiveRehabUpdate | null>(null);
  const [transportError, setTransportError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [savedSessionId, setSavedSessionId] = useState<number | null>(null);
  const { status, error: cameraError, start: startCamera, stop: stopCamera } =
    useCameraSource(videoRef, {
      unavailable: copy.cameraUnavailable,
      denied: copy.cameraDenied,
      startError: copy.cameraStartError,
    });

  const capture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context) return;
    context.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    canvas.toBlob(
      (blob) => {
        if (blob) schedulerRef.current?.enqueue(blob);
      },
      "image/jpeg",
      0.72,
    );
  }, []);

  const stop = useCallback(async () => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    schedulerRef.current?.dispose();
    schedulerRef.current = null;
    const sessionId = sessionIdRef.current;
    sessionIdRef.current = null;
    if (sessionId) {
      await deleteLiveRehabSession(sessionId).catch(() => undefined);
    }
    stopCamera();
    setUpdate(null);
    setSavedSessionId(null);
  }, [stopCamera]);

  const start = useCallback(async () => {
    setTransportError(null);
    setSavedSessionId(null);
    try {
      await startCamera();
      const session = await createLiveRehabSession(protocol, 5);
      sessionIdRef.current = session.sessionId;
      schedulerRef.current = createFrameScheduler<Blob>(async (blob) => {
        const sessionId = sessionIdRef.current;
        if (!sessionId) return;
        const calibrate = calibrateNextRef.current;
        calibrateNextRef.current = false;
        try {
          setUpdate(await sendLiveRehabFrame(sessionId, blob, calibrate));
          setTransportError(null);
        } catch {
          setTransportError(copy.liveError);
        }
      });
      timerRef.current = window.setInterval(capture, ANALYSIS_INTERVAL_MS);
    } catch {
      stopCamera();
    }
  }, [capture, copy.liveError, protocol, startCamera, stopCamera]);

  useEffect(() => {
    const onFullscreen = () =>
      setFullscreen(document.fullscreenElement === workspaceRef.current);
    document.addEventListener("fullscreenchange", onFullscreen);
    return () => document.removeEventListener("fullscreenchange", onFullscreen);
  }, []);

  useEffect(() => {
    if (!fullscreen || document.fullscreenElement) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [fullscreen]);

  useEffect(() => {
    return () => {
      void stop();
    };
  }, [stop]);

  const toggleFullscreen = async () => {
    if (document.fullscreenElement) {
      await document.exitFullscreen?.();
      return;
    }
    if (fullscreen) {
      setFullscreen(false);
      return;
    }
    const workspace = workspaceRef.current;
    if (workspace && typeof workspace.requestFullscreen === "function") {
      try {
        await workspace.requestFullscreen();
        return;
      } catch {
        // Fall back to an app-level fullscreen surface when browser policy blocks the API.
      }
    }
    setFullscreen(true);
  };

  const isLive = status === "live" && sessionIdRef.current !== null;
  const cameraLevel = update?.camera_level ?? null;
  const levelLabel =
    cameraLevel?.status === "level"
      ? copy.level
      : cameraLevel?.status === "adjust"
        ? copy.adjust
        : cameraLevel?.status === "recalibrate"
          ? copy.recalibrate
          : copy.uncalibrated;

  return (
    <div
      ref={workspaceRef}
      className={`group relative min-h-[920px] overflow-hidden rounded-2xl border border-white/[0.08] bg-[#070b10] shadow-2xl shadow-black/30 sm:min-h-[790px] md:min-h-[690px] xl:min-h-[610px] fullscreen:min-h-screen fullscreen:rounded-none ${
        fullscreen ? "fixed inset-0 z-[100] min-h-screen rounded-none" : ""
      }`}
    >
      <div
        className="absolute inset-0 opacity-50"
        style={{
          background:
            "radial-gradient(circle at 50% 32%, rgba(34,211,238,.12), transparent 52%), linear-gradient(135deg, #101923, #070b10 70%)",
        }}
      />
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className={`absolute inset-0 h-full w-full object-cover scale-x-[-1] transition-opacity duration-500 ${
          status === "live" ? "opacity-100" : "opacity-0"
        }`}
      />
      <canvas
        ref={canvasRef}
        width={CAPTURE_WIDTH}
        height={CAPTURE_HEIGHT}
        className="hidden"
      />
      <PostureOverlay
        visible={showPosture}
        posture={update?.posture ?? null}
        locale={locale}
      />

      <div className="absolute left-4 right-4 top-4 z-20 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            role="switch"
            aria-label={copy.postureMapAria}
            aria-checked={showPosture}
            onClick={() => setShowPosture((value) => !value)}
            className={`flex h-9 items-center gap-2 rounded-lg border px-3 text-xs font-medium backdrop-blur-xl transition ${
              showPosture
                ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-100"
                : "border-white/10 bg-black/45 text-slate-400"
            }`}
          >
            <span
              className={`relative h-4 w-7 rounded-full transition ${
                showPosture ? "bg-cyan-400" : "bg-slate-700"
              }`}
            >
              <span
                className={`absolute top-0.5 h-3 w-3 rounded-full bg-slate-950 transition ${
                  showPosture ? "left-3.5" : "left-0.5"
                }`}
              />
            </span>
            <ScanLine className="h-3.5 w-3.5" />
            {copy.postureMap}
          </button>
          <button
            type="button"
            disabled={!isLive || !update?.report || savedSessionId !== null}
            onClick={async () => {
              const sessionId = sessionIdRef.current;
              if (!sessionId) return;
              try {
                const saved = await saveLiveRehabSession(sessionId);
                setSavedSessionId(saved.sessionId);
              } catch {
                setTransportError(copy.saveError);
              }
            }}
            className="flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-black/45 px-3 text-xs font-medium text-slate-200 backdrop-blur-xl transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Save className="h-3.5 w-3.5 text-cyan-300" />
            {savedSessionId
              ? `${copy.saved} #${savedSessionId}`
              : copy.save}
          </button>
          <button
            type="button"
            onClick={() => {
              calibrateNextRef.current = true;
              capture();
            }}
            disabled={!isLive}
            className="flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-black/45 px-3 text-xs font-medium text-slate-200 backdrop-blur-xl transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Crosshair className="h-3.5 w-3.5 text-emerald-300" />
            {copy.calibrate}
          </button>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            aria-label={
              fullscreen ? copy.exitFullscreen : copy.fullscreen
            }
            onClick={() => void toggleFullscreen()}
            className="flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-black/45 px-3 text-xs font-medium text-slate-200 backdrop-blur-xl transition hover:bg-white/10"
          >
            {fullscreen ? (
              <Minimize2 className="h-3.5 w-3.5" />
            ) : (
              <Maximize2 className="h-3.5 w-3.5" />
            )}
            <span className="hidden sm:inline">
              {fullscreen ? copy.exit : copy.fullscreen}
            </span>
          </button>
          <div className="flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-black/45 px-3 text-[10px] font-bold uppercase tracking-[0.13em] text-slate-300 backdrop-blur-xl">
            <span
              className={`h-2 w-2 rounded-full ${
                isLive
                  ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.8)]"
                  : "bg-slate-600"
              }`}
            />
            {isLive ? copy.active : copy.standby}
          </div>
        </div>
      </div>

      {status !== "live" ? (
        <div className="absolute inset-0 z-10 flex items-center justify-center px-6 pb-[380px] pt-28 sm:pb-[280px] md:pb-0 md:pt-0">
          <div className="max-w-md text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-400/10 shadow-[0_0_40px_rgba(34,211,238,.1)]">
              <Camera className="h-7 w-7 text-cyan-300" />
            </div>
            <h3 className="mt-5 text-xl font-semibold text-slate-50">
              {copy.studioTitle}
            </h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              {copy.studioBody}
            </p>
            <button
              type="button"
              onClick={() => void start()}
              disabled={status === "requesting"}
              className="mt-6 inline-flex h-10 items-center gap-2 rounded-lg bg-cyan-400 px-4 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 active:scale-[0.98] disabled:opacity-60"
            >
              <Power className="h-4 w-4" />
              {status === "requesting" ? copy.connecting : copy.startCamera}
            </button>
            {cameraError ? (
              <p className="mt-3 text-xs text-rose-300">{cameraError}</p>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="absolute bottom-4 left-3 right-3 z-20 flex flex-col items-start justify-between gap-3 sm:left-4 sm:right-4 xl:flex-row xl:items-end">
        <LiveMetricRail
          report={update?.report ?? null}
          cameraLevel={cameraLevel}
          poseDetected={update?.pose_detected ?? null}
          locale={locale}
        />
        <div className="w-full max-w-[270px] rounded-xl border border-white/[0.09] bg-[#080c12]/80 p-3.5 backdrop-blur-xl">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.12em] text-slate-500">
            <span>{copy.cameraAngle}</span>
            <span
              className={
                cameraLevel?.status === "level"
                  ? "text-emerald-300"
                  : cameraLevel?.status === "adjust"
                    ? "text-amber-300"
                    : "text-slate-400"
              }
            >
              {cameraLevel?.angle_deg === null ||
              cameraLevel?.angle_deg === undefined
                ? "—"
                : `${cameraLevel.angle_deg.toFixed(1)}°`}{" "}
              · {levelLabel}
            </span>
          </div>
          <div className="relative mt-3 h-px bg-slate-700">
            <span
              className={`absolute -top-1.5 left-1/2 h-3 w-8 -translate-x-1/2 rounded-full border ${
                cameraLevel?.status === "level"
                  ? "border-emerald-400 bg-emerald-400/15"
                  : "border-amber-400 bg-amber-400/15"
              }`}
            />
          </div>
          <p className="mt-2 text-[10px] text-slate-600">
            {copy.calibrationNote}
          </p>
        </div>
      </div>

      {isLive ? (
        <button
          type="button"
          onClick={() => void stop()}
          className="absolute right-4 top-16 z-20 rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-rose-200 backdrop-blur-xl hover:bg-rose-400/15"
        >
          {copy.stop}
        </button>
      ) : null}
      {transportError ? (
        <div className="absolute left-1/2 top-16 z-20 -translate-x-1/2 rounded-lg border border-rose-400/20 bg-rose-950/80 px-3 py-2 text-xs text-rose-200 backdrop-blur-xl">
          {transportError}
        </div>
      ) : null}
    </div>
  );
}
