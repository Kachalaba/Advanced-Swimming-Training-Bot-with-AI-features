"use client";

import {
  CheckCircle2,
  Download,
  FileArchive,
  FileVideo,
  Loader2,
  Save,
  ScanLine,
  Scissors,
  Upload,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  saveToolJob,
  subscribeToolJob,
  toolDownloadUrl,
  uploadFrameExtraction,
  uploadTrimVideo,
  type ToolJobEvent,
  type ToolOperation,
  type ToolResultEvent,
} from "@/lib/tools";

type FrameMode = "interval" | "count";

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function VideoToolsWorkspace() {
  const unsubscribeRef = useRef<null | (() => void)>(null);
  const [operation, setOperation] = useState<ToolOperation>("trim");
  const [file, setFile] = useState<File | null>(null);
  const [startSec, setStartSec] = useState(0);
  const [endSec, setEndSec] = useState(5);
  const [frameMode, setFrameMode] = useState<FrameMode>("interval");
  const [intervalSec, setIntervalSec] = useState(1);
  const [frameCount, setFrameCount] = useState(12);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [label, setLabel] = useState("Ready to process");
  const [result, setResult] = useState<ToolResultEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedSessionId, setSavedSessionId] = useState<number | null>(null);
  const busy = progress > 0 && progress < 100 && !error && !result;

  useEffect(() => () => unsubscribeRef.current?.(), []);

  function selectOperation(next: ToolOperation) {
    if (busy) return;
    setOperation(next);
    setJobId(null);
    setProgress(0);
    setLabel("Ready to process");
    setResult(null);
    setError(null);
    setSavedSessionId(null);
  }

  function handleEvent(event: ToolJobEvent) {
    if (event.type === "progress") {
      setProgress(event.pct);
      setLabel(event.label);
    } else if (event.type === "result") {
      setProgress(100);
      setLabel("Processing complete");
      setResult(event);
    } else {
      setProgress(0);
      setError(event.message);
      setLabel("Processing failed");
    }
  }

  async function processVideo() {
    if (!file || busy) return;
    unsubscribeRef.current?.();
    setError(null);
    setResult(null);
    setSavedSessionId(null);
    setProgress(1);
    setLabel("Uploading source video");
    try {
      const upload =
        operation === "trim"
          ? await uploadTrimVideo(file, startSec, endSec)
          : await uploadFrameExtraction(
              file,
              frameMode === "interval"
                ? { mode: "interval", intervalSec }
                : { mode: "count", frameCount },
            );
      setJobId(upload.jobId);
      unsubscribeRef.current = subscribeToolJob(upload.jobId, handleEvent);
    } catch (cause) {
      setProgress(0);
      setLabel("Upload failed");
      setError(cause instanceof Error ? cause.message : "Could not process video");
    }
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-2 sm:grid-cols-2">
        {[
          {
            value: "trim" as const,
            title: "Trim & cut",
            description: "Export one browser-ready MP4 segment.",
            icon: Scissors,
          },
          {
            value: "frame_extractor" as const,
            title: "Frame extractor",
            description: "Export JPEG frames with a JSON manifest.",
            icon: ScanLine,
          },
        ].map((tool) => {
          const Icon = tool.icon;
          const selected = operation === tool.value;
          return (
            <button
              key={tool.value}
              type="button"
              disabled={busy}
              aria-pressed={selected}
              aria-label={tool.title}
              onClick={() => selectOperation(tool.value)}
              className={`rounded-xl border p-4 text-left transition ${
                selected
                  ? "border-cyan-400/30 bg-cyan-400/[0.08]"
                  : "border-white/[0.06] bg-white/[0.02] hover:border-white/15"
              } disabled:cursor-wait`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                    selected ? "bg-cyan-400/10 text-cyan-300" : "bg-white/5 text-slate-400"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-100">{tool.title}</p>
                  <p className="mt-0.5 text-xs text-slate-500">{tool.description}</p>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="grid gap-4 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 lg:grid-cols-[1.2fr_1fr]">
        <div>
          <label
            htmlFor="tool-source-video"
            className="flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-black/10 px-5 text-center transition hover:border-cyan-400/30 hover:bg-cyan-400/[0.025]"
          >
            <input
              id="tool-source-video"
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
              className="sr-only"
              aria-label="Choose source video"
              disabled={busy}
              onChange={(event) => {
                const nextFile = event.target.files?.[0] ?? null;
                setFile(nextFile);
                setJobId(null);
                setResult(null);
                setError(null);
                setProgress(0);
                setSavedSessionId(null);
              }}
            />
            {file ? (
              <FileVideo className="h-6 w-6 text-cyan-300" />
            ) : (
              <Upload className="h-6 w-6 text-slate-400" />
            )}
            <p className="mt-3 text-sm font-medium text-slate-200">
              {file?.name ?? "Choose a source video"}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              MP4, MOV, AVI or MKV up to 512 MB
            </p>
          </label>
        </div>

        <div className="space-y-4">
          {operation === "trim" ? (
            <div className="grid grid-cols-2 gap-3">
              <label className="text-xs font-medium text-slate-400">
                Start, sec
                <input
                  type="number"
                  min={0}
                  step={0.1}
                  value={startSec}
                  aria-label="Start time in seconds"
                  onChange={(event) => setStartSec(Number(event.target.value))}
                  className="mt-1.5 h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                />
              </label>
              <label className="text-xs font-medium text-slate-400">
                End, sec
                <input
                  type="number"
                  min={0.1}
                  step={0.1}
                  value={endSec}
                  aria-label="End time in seconds"
                  onChange={(event) => setEndSec(Number(event.target.value))}
                  className="mt-1.5 h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                />
              </label>
              <p className="col-span-2 text-xs leading-relaxed text-slate-500">
                The clip is re-encoded to H.264 with fast-start playback.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { value: "interval" as const, label: "Every N seconds" },
                  { value: "count" as const, label: "Exact count" },
                ].map((mode) => (
                  <label
                    key={mode.value}
                    className={`cursor-pointer rounded-lg border px-3 py-2.5 text-xs font-medium ${
                      frameMode === mode.value
                        ? "border-cyan-400/30 bg-cyan-400/10 text-cyan-200"
                        : "border-white/[0.06] text-slate-400"
                    }`}
                  >
                    <input
                      type="radio"
                      name="frame-mode"
                      value={mode.value}
                      checked={frameMode === mode.value}
                      onChange={() => setFrameMode(mode.value)}
                      className="sr-only"
                    />
                    {mode.label}
                  </label>
                ))}
              </div>
              {frameMode === "interval" ? (
                <label className="block text-xs font-medium text-slate-400">
                  Interval, sec
                  <input
                    type="number"
                    min={0.1}
                    step={0.1}
                    value={intervalSec}
                    aria-label="Frame interval in seconds"
                    onChange={(event) => setIntervalSec(Number(event.target.value))}
                    className="mt-1.5 h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                  />
                </label>
              ) : (
                <label className="block text-xs font-medium text-slate-400">
                  Frames
                  <input
                    type="number"
                    min={1}
                    max={200}
                    step={1}
                    value={frameCount}
                    aria-label="Number of frames"
                    onChange={(event) => setFrameCount(Number(event.target.value))}
                    className="mt-1.5 h-10 w-full rounded-lg border border-white/10 bg-black/20 px-3 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                  />
                </label>
              )}
              <p className="text-xs leading-relaxed text-slate-500">
                Up to 200 high-quality JPEGs are packaged in one ZIP archive.
              </p>
            </div>
          )}

          <button
            type="button"
            disabled={!file || busy}
            onClick={() => void processVideo()}
            className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-cyan-400 px-4 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:bg-white/[0.06] disabled:text-slate-500"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {busy ? label : "Process video"}
          </button>
        </div>
      </div>

      {progress > 0 ? (
        <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">{label}</span>
            <span className="font-mono text-slate-500">{progress}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full bg-cyan-400 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="rounded-xl border border-rose-400/20 bg-rose-400/[0.06] p-4 text-sm text-rose-200">
          {error}
        </p>
      ) : null}

      {result && jobId ? (
        <div className="rounded-xl border border-emerald-400/15 bg-emerald-400/[0.045] p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex min-w-0 items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-400/10 text-emerald-300">
                {result.operation === "trim" ? (
                  <FileVideo className="h-5 w-5" />
                ) : (
                  <FileArchive className="h-5 w-5" />
                )}
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                  <p className="text-sm font-semibold text-slate-100">
                    Result ready
                  </p>
                </div>
                <p className="mt-1 truncate text-xs text-slate-400">
                  {result.artifact_name} · {formatBytes(result.size_bytes)}
                </p>
                <p className="mt-1 text-[11px] text-slate-500">
                  {result.operation === "trim"
                    ? `${String(result.metadata.duration_sec ?? "—")} sec clip`
                    : `${String(result.metadata.frame_count ?? "—")} extracted frames`}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <a
                href={toolDownloadUrl(jobId)}
                download
                className="inline-flex h-9 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 text-xs font-semibold text-slate-200 transition hover:bg-white/[0.08]"
              >
                <Download className="h-3.5 w-3.5" />
                Download result
              </a>
              <button
                type="button"
                disabled={savedSessionId !== null}
                aria-label={
                  savedSessionId
                    ? `Saved to history #${savedSessionId}`
                    : "Save to history"
                }
                onClick={async () => {
                  try {
                    const saved = await saveToolJob(jobId);
                    setSavedSessionId(saved.sessionId);
                  } catch (cause) {
                    setError(
                      cause instanceof Error
                        ? cause.message
                        : "Could not save result to history",
                    );
                  }
                }}
                className="inline-flex h-9 items-center gap-2 rounded-lg border border-cyan-400/25 bg-cyan-400/10 px-3 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-400/15 disabled:border-emerald-400/20 disabled:bg-emerald-400/10 disabled:text-emerald-200"
              >
                <Save className="h-3.5 w-3.5" />
                {savedSessionId
                  ? `Saved to history #${savedSessionId}`
                  : "Save to history"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
