"use client";

import { CheckCircle2, Footprints, Loader2, Route, Timer, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { uploadRunningVideo } from "@/lib/analysis";

type RunningScenario = "track" | "treadmill" | "trail";

const SCENARIOS: Record<
  RunningScenario,
  {
    label: string;
    description: string;
    guidance: string;
    checks: string[];
  }
> = {
  track: {
    label: "Track",
    description: "Best for baseline cadence, knee lift and arm-swing symmetry.",
    guidance: "Camera side-on, 10-20 meters of visible running lane, full body in frame.",
    checks: ["Camera side-on", "10-20 meters visible", "One runner only"],
  },
  treadmill: {
    label: "Treadmill",
    description: "Best for repeatable clinic or lab comparison at a known pace.",
    guidance: "Keep the belt and shoes visible, use steady speed, and avoid handrail contact.",
    checks: ["Belt and shoes visible", "Steady speed", "No handrail contact"],
  },
  trail: {
    label: "Trail",
    description: "Best for outdoor form review when terrain is part of the context.",
    guidance: "Choose a flat segment, keep lighting even, and avoid crowds crossing frame.",
    checks: ["Flat segment", "Even lighting", "No crossing traffic"],
  },
};

export function RunningUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [scenario, setScenario] = useState<RunningScenario>("track");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeScenario = SCENARIOS[scenario];

  async function handleFile(file: File) {
    setError(null);
    setUploading(true);
    try {
      const { jobId } = await uploadRunningVideo(file);
      router.push(`/running/${jobId}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setUploading(false);
    }
  }

  return (
    <div className="space-y-5">
      <input
        aria-label="Running video file"
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      <div>
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-cyan-200/70">
              Capture setup
            </p>
            <h3 className="mt-1 text-base font-semibold text-slate-100">
              Choose the running scenario
            </h3>
          </div>
          <div className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-100">
            Readiness score 3/3
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-3">
          {(Object.entries(SCENARIOS) as Array<
            [RunningScenario, (typeof SCENARIOS)[RunningScenario]]
          >).map(([id, item]) => (
            <button
              key={id}
              type="button"
              aria-pressed={scenario === id}
              onClick={() => setScenario(id)}
              className={`rounded-2xl border p-4 text-left transition ${
                scenario === id
                  ? "border-cyan-300/50 bg-cyan-400/10 shadow-[0_0_28px_rgba(34,211,238,0.12)]"
                  : "border-white/[0.08] bg-white/[0.03] hover:border-white/20"
              }`}
            >
              <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
                <Footprints className="h-4 w-4 text-cyan-300" />
                {item.label}
              </div>
              <p className="text-xs leading-relaxed text-slate-400">
                {item.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-2xl border border-cyan-300/15 bg-cyan-400/[0.06] p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-cyan-100">
            <Route className="h-4 w-4" />
            Recommended capture
          </div>
          <p className="text-sm leading-relaxed text-slate-300">
            {activeScenario.guidance}
          </p>
        </div>
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Timer className="h-4 w-4 text-cyan-300" />
            Pre-flight checks
          </div>
          <div className="grid gap-2">
            {activeScenario.checks.map((check) => (
              <div
                key={check}
                className="flex items-center gap-2 text-xs text-slate-300"
              >
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
                {check}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label="Upload running video"
        onKeyDown={(event) => {
          if (!uploading && (event.key === "Enter" || event.key === " ")) {
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200 ${
          dragging
            ? "border-cyan-400 bg-cyan-400/5"
            : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
        } ${uploading ? "pointer-events-none opacity-70" : ""}`}
      >
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-white/5">
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin text-cyan-400" />
          ) : (
            <Upload className="h-5 w-5 text-slate-400" />
          )}
        </div>
        <p className="text-sm font-medium text-slate-200">
          {uploading ? "Uploading…" : `Drop a ${activeScenario.label.toLowerCase()} running video here`}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          MP4, MOV, AVI, MKV up to 512 MB · Side-on view recommended
        </p>
        {error ? (
          <p className="mt-3 text-xs text-rose-400">{error}</p>
        ) : null}
      </div>
    </div>
  );
}
