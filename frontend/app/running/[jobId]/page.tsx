"use client";

import {
  AlertCircle,
  ArrowLeft,
  Check,
  Footprints,
  Loader2,
  Save,
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { api, type Athlete } from "@/lib/api";
import {
  annotatedVideoUrl,
  saveRunningAnalysis,
  subscribeAnalysis,
  type AnalysisEvent,
  type ResultEvent,
} from "@/lib/analysis";

function num(v: unknown, digits = 0): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function pct(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return `${v.toFixed(0)}%`;
}

export default function RunningResultPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;
  const [progress, setProgress] = useState<{ pct: number; label: string }>({
    pct: 0,
    label: "Connecting…",
  });
  const [result, setResult] = useState<ResultEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [athletes, setAthletes] = useState<Athlete[]>([]);
  const [selectedAthleteId, setSelectedAthleteId] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedSessionId, setSavedSessionId] = useState<number | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const unsub = subscribeAnalysis(jobId, (e: AnalysisEvent) => {
      if (e.type === "progress") {
        setProgress({ pct: e.pct, label: e.label });
      } else if (e.type === "result") {
        setResult(e);
      } else if (e.type === "error") {
        setError(e.message);
      }
    });
    return unsub;
  }, [jobId]);

  useEffect(() => {
    let active = true;
    api
      .listAthletes()
      .then((items) => {
        if (!active) return;
        setAthletes(items);
        setSelectedAthleteId((current) => current || items[0]?.id || "");
      })
      .catch((athleteError: unknown) => {
        if (active) {
          setSaveError(
            athleteError instanceof Error
              ? athleteError.message
              : "Could not load athletes",
          );
        }
      });
    return () => {
      active = false;
    };
  }, []);

  async function saveSession() {
    if (!selectedAthleteId || saving || savedSessionId !== null) return;
    setSaving(true);
    setSaveError(null);
    try {
      const saved = await saveRunningAnalysis(jobId, {
        athleteId: Number(selectedAthleteId),
      });
      setSavedSessionId(saved.sessionId);
    } catch (saveFailure) {
      setSaveError(
        saveFailure instanceof Error
          ? saveFailure.message
          : "Could not save session",
      );
    } finally {
      setSaving(false);
    }
  }

  const a = (result?.analysis ?? {}) as Record<string, unknown>;
  const done = !!result;

  return (
    <div className="animate-slide-up space-y-6">
      <div>
        <Link
          href="/running"
          className="text-xs text-slate-500 hover:text-slate-300 mb-2 flex items-center gap-1 transition-colors w-fit"
        >
          <ArrowLeft className="w-3 h-3" /> Back to running
        </Link>
        <div className="flex items-center gap-2 mb-1.5">
          <StatusBadge variant="info" icon={Footprints}>
            Running
          </StatusBadge>
          {done ? (
            <StatusBadge variant="success" icon={Check}>
              Analysis complete
            </StatusBadge>
          ) : error ? (
            <StatusBadge variant="danger" icon={AlertCircle}>
              Failed
            </StatusBadge>
          ) : (
            <StatusBadge variant="info" icon={Loader2}>
              Analyzing…
            </StatusBadge>
          )}
        </div>
        <h1 className="text-xl font-bold tracking-tight">
          Session {jobId.slice(0, 8)}
        </h1>
      </div>

      {!done && !error ? (
        <ChartContainer
          title={progress.label}
          subtitle={`Job ${jobId} · ${progress.pct}%`}
        >
          <div className="space-y-3">
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-400 transition-all duration-300 ease-out shadow-[0_0_12px_rgba(34,211,238,0.6)]"
                style={{ width: `${progress.pct}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">
              The pipeline runs YOLO detection, MediaPipe pose with lock-on
              tracking, and the running analyzer. This typically takes 30–60s
              for a 30s clip — longer videos scale linearly.
            </p>
          </div>
        </ChartContainer>
      ) : null}

      {error ? (
        <ChartContainer title="Analysis failed" subtitle="Server error">
          <p className="text-sm text-rose-400">{error}</p>
        </ChartContainer>
      ) : null}

      {done ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="rounded-xl overflow-hidden border border-white/[0.06] bg-black">
              <video
                src={annotatedVideoUrl(jobId)}
                controls
                playsInline
                className="w-full aspect-video bg-black"
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Annotated output ·{" "}
              {result.frames_with_pose}/{result.frames_total} frames with pose
            </p>
          </div>

          <div className="space-y-4">
            <ChartContainer title="Headline metrics" subtitle="Session average">
              <div className="space-y-3">
                {[
                  { label: "Cadence", value: num(a.cadence), unit: "spm" },
                  { label: "Foot strike", value: String(a.foot_strike_type ?? "—") },
                  { label: "Knee lift", value: num(a.avg_knee_lift), unit: "°" },
                  { label: "Forward lean", value: num(a.forward_lean, 1), unit: "°" },
                  { label: "Arm symmetry", value: pct(a.arm_symmetry) },
                  {
                    label: "Vertical movement",
                    value: num(a.vertical_oscillation_px, 1),
                    unit: "px",
                  },
                  { label: "Total steps", value: num(a.total_steps) },
                ].map((m, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-1.5 border-b border-white/[0.04] last:border-0"
                  >
                    <p className="text-xs text-slate-400">{m.label}</p>
                    <span className="text-base font-bold text-slate-100 tnum">
                      {m.value}
                      {m.unit ? (
                        <span className="text-xs text-slate-500 font-medium ml-0.5">
                          {m.unit}
                        </span>
                      ) : null}
                    </span>
                  </div>
                ))}
              </div>
            </ChartContainer>

            <ChartContainer title="Detection quality" subtitle="How clean is the data">
              <div className="space-y-2 text-xs">
                {[
                  { label: "Frames analysed", value: String(result.frames_total) },
                  {
                    label: "Frames with pose",
                    value: String(result.frames_with_pose),
                  },
                  {
                    label: "Coverage",
                    value:
                      result.frames_total > 0
                        ? `${Math.round((result.frames_with_pose / result.frames_total) * 100)}%`
                        : "—",
                  },
                ].map((row) => (
                  <div
                    key={row.label}
                    className="flex items-center justify-between py-1.5 border-b border-white/[0.04] last:border-0"
                  >
                    <span className="text-slate-400">{row.label}</span>
                    <span className="text-slate-200 font-medium tnum">
                      {row.value}
                    </span>
                  </div>
                ))}
              </div>
            </ChartContainer>

            <ChartContainer
              title="Save to athlete history"
              subtitle="Persist this measurement for future comparison"
            >
              <div className="space-y-3">
                <label
                  htmlFor="running-athlete"
                  className="block text-xs font-medium text-slate-400"
                >
                  Save for athlete
                </label>
                <select
                  id="running-athlete"
                  value={selectedAthleteId}
                  onChange={(event) =>
                    setSelectedAthleteId(event.target.value)
                  }
                  disabled={saving || savedSessionId !== null}
                  className="h-10 w-full rounded-lg border border-white/[0.08] bg-bg px-3 text-sm text-slate-100 outline-none focus:border-cyan-400/50"
                >
                  {athletes.length === 0 ? (
                    <option value="">No athletes available</option>
                  ) : null}
                  {athletes.map((athlete) => (
                    <option key={athlete.id} value={athlete.id}>
                      {athlete.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={saveSession}
                  disabled={
                    !selectedAthleteId || saving || savedSessionId !== null
                  }
                  className="flex h-9 w-full items-center justify-center gap-2 rounded-lg bg-cyan-400 px-4 text-xs font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {saving ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : savedSessionId !== null ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : (
                    <Save className="h-3.5 w-3.5" />
                  )}
                  {savedSessionId !== null ? "Session saved" : "Save session"}
                </button>
                {savedSessionId !== null ? (
                  <p className="text-xs text-emerald-300">
                    Saved as session #{savedSessionId}
                  </p>
                ) : null}
                {saveError ? (
                  <p className="text-xs text-rose-300">{saveError}</p>
                ) : null}
              </div>
            </ChartContainer>
          </div>
        </div>
      ) : null}
    </div>
  );
}
