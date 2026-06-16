"use client";

import {
  AlertCircle,
  ArrowLeft,
  Check,
  Dumbbell,
  Info,
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
  drylandAnnotatedVideoUrl,
  saveDrylandAnalysis,
  subscribeDrylandAnalysis,
  type AnalysisEvent,
  type DrylandResultEvent,
} from "@/lib/analysis";

function num(value: unknown, digits = 0): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function exerciseLabel(value: string | undefined): string {
  if (value === "push_up") return "Push-up";
  if (value === "lunge") return "Lunge";
  return "Squat";
}

export default function DrylandResultPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;
  const [progress, setProgress] = useState({
    pct: 0,
    label: "Preparing dryland analysis",
  });
  const [result, setResult] = useState<DrylandResultEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [athletes, setAthletes] = useState<Athlete[]>([]);
  const [selectedAthleteId, setSelectedAthleteId] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedSessionId, setSavedSessionId] = useState<number | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    return subscribeDrylandAnalysis(jobId, (event: AnalysisEvent) => {
      if (event.type === "progress") {
        setProgress({ pct: event.pct, label: event.label });
      } else if (event.type === "result") {
        setResult(event as DrylandResultEvent);
      } else {
        setError(event.message);
      }
    });
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
      const saved = await saveDrylandAnalysis(jobId, {
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

  const done = result !== null;
  const analysis = result?.analysis;
  const exercise = exerciseLabel(result?.exercise_type ?? analysis?.exercise_type);
  const metrics = [
    { label: "Confirmed reps", value: num(analysis?.total_reps), unit: "" },
    { label: "Average tempo", value: num(analysis?.avg_tempo, 1), unit: "s" },
    {
      label: "Average ROM",
      value: num(analysis?.avg_range_of_motion, 1),
      unit: "°",
    },
    {
      label: "Stability score",
      value: num(analysis?.stability_score, 0),
      unit: "/100",
    },
    { label: "Minimum angle", value: num(analysis?.min_angle, 0), unit: "°" },
    { label: "Maximum angle", value: num(analysis?.max_angle, 0), unit: "°" },
  ];

  return (
    <div className="animate-slide-up space-y-6">
      <div>
        <Link
          href="/dryland"
          className="mb-2 flex w-fit items-center gap-1 text-xs text-slate-500 transition-colors hover:text-slate-300"
        >
          <ArrowLeft className="h-3 w-3" /> Back to dryland
        </Link>
        <div className="mb-1.5 flex items-center gap-2">
          <StatusBadge variant="info" icon={Dumbbell}>
            Dryland
          </StatusBadge>
          {done ? (
            <StatusBadge variant="success" icon={Check}>
              Evidence ready
            </StatusBadge>
          ) : error ? (
            <StatusBadge variant="danger" icon={AlertCircle}>
              Capture rejected
            </StatusBadge>
          ) : (
            <StatusBadge variant="info" icon={Loader2}>
              Analyzing…
            </StatusBadge>
          )}
        </div>
        <h1 className="text-xl font-bold tracking-tight">
          {done ? `${exercise} evidence` : `Dryland session ${jobId.slice(0, 8)}`}
        </h1>
      </div>

      {!done && !error ? (
        <ChartContainer
          title={progress.label}
          subtitle={`Job ${jobId} · ${progress.pct}%`}
        >
          <div className="space-y-3">
            <div className="h-2 overflow-hidden rounded-full bg-white/5">
              <div
                className="h-full bg-violet-400 shadow-[0_0_12px_rgba(139,92,246,0.55)] transition-all duration-300"
                style={{ width: `${progress.pct}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">
              SPRINT checks pose coverage and exercise-specific metric readiness
              before it confirms full repetitions.
            </p>
          </div>
        </ChartContainer>
      ) : null}

      {error ? (
        <ChartContainer
          title="Capture quality is not sufficient"
          subtitle="No unsupported dryland score was produced"
        >
          <p className="text-sm leading-relaxed text-rose-300">{error}</p>
          <Link
            href="/dryland"
            className="mt-4 inline-flex h-9 items-center rounded-lg bg-white/[0.06] px-4 text-xs font-medium text-slate-200"
          >
            Record another clip
          </Link>
        </ChartContainer>
      ) : null}

      {done && analysis ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-black">
              <video
                src={drylandAnnotatedVideoUrl(jobId)}
                controls
                playsInline
                className="aspect-video w-full bg-black"
              />
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <span>
                {result.frames_with_pose}/{result.frames_total} frames with pose
              </span>
              <span>·</span>
              <span>{num(result.quality.pose_coverage, 1)}% pose coverage</span>
              <span>·</span>
              <span>
                {result.quality.metric_ready_frames}/
                {result.quality.minimum_required_frames} metric-ready frames
              </span>
            </div>

            <ChartContainer
              title="Per-rep evidence"
              subtitle={`${exercise} · ${analysis.tracked_joint} tracking`}
            >
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] text-left text-xs">
                  <thead className="text-slate-500">
                    <tr className="border-b border-white/[0.06]">
                      <th className="py-2 pr-3 font-medium">Rep</th>
                      <th className="py-2 pr-3 font-medium">Side</th>
                      <th className="py-2 pr-3 font-medium">Tempo</th>
                      <th className="py-2 pr-3 font-medium">ROM</th>
                      <th className="py-2 pr-3 font-medium">Min</th>
                      <th className="py-2 pr-3 font-medium">Frames</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.reps.map((rep) => (
                      <tr
                        key={rep.rep_number}
                        className="border-b border-white/[0.04] last:border-0"
                      >
                        <td className="py-2 pr-3 text-slate-100">
                          #{rep.rep_number}
                        </td>
                        <td className="py-2 pr-3 capitalize text-slate-300">
                          {rep.active_side || "—"}
                        </td>
                        <td className="py-2 pr-3 text-slate-300 tnum">
                          {num(rep.duration_sec, 1)}s
                        </td>
                        <td className="py-2 pr-3 text-slate-300 tnum">
                          {num(rep.range_of_motion, 1)}°
                        </td>
                        <td className="py-2 pr-3 text-slate-300 tnum">
                          {num(rep.min_angle, 0)}°
                        </td>
                        <td className="py-2 pr-3 text-slate-500 tnum">
                          {rep.start_frame} → {rep.effort_frame} →{" "}
                          {rep.end_frame}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ChartContainer>

            <div className="flex gap-3 rounded-xl border border-violet-300/10 bg-violet-400/[0.04] p-4">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-violet-300" />
              <p className="text-xs leading-relaxed text-slate-400">
                This is movement-screening evidence from one fixed-view clip.
                Use it to compare repeatable sessions; confirm clinical
                decisions with a qualified professional.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <ChartContainer title="Dryland evidence" subtitle="Measured from this clip">
              <div className="space-y-2">
                {metrics.map((metric) => (
                  <div
                    key={metric.label}
                    className="flex items-center justify-between border-b border-white/[0.04] py-1.5 last:border-0"
                  >
                    <span className="text-xs text-slate-400">{metric.label}</span>
                    <span className="text-base font-bold text-slate-100 tnum">
                      {metric.value}
                      {metric.unit ? (
                        <span className="ml-0.5 text-xs font-medium text-slate-500">
                          {metric.unit}
                        </span>
                      ) : null}
                    </span>
                  </div>
                ))}
              </div>
            </ChartContainer>

            <ChartContainer
              title="Save to athlete history"
              subtitle="Build a comparable strength baseline"
            >
              <div className="space-y-3">
                <label
                  htmlFor="dryland-athlete"
                  className="block text-xs font-medium text-slate-400"
                >
                  Save for athlete
                </label>
                <select
                  id="dryland-athlete"
                  value={selectedAthleteId}
                  onChange={(event) =>
                    setSelectedAthleteId(event.target.value)
                  }
                  disabled={saving || savedSessionId !== null}
                  className="h-10 w-full rounded-lg border border-white/[0.08] bg-bg px-3 text-sm text-slate-100 outline-none focus:border-violet-400/50"
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
                  className="flex h-9 w-full items-center justify-center gap-2 rounded-lg bg-violet-400 px-4 text-xs font-semibold text-slate-950 transition hover:bg-violet-300 disabled:cursor-not-allowed disabled:opacity-50"
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
