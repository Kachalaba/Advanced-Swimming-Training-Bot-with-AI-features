"use client";

import {
  AlertCircle,
  ArrowLeft,
  Bike,
  Check,
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
  cyclingAnnotatedVideoUrl,
  saveCyclingAnalysis,
  subscribeCyclingAnalysis,
  type AnalysisEvent,
  type ResultEvent,
} from "@/lib/analysis";

function num(value: unknown, digits = 0): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export default function CyclingResultPage() {
  const params = useParams<{ jobId: string }>();
  const jobId = params.jobId;
  const [progress, setProgress] = useState({
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
    return subscribeCyclingAnalysis(jobId, (event: AnalysisEvent) => {
      if (event.type === "progress") {
        setProgress({ pct: event.pct, label: event.label });
      } else if (event.type === "result") {
        setResult(event);
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
      const saved = await saveCyclingAnalysis(jobId, {
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

  const analysis = (result?.analysis ?? {}) as Record<string, unknown>;
  const done = result !== null;
  const metrics = [
    { label: "Cadence", value: num(analysis.cadence), unit: "rpm" },
    {
      label: "Knee angle · top",
      value: num(analysis.avg_knee_angle_top),
      unit: "°",
    },
    {
      label: "Knee extension · bottom",
      value: num(analysis.avg_knee_angle_bottom),
      unit: "°",
    },
    { label: "Knee range", value: num(analysis.knee_range), unit: "°" },
    {
      label: "Upper-body stability",
      value: num(analysis.upper_body_stability),
      unit: "%",
    },
    {
      label: "Pedal smoothness",
      value: num(analysis.pedal_smoothness),
      unit: "/100",
    },
    {
      label: "Bike-fit score",
      value: num(analysis.bike_fit_score),
      unit: "/100",
    },
  ];

  return (
    <div className="animate-slide-up space-y-6">
      <div>
        <Link
          href="/cycling"
          className="mb-2 flex w-fit items-center gap-1 text-xs text-slate-500 transition-colors hover:text-slate-300"
        >
          <ArrowLeft className="h-3 w-3" /> Back to cycling
        </Link>
        <div className="mb-1.5 flex items-center gap-2">
          <StatusBadge variant="success" icon={Bike}>
            Cycling
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
          Bike-fit session {jobId.slice(0, 8)}
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
                className="h-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.5)] transition-all duration-300"
                style={{ width: `${progress.pct}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">
              SPRINT locks onto the cyclist, measures the complete pedal cycle,
              then rejects clips without enough pose evidence.
            </p>
          </div>
        </ChartContainer>
      ) : null}

      {error ? (
        <ChartContainer title="Capture quality is not sufficient" subtitle="No unsupported fit score was produced">
          <p className="text-sm leading-relaxed text-rose-300">{error}</p>
          <Link
            href="/cycling"
            className="mt-4 inline-flex h-9 items-center rounded-lg bg-white/[0.06] px-4 text-xs font-medium text-slate-200"
          >
            Record another clip
          </Link>
        </ChartContainer>
      ) : null}

      {done ? (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-black">
              <video
                src={cyclingAnnotatedVideoUrl(jobId)}
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
              <span>
                {num(result.quality?.pose_coverage, 1)}% pose coverage
              </span>
            </div>
            <div className="flex gap-3 rounded-xl border border-amber-400/10 bg-amber-400/[0.04] p-4">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
              <p className="text-xs leading-relaxed text-slate-400">
                This is a repeatable side-view screening measurement, not a
                medical diagnosis or a power-meter estimate. Confirm material
                fit changes with a qualified bike fitter.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <ChartContainer title="Bike-fit evidence" subtitle="Measured from this clip">
              <div className="space-y-2">
                {metrics.map((metric) => (
                  <div
                    key={metric.label}
                    className="flex items-center justify-between border-b border-white/[0.04] py-1.5 last:border-0"
                  >
                    <span className="text-xs text-slate-400">{metric.label}</span>
                    <span className="text-base font-bold text-slate-100 tnum">
                      {metric.value}
                      <span className="ml-0.5 text-xs font-medium text-slate-500">
                        {metric.unit}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            </ChartContainer>

            <ChartContainer title="Save to athlete history" subtitle="Build a comparable bike-fit baseline">
              <div className="space-y-3">
                <label
                  htmlFor="cycling-athlete"
                  className="block text-xs font-medium text-slate-400"
                >
                  Save for athlete
                </label>
                <select
                  id="cycling-athlete"
                  value={selectedAthleteId}
                  onChange={(event) =>
                    setSelectedAthleteId(event.target.value)
                  }
                  disabled={saving || savedSessionId !== null}
                  className="h-10 w-full rounded-lg border border-white/[0.08] bg-bg px-3 text-sm text-slate-100 outline-none focus:border-emerald-400/50"
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
                  className="flex h-9 w-full items-center justify-center gap-2 rounded-lg bg-emerald-400 px-4 text-xs font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
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
