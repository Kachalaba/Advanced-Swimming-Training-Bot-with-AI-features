"use client";

import {
  AlertCircle,
  ArrowLeft,
  Check,
  Clock3,
  Eye,
  Gauge,
  Loader2,
  Play,
  Save,
  Sparkles,
  Target,
  Waves,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { api, type Athlete } from "@/lib/api";
import {
  saveSwimmingAnalysis,
  subscribeSwimmingAnalysis,
  swimmingVideoUrl,
  type SwimmingAnalysisEvent,
  type SwimmingConfidenceLevel,
  type SwimmingErrorEvent,
  type SwimmingResultEvent,
  type SwimmingZone,
} from "@/lib/swimming";

const ZONE_LABELS: Record<SwimmingZone["id"], string> = {
  body_position: "Body position",
  rotation: "Torso rotation",
  catch: "Catch and arm path",
  breathing: "Breathing and head return",
  kick: "Kick",
};

const STAGE_DESCRIPTIONS: Record<string, string> = {
  quality_gate: "Checking blur, brightness, framing and camera angle",
  tracking: "Following one swimmer without silent identity switching",
  waterline: "Separating above-water and underwater visual evidence",
  pose: "Keeping only reliable body segments",
  cycles: "Ranking complete stroke cycles by evidence quality",
  technique: "Evaluating the five technique zones",
  coaching: "Connecting the main issue to a corrective drill",
  rendering: "Creating the annotated evidence video",
  completed: "Analysis complete",
};

function confidenceVariant(
  level: SwimmingConfidenceLevel,
): "success" | "warn" | "neutral" {
  if (level === "high") return "success";
  if (level === "medium") return "warn";
  return "neutral";
}

function formatMetricName(name: string): string {
  return name.replaceAll("_", " ");
}

function formatMetricValue(value: number | string): string {
  if (typeof value === "string") return value;
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) < 1 ? value.toFixed(3) : value.toFixed(1);
}

function ZoneCard({
  zone,
  onSeek,
}: {
  zone: SwimmingZone;
  onSeek: (seconds: number) => void;
}) {
  const label = ZONE_LABELS[zone.id];
  const evidence = zone.evidence[0];
  const statusVariant =
    zone.status === "good"
      ? "success"
      : zone.status === "needs_attention"
        ? "warn"
        : "neutral";
  const statusLabel =
    zone.status === "good"
      ? "Good"
      : zone.status === "needs_attention"
        ? "Needs attention"
        : "Insufficient data";

  return (
    <article className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">{label}</h3>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <StatusBadge variant={statusVariant}>{statusLabel}</StatusBadge>
            <StatusBadge variant={confidenceVariant(zone.confidence_level)}>
              {zone.confidence_level === "insufficient"
                ? "No confidence"
                : `${zone.confidence_level} confidence`}
            </StatusBadge>
          </div>
        </div>
        {zone.score !== null ? (
          <span className="font-mono text-xl font-bold text-slate-100 tnum">
            {Math.round(zone.score)}
          </span>
        ) : (
          <span className="text-xs text-slate-500">No score</span>
        )}
      </div>

      {Object.keys(zone.metrics).length > 0 ? (
        <div className="mt-4 grid grid-cols-2 gap-2">
          {Object.entries(zone.metrics)
            .slice(0, 2)
            .map(([name, value]) => (
              <div
                key={name}
                className="rounded-lg border border-white/[0.04] bg-black/10 px-3 py-2"
              >
                <p className="text-[10px] uppercase tracking-wider text-slate-600">
                  {formatMetricName(name)}
                </p>
                <p className="mt-1 font-mono text-sm font-semibold text-slate-200 tnum">
                  {formatMetricValue(value)}
                </p>
              </div>
            ))}
        </div>
      ) : null}

      {evidence ? (
        <button
          type="button"
          aria-label={`Show ${label} at ${evidence.peak_sec.toFixed(1)} seconds`}
          onClick={() => onSeek(evidence.peak_sec)}
          className="mt-4 inline-flex h-8 items-center gap-1.5 rounded-lg border border-cyan-400/20 bg-cyan-400/[0.07] px-3 text-xs font-medium text-cyan-200 transition hover:bg-cyan-400/10"
        >
          <Play className="h-3 w-3" />
          View at {evidence.peak_sec.toFixed(1)}s
        </button>
      ) : null}
    </article>
  );
}

function ProgressState({
  progress,
}: {
  progress: { stage: string; pct: number; label: string };
}) {
  return (
    <ChartContainer
      title={progress.label}
      subtitle={STAGE_DESCRIPTIONS[progress.stage] ?? "Analyzing video"}
      action={<span className="font-mono text-xs text-cyan-300 tnum">{progress.pct}%</span>}
    >
      <div className="h-2 overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full bg-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.45)] transition-all duration-300"
          style={{ width: `${progress.pct}%` }}
        />
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 md:grid-cols-4">
        {["Track swimmer", "Find waterline", "Select cycles", "Build coaching"].map(
          (step) => (
            <div
              key={step}
              className="rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-2 text-[11px] text-slate-500"
            >
              {step}
            </div>
          ),
        )}
      </div>
    </ChartContainer>
  );
}

function ErrorState({ error }: { error: SwimmingErrorEvent }) {
  return (
    <div className="rounded-2xl border border-rose-400/20 bg-rose-400/[0.05] p-6">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-rose-400/10 text-rose-300">
          <AlertCircle className="h-5 w-5" />
        </div>
        <div>
          <StatusBadge variant="danger">Analysis stopped</StatusBadge>
          <h2 className="mt-3 text-lg font-semibold text-slate-100">
            {error.message}
          </h2>
          {error.reshoot_guidance ? (
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
              {error.reshoot_guidance}
            </p>
          ) : null}
          <Link
            href="/swimming"
            className="mt-5 inline-flex h-9 items-center gap-2 rounded-lg bg-cyan-400 px-4 text-xs font-semibold text-slate-950 transition hover:bg-cyan-300"
          >
            Record another clip
          </Link>
        </div>
      </div>
    </div>
  );
}

export function SwimmingAnalysisWorkspace({
  jobId,
  initialResult,
}: {
  jobId: string;
  initialResult?: SwimmingResultEvent;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [progress, setProgress] = useState({
    stage: "quality_gate",
    pct: 0,
    label: "Connecting to analysis",
  });
  const [result, setResult] = useState<SwimmingResultEvent | null>(
    initialResult ?? null,
  );
  const [error, setError] = useState<SwimmingErrorEvent | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedSessionId, setSavedSessionId] = useState<number | null>(null);
  const [athletes, setAthletes] = useState<Athlete[]>([]);
  const [selectedAthleteId, setSelectedAthleteId] = useState("");
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (initialResult) return;
    return subscribeSwimmingAnalysis(jobId, (event: SwimmingAnalysisEvent) => {
      if (event.type === "progress") {
        setProgress({
          stage: event.stage,
          pct: event.pct,
          label: event.label,
        });
      } else if (event.type === "result") {
        setResult(event);
      } else {
        setError(event);
      }
    });
  }, [initialResult, jobId]);

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
        if (!active) return;
        setSaveError(
          athleteError instanceof Error
            ? athleteError.message
            : "Could not load athletes",
        );
      });
    return () => {
      active = false;
    };
  }, []);

  function seekTo(seconds: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = seconds;
  }

  async function saveResult() {
    if (!selectedAthleteId || saving || savedSessionId !== null) return;
    setSaving(true);
    setSaveError(null);
    try {
      const saved = await saveSwimmingAnalysis(jobId, {
        athleteId: Number(selectedAthleteId),
      });
      setSavedSessionId(saved.sessionId);
    } catch (saveError) {
      setSaveError(
        saveError instanceof Error
          ? saveError.message
          : "Could not save this analysis.",
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="animate-slide-up space-y-6">
      <header>
        <Link
          href="/swimming"
          className="mb-3 flex w-fit items-center gap-1 text-xs text-slate-500 transition hover:text-slate-300"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to swimming
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge variant="info" icon={Waves}>
            Freestyle · Side view
          </StatusBadge>
          {result ? (
            <StatusBadge variant="success" icon={Check}>
              Analysis complete
            </StatusBadge>
          ) : error ? (
            <StatusBadge variant="danger" icon={AlertCircle}>
              Needs a new clip
            </StatusBadge>
          ) : (
            <StatusBadge variant="info" icon={Loader2}>
              Analyzing
            </StatusBadge>
          )}
        </div>
        <h1 className="mt-3 text-2xl font-bold tracking-tight text-slate-50">
          Swimming analysis
        </h1>
        <p className="mt-1 text-xs text-slate-500">
          Session {jobId.slice(0, 8)} · Waterline-aware hybrid
        </p>
      </header>

      {!result && !error ? <ProgressState progress={progress} /> : null}
      {error ? <ErrorState error={error} /> : null}

      {result ? (
        <>
          {result.primary_issue ? (
            <section className="relative overflow-hidden rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-cyan-400/[0.08] via-surface to-bg p-6">
              <div className="absolute -right-16 -top-20 h-56 w-56 rounded-full bg-cyan-400/[0.08] blur-3xl" />
              <div className="relative">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge variant="warn" icon={Target}>
                    Main issue
                  </StatusBadge>
                  <StatusBadge
                    variant={confidenceVariant(
                      result.primary_issue.confidence_level,
                    )}
                  >
                    {result.primary_issue.confidence_level} confidence
                  </StatusBadge>
                  <span className="text-xs text-slate-500">
                    Confirmed in {result.primary_issue.confirming_cycles} cycles
                  </span>
                </div>
                <h2 className="mt-4 text-xl font-bold tracking-tight text-slate-50 md:text-2xl">
                  {result.primary_issue.title}
                </h2>
                <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-300">
                  {result.primary_issue.why_it_matters}
                </p>
                {result.primary_issue.evidence[0] ? (
                  <button
                    type="button"
                    onClick={() =>
                      seekTo(result.primary_issue?.evidence[0].peak_sec ?? 0)
                    }
                    className="mt-5 inline-flex h-9 items-center gap-2 rounded-lg bg-cyan-400 px-4 text-xs font-semibold text-slate-950 transition hover:bg-cyan-300"
                  >
                    <Play className="h-3.5 w-3.5 fill-current" />
                    Show exact moment ·{" "}
                    {result.primary_issue.evidence[0].peak_sec.toFixed(1)}s
                  </button>
                ) : null}
              </div>
            </section>
          ) : (
            <ChartContainer
              title="No repeated primary issue confirmed"
              subtitle="The system did not find the same medium-confidence problem in two cycles"
            >
              <p className="text-sm text-slate-400">
                Review the available technique zones below without treating a
                single unusual frame as a diagnosis.
              </p>
            </ChartContainer>
          )}

          <section className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.7fr)_minmax(300px,.7fr)]">
            <div>
              <div className="overflow-hidden rounded-xl border border-white/[0.07] bg-black">
                <video
                  ref={videoRef}
                  data-testid="swimming-video"
                  src={swimmingVideoUrl(jobId)}
                  controls
                  playsInline
                  preload="metadata"
                  className="aspect-video w-full bg-black"
                />
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {result.cycles.map((cycle, index) => (
                  <button
                    key={cycle.id}
                    type="button"
                    aria-label={`Show cycle ${index + 1} at ${cycle.peak_sec.toFixed(1)} seconds`}
                    onClick={() => seekTo(cycle.peak_sec)}
                    className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-white/[0.07] bg-white/[0.03] px-3 text-xs text-slate-300 transition hover:border-cyan-400/20 hover:text-cyan-200"
                  >
                    Cycle {index + 1}
                    <span className="font-mono text-[10px] text-slate-600 tnum">
                      {Math.round(cycle.quality * 100)}%
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <ChartContainer
              title="Evidence quality"
              subtitle={`${result.coverage.available_zones} of ${result.coverage.total_zones} zones analyzed`}
              action={<Eye className="h-4 w-4 text-cyan-300" />}
            >
              <div className="space-y-3 text-xs">
                <div className="flex items-center justify-between border-b border-white/[0.05] pb-3">
                  <span className="text-slate-500">Selected cycles</span>
                  <span className="font-mono font-semibold text-slate-200 tnum">
                    {result.cycles.length}
                  </span>
                </div>
                <div className="flex items-center justify-between border-b border-white/[0.05] pb-3">
                  <span className="text-slate-500">Pose coverage</span>
                  <span className="font-mono font-semibold text-slate-200 tnum">
                    {result.frames_total > 0
                      ? Math.round(
                          (result.frames_with_pose / result.frames_total) * 100,
                        )
                      : 0}
                    %
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-500">Analysis contract</span>
                  <span className="font-mono text-slate-300">
                    v{result.contract_version}
                  </span>
                </div>
              </div>
              {result.quality.warnings.length > 0 ? (
                <div className="mt-4 space-y-2">
                  {result.quality.warnings.map((warning) => (
                    <p
                      key={warning}
                      className="rounded-lg border border-amber-400/15 bg-amber-400/[0.05] px-3 py-2 text-[11px] leading-relaxed text-amber-200"
                    >
                      {warning}
                    </p>
                  ))}
                </div>
              ) : null}
            </ChartContainer>
          </section>

          <section>
            <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">
                  Technique zones
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Missing evidence is excluded, never converted into a zero.
                </p>
              </div>
              <span className="rounded-md border border-white/[0.06] bg-white/[0.03] px-2 py-1 text-xs text-slate-400">
                {result.coverage.available_zones} of{" "}
                {result.coverage.total_zones} zones analyzed
              </span>
            </div>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
              {result.zones.map((zone) => (
                <ZoneCard key={zone.id} zone={zone} onSeek={seekTo} />
              ))}
            </div>
          </section>

          {result.prescription ? (
            <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <ChartContainer
                title="Corrective drill"
                subtitle="One focused change before adding complexity"
                action={<Target className="h-4 w-4 text-cyan-300" />}
              >
                <h3 className="text-base font-semibold text-slate-100">
                  {result.prescription.drill.name}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  {result.prescription.drill.purpose}
                </p>
                <div className="mt-4 space-y-3 text-xs">
                  <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-3">
                    <p className="font-medium text-slate-300">How to do it</p>
                    <p className="mt-1 leading-relaxed text-slate-500">
                      {result.prescription.drill.execution}
                    </p>
                  </div>
                  <div className="rounded-lg border border-amber-400/10 bg-amber-400/[0.035] p-3">
                    <p className="font-medium text-amber-200">Watch for</p>
                    <p className="mt-1 leading-relaxed text-slate-500">
                      {result.prescription.drill.common_mistake}
                    </p>
                  </div>
                  <div className="rounded-lg border border-emerald-400/10 bg-emerald-400/[0.035] p-3">
                    <p className="font-medium text-emerald-200">Success cue</p>
                    <p className="mt-1 leading-relaxed text-slate-500">
                      {result.prescription.drill.success_cue}
                    </p>
                  </div>
                </div>
              </ChartContainer>

              <ChartContainer
                title="Next-workout mini-set"
                subtitle={result.prescription.mini_set.title}
                action={<Sparkles className="h-4 w-4 text-cyan-300" />}
              >
                <div className="rounded-xl border border-cyan-400/15 bg-cyan-400/[0.045] p-5">
                  <p className="font-mono text-3xl font-bold tracking-tight text-cyan-200 tnum">
                    {result.prescription.mini_set.repetitions} ×{" "}
                    {result.prescription.mini_set.distance_m} m
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <StatusBadge variant="info" icon={Clock3}>
                      {result.prescription.mini_set.rest_sec}s rest
                    </StatusBadge>
                    <StatusBadge variant="neutral" icon={Gauge}>
                      {result.prescription.mini_set.intensity}
                    </StatusBadge>
                  </div>
                  <p className="mt-5 text-xs uppercase tracking-wider text-slate-600">
                    Focus
                  </p>
                  <p className="mt-1 text-sm leading-relaxed text-slate-300">
                    {result.prescription.mini_set.focus}
                  </p>
                </div>
              </ChartContainer>
            </section>
          ) : null}

          <div className="flex flex-wrap items-end justify-between gap-4 rounded-xl border border-white/[0.06] bg-surface p-4">
            <div>
              <p className="text-sm font-medium text-slate-200">
                Keep this result in athlete history
              </p>
              <p className="mt-0.5 text-xs text-slate-500">
                Save zone confidence, selected cycles, main issue and video.
              </p>
            </div>
            <div className="w-full space-y-2 sm:w-auto sm:min-w-64">
              <label
                htmlFor="swimming-athlete"
                className="block text-xs font-medium text-slate-400"
              >
                Save for athlete
              </label>
              <div className="flex flex-col gap-2 sm:flex-row">
                <select
                  id="swimming-athlete"
                  value={selectedAthleteId}
                  onChange={(event) =>
                    setSelectedAthleteId(event.target.value)
                  }
                  disabled={saving || savedSessionId !== null}
                  className="h-9 min-w-44 rounded-lg border border-white/[0.08] bg-bg px-3 text-xs text-slate-100 outline-none focus:border-cyan-400/50"
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
                  disabled={
                    !selectedAthleteId || saving || savedSessionId !== null
                  }
                  onClick={() => void saveResult()}
                  className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-cyan-400/25 bg-cyan-400/10 px-4 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-400/15 disabled:cursor-default disabled:border-emerald-400/20 disabled:bg-emerald-400/10 disabled:text-emerald-200"
                >
                  {savedSessionId !== null ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : saving ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Save className="h-3.5 w-3.5" />
                  )}
                  {savedSessionId !== null
                    ? `Saved to history #${savedSessionId}`
                    : saving
                      ? "Saving…"
                      : "Save to history"}
                </button>
              </div>
              {saveError ? (
                <p className="text-xs text-rose-300">{saveError}</p>
              ) : null}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
