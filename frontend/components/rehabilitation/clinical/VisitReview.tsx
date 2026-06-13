"use client";

import { Activity, AlertTriangle, CheckCircle2 } from "lucide-react";

import type {
  CaptureQuality,
  ClinicalProgressObservation,
  RehabEpisode,
} from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";

import type { RehabAnalysisSnapshot } from "../rehabHandoff";

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${Math.round(value)}°`;
}

export function VisitReview({
  episode,
  baseline,
  snapshot,
  locale,
  observation,
  quality,
  qualityDetails,
  warningAcknowledged,
  submitting,
  onObservationChange,
  onQualityChange,
  onQualityDetailsChange,
  onWarningAcknowledgedChange,
  onFinalize,
}: {
  episode: RehabEpisode;
  baseline: ClinicalProgressObservation | null;
  snapshot: RehabAnalysisSnapshot;
  locale: RehabLocale;
  observation: string;
  quality: CaptureQuality;
  qualityDetails: string;
  warningAcknowledged: boolean;
  submitting: boolean;
  onObservationChange: (value: string) => void;
  onQualityChange: (value: CaptureQuality) => void;
  onQualityDetailsChange: (value: string) => void;
  onWarningAcknowledgedChange: (value: boolean) => void;
  onFinalize: () => void;
}) {
  const copy = clinicalCopy[locale];
  const currentLeft = snapshot.report.target_metrics.left.rom;
  const currentRight = snapshot.report.target_metrics.right.rom;
  const warningRequired = quality === "accepted_with_warning";
  const canFinalize =
    observation.trim().length > 0 &&
    quality !== "repeat_required" &&
    (!warningRequired || warningAcknowledged) &&
    !submitting;

  return (
    <div className="grid gap-5 xl:grid-cols-[1.05fr_.95fr]">
      <section className="rounded-2xl border border-white/[0.07] bg-[#091017] p-5">
        <div className="flex items-center gap-2 text-cyan-300">
          <Activity className="h-4 w-4" />
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em]">
            {copy.visit.baselineComparison}
          </p>
        </div>
        <h2 className="mt-3 text-xl font-semibold text-slate-50">
          {episode.title}
        </h2>
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {[
            {
              label: rehabCopy[locale].demo.metrics.leftRom,
              value: `${Math.round(currentLeft)}°`,
              delta: baseline
                ? signed(currentLeft - baseline.leftRom)
                : "—",
            },
            {
              label: rehabCopy[locale].demo.metrics.rightRom,
              value: `${Math.round(currentRight)}°`,
              delta: baseline
                ? signed(currentRight - baseline.rightRom)
                : "—",
            },
          ].map((metric) => (
            <div
              key={metric.label}
              className="rounded-xl border border-white/[0.06] bg-black/10 p-4"
            >
              <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">
                {metric.label}
              </p>
              <div className="mt-2 flex items-end justify-between gap-3">
                <span className="text-2xl font-semibold text-slate-50">
                  {metric.value}
                </span>
                <span className="text-sm font-semibold text-emerald-300">
                  {metric.delta}
                </span>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <div className="rounded-xl border border-white/[0.06] bg-black/10 p-4">
            <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">
              {rehabCopy[locale].demo.metrics.symmetry}
            </p>
            <p className="mt-2 text-2xl font-semibold text-cyan-200">
              {Math.round(snapshot.report.symmetry.score)}%
            </p>
          </div>
          <div className="rounded-xl border border-white/[0.06] bg-black/10 p-4">
            <p className="text-[10px] uppercase tracking-[0.13em] text-slate-500">
              {rehabCopy[locale].demo.metrics.repetitions}
            </p>
            <p className="mt-2 text-2xl font-semibold text-cyan-200">
              {snapshot.report.total_correct_reps}
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-white/[0.07] bg-[#091017] p-5">
        <label className="block text-xs font-semibold text-slate-300">
          {copy.visit.specialistObservation}
          <textarea
            aria-label={copy.visit.specialistObservation}
            rows={5}
            value={observation}
            onChange={(event) => onObservationChange(event.target.value)}
            className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-[#0c141d] px-3 py-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
          />
        </label>
        <label className="mt-4 block text-xs font-semibold text-slate-300">
          {copy.visit.quality}
          <select
            aria-label={copy.visit.quality}
            value={quality}
            onChange={(event) =>
              onQualityChange(event.target.value as CaptureQuality)
            }
            className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
          >
            {(
              [
                "acceptable",
                "accepted_with_warning",
                "repeat_required",
              ] as const
            ).map((value) => (
              <option key={value} value={value}>
                {copy.quality[value]}
              </option>
            ))}
          </select>
        </label>
        <label className="mt-4 block text-xs font-semibold text-slate-300">
          {copy.visit.qualityDetails}
          <textarea
            aria-label={copy.visit.qualityDetails}
            rows={2}
            value={qualityDetails}
            onChange={(event) => onQualityDetailsChange(event.target.value)}
            className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-[#0c141d] px-3 py-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
          />
        </label>
        {warningRequired ? (
          <label className="mt-4 flex items-start gap-3 rounded-xl border border-amber-300/15 bg-amber-300/[0.05] p-3 text-xs leading-relaxed text-amber-100">
            <input
              type="checkbox"
              checked={warningAcknowledged}
              onChange={(event) =>
                onWarningAcknowledgedChange(event.target.checked)
              }
              className="mt-0.5 h-4 w-4 accent-amber-300"
            />
            {copy.visit.warningAcknowledgement}
          </label>
        ) : null}
        {quality === "repeat_required" ? (
          <div className="mt-4 flex gap-2 rounded-xl border border-rose-400/15 bg-rose-400/[0.05] p-3 text-xs text-rose-200">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            {copy.visit.repeatMeasurement}
          </div>
        ) : null}
        <button
          type="button"
          disabled={!canFinalize}
          onClick={onFinalize}
          className="mt-5 inline-flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-emerald-300 px-5 text-sm font-semibold text-[#031016] disabled:cursor-not-allowed disabled:opacity-35"
        >
          <CheckCircle2 className="h-4 w-4" />
          {copy.visit.finalize}
        </button>
      </section>
    </div>
  );
}
