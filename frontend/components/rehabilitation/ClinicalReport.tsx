"use client";

import { FileDown, ShieldCheck, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { clinicalCopy } from "@/lib/clinicalCopy";
import { rehabCopy } from "@/lib/rehabCopy";

import type { RehabDelta, RehabHandoff } from "./rehabHandoff";

function Metric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="report-section rounded-xl border border-slate-200 p-3">
      <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">
        {label}
      </div>
      <div className="mt-1 font-mono text-2xl font-semibold text-slate-950 tnum">
        {value}
      </div>
    </div>
  );
}

function signed(value: number, suffix = ""): string {
  return `${value >= 0 ? "+" : ""}${Math.round(value * 10) / 10}${suffix}`;
}

function comparisonText(
  label: string,
  delta: RehabDelta | null,
  unavailable: string,
): string {
  if (!delta) return `${label}: ${unavailable}`;
  return `${label}: L ${signed(delta.leftRom, "°")} · R ${signed(
    delta.rightRom,
    "°",
  )} · symmetry ${signed(delta.symmetry, " pp")}`;
}

export function ClinicalReport({
  handoff,
  onClose,
}: {
  handoff: RehabHandoff;
  onClose: () => void;
}) {
  const copy = rehabCopy[handoff.locale];
  const clinicalLabels = clinicalCopy[handoff.locale];
  const handoffCopy = copy.handoff;
  const closeRef = useRef<HTMLButtonElement>(null);
  const [patientCode, setPatientCode] = useState("");
  const [clinicianNote, setClinicianNote] = useState("");
  const [printError, setPrintError] = useState<string | null>(null);
  const dateLocale = handoff.locale === "uk" ? "uk-UA" : "en-GB";
  const recordedAt = new Intl.DateTimeFormat(dateLocale, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(handoff.recordedAt));
  const unavailable = handoffCopy.notAvailable;

  useEffect(() => {
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const print = () => {
    setPrintError(null);
    try {
      window.print();
    } catch {
      setPrintError(handoffCopy.printError);
    }
  };

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={handoffCopy.reportTitle}
      className="fixed inset-0 z-[120] overflow-y-auto bg-[#03070b]/95 px-3 py-4 backdrop-blur-xl sm:px-6 sm:py-8"
    >
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 pb-4 print:hidden">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300">
            SPRINT AI · Clinical Handoff
          </div>
          <h2 className="mt-1 text-xl font-semibold text-slate-50">
            {handoffCopy.reportTitle}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={print}
            className="inline-flex h-10 items-center gap-2 rounded-xl bg-cyan-300 px-4 text-sm font-semibold text-[#031016] transition hover:bg-cyan-200"
          >
            <FileDown className="h-4 w-4" />
            {handoffCopy.exportPdf}
          </button>
          <button
            ref={closeRef}
            type="button"
            aria-label={handoffCopy.closeReport}
            onClick={onClose}
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/[0.08]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {printError ? (
        <p className="mx-auto mb-3 max-w-5xl rounded-lg border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-xs text-rose-200 print:hidden">
          {printError}
        </p>
      ) : null}

      <article
        data-print-root
        className="relative mx-auto min-h-[1120px] max-w-5xl overflow-hidden bg-white px-6 py-7 text-slate-900 shadow-[0_30px_100px_rgba(0,0,0,.45)] sm:px-10 sm:py-10"
      >
        {handoff.source === "demo" ? (
          <div className="pointer-events-none absolute right-[-46px] top-10 rotate-45 border-y border-cyan-700/20 bg-cyan-50 px-14 py-1.5 text-[10px] font-bold tracking-[0.18em] text-cyan-800">
            {handoffCopy.simulatedWatermark}
          </div>
        ) : null}

        <header className="report-section flex flex-col justify-between gap-5 border-b border-slate-200 pb-6 sm:flex-row sm:items-start">
          <div>
            <div className="text-sm font-black tracking-tight text-slate-950">
              SPRINT AI
            </div>
            <div className="mt-1 text-[9px] font-semibold uppercase tracking-[0.2em] text-cyan-700">
              Clinical movement intelligence
            </div>
            <h1 className="mt-5 text-3xl font-semibold tracking-tight text-slate-950">
              {handoffCopy.reportTitle}
            </h1>
          </div>
          <div className="text-left text-xs leading-relaxed text-slate-500 sm:text-right">
            <div>{handoffCopy.generatedAt}</div>
            <div className="mt-1 font-mono font-semibold text-slate-800">
              {recordedAt}
            </div>
          </div>
        </header>

        <section className="report-section mt-6 grid gap-3 sm:grid-cols-2">
          <div className="rounded-xl bg-slate-50 p-4">
            <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
              {handoffCopy.source}
            </div>
            <div className="mt-1 text-sm font-semibold text-slate-900">
              {handoffCopy.sourceLabels[handoff.source]}
            </div>
          </div>
          <div className="rounded-xl bg-slate-50 p-4">
            <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
              {handoffCopy.protocol}
            </div>
            <div className="mt-1 text-sm font-semibold text-slate-900">
              {copy.protocols[handoff.protocol]}
            </div>
          </div>
        </section>

        {handoff.clinical ? (
          <section className="report-section mt-6 rounded-2xl border border-cyan-900/10 bg-cyan-50/50 p-5">
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                [handoffCopy.patient, handoff.clinical.patientName],
                [handoffCopy.episode, handoff.clinical.episodeTitle],
                [
                  handoffCopy.functionalGoal,
                  handoff.clinical.functionalGoal,
                ],
                [
                  handoffCopy.measurementQuality,
                  clinicalLabels.quality[handoff.clinical.captureQuality],
                ],
              ].map(([label, value]) => (
                <div key={label}>
                  <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
                    {label}
                  </div>
                  <div className="mt-1 text-sm font-semibold text-slate-900">
                    {value}
                  </div>
                </div>
              ))}
            </div>
            {handoff.clinical.qualityDetails ? (
              <p className="mt-4 rounded-lg bg-white/70 px-3 py-2 text-xs leading-relaxed text-slate-600">
                {handoff.clinical.qualityDetails}
              </p>
            ) : null}
          </section>
        ) : (
          <section className="report-section mt-6 rounded-2xl border border-cyan-900/10 bg-cyan-50/50 p-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="block">
                <span className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
                  {handoffCopy.patientCode}
                </span>
                <input
                  aria-label={handoffCopy.patientCode}
                  value={patientCode}
                  onChange={(event) => setPatientCode(event.target.value)}
                  placeholder={handoffCopy.patientCodePlaceholder}
                  className="mt-2 h-10 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus:border-cyan-600"
                />
              </label>
              <label className="block">
                <span className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
                  {handoffCopy.clinicianNote}
                </span>
                <textarea
                  aria-label={handoffCopy.clinicianNote}
                  value={clinicianNote}
                  onChange={(event) => setClinicianNote(event.target.value)}
                  placeholder={handoffCopy.clinicianNotePlaceholder}
                  rows={2}
                  className="mt-2 w-full resize-none rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none placeholder:text-slate-400 focus:border-cyan-600"
                />
              </label>
            </div>
            <p className="mt-2 text-[10px] text-slate-500 print:hidden">
              {handoffCopy.localOnly}
            </p>
          </section>
        )}

        <section className="report-section mt-7">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">
            {handoffCopy.measurements}
          </h2>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric label={copy.demo.metrics.leftRom} value={`${handoff.leftRom}°`} />
            <Metric label={copy.demo.metrics.rightRom} value={`${handoff.rightRom}°`} />
            <Metric label={copy.demo.metrics.symmetry} value={`${handoff.symmetry}%`} />
            <Metric label={copy.demo.metrics.repetitions} value={String(handoff.repetitions)} />
            <Metric label={copy.demo.target} value={`${handoff.completionScore}%`} />
            <Metric
              label={handoffCopy.confidence}
              value={
                handoff.confidence === null
                  ? unavailable
                  : `${handoff.confidence}%`
              }
            />
            <Metric
              label={handoffCopy.poseCoverage}
              value={
                handoff.poseCoverage === null
                  ? unavailable
                  : `${handoff.poseCoverage}%`
              }
            />
            <Metric
              label={handoffCopy.evidenceTimestamp}
              value={handoff.evidenceTimestamp ?? unavailable}
            />
          </div>
        </section>

        {handoff.clinical ? (
          <section className="report-section mt-7 rounded-2xl border border-slate-200 p-5">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.17em] text-cyan-800">
              {handoffCopy.baselineComparison}
            </h2>
            <p className="mt-3 text-sm font-semibold text-slate-800">
              {comparisonText(
                handoffCopy.baselineComparison,
                handoff.clinical.baselineDelta,
                unavailable,
              )}
            </p>
            <p className="mt-2 text-xs text-slate-600">
              {comparisonText(
                handoffCopy.previousComparison,
                handoff.clinical.previousDelta,
                unavailable,
              )}
            </p>
          </section>
        ) : null}

        <section className="report-section mt-7 rounded-2xl border border-slate-200 p-5">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.17em] text-cyan-800">
            <ShieldCheck className="h-4 w-4" />
            {handoffCopy.clinicalObservation}
          </div>
          <h2 className="mt-3 text-xl font-semibold text-slate-950">
            {handoff.findingTitle}
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">
            {handoff.findingBody}
          </p>
          {handoff.clinical ? (
            <div className="mt-4 border-t border-slate-200 pt-4">
              <div className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500">
                {handoffCopy.specialistObservation}
              </div>
              <p className="mt-2 text-sm font-semibold leading-relaxed text-slate-800">
                {handoff.clinical.specialistObservation}
              </p>
            </div>
          ) : null}
        </section>

        <section className="report-section mt-7 border-t border-slate-200 pt-5">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.17em] text-amber-700">
            {copy.limitationTitle}
          </h2>
          <p className="mt-2 text-xs leading-relaxed text-slate-600">
            {handoff.disclaimer}
          </p>
        </section>

        <footer className="report-section mt-8 flex items-start justify-between gap-6 border-t border-slate-200 pt-5 text-[10px] leading-relaxed text-slate-500">
          <p className="max-w-2xl">{handoffCopy.reportFooter}</p>
          <span className="shrink-0 font-mono">SPRINT AI · v0.1</span>
        </footer>
      </article>
    </div>,
    document.body,
  );
}
