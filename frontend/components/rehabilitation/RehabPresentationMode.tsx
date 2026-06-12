"use client";

import {
  FileText,
  Maximize2,
  Play,
  ShieldCheck,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { createPortal } from "react-dom";

import { rehabCopy } from "@/lib/rehabCopy";

import { demoFrames, type RehabDemoFrame } from "./demoSession";
import { RehabBodyFigure } from "./RehabBodyFigure";
import { RehabDemoStage } from "./RehabDemoStage";
import type { RehabHandoff } from "./rehabHandoff";

function PresentationMetric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "cyan" | "green";
}) {
  return (
    <div className="border-l border-white/[0.08] px-4 first:border-l-0">
      <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">
        {label}
      </div>
      <div
        className={`mt-1 font-mono text-2xl font-semibold tnum ${
          accent === "green" ? "text-emerald-200" : "text-cyan-100"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function RealEvidenceStage({ handoff }: { handoff: RehabHandoff }) {
  const copy = rehabCopy[handoff.locale].handoff;
  const frame = useMemo<RehabDemoFrame>(
    () => ({
      timestamp: 0,
      leftRom: handoff.leftRom,
      rightRom: handoff.rightRom,
      symmetry: handoff.symmetry,
      confidence: handoff.confidence ?? 0,
      repetitions: handoff.repetitions,
      phase: "complete",
    }),
    [handoff],
  );

  return (
    <section className="relative min-h-[460px] overflow-hidden rounded-[26px] border border-cyan-300/15 bg-[#060a0f]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_38%,rgba(34,211,238,.13),transparent_42%),linear-gradient(145deg,rgba(15,23,34,.96),rgba(3,7,11,.98))]" />
      <div className="absolute left-5 top-5 z-20 flex items-center gap-2 rounded-full border border-white/10 bg-black/35 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-300 backdrop-blur-xl">
        <ShieldCheck className="h-3.5 w-3.5 text-emerald-300" />
        {copy.evidenceSummary}
      </div>
      <div className="absolute inset-x-0 bottom-16 top-10">
        <RehabBodyFigure frame={frame} />
      </div>
      <div className="absolute bottom-5 left-5 right-5 z-20 rounded-2xl border border-white/[0.08] bg-black/55 p-4 backdrop-blur-xl">
        <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-cyan-300">
          {copy.clinicalObservation}
        </div>
        <p className="mt-2 text-sm font-medium text-slate-100">
          {handoff.findingTitle}
        </p>
      </div>
    </section>
  );
}

export function RehabPresentationMode({
  handoff,
  onClose,
  onOpenReport,
  onReplayDemo,
}: {
  handoff: RehabHandoff;
  onClose: () => void;
  onOpenReport: () => void;
  onReplayDemo?: () => void;
}) {
  const copy = rehabCopy[handoff.locale];
  const handoffCopy = copy.handoff;
  const rootRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const root = rootRef.current;
    if (root && typeof root.requestFullscreen === "function") {
      void root.requestFullscreen().catch(() => undefined);
    }
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const close = async () => {
    if (document.fullscreenElement) {
      await document.exitFullscreen?.().catch(() => undefined);
    }
    onClose();
  };

  const metrics = [
    [copy.demo.metrics.leftRom, `${handoff.leftRom}°`, "cyan"],
    [copy.demo.metrics.rightRom, `${handoff.rightRom}°`, "green"],
    [copy.demo.metrics.symmetry, `${handoff.symmetry}%`, "cyan"],
    [copy.demo.metrics.repetitions, String(handoff.repetitions), "cyan"],
    [handoffCopy.completion, `${handoff.completionScore}%`, "green"],
  ] as const;

  return createPortal(
    <div
      ref={rootRef}
      role="dialog"
      aria-modal="true"
      aria-label={handoffCopy.presentation}
      className="fixed inset-0 z-[110] overflow-y-auto bg-[#03070b] px-4 py-4 text-slate-100 sm:px-6 sm:py-6"
    >
      <div className="mx-auto flex min-h-full max-w-[1500px] flex-col">
        <header className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.07] pb-4">
          <div className="flex items-center gap-4">
            <div>
              <div className="text-base font-black tracking-tight">SPRINT AI</div>
              <div className="text-[9px] font-semibold uppercase tracking-[0.2em] text-cyan-300">
                Clinical movement intelligence
              </div>
            </div>
            <span className="hidden h-8 w-px bg-white/10 sm:block" />
            <div className="hidden text-xs text-slate-500 sm:block">
              {handoffCopy.sourceLabels[handoff.source]} ·{" "}
              <span className="text-slate-300">
                {copy.protocols[handoff.protocol]}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-[10px] uppercase tracking-[0.14em] text-slate-400 md:flex">
              <Maximize2 className="h-3.5 w-3.5 text-cyan-300" />
              {handoffCopy.presentation}
            </span>
            <button
              ref={closeRef}
              type="button"
              aria-label={handoffCopy.closePresentation}
              onClick={() => void close()}
              className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/[0.08]"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        <main className="grid flex-1 items-center gap-7 py-6 xl:grid-cols-[.62fr_1.38fr]">
          <section>
            <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-300">
              {handoffCopy.sourceLabels[handoff.source]}
            </div>
            <h1 className="mt-4 max-w-xl text-4xl font-semibold leading-[1.05] tracking-[-0.04em] text-slate-50 md:text-5xl">
              {handoff.findingTitle}
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-relaxed text-slate-400 md:text-base">
              {handoff.findingBody}
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              {handoff.source === "demo" && onReplayDemo ? (
                <button
                  type="button"
                  onClick={onReplayDemo}
                  className="inline-flex h-11 items-center gap-2 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] transition hover:bg-cyan-200"
                >
                  <Play className="h-4 w-4 fill-current" />
                  {handoffCopy.replayDemo}
                </button>
              ) : null}
              <button
                type="button"
                onClick={onOpenReport}
                className="inline-flex h-11 items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-5 text-sm font-semibold text-slate-100 transition hover:border-cyan-300/30 hover:bg-white/[0.07]"
              >
                <FileText className="h-4 w-4 text-cyan-300" />
                {handoffCopy.openReport}
              </button>
            </div>
          </section>

          {handoff.source === "demo" ? (
            <RehabDemoStage
              frame={demoFrames[demoFrames.length - 1]}
              locale={handoff.locale}
              running={false}
            />
          ) : (
            <RealEvidenceStage handoff={handoff} />
          )}
        </main>

        <footer>
          <div className="grid overflow-hidden rounded-2xl border border-white/[0.07] bg-[#091017] sm:grid-cols-5">
            {metrics.map(([label, value, accent]) => (
              <PresentationMetric
                key={label}
                label={label}
                value={value}
                accent={accent}
              />
            ))}
          </div>
          <p className="mx-auto mt-4 max-w-4xl text-center text-[10px] leading-relaxed text-slate-600">
            {handoff.disclaimer}
          </p>
        </footer>
      </div>
    </div>,
    document.body,
  );
}
