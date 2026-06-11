import { Activity, CircleDot, ScanLine } from "lucide-react";

import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";

import type { RehabDemoFrame } from "./demoSession";
import { formatDemoTimestamp } from "./demoSession";
import { RehabBodyFigure } from "./RehabBodyFigure";
import { RomArc } from "./RomArc";

export function RehabDemoStage({
  frame,
  locale,
  running,
}: {
  frame: RehabDemoFrame;
  locale: RehabLocale;
  running: boolean;
}) {
  const copy = rehabCopy[locale].demo;
  const timelineProgress = Math.min(100, (frame.timestamp / 4.8) * 100);

  return (
    <section className="relative min-h-[460px] overflow-hidden rounded-[24px] border border-cyan-300/15 bg-[#060a0f] shadow-[0_32px_100px_rgba(0,0,0,.52)] md:min-h-[500px] md:rounded-[28px]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_55%_36%,rgba(34,211,238,.12),transparent_42%),linear-gradient(145deg,rgba(15,23,34,.96),rgba(3,7,11,.98))]" />
      <div className="absolute inset-x-3 top-3 z-20 flex items-center justify-between gap-2 md:inset-x-5 md:top-5 md:gap-3">
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-black/35 px-2.5 py-1.5 text-[8px] font-semibold uppercase tracking-[0.12em] text-slate-300 backdrop-blur-xl md:px-3 md:text-[10px] md:tracking-[0.16em]">
          <span className="h-1.5 w-1.5 rounded-full bg-cyan-300 shadow-[0_0_10px_#67e8f9]" />
          {copy.simulated}
        </div>
        <div className="flex items-center gap-1.5 font-mono text-[8px] uppercase tracking-[0.1em] text-slate-400 md:gap-2 md:text-[10px] md:tracking-[0.12em]">
          <ScanLine className="h-3.5 w-3.5 text-cyan-300" />
          {running ? copy.analyzing : copy.evidenceReady}
        </div>
      </div>

      <RomArc value={frame.leftRom} side="left" />
      <RomArc value={frame.rightRom} side="right" />

      <div className="absolute inset-x-0 bottom-16 top-12">
        <RehabBodyFigure frame={frame} />
      </div>

      <div className="absolute bottom-5 left-5 right-5 z-20 rounded-2xl border border-white/[0.08] bg-black/45 p-3 backdrop-blur-xl">
        <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-slate-500">
          <span className="flex items-center gap-1.5">
            <CircleDot className="h-3 w-3 text-emerald-300" />
            {copy.cycleEvidence}
          </span>
          <span className="font-mono text-cyan-100">
            {formatDemoTimestamp(frame.timestamp)}
          </span>
        </div>
        <div className="relative mt-3 h-1 overflow-hidden rounded-full bg-white/[0.06]">
          <span
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-cyan-400 to-emerald-300 transition-[width] duration-500"
            style={{ width: `${timelineProgress}%` }}
          />
        </div>
        <div className="mt-3 flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5 text-slate-400">
            <Activity className="h-3.5 w-3.5 text-cyan-300" />
            {copy.shoulderFlexion}
          </span>
          <span className="font-mono font-semibold text-emerald-200">
            {frame.confidence}% {copy.confidence}
          </span>
        </div>
      </div>
    </section>
  );
}
