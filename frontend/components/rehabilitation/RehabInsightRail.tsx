import { Activity, BadgeCheck, Repeat2, ScanLine } from "lucide-react";

import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";

import type { RehabDemoFrame } from "./demoSession";

export function RehabInsightRail({
  frame,
  locale,
}: {
  frame: RehabDemoFrame;
  locale: RehabLocale;
}) {
  const copy = rehabCopy[locale].demo.metrics;
  const metrics = [
    { label: copy.leftRom, value: `${frame.leftRom}°`, tone: "text-cyan-200" },
    {
      label: copy.rightRom,
      value: `${frame.rightRom}°`,
      tone: "text-emerald-200",
    },
    {
      label: copy.symmetry,
      value: `${frame.symmetry}%`,
      tone: "text-slate-50",
    },
    {
      label: copy.repetitions,
      value: String(frame.repetitions),
      tone: "text-slate-50",
    },
  ];

  return (
    <section className="grid gap-px overflow-hidden rounded-2xl border border-white/[0.07] bg-white/[0.07] sm:grid-cols-2 xl:grid-cols-5">
      {metrics.map((metric, index) => (
        <div
          key={metric.label}
          className="relative bg-[#0a1017] px-4 py-4"
        >
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-500">
            {index === 3 ? (
              <Repeat2 className="h-3.5 w-3.5" />
            ) : (
              <Activity className="h-3.5 w-3.5" />
            )}
            {metric.label}
          </div>
          <div className={`mt-2 font-mono text-2xl font-semibold tnum ${metric.tone}`}>
            {metric.value}
          </div>
        </div>
      ))}
      <div className="bg-[#0a1017] px-4 py-4">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-500">
          <ScanLine className="h-3.5 w-3.5" />
          {copy.pose}
        </div>
        <div className="mt-2 flex items-center gap-2 text-sm font-semibold text-emerald-200">
          <BadgeCheck className="h-5 w-5" />
          {copy.detected}
        </div>
      </div>
    </section>
  );
}
