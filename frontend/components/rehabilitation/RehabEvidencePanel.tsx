import { ArrowRight, Crosshair, Sparkles } from "lucide-react";

import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";

export function RehabEvidencePanel({ locale }: { locale: RehabLocale }) {
  const copy = rehabCopy[locale].demo;

  return (
    <section className="grid gap-4 lg:grid-cols-[1.35fr_.65fr]">
      <article className="relative overflow-hidden rounded-2xl border border-cyan-300/15 bg-[linear-gradient(135deg,rgba(10,19,28,.98),rgba(6,10,15,.98))] p-5 md:p-6">
        <div className="absolute right-0 top-0 h-40 w-40 rounded-full bg-cyan-300/[0.08] blur-3xl" />
        <div className="relative">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-cyan-300">
            <Sparkles className="h-3.5 w-3.5" />
            {copy.findingLabel}
          </div>
          <h2 className="mt-3 max-w-2xl text-xl font-semibold tracking-tight text-slate-50 md:text-2xl">
            {copy.findingTitle}
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
            {copy.findingBody}
          </p>
          <div className="mt-5 flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-300/[0.07] px-3 py-1.5 font-mono text-xs text-emerald-200">
              <Crosshair className="h-3.5 w-3.5" />
              {copy.evidenceFrame} · 0:03.2
            </span>
            <span className="font-mono text-xs text-slate-500">
              154° <ArrowRight className="mx-1 inline h-3.5 w-3.5" /> 136°
            </span>
          </div>
        </div>
      </article>

      <article className="rounded-2xl border border-white/[0.07] bg-[#0a1017] p-5 md:p-6">
        <div className="text-[10px] font-semibold uppercase tracking-[0.17em] text-slate-500">
          {copy.comparisonTitle}
        </div>
        <div className="mt-5 flex items-end justify-between gap-4">
          <div>
            <div className="text-xs text-slate-500">{copy.before}</div>
            <div className="mt-1 font-mono text-3xl font-semibold text-slate-400 tnum">
              118°
            </div>
          </div>
          <ArrowRight className="mb-2 h-5 w-5 text-cyan-400" />
          <div className="text-right">
            <div className="text-xs text-slate-500">{copy.after}</div>
            <div className="mt-1 font-mono text-3xl font-semibold text-cyan-100 tnum">
              154°
            </div>
          </div>
        </div>
        <div className="mt-5 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
          <div className="h-full w-[86%] rounded-full bg-gradient-to-r from-cyan-400 to-emerald-300" />
        </div>
        <div className="mt-2 flex justify-between text-[10px] uppercase tracking-[0.13em] text-slate-500">
          <span>{copy.target}</span>
          <span className="font-mono text-emerald-200">86%</span>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500">
          {copy.comparisonBody}
        </p>
      </article>
    </section>
  );
}
