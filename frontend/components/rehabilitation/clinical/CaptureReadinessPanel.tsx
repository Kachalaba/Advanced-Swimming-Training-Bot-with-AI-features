"use client";

import { AlertTriangle, CheckCircle2, ShieldX } from "lucide-react";

import type { CaptureReadinessResult } from "@/lib/captureReadiness";
import { clinicalCopy } from "@/lib/clinicalCopy";
import type { RehabLocale } from "@/lib/rehabCopy";

export function CaptureReadinessPanel({
  result,
  locale,
  acknowledged,
  onAcknowledgedChange,
}: {
  result: CaptureReadinessResult | null;
  locale: RehabLocale;
  acknowledged: boolean;
  onAcknowledgedChange: (value: boolean) => void;
}) {
  const copy = clinicalCopy[locale];
  const state = result?.state ?? "waiting";
  const presentation = {
    waiting: {
      icon: AlertTriangle,
      label: copy.readiness.waiting,
      accent: "border-white/[0.07] bg-white/[0.025] text-slate-400",
    },
    ready: {
      icon: CheckCircle2,
      label: copy.readiness.ready,
      accent:
        "border-emerald-400/20 bg-emerald-400/[0.06] text-emerald-200",
    },
    warning: {
      icon: AlertTriangle,
      label: copy.readiness.warning,
      accent: "border-amber-400/20 bg-amber-400/[0.06] text-amber-200",
    },
    blocked: {
      icon: ShieldX,
      label: copy.readiness.blocked,
      accent: "border-rose-400/20 bg-rose-400/[0.06] text-rose-200",
    },
  }[state];
  const Icon = presentation.icon;

  return (
    <section
      aria-live="polite"
      className={`rounded-2xl border p-5 ${presentation.accent}`}
    >
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-current/15 bg-black/10">
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] opacity-70">
            {copy.readiness.title}
          </p>
          <h2 className="mt-1 text-lg font-semibold">{presentation.label}</h2>
        </div>
      </div>

      <div className="mt-4 space-y-2">
        {result ? (
          (result.issues.length > 0 ? result.issues : [result.code]).map(
            (code) => (
              <div
                key={code}
                className="flex items-start gap-2 rounded-xl border border-current/10 bg-black/10 px-3 py-2 text-xs leading-relaxed"
              >
                {result.state === "ready" ? (
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                ) : (
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                )}
                {copy.readiness.codes[code]}
              </div>
            ),
          )
        ) : (
          <p className="text-xs leading-relaxed opacity-75">
            {copy.readiness.waiting}
          </p>
        )}
      </div>

      {result?.state === "warning" ? (
        <label className="mt-4 flex cursor-pointer items-start gap-3 rounded-xl border border-amber-300/15 bg-black/10 p-3 text-xs leading-relaxed text-amber-100">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(event) =>
              onAcknowledgedChange(event.target.checked)
            }
            className="mt-0.5 h-4 w-4 accent-amber-300"
          />
          {copy.visit.warningAcknowledgement}
        </label>
      ) : null}
    </section>
  );
}
