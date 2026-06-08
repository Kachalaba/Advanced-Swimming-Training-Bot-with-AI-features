"use client";

import {
  Activity,
  Camera,
  CircleGauge,
  HeartPulse,
  Languages,
  ShieldAlert,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { useEffect, useState } from "react";

import { LiveRehabWorkspace } from "@/components/rehabilitation/LiveRehabWorkspace";
import { RehabUploader } from "@/components/rehabilitation/RehabUploader";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  REHAB_LOCALE_STORAGE_KEY,
  rehabCopy,
  type RehabLocale,
} from "@/lib/rehabCopy";
import {
  rehabProtocols,
  type RehabProtocol,
} from "@/lib/rehabilitation";

type InputMode = "live" | "upload";

export default function RehabilitationPage() {
  const [protocol, setProtocol] =
    useState<RehabProtocol>("shoulder_flexion");
  const [mode, setMode] = useState<InputMode>("live");
  const [locale, setLocale] = useState<RehabLocale>("uk");
  const copy = rehabCopy[locale];

  useEffect(() => {
    const stored = window.localStorage.getItem(REHAB_LOCALE_STORAGE_KEY);
    if (stored === "uk" || stored === "en") setLocale(stored);
  }, []);

  const changeLocale = (value: RehabLocale) => {
    setLocale(value);
    window.localStorage.setItem(REHAB_LOCALE_STORAGE_KEY, value);
  };

  return (
    <div className="animate-slide-up space-y-6" lang={locale}>
      <section className="relative overflow-hidden rounded-2xl border border-white/[0.07] bg-gradient-to-br from-surface via-surface to-bg p-6 md:p-8">
        <div className="absolute -right-20 -top-28 h-80 w-80 rounded-full bg-cyan-400/[0.08] blur-3xl" />
        <div
          className="absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(34,211,238,.24) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.24) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
            maskImage:
              "radial-gradient(ellipse at top right, black, transparent 68%)",
          }}
        />
        <div className="relative flex flex-col justify-between gap-6 xl:flex-row xl:items-end">
          <div>
            <div className="mb-3 flex flex-wrap gap-2">
              <StatusBadge variant="warn" icon={HeartPulse}>
                {copy.prototypeBadge}
              </StatusBadge>
              <StatusBadge variant="success" icon={ShieldCheck}>
                {copy.localBadge}
              </StatusBadge>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-50 md:text-4xl">
              {copy.title}
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
              {copy.subtitle}
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="space-y-1.5">
              <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.13em] text-slate-500">
                <Languages className="h-3 w-3" />
                {copy.language}
              </span>
              <SegmentedControl<RehabLocale>
                value={locale}
                onChange={changeLocale}
                options={[
                  { value: "uk", label: "Українська" },
                  { value: "en", label: "English" },
                ]}
              />
            </div>
            <label className="space-y-1.5">
              <span className="block text-[10px] font-semibold uppercase tracking-[0.13em] text-slate-500">
                {copy.protocol}
              </span>
              <select
                value={protocol}
                onChange={(event) =>
                  setProtocol(event.target.value as RehabProtocol)
                }
                className="h-9 min-w-[210px] rounded-lg border border-white/10 bg-[#0b1118] px-3 text-xs font-medium text-slate-200 outline-none transition focus:border-cyan-400/50"
              >
                {rehabProtocols.map((value) => (
                  <option key={value} value={value}>
                    {copy.protocols[value]}
                  </option>
                ))}
              </select>
            </label>
            <SegmentedControl<InputMode>
              value={mode}
              onChange={setMode}
              options={[
                { value: "live", label: copy.liveMode },
                { value: "upload", label: copy.uploadMode },
              ]}
            />
          </div>
        </div>
      </section>

      <section className="flex gap-3 rounded-xl border border-amber-400/20 bg-amber-400/[0.06] p-4">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
        <div>
          <h2 className="text-sm font-semibold text-amber-100">
            {copy.limitationTitle}
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-amber-100/65">
            {copy.limitationBody}
          </p>
        </div>
      </section>

      {mode === "live" ? (
        <LiveRehabWorkspace
          key={protocol}
          protocol={protocol}
          locale={locale}
        />
      ) : (
        <section className="rounded-2xl border border-white/[0.07] bg-surface p-5 md:p-6">
          <RehabUploader protocol={protocol} locale={locale} />
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        {[
          {
            icon: CircleGauge,
            title: copy.featureRomTitle,
            text: copy.featureRomText,
          },
          {
            icon: Activity,
            title: copy.featurePostureTitle,
            text: copy.featurePostureText,
          },
          {
            icon: mode === "live" ? Camera : Upload,
            title: copy.featureQualityTitle,
            text: copy.featureQualityText,
          },
        ].map(({ icon: Icon, title, text }) => (
          <div
            key={title}
            className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"
          >
            <Icon className="h-4 w-4 text-cyan-300" />
            <h2 className="mt-3 text-sm font-semibold text-slate-100">{title}</h2>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{text}</p>
          </div>
        ))}
      </section>

      <p className="text-center text-[11px] leading-relaxed text-slate-600">
        {copy.footer}
      </p>
    </div>
  );
}
