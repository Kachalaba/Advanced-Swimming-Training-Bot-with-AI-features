"use client";

import {
  Activity,
  ArrowLeft,
  CalendarDays,
  CheckCircle2,
  Database,
  HeartPulse,
  RefreshCw,
  ShieldAlert,
  UserRound,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { RehabProgressChart } from "@/components/rehabilitation/RehabProgressChart";
import { ChartContainer } from "@/components/ui/ChartContainer";
import { EmptyState } from "@/components/ui/EmptyState";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  api,
  type Athlete,
  type RehabProgressResponse,
} from "@/lib/api";
import {
  buildProgressObservation,
  compareRehabProgress,
  sessionsForProtocol,
  summarizeRehabProtocols,
} from "@/lib/rehabProgress";
import {
  REHAB_LOCALE_STORAGE_KEY,
  rehabCopy,
  type RehabLocale,
} from "@/lib/rehabCopy";
import { rehabProgressCopy } from "@/lib/rehabProgressCopy";
import type { RehabProtocol } from "@/lib/rehabilitation";

function formatDate(value: string, locale: RehabLocale): string {
  return new Intl.DateTimeFormat(locale === "uk" ? "uk-UA" : "en-US", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function signed(value: number, suffix: string): string {
  return `${value > 0 ? "+" : ""}${value}${suffix}`;
}

function ComparisonMetric({
  label,
  baseline,
  current,
  delta,
  valueSuffix,
  deltaSuffix = valueSuffix,
}: {
  label: string;
  baseline: number;
  current: number;
  delta: number;
  valueSuffix: string;
  deltaSuffix?: string;
}) {
  const changeColor =
    delta > 0
      ? "text-emerald-300"
      : delta < 0
        ? "text-amber-300"
        : "text-slate-400";
  return (
    <article className="rounded-xl border border-white/[0.06] bg-[#0a1118] p-4">
      <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        {label}
      </p>
      <div className="mt-4 grid grid-cols-[1fr_auto_1fr] items-end gap-3">
        <div>
          <p className="text-2xl font-bold tabular-nums text-slate-400">
            {baseline}
            {valueSuffix}
          </p>
        </div>
        <div className={`pb-1 text-xs font-bold tabular-nums ${changeColor}`}>
          {signed(delta, deltaSuffix)}
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold tabular-nums text-slate-50">
            {current}
            {valueSuffix}
          </p>
        </div>
      </div>
    </article>
  );
}

export default function RehabProgressPage() {
  const [locale, setLocale] = useState<RehabLocale>("uk");
  const [athletes, setAthletes] = useState<Athlete[]>([]);
  const [athleteId, setAthleteId] = useState("");
  const [progress, setProgress] = useState<RehabProgressResponse | null>(null);
  const [protocol, setProtocol] = useState<RehabProtocol | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [retryKey, setRetryKey] = useState(0);
  const copy = rehabProgressCopy[locale];

  useEffect(() => {
    const stored = window.localStorage.getItem(REHAB_LOCALE_STORAGE_KEY);
    if (stored === "uk" || stored === "en") setLocale(stored);
  }, []);

  const changeLocale = (value: RehabLocale) => {
    setLocale(value);
    window.localStorage.setItem(REHAB_LOCALE_STORAGE_KEY, value);
    window.dispatchEvent(
      new CustomEvent("rehab-locale-change", { detail: value }),
    );
  };

  useEffect(() => {
    let active = true;
    api
      .listAthletes()
      .then((items) => {
        if (!active) return;
        setAthletes(items);
        setAthleteId((current) => current || items[0]?.id || "");
        if (!items.length) setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setError(true);
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [retryKey]);

  const loadProgress = useCallback(() => {
    if (!athleteId) return;
    setLoading(true);
    setError(false);
    api
      .rehabilitationProgress(athleteId)
      .then((result) => {
        setProgress(result);
        setProtocol((current) =>
          current && result.protocols.includes(current)
            ? current
            : result.protocols[0] ?? null,
        );
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [athleteId]);

  useEffect(() => {
    loadProgress();
  }, [loadProgress, retryKey]);

  const summaries = useMemo(
    () => summarizeRehabProtocols(progress?.sessions ?? []),
    [progress],
  );
  const selectedSessions = useMemo(
    () =>
      protocol
        ? sessionsForProtocol(progress?.sessions ?? [], protocol)
        : [],
    [progress, protocol],
  );
  const comparison = useMemo(
    () =>
      protocol
        ? compareRehabProgress(progress?.sessions ?? [], protocol)
        : null,
    [progress, protocol],
  );
  const baseline = comparison?.baseline ?? selectedSessions[0] ?? null;
  const current =
    comparison?.current ??
    selectedSessions[selectedSessions.length - 1] ??
    null;
  const protocolCopy = rehabCopy[locale].protocols;

  const retry = () => {
    setProgress(null);
    setError(false);
    setLoading(true);
    setRetryKey((value) => value + 1);
  };

  return (
    <div className="animate-slide-up space-y-6 pb-8">
      <header className="rounded-2xl border border-white/[0.07] bg-[radial-gradient(circle_at_top_right,rgba(34,211,238,0.11),transparent_34%),#080f16] p-5 md:p-7">
        <div className="flex flex-col justify-between gap-6 xl:flex-row xl:items-start">
          <div className="max-w-3xl">
            <div className="mb-4 flex flex-wrap gap-2">
              <StatusBadge variant="warn" icon={ShieldAlert}>
                {copy.prototype}
              </StatusBadge>
              <StatusBadge variant="success" icon={Database}>
                {copy.localData}
              </StatusBadge>
            </div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300">
              {copy.eyebrow}
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-50 md:text-4xl">
              {copy.title}
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
              {copy.subtitle}
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-[minmax(180px,1fr)_auto]">
            <label className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
              {copy.patient}
              <select
                aria-label={copy.patient}
                value={athleteId}
                onChange={(event) => {
                  setProgress(null);
                  setProtocol(null);
                  setAthleteId(event.target.value);
                }}
                className="mt-2 h-10 w-full rounded-xl border border-white/10 bg-[#0b1219] px-3 text-sm font-semibold text-slate-100 outline-none focus:border-cyan-300/40"
              >
                {athletes.map((athlete) => (
                  <option key={athlete.id} value={athlete.id}>
                    {athlete.name}
                  </option>
                ))}
              </select>
            </label>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                {copy.language}
              </p>
              <div className="mt-2 flex rounded-xl border border-white/10 bg-[#0b1219] p-1">
                {(["uk", "en"] as RehabLocale[]).map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => changeLocale(value)}
                    className={`h-8 rounded-lg px-3 text-xs font-semibold transition ${
                      locale === value
                        ? "bg-cyan-300/12 text-cyan-200"
                        : "text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    {value === "uk" ? "Українська" : "English"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
        <Link
          href="/rehabilitation"
          className="mt-6 inline-flex h-10 items-center gap-2 rounded-xl border border-white/10 bg-white/[0.035] px-4 text-xs font-semibold text-slate-200 transition hover:bg-white/[0.07]"
        >
          <ArrowLeft className="h-4 w-4" />
          {copy.newAnalysis}
        </Link>
      </header>

      {loading && !progress ? (
        <div className="flex min-h-[360px] items-center justify-center rounded-2xl border border-white/[0.06] bg-[#091017]">
          <RefreshCw className="mr-3 h-5 w-5 animate-spin text-cyan-300" />
          <p className="text-sm text-slate-400">{copy.loading}</p>
        </div>
      ) : error ? (
        <div className="rounded-2xl border border-rose-400/15 bg-rose-400/[0.04]">
          <EmptyState
            icon={ShieldAlert}
            title={copy.loadError}
            message={copy.loadErrorBody}
            action={
              <button
                type="button"
                onClick={retry}
                className="inline-flex h-9 items-center gap-2 rounded-lg bg-cyan-300 px-4 text-xs font-bold text-slate-950"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                {copy.retry}
              </button>
            }
          />
        </div>
      ) : !progress?.sessions.length ? (
        <div className="rounded-2xl border border-white/[0.06] bg-[#091017]">
          <EmptyState
            icon={HeartPulse}
            title={copy.emptyTitle}
            message={copy.emptyBody}
            action={
              <Link
                href="/rehabilitation"
                className="rounded-lg bg-cyan-300 px-4 py-2 text-xs font-bold text-slate-950"
              >
                {copy.newAnalysis}
              </Link>
            }
          />
        </div>
      ) : (
        <>
          <section className="grid gap-3 lg:grid-cols-3">
            {summaries.map((summary) => (
              <button
                key={summary.protocol}
                type="button"
                onClick={() => setProtocol(summary.protocol)}
                className={`rounded-xl border p-4 text-left transition ${
                  protocol === summary.protocol
                    ? "border-cyan-300/25 bg-cyan-300/[0.08]"
                    : "border-white/[0.06] bg-[#091017] hover:border-white/[0.12]"
                }`}
              >
                <p className="font-semibold text-slate-100">
                  {protocolCopy[summary.protocol]}
                </p>
                <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500">
                  <span>
                    {summary.count} {copy.sessions}
                  </span>
                  <span>
                    {copy.latest} {formatDate(summary.latestDate, locale)}
                  </span>
                </div>
              </button>
            ))}
          </section>

          {baseline && current ? (
            <>
              <section className="rounded-2xl border border-white/[0.06] bg-[#091017] p-5">
                <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-300">
                      {protocol ? protocolCopy[protocol] : ""}
                    </p>
                    <h2 className="mt-1 text-xl font-bold text-slate-50">
                      {copy.change}
                    </h2>
                  </div>
                  <div className="flex gap-5 text-xs text-slate-500">
                    <span>
                      {copy.baseline}: {formatDate(baseline.date, locale)}
                    </span>
                    <span>
                      {copy.current}: {formatDate(current.date, locale)}
                    </span>
                  </div>
                </div>

                {comparison ? (
                  <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    <ComparisonMetric
                      label={copy.leftRom}
                      baseline={baseline.leftRom}
                      current={current.leftRom}
                      delta={comparison.deltas.leftRom}
                      valueSuffix="°"
                    />
                    <ComparisonMetric
                      label={copy.rightRom}
                      baseline={baseline.rightRom}
                      current={current.rightRom}
                      delta={comparison.deltas.rightRom}
                      valueSuffix="°"
                    />
                    <ComparisonMetric
                      label={copy.symmetry}
                      baseline={baseline.symmetry}
                      current={current.symmetry}
                      delta={comparison.deltas.symmetry}
                      valueSuffix="%"
                      deltaSuffix={locale === "uk" ? " в.п." : " pp"}
                    />
                    <ComparisonMetric
                      label={copy.completion}
                      baseline={baseline.completionScore}
                      current={current.completionScore}
                      delta={comparison.deltas.completionScore}
                      valueSuffix="%"
                      deltaSuffix={locale === "uk" ? " в.п." : " pp"}
                    />
                  </div>
                ) : (
                  <div className="mt-5 rounded-xl border border-amber-300/15 bg-amber-300/[0.04] p-4">
                    <p className="font-semibold text-amber-200">
                      {copy.oneSessionTitle}
                    </p>
                    <p className="mt-1 text-sm text-slate-400">
                      {copy.oneSessionBody}
                    </p>
                  </div>
                )}
              </section>

              {selectedSessions.length > 1 ? (
                <ChartContainer
                  title={copy.trendTitle}
                  subtitle={copy.trendSubtitle}
                >
                  <RehabProgressChart
                    sessions={selectedSessions}
                    locale={locale}
                  />
                </ChartContainer>
              ) : null}

              <section className="grid gap-4 lg:grid-cols-[1fr_1.5fr]">
                <article className="rounded-2xl border border-emerald-300/12 bg-[linear-gradient(145deg,rgba(52,211,153,0.07),rgba(9,16,23,0.96))] p-5">
                  <div className="flex items-center gap-2 text-emerald-300">
                    <CheckCircle2 className="h-4 w-4" />
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em]">
                      {copy.observationTitle}
                    </p>
                  </div>
                  <p className="mt-4 text-sm leading-7 text-slate-200">
                    {buildProgressObservation(comparison, locale)}
                  </p>
                  <p className="mt-5 border-t border-white/[0.06] pt-4 text-xs leading-relaxed text-slate-500">
                    {copy.limitation}
                  </p>
                </article>

                <ChartContainer
                  title={copy.timelineTitle}
                  subtitle={copy.timelineSubtitle}
                >
                  <div className="space-y-2">
                    {selectedSessions.toReversed().map((session, index) => (
                      <article
                        key={session.id}
                        className="grid gap-3 rounded-xl border border-white/[0.05] bg-white/[0.025] p-4 sm:grid-cols-[auto_1fr_auto] sm:items-center"
                      >
                        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-300/10 text-cyan-300">
                          {index === 0 ? (
                            <Activity className="h-4 w-4" />
                          ) : (
                            <CalendarDays className="h-4 w-4" />
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-slate-100">
                            {formatDate(session.date, locale)}
                          </p>
                          <p className="mt-1 text-[11px] text-slate-500">
                            {session.validFrames !== null
                              ? `${session.validFrames} ${copy.frames} · `
                              : ""}
                            {session.hasVideo ? copy.video : copy.noVideo}
                          </p>
                        </div>
                        <div className="grid grid-cols-4 gap-4 text-right text-xs tabular-nums">
                          <span className="text-cyan-300">
                            {session.leftRom}°
                          </span>
                          <span className="text-indigo-300">
                            {session.rightRom}°
                          </span>
                          <span className="text-emerald-300">
                            {session.symmetry}%
                          </span>
                          <span className="text-slate-300">
                            {session.completionScore}%
                          </span>
                        </div>
                      </article>
                    ))}
                  </div>
                </ChartContainer>
              </section>
            </>
          ) : null}
        </>
      )}

      <footer className="flex items-start gap-3 rounded-xl border border-white/[0.05] bg-white/[0.02] p-4 text-xs leading-relaxed text-slate-500">
        <UserRound className="mt-0.5 h-4 w-4 shrink-0 text-slate-600" />
        {copy.limitation}
      </footer>
    </div>
  );
}
