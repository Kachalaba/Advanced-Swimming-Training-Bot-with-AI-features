"use client";

import {
  Activity,
  ArrowLeft,
  CalendarPlus,
  Goal,
  RefreshCw,
  ShieldAlert,
  Stethoscope,
  UserRound,
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { RehabProgressChart } from "@/components/rehabilitation/RehabProgressChart";
import { ClinicalVisitTimeline } from "@/components/rehabilitation/clinical/ClinicalVisitTimeline";
import { EpisodeForm } from "@/components/rehabilitation/clinical/EpisodeForm";
import { ChartContainer } from "@/components/ui/ChartContainer";
import { EmptyState } from "@/components/ui/EmptyState";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  clinicalApi,
  type EpisodeDetail,
  type PatientDetail,
  type RehabEpisode,
} from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import { compareRehabProgress } from "@/lib/rehabProgress";
import {
  REHAB_LOCALE_STORAGE_KEY,
  rehabCopy,
  type RehabLocale,
} from "@/lib/rehabCopy";

function formatDate(value: string, locale: RehabLocale): string {
  return new Intl.DateTimeFormat(locale === "uk" ? "uk-UA" : "en-GB", {
    dateStyle: "medium",
  }).format(new Date(value));
}

function Metric({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-black/10 p-4">
      <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        {label}
      </p>
      <p className={`mt-2 text-2xl font-semibold tabular-nums ${accent}`}>
        {value}
      </p>
    </div>
  );
}

export default function PatientDetailPage() {
  const params = useParams<{ patientId: string }>();
  const patientId = params.patientId;
  const [locale, setLocale] = useState<RehabLocale>("uk");
  const [patient, setPatient] = useState<PatientDetail | null>(null);
  const [episode, setEpisode] = useState<EpisodeDetail | null>(null);
  const [showEpisodeForm, setShowEpisodeForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const copy = clinicalCopy[locale];
  const progress = useMemo(() => episode?.progress ?? [], [episode?.progress]);
  const comparison = useMemo(
    () => (episode ? compareRehabProgress(progress, episode.protocol) : null),
    [episode, progress],
  );
  const baseline = comparison?.baseline ?? progress[0] ?? null;
  const current =
    comparison?.current ?? progress[progress.length - 1] ?? null;

  useEffect(() => {
    const stored = window.localStorage.getItem(REHAB_LOCALE_STORAGE_KEY);
    if (stored === "uk" || stored === "en") setLocale(stored);
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      const loadedPatient = await clinicalApi.getPatient(patientId);
      setPatient(loadedPatient);
      setEpisode(
        loadedPatient.activeEpisode
          ? await clinicalApi.getEpisode(loadedPatient.activeEpisode.id)
          : null,
      );
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  useEffect(() => {
    void load();
  }, [load]);

  const changeLocale = (value: RehabLocale) => {
    setLocale(value);
    window.localStorage.setItem(REHAB_LOCALE_STORAGE_KEY, value);
    window.dispatchEvent(
      new CustomEvent("rehab-locale-change", { detail: value }),
    );
  };

  const onEpisodeCreated = async (created: RehabEpisode) => {
    setShowEpisodeForm(false);
    setPatient((currentPatient) =>
      currentPatient
        ? {
            ...currentPatient,
            activeEpisode: created,
            episodes: [created, ...currentPatient.episodes],
          }
        : currentPatient,
    );
    setEpisode(await clinicalApi.getEpisode(created.id));
  };

  if (loading) {
    return (
      <div className="flex min-h-[420px] items-center justify-center gap-3 text-sm text-slate-400">
        <RefreshCw className="h-5 w-5 animate-spin text-cyan-300" />
        {copy.loading}
      </div>
    );
  }

  if (error || !patient) {
    return (
      <div className="rounded-2xl border border-rose-400/15 bg-rose-400/[0.04]">
        <EmptyState
          icon={ShieldAlert}
          title={copy.patient.loadError}
          message={copy.localOnly}
          action={
            <button
              type="button"
              onClick={() => void load()}
              className="rounded-xl bg-white/[0.06] px-4 py-2 text-xs font-semibold text-slate-200"
            >
              {copy.retry}
            </button>
          }
        />
      </div>
    );
  }

  return (
    <>
      <div
        aria-hidden={showEpisodeForm || undefined}
        className="animate-slide-up space-y-6 pb-10"
        lang={locale}
      >
        <header className="rounded-[28px] border border-white/[0.07] bg-[radial-gradient(circle_at_85%_0%,rgba(34,211,238,.12),transparent_34%),#080f16] p-5 md:p-8">
          <div className="flex flex-col justify-between gap-6 xl:flex-row xl:items-start">
            <div>
              <Link
                href="/rehabilitation/clinical"
                className="inline-flex items-center gap-2 text-xs font-semibold text-slate-500 transition hover:text-cyan-200"
              >
                <ArrowLeft className="h-4 w-4" />
                {copy.workspace.title}
              </Link>
              <div className="mt-5 flex flex-wrap gap-2">
                <StatusBadge variant="success">
                  {copy.localOnly}
                </StatusBadge>
                <StatusBadge variant="info" icon={UserRound}>
                  {copy.affectedSides[patient.affectedSide]}
                </StatusBadge>
              </div>
              <h1 className="mt-4 text-3xl font-semibold tracking-tight text-slate-50 md:text-5xl">
                {patient.displayName}
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400">
                {patient.clinicalContext || copy.patient.context}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <SegmentedControl
                value={locale}
                onChange={changeLocale}
                options={[
                  { value: "uk", label: "Українська" },
                  { value: "en", label: "English" },
                ]}
              />
              {episode ? (
                <Link
                  href={`/rehabilitation/clinical/episodes/${episode.id}/visits/new`}
                  className="inline-flex h-10 items-center gap-2 rounded-xl bg-cyan-300 px-4 text-sm font-semibold text-[#031016]"
                >
                  <CalendarPlus className="h-4 w-4" />
                  {copy.patient.newVisit}
                </Link>
              ) : (
                <button
                  type="button"
                  onClick={() => setShowEpisodeForm(true)}
                  className="inline-flex h-10 items-center gap-2 rounded-xl bg-cyan-300 px-4 text-sm font-semibold text-[#031016]"
                >
                  <CalendarPlus className="h-4 w-4" />
                  {copy.patient.startEpisode}
                </button>
              )}
            </div>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
          <article className="rounded-2xl border border-white/[0.07] bg-[#091017] p-5">
            <div className="flex items-center gap-2 text-cyan-300">
              <Stethoscope className="h-4 w-4" />
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em]">
                {copy.patient.context}
              </p>
            </div>
            <p className="mt-4 text-sm leading-relaxed text-slate-300">
              {patient.clinicalContext}
            </p>
            <div className="mt-5 border-t border-white/[0.06] pt-4">
              <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-amber-300">
                {copy.patient.precautions}
              </p>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">
                {patient.precautions || copy.patient.noPrecautions}
              </p>
            </div>
          </article>

          {episode ? (
            <article className="rounded-2xl border border-cyan-300/12 bg-[linear-gradient(145deg,rgba(34,211,238,.07),rgba(9,16,23,.97))] p-5">
              <div className="flex items-center gap-2 text-cyan-300">
                <Goal className="h-4 w-4" />
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em]">
                  {rehabCopy[locale].protocols[episode.protocol]}
                </p>
              </div>
              <h2 className="mt-3 text-xl font-semibold text-slate-50">
                {episode.title}
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-300">
                {episode.functionalGoal}
              </p>
              <div className="mt-5 flex flex-wrap gap-3 text-xs text-slate-500">
                <span>
                  {copy.patient.target}: {episode.targetLeftRom ?? "—"}° /{" "}
                  {episode.targetRightRom ?? "—"}°
                </span>
                <span>{formatDate(episode.startedAt, locale)}</span>
              </div>
            </article>
          ) : (
            <div className="rounded-2xl border border-white/[0.07] bg-[#091017]">
              <EmptyState
                icon={Goal}
                title={copy.workspace.noEpisode}
                message={copy.workspace.noPatientsBody}
                action={
                  <button
                    type="button"
                    onClick={() => setShowEpisodeForm(true)}
                    className="rounded-xl bg-cyan-300 px-4 py-2 text-xs font-semibold text-[#031016]"
                  >
                    {copy.patient.startEpisode}
                  </button>
                }
              />
            </div>
          )}
        </section>

        {episode && current ? (
          <>
            <section className="rounded-2xl border border-white/[0.07] bg-[#091017] p-5">
              <div className="flex items-end justify-between gap-4">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-300">
                    {copy.patient.progress}
                  </p>
                  <h2 className="mt-1 text-xl font-semibold text-slate-50">
                    {episode.title}
                  </h2>
                </div>
                <p className="text-xs text-slate-500">
                  {progress.length} {rehabCopy[locale].handoff.progress}
                </p>
              </div>
              <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <Metric
                  label={rehabCopy[locale].demo.metrics.leftRom}
                  value={`${current.leftRom}°`}
                  accent="text-cyan-200"
                />
                <Metric
                  label={rehabCopy[locale].demo.metrics.rightRom}
                  value={`${current.rightRom}°`}
                  accent="text-indigo-200"
                />
                <Metric
                  label={rehabCopy[locale].demo.metrics.symmetry}
                  value={`${current.symmetry}%`}
                  accent="text-emerald-200"
                />
                <Metric
                  label={rehabCopy[locale].demo.target}
                  value={`${current.completionScore}%`}
                  accent="text-slate-100"
                />
              </div>
              {baseline && comparison ? (
                <p className="mt-4 text-xs text-slate-500">
                  {copy.patient.baseline}: {baseline.leftRom}° /{" "}
                  {baseline.rightRom}° · {comparison.deltas.leftRom > 0 ? "+" : ""}
                  {comparison.deltas.leftRom}° /{" "}
                  {comparison.deltas.rightRom > 0 ? "+" : ""}
                  {comparison.deltas.rightRom}°
                </p>
              ) : null}
            </section>

            {progress.length > 1 ? (
              <ChartContainer
                title={copy.patient.progress}
                subtitle={episode.functionalGoal}
              >
                <RehabProgressChart sessions={progress} locale={locale} />
              </ChartContainer>
            ) : null}
          </>
        ) : null}

        {episode ? (
          <ChartContainer
            title={copy.patient.visits}
            subtitle={episode.functionalGoal}
          >
            {episode.visits.length ? (
              <ClinicalVisitTimeline
                visits={episode.visits}
                locale={locale}
              />
            ) : (
              <div className="flex min-h-40 items-center justify-center text-sm text-slate-500">
                <Activity className="mr-2 h-4 w-4" />
                {copy.workspace.noVisits}
              </div>
            )}
          </ChartContainer>
        ) : null}

        <footer className="flex items-start gap-3 rounded-xl border border-white/[0.05] bg-white/[0.02] p-4 text-xs leading-relaxed text-slate-500">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-300/70" />
          {copy.safety}
        </footer>
      </div>

      {showEpisodeForm ? (
        <EpisodeForm
          patientId={patient.id}
          locale={locale}
          onCancel={() => setShowEpisodeForm(false)}
          onCreated={(created) => void onEpisodeCreated(created)}
        />
      ) : null}
    </>
  );
}
