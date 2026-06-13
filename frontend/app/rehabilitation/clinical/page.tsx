"use client";

import {
  Activity,
  Archive,
  ArrowLeft,
  CalendarDays,
  Database,
  FolderHeart,
  Plus,
  RefreshCw,
  ShieldAlert,
  UserRound,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { PatientForm } from "@/components/rehabilitation/clinical/PatientForm";
import { EmptyState } from "@/components/ui/EmptyState";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { api, type Athlete } from "@/lib/api";
import { clinicalApi, type PatientProfile } from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import {
  REHAB_LOCALE_STORAGE_KEY,
  rehabCopy,
  type RehabLocale,
} from "@/lib/rehabCopy";

type PatientFilter = "active" | "archived";

function formatDate(value: string, locale: RehabLocale): string {
  return new Intl.DateTimeFormat(locale === "uk" ? "uk-UA" : "en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(new Date(value));
}

export default function ClinicalWorkspacePage() {
  const [locale, setLocale] = useState<RehabLocale>("uk");
  const [filter, setFilter] = useState<PatientFilter>("active");
  const [athletes, setAthletes] = useState<Athlete[]>([]);
  const [patients, setPatients] = useState<PatientProfile[]>([]);
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const copy = clinicalCopy[locale];
  const protocolCopy = rehabCopy[locale].protocols;

  useEffect(() => {
    const stored = window.localStorage.getItem(REHAB_LOCALE_STORAGE_KEY);
    if (stored === "uk" || stored === "en") setLocale(stored);
    api.listAthletes().then(setAthletes).catch(() => setAthletes([]));
  }, []);

  const loadPatients = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      setPatients(await clinicalApi.listPatients(filter === "archived"));
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    void loadPatients();
  }, [loadPatients]);

  const changeLocale = (value: RehabLocale) => {
    setLocale(value);
    window.localStorage.setItem(REHAB_LOCALE_STORAGE_KEY, value);
    window.dispatchEvent(
      new CustomEvent("rehab-locale-change", { detail: value }),
    );
  };

  return (
    <div className="animate-slide-up space-y-6 pb-10" lang={locale}>
      <header className="relative overflow-hidden rounded-[28px] border border-white/[0.07] bg-[radial-gradient(circle_at_85%_0%,rgba(34,211,238,.13),transparent_34%),#080f16] p-5 md:p-8">
        <div className="pointer-events-none absolute -left-20 bottom-0 h-56 w-56 rounded-full bg-emerald-400/[0.05] blur-3xl" />
        <div className="relative flex flex-col justify-between gap-7 xl:flex-row xl:items-start">
          <div className="max-w-3xl">
            <div className="flex flex-wrap gap-2">
              <StatusBadge variant="warn" icon={ShieldAlert}>
                {copy.prototype}
              </StatusBadge>
              <StatusBadge variant="success" icon={Database}>
                {copy.localOnly}
              </StatusBadge>
            </div>
            <p className="mt-6 text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-300">
              {copy.workspace.eyebrow}
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-50 md:text-5xl">
              {copy.workspace.title}
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-400 md:text-base">
              {copy.workspace.subtitle}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <SegmentedControl
              value={locale}
              onChange={changeLocale}
              options={[
                { value: "uk", label: "Українська" },
                { value: "en", label: "English" },
              ]}
            />
            <button
              type="button"
              onClick={() => setShowPatientForm(true)}
              disabled={!athletes.length}
              className="inline-flex h-10 items-center gap-2 rounded-xl bg-cyan-300 px-4 text-sm font-semibold text-[#031016] transition hover:bg-cyan-200 disabled:opacity-40"
            >
              <Plus className="h-4 w-4" />
              {copy.workspace.newPatient}
            </button>
          </div>
        </div>
      </header>

      <section className="rounded-2xl border border-white/[0.07] bg-[#080f16]">
        <div className="flex flex-col justify-between gap-4 border-b border-white/[0.06] p-4 sm:flex-row sm:items-center">
          <SegmentedControl
            value={filter}
            onChange={setFilter}
            options={[
              { value: "active", label: copy.workspace.active },
              { value: "archived", label: copy.workspace.archived },
            ]}
          />
          <Link
            href="/rehabilitation"
            className="inline-flex items-center gap-2 text-xs font-semibold text-slate-400 transition hover:text-cyan-200"
          >
            <ArrowLeft className="h-4 w-4" />
            {copy.backToRehab}
          </Link>
        </div>

        {loading ? (
          <div className="flex min-h-72 items-center justify-center gap-3 text-sm text-slate-400">
            <Activity className="h-5 w-5 animate-pulse text-cyan-300" />
            {copy.loading}
          </div>
        ) : error ? (
          <EmptyState
            icon={RefreshCw}
            title={copy.workspace.loadError}
            message={copy.localOnly}
            action={
              <button
                type="button"
                onClick={() => void loadPatients()}
                className="rounded-xl bg-white/[0.06] px-4 py-2 text-xs font-semibold text-slate-200"
              >
                {copy.retry}
              </button>
            }
          />
        ) : patients.length === 0 ? (
          <EmptyState
            icon={filter === "archived" ? Archive : UserRound}
            title={copy.workspace.noPatientsTitle}
            message={copy.workspace.noPatientsBody}
            action={
              filter === "active" ? (
                <button
                  type="button"
                  onClick={() => setShowPatientForm(true)}
                  className="rounded-xl bg-cyan-300 px-4 py-2 text-xs font-semibold text-[#031016]"
                >
                  {copy.workspace.newPatient}
                </button>
              ) : null
            }
          />
        ) : (
          <div className="grid gap-3 p-4 lg:grid-cols-2 2xl:grid-cols-3">
            {patients.map((patient) => (
              <article
                key={patient.id}
                className="group rounded-2xl border border-white/[0.07] bg-[#0b131c] p-5 transition hover:border-cyan-300/20 hover:bg-[#0c1620]"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-cyan-300/15 bg-cyan-300/[0.07] text-cyan-200">
                      <UserRound className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <h2 className="truncate text-base font-semibold text-slate-50">
                        {patient.displayName}
                      </h2>
                      <p className="mt-1 text-xs text-slate-500">
                        {copy.affectedSides[patient.affectedSide]}
                      </p>
                    </div>
                  </div>
                  <StatusBadge
                    variant={patient.status === "active" ? "success" : "neutral"}
                  >
                    {patient.status === "active"
                      ? copy.workspace.active
                      : copy.workspace.archived}
                  </StatusBadge>
                </div>

                <div className="mt-5 grid gap-3">
                  <div className="rounded-xl border border-white/[0.05] bg-black/10 p-3">
                    <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                      <FolderHeart className="h-3.5 w-3.5 text-cyan-300" />
                      {copy.workspace.activeEpisode}
                    </div>
                    <p className="mt-2 text-sm font-semibold text-slate-200">
                      {patient.activeEpisode?.title ??
                        copy.workspace.noEpisode}
                    </p>
                    {patient.activeEpisode ? (
                      <p className="mt-1 text-xs text-slate-500">
                        {protocolCopy[patient.activeEpisode.protocol]}
                      </p>
                    ) : null}
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <CalendarDays className="h-4 w-4" />
                    {patient.latestVisit
                      ? `${copy.workspace.latestVisit}: ${formatDate(
                          patient.latestVisit.visitedAt,
                          locale,
                        )}`
                      : copy.workspace.noVisits}
                  </div>
                </div>

                <Link
                  href={`/rehabilitation/clinical/patients/${patient.id}`}
                  className="mt-5 inline-flex h-10 w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.035] text-xs font-semibold text-slate-200 transition group-hover:border-cyan-300/20 group-hover:text-cyan-100"
                >
                  {copy.workspace.openPatient}
                </Link>
              </article>
            ))}
          </div>
        )}
      </section>

      {showPatientForm ? (
        <PatientForm
          locale={locale}
          athletes={athletes}
          onCancel={() => setShowPatientForm(false)}
          onCreated={(patient) => {
            setPatients((current) => [...current, patient]);
            setShowPatientForm(false);
          }}
        />
      ) : null}
    </div>
  );
}
