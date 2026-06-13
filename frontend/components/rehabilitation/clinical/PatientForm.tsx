"use client";

import { X } from "lucide-react";
import { useMemo, useState } from "react";

import type { Athlete } from "@/lib/api";
import {
  clinicalApi,
  type AffectedSide,
  type PatientProfile,
} from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import type { RehabLocale } from "@/lib/rehabCopy";

export function PatientForm({
  locale,
  athletes,
  onCancel,
  onCreated,
}: {
  locale: RehabLocale;
  athletes: Athlete[];
  onCancel: () => void;
  onCreated: (patient: PatientProfile) => void;
}) {
  const copy = clinicalCopy[locale];
  const initialAthlete = athletes[0];
  const [athleteId, setAthleteId] = useState(initialAthlete?.id ?? "");
  const [displayName, setDisplayName] = useState(initialAthlete?.name ?? "");
  const [affectedSide, setAffectedSide] =
    useState<AffectedSide>("unspecified");
  const [clinicalContext, setClinicalContext] = useState("");
  const [precautions, setPrecautions] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(false);
  const selectedAthlete = useMemo(
    () => athletes.find((athlete) => athlete.id === athleteId),
    [athleteId, athletes],
  );

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedAthlete || !displayName.trim()) return;
    setSubmitting(true);
    setError(false);
    try {
      const patient = await clinicalApi.createPatient({
        athleteId: Number(selectedAthlete.id),
        displayName: displayName.trim(),
        affectedSide,
        clinicalContext: clinicalContext.trim(),
        precautions: precautions.trim(),
      });
      onCreated(patient);
    } catch {
      setError(true);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="patient-form-title"
      className="fixed inset-0 z-[100] overflow-y-auto bg-[#03070b]/90 px-3 py-6 backdrop-blur-xl"
    >
      <form
        onSubmit={submit}
        className="mx-auto max-w-2xl rounded-2xl border border-white/10 bg-[#091017] p-5 shadow-[0_30px_100px_rgba(0,0,0,.5)] md:p-7"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300">
              SPRINT AI · Clinical Pilot
            </p>
            <h2
              id="patient-form-title"
              className="mt-2 text-2xl font-semibold text-slate-50"
            >
              {copy.patientForm.title}
            </h2>
          </div>
          <button
            type="button"
            aria-label={copy.patientForm.cancel}
            onClick={onCancel}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 text-slate-400 transition hover:bg-white/[0.06]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-6 grid gap-5 sm:grid-cols-2">
          <label className="text-xs font-semibold text-slate-300">
            {copy.patientForm.athlete}
            <select
              aria-label={copy.patientForm.athlete}
              value={athleteId}
              onChange={(event) => {
                const value = event.target.value;
                setAthleteId(value);
                const athlete = athletes.find((item) => item.id === value);
                if (athlete) setDisplayName(athlete.name);
              }}
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            >
              {athletes.map((athlete) => (
                <option key={athlete.id} value={athlete.id}>
                  {athlete.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300">
            {copy.patientForm.displayName}
            <input
              aria-label={copy.patientForm.displayName}
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
          <label className="text-xs font-semibold text-slate-300">
            {copy.patientForm.affectedSide}
            <select
              aria-label={copy.patientForm.affectedSide}
              value={affectedSide}
              onChange={(event) =>
                setAffectedSide(event.target.value as AffectedSide)
              }
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            >
              {(
                [
                  "unspecified",
                  "left",
                  "right",
                  "bilateral",
                ] as AffectedSide[]
              ).map((side) => (
                <option key={side} value={side}>
                  {copy.affectedSides[side]}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300 sm:col-span-2">
            {copy.patientForm.clinicalContext}
            <textarea
              aria-label={copy.patientForm.clinicalContext}
              value={clinicalContext}
              onChange={(event) => setClinicalContext(event.target.value)}
              rows={3}
              className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-[#0c141d] px-3 py-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
          <label className="text-xs font-semibold text-slate-300 sm:col-span-2">
            {copy.patientForm.precautions}
            <textarea
              aria-label={copy.patientForm.precautions}
              value={precautions}
              onChange={(event) => setPrecautions(event.target.value)}
              rows={2}
              className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-[#0c141d] px-3 py-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
        </div>

        {error ? (
          <p className="mt-4 rounded-xl border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-xs text-rose-200">
            {copy.patientForm.submitError}
          </p>
        ) : null}

        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="h-10 rounded-xl border border-white/10 px-4 text-sm font-semibold text-slate-300"
          >
            {copy.patientForm.cancel}
          </button>
          <button
            type="submit"
            disabled={submitting || !selectedAthlete || !displayName.trim()}
            className="h-10 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {copy.patientForm.create}
          </button>
        </div>
      </form>
    </div>
  );
}
