"use client";

import { X } from "lucide-react";
import { useState } from "react";

import { clinicalApi, type RehabEpisode } from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";
import {
  rehabProtocols,
  type RehabProtocol,
} from "@/lib/rehabilitation";

export function EpisodeForm({
  patientId,
  locale,
  onCancel,
  onCreated,
}: {
  patientId: number;
  locale: RehabLocale;
  onCancel: () => void;
  onCreated: (episode: RehabEpisode) => void;
}) {
  const copy = clinicalCopy[locale];
  const [title, setTitle] = useState("");
  const [protocol, setProtocol] = useState<RehabProtocol>("shoulder_flexion");
  const [functionalGoal, setFunctionalGoal] = useState("");
  const [targetLeftRom, setTargetLeftRom] = useState("");
  const [targetRightRom, setTargetRightRom] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!title.trim() || !functionalGoal.trim()) return;
    setSubmitting(true);
    setError(false);
    try {
      onCreated(
        await clinicalApi.createEpisode(patientId, {
          title: title.trim(),
          protocol,
          functionalGoal: functionalGoal.trim(),
          targetLeftRom: targetLeftRom ? Number(targetLeftRom) : null,
          targetRightRom: targetRightRom ? Number(targetRightRom) : null,
        }),
      );
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
      aria-labelledby="episode-form-title"
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
              id="episode-form-title"
              className="mt-2 text-2xl font-semibold text-slate-50"
            >
              {copy.episodeForm.title}
            </h2>
          </div>
          <button
            type="button"
            aria-label={copy.episodeForm.cancel}
            onClick={onCancel}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 text-slate-400"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-6 grid gap-5 sm:grid-cols-2">
          <label className="text-xs font-semibold text-slate-300 sm:col-span-2">
            {copy.episodeForm.episodeTitle}
            <input
              aria-label={copy.episodeForm.episodeTitle}
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
          <label className="text-xs font-semibold text-slate-300 sm:col-span-2">
            {copy.episodeForm.protocol}
            <select
              aria-label={copy.episodeForm.protocol}
              value={protocol}
              onChange={(event) =>
                setProtocol(event.target.value as RehabProtocol)
              }
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            >
              {rehabProtocols.map((value) => (
                <option key={value} value={value}>
                  {rehabCopy[locale].protocols[value]}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-semibold text-slate-300 sm:col-span-2">
            {copy.episodeForm.functionalGoal}
            <textarea
              aria-label={copy.episodeForm.functionalGoal}
              value={functionalGoal}
              onChange={(event) => setFunctionalGoal(event.target.value)}
              rows={3}
              className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-[#0c141d] px-3 py-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
          <label className="text-xs font-semibold text-slate-300">
            {copy.episodeForm.targetLeft}
            <input
              aria-label={copy.episodeForm.targetLeft}
              type="number"
              min="0"
              max="360"
              value={targetLeftRom}
              onChange={(event) => setTargetLeftRom(event.target.value)}
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
          <label className="text-xs font-semibold text-slate-300">
            {copy.episodeForm.targetRight}
            <input
              aria-label={copy.episodeForm.targetRight}
              type="number"
              min="0"
              max="360"
              value={targetRightRom}
              onChange={(event) => setTargetRightRom(event.target.value)}
              className="mt-2 h-11 w-full rounded-xl border border-white/10 bg-[#0c141d] px-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
        </div>

        {error ? (
          <p className="mt-4 rounded-xl border border-rose-400/20 bg-rose-400/10 px-3 py-2 text-xs text-rose-200">
            {copy.episodeForm.submitError}
          </p>
        ) : null}

        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="h-10 rounded-xl border border-white/10 px-4 text-sm font-semibold text-slate-300"
          >
            {copy.episodeForm.cancel}
          </button>
          <button
            type="submit"
            disabled={
              submitting || !title.trim() || !functionalGoal.trim()
            }
            className="h-10 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] disabled:opacity-40"
          >
            {copy.episodeForm.create}
          </button>
        </div>
      </form>
    </div>
  );
}
