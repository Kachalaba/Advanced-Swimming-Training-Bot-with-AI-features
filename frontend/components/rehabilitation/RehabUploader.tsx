"use client";

import { CheckCircle2, Loader2, Save, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";
import {
  rehabAnnotatedVideoUrl,
  saveUploadedRehabSession,
  subscribeRehabAnalysis,
  uploadRehabVideo,
  type RehabAnalysisEvent,
  type RehabProtocol,
  type RehabSaveTarget,
} from "@/lib/rehabilitation";

import type { RehabAnalysisSnapshot } from "./rehabHandoff";

export function RehabUploader({
  protocol,
  locale = "uk",
  onAnalysisChange,
  saveTarget,
  onSessionSaved,
  initialFile,
}: {
  protocol: RehabProtocol;
  locale?: RehabLocale;
  onAnalysisChange?: (snapshot: RehabAnalysisSnapshot | null) => void;
  saveTarget?: RehabSaveTarget;
  onSessionSaved?: (sessionId: number) => void;
  initialFile?: File | null;
}) {
  const copy = rehabCopy[locale].upload;
  const inputRef = useRef<HTMLInputElement>(null);
  const unsubscribeRef = useRef<null | (() => void)>(null);
  const analyzedInitialFileRef = useRef<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [label, setLabel] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<
    Extract<RehabAnalysisEvent, { type: "result" }> | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [savedSessionId, setSavedSessionId] = useState<number | null>(null);
  const busy = progress > 0 && progress < 100 && !error;

  useEffect(() => {
    onAnalysisChange?.(null);
    return () => {
      unsubscribeRef.current?.();
      onAnalysisChange?.(null);
    };
  }, [onAnalysisChange]);

  const analyze = useCallback(async (file: File) => {
    onAnalysisChange?.(null);
    setError(null);
    setResult(null);
    setSavedSessionId(null);
    setProgress(1);
    setLabel(null);
    try {
      const upload = await uploadRehabVideo(file, protocol);
      setJobId(upload.jobId);
      unsubscribeRef.current = subscribeRehabAnalysis(upload.jobId, (event) => {
        if (event.type === "progress") {
          setProgress(event.pct);
          setLabel(event.label);
        } else if (event.type === "result") {
          setProgress(100);
          setLabel("Done");
          setResult(event);
          onAnalysisChange?.({
            report: event.report,
            confidence: null,
            poseCoverage: event.frames_total
              ? (event.frames_with_pose / event.frames_total) * 100
              : null,
          });
        } else {
          setError(event.message);
        }
      });
    } catch {
      setError(copy.uploadError);
      setProgress(0);
    }
  }, [copy.uploadError, onAnalysisChange, protocol]);

  useEffect(() => {
    if (
      !initialFile ||
      analyzedInitialFileRef.current === initialFile
    ) {
      return;
    }
    analyzedInitialFileRef.current = initialFile;
    void analyze(initialFile);
  }, [analyze, initialFile]);

  const progressLabel =
    progress === 0
      ? copy.ready
      : progress === 1 && label === null
        ? copy.uploading
        : label
          ? (copy.progress as Record<string, string>)[label] ?? label
          : copy.uploading;

  return (
    <div className="space-y-5">
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void analyze(file);
        }}
      />
      <button
        type="button"
        onClick={() => !busy && inputRef.current?.click()}
        className="group relative flex min-h-[320px] w-full flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed border-white/10 bg-gradient-to-br from-white/[0.035] to-transparent px-8 text-center transition hover:border-cyan-400/35 hover:bg-cyan-400/[0.035]"
      >
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_20%,rgba(34,211,238,.08),transparent_45%)]" />
        <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.05]">
          {busy ? (
            <Loader2 className="h-6 w-6 animate-spin text-cyan-300" />
          ) : result ? (
            <CheckCircle2 className="h-6 w-6 text-emerald-300" />
          ) : (
            <Upload className="h-6 w-6 text-slate-300 transition group-hover:-translate-y-0.5 group-hover:text-cyan-300" />
          )}
        </div>
        <h3 className="relative mt-5 text-base font-semibold text-slate-100">
          {result ? copy.resultTitle : copy.uploadTitle}
        </h3>
        <p className="relative mt-2 max-w-md text-sm text-slate-500">
          {copy.uploadBody}
        </p>
        {progress > 0 ? (
          <div className="relative mt-5 w-full max-w-sm">
            <div className="h-1 overflow-hidden rounded-full bg-white/5">
              <div
                className="h-full rounded-full bg-cyan-400 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="mt-2 flex justify-between text-[10px] uppercase tracking-wider text-slate-500">
              <span>{progressLabel}</span>
              <span className="font-mono">{progress}%</span>
            </div>
          </div>
        ) : null}
        {error ? <p className="relative mt-4 text-xs text-rose-300">{error}</p> : null}
      </button>

      {result ? (
        <div className="grid gap-3 sm:grid-cols-4">
          {[
            [copy.repetitions, result.report.total_correct_reps],
            [copy.targetCompletion, `${result.report.completion_score.toFixed(0)}%`],
            [copy.symmetry, `${result.report.symmetry.score.toFixed(0)}%`],
            [
              copy.poseCoverage,
              `${
                result.frames_total
                  ? Math.round(
                      (result.frames_with_pose / result.frames_total) * 100,
                    )
                  : 0
              }%`,
            ],
          ].map(([metric, value]) => (
            <div
              key={metric}
              className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-4"
            >
              <p className="text-[10px] uppercase tracking-wider text-slate-500">
                {metric}
              </p>
              <p className="mt-1 font-mono text-xl font-bold text-slate-100">
                {value}
              </p>
            </div>
          ))}
          {jobId && result.video_path ? (
            <video
              controls
              className="mt-3 w-full rounded-xl border border-white/10 sm:col-span-4"
              src={rehabAnnotatedVideoUrl(jobId)}
            />
          ) : null}
          {jobId ? (
            <button
              type="button"
              disabled={savedSessionId !== null}
              onClick={async () => {
                try {
                  const saved = await saveUploadedRehabSession(
                    jobId,
                    saveTarget,
                  );
                  setSavedSessionId(saved.sessionId);
                  onSessionSaved?.(saved.sessionId);
                } catch {
                  setError(copy.saveError);
                }
              }}
              className="mt-1 inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-cyan-400/25 bg-cyan-400/10 px-4 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-400/15 disabled:cursor-default disabled:border-emerald-400/25 disabled:bg-emerald-400/10 disabled:text-emerald-200 sm:col-span-4"
            >
              <Save className="h-3.5 w-3.5" />
              {savedSessionId
                ? `${copy.saved} #${savedSessionId}`
                : copy.save}
            </button>
          ) : null}
          <p className="text-center text-[11px] leading-relaxed text-slate-500 sm:col-span-4">
            {copy.targetNote}
          </p>
        </div>
      ) : null}
    </div>
  );
}
