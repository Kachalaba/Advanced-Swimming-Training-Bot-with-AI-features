"use client";

import { CheckCircle2, Loader2, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  rehabAnnotatedVideoUrl,
  subscribeRehabAnalysis,
  uploadRehabVideo,
  type RehabAnalysisEvent,
  type RehabProtocol,
} from "@/lib/rehabilitation";

export function RehabUploader({ protocol }: { protocol: RehabProtocol }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const unsubscribeRef = useRef<null | (() => void)>(null);
  const [progress, setProgress] = useState(0);
  const [label, setLabel] = useState("Готов к загрузке");
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<
    Extract<RehabAnalysisEvent, { type: "result" }> | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const busy = progress > 0 && progress < 100 && !error;

  useEffect(() => () => unsubscribeRef.current?.(), []);

  async function analyze(file: File) {
    setError(null);
    setResult(null);
    setProgress(1);
    setLabel("Загрузка видео");
    try {
      const upload = await uploadRehabVideo(file, protocol);
      setJobId(upload.jobId);
      unsubscribeRef.current = subscribeRehabAnalysis(upload.jobId, (event) => {
        if (event.type === "progress") {
          setProgress(event.pct);
          setLabel(event.label);
        } else if (event.type === "result") {
          setProgress(100);
          setLabel("Анализ завершён");
          setResult(event);
        } else {
          setError(event.message);
        }
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Не удалось загрузить видео");
      setProgress(0);
    }
  }

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
          {result ? "Видео проанализировано" : "Загрузить реабилитационную сессию"}
        </h3>
        <p className="relative mt-2 max-w-md text-sm text-slate-500">
          MP4, MOV, AVI или MKV до 512 MB. Фронтальный ракурс лучше всего
          подходит для постуральной карты.
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
              <span>{label}</span>
              <span className="font-mono">{progress}%</span>
            </div>
          </div>
        ) : null}
        {error ? <p className="relative mt-4 text-xs text-rose-300">{error}</p> : null}
      </button>

      {result ? (
        <div className="grid gap-3 sm:grid-cols-4">
          {[
            ["Повторы", result.report.total_correct_reps],
            ["Выполнение", `${result.report.completion_score.toFixed(0)}%`],
            ["Симметрия", `${result.report.symmetry.score.toFixed(0)}%`],
            ["Pose-кадры", `${result.frames_with_pose}/${result.frames_total}`],
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
        </div>
      ) : null}
    </div>
  );
}
