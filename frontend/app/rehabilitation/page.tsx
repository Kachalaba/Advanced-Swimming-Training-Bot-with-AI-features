"use client";

import {
  BriefcaseMedical,
  Camera,
  ChevronDown,
  FileText,
  History,
  Languages,
  MonitorPlay,
  Play,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Upload,
} from "lucide-react";
import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ClinicalReport } from "@/components/rehabilitation/ClinicalReport";
import { LiveRehabWorkspace } from "@/components/rehabilitation/LiveRehabWorkspace";
import { RehabDemoStage } from "@/components/rehabilitation/RehabDemoStage";
import { RehabEvidencePanel } from "@/components/rehabilitation/RehabEvidencePanel";
import { RehabInsightRail } from "@/components/rehabilitation/RehabInsightRail";
import { RehabPresentationMode } from "@/components/rehabilitation/RehabPresentationMode";
import { RehabUploader } from "@/components/rehabilitation/RehabUploader";
import { demoFrames } from "@/components/rehabilitation/demoSession";
import {
  createDemoHandoff,
  createReportHandoff,
  type RehabAnalysisSnapshot,
  type RehabHandoff,
} from "@/components/rehabilitation/rehabHandoff";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { clinicalCopy } from "@/lib/clinicalCopy";
import {
  REHAB_LOCALE_STORAGE_KEY,
  rehabCopy,
  type RehabLocale,
} from "@/lib/rehabCopy";
import { rehabProtocols, type RehabProtocol } from "@/lib/rehabilitation";

type InputMode = "demo" | "live" | "upload";
type HandoffOverlay = "presentation" | "report" | null;
type CapturedAnalysis = {
  snapshot: RehabAnalysisSnapshot;
  recordedAt: string;
};

export default function RehabilitationPage() {
  const [protocol, setProtocol] =
    useState<RehabProtocol>("shoulder_flexion");
  const [mode, setMode] = useState<InputMode>("demo");
  const [locale, setLocale] = useState<RehabLocale>("uk");
  const [demoIndex, setDemoIndex] = useState(demoFrames.length - 1);
  const [demoRunning, setDemoRunning] = useState(false);
  const [liveAnalysis, setLiveAnalysis] = useState<CapturedAnalysis | null>(
    null,
  );
  const [uploadAnalysis, setUploadAnalysis] =
    useState<CapturedAnalysis | null>(null);
  const [handoffOverlay, setHandoffOverlay] =
    useState<HandoffOverlay>(null);
  const handoffTriggerRef = useRef<HTMLButtonElement | null>(null);
  const copy = rehabCopy[locale];
  const frame = demoFrames[demoIndex];

  const currentHandoff = useMemo<RehabHandoff | null>(() => {
    if (mode === "demo") return createDemoHandoff(locale, protocol);
    const captured = mode === "live" ? liveAnalysis : uploadAnalysis;
    if (!captured) return null;
    return createReportHandoff({
      source: mode,
      locale,
      protocol,
      report: captured.snapshot.report,
      confidence: captured.snapshot.confidence,
      poseCoverage: captured.snapshot.poseCoverage,
      recordedAt: captured.recordedAt,
    });
  }, [liveAnalysis, locale, mode, protocol, uploadAnalysis]);

  useEffect(() => {
    const stored = window.localStorage.getItem(REHAB_LOCALE_STORAGE_KEY);
    if (stored === "uk" || stored === "en") setLocale(stored);
  }, []);

  useEffect(() => {
    if (!demoRunning) return;
    if (demoIndex >= demoFrames.length - 1) {
      setDemoRunning(false);
      return;
    }
    const timer = window.setTimeout(
      () => setDemoIndex((value) => value + 1),
      620,
    );
    return () => window.clearTimeout(timer);
  }, [demoIndex, demoRunning]);

  const changeLocale = (value: RehabLocale) => {
    setLocale(value);
    window.localStorage.setItem(REHAB_LOCALE_STORAGE_KEY, value);
    window.dispatchEvent(
      new CustomEvent("rehab-locale-change", { detail: value }),
    );
  };

  const changeProtocol = (value: RehabProtocol) => {
    setProtocol(value);
    setLiveAnalysis(null);
    setUploadAnalysis(null);
    setHandoffOverlay(null);
  };

  const captureAnalysis = useCallback(
    (
      setter: (value: CapturedAnalysis | null) => void,
      snapshot: RehabAnalysisSnapshot | null,
    ) => {
      setter(
        snapshot
          ? { snapshot, recordedAt: new Date().toISOString() }
          : null,
      );
    },
    [],
  );

  const onLiveAnalysisChange = useCallback(
    (snapshot: RehabAnalysisSnapshot | null) =>
      captureAnalysis(setLiveAnalysis, snapshot),
    [captureAnalysis],
  );

  const onUploadAnalysisChange = useCallback(
    (snapshot: RehabAnalysisSnapshot | null) =>
      captureAnalysis(setUploadAnalysis, snapshot),
    [captureAnalysis],
  );

  const openHandoff = (
    overlay: Exclude<HandoffOverlay, null>,
    trigger: HTMLButtonElement,
  ) => {
    if (!currentHandoff) return;
    handoffTriggerRef.current = trigger;
    setHandoffOverlay(overlay);
  };

  const closeHandoff = () => {
    setHandoffOverlay(null);
    window.setTimeout(() => handoffTriggerRef.current?.focus(), 0);
  };

  const runDemo = () => {
    setMode("demo");
    setDemoIndex(0);
    setDemoRunning(true);
  };

  return (
    <div className="animate-slide-up space-y-5" lang={locale}>
      <section className="relative overflow-hidden rounded-[30px] border border-white/[0.07] bg-[#080d13] p-4 shadow-[0_35px_120px_rgba(0,0,0,.32)] md:p-6 xl:p-8">
        <div className="pointer-events-none absolute -left-24 top-10 h-72 w-72 rounded-full bg-cyan-400/[0.06] blur-3xl" />
        <div className="relative mb-5 flex flex-col justify-between gap-4 xl:flex-row xl:items-center">
          <div className="flex flex-wrap gap-2">
            <StatusBadge variant="warn">{copy.prototypeBadge}</StatusBadge>
            <StatusBadge variant="success" icon={ShieldCheck}>
              {copy.localBadge}
            </StatusBadge>
          </div>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
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
                  changeProtocol(event.target.value as RehabProtocol)
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
                { value: "demo", label: copy.demoMode },
                { value: "live", label: copy.liveMode },
                { value: "upload", label: copy.uploadMode },
              ]}
            />
          </div>
        </div>

        {mode === "demo" ? (
          <div className="relative grid items-center gap-7 xl:grid-cols-[.76fr_1.24fr]">
            <div className="px-1 py-4 md:px-3 xl:py-8">
              <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-300">
                {copy.heroEyebrow}
              </div>
              <h1 className="mt-4 max-w-xl text-4xl font-semibold leading-[1.04] tracking-[-0.04em] text-slate-50 md:text-5xl xl:text-6xl">
                {copy.heroTitle}
              </h1>
              <p className="mt-5 max-w-xl text-sm leading-relaxed text-slate-400 md:text-base">
                {copy.heroSubtitle}
              </p>
              <div className="mt-7 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={runDemo}
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] shadow-[0_0_35px_rgba(103,232,249,.16)] transition hover:bg-cyan-200 active:scale-[0.98]"
                >
                  {demoRunning ? (
                    <RotateCcw className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4 fill-current" />
                  )}
                  {demoIndex === demoFrames.length - 1
                    ? copy.replayDemo
                    : copy.runDemo}
                </button>
                <button
                  type="button"
                  onClick={() => setMode("live")}
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-5 text-sm font-semibold text-slate-200 transition hover:border-cyan-300/30 hover:bg-white/[0.06]"
                >
                  <Camera className="h-4 w-4 text-cyan-300" />
                  {copy.openLive}
                </button>
              </div>
              <p className="mt-4 text-xs text-slate-600">{copy.demoHint}</p>
            </div>
            <RehabDemoStage
              frame={frame}
              locale={locale}
              running={demoRunning}
            />
          </div>
        ) : mode === "live" ? (
          <div className="mt-2">
            <LiveRehabWorkspace
              key={`${protocol}-${locale}`}
              protocol={protocol}
              locale={locale}
              onAnalysisChange={onLiveAnalysisChange}
            />
          </div>
        ) : (
          <section className="mt-2 rounded-2xl border border-white/[0.07] bg-[#0a1017] p-5 md:p-6">
            <div className="mb-4 flex items-center gap-2 text-xs font-medium text-slate-400">
              <Upload className="h-4 w-4 text-cyan-300" />
              {copy.uploadMode}
            </div>
            <RehabUploader
              protocol={protocol}
              locale={locale}
              onAnalysisChange={onUploadAnalysisChange}
            />
          </section>
        )}
      </section>

      <section className="flex flex-col justify-between gap-4 rounded-2xl border border-white/[0.07] bg-[#091017] p-4 md:flex-row md:items-center">
        <div>
          <div
            className={`text-[10px] font-semibold uppercase tracking-[0.17em] ${
              currentHandoff ? "text-emerald-300" : "text-slate-500"
            }`}
          >
            {currentHandoff ? copy.handoff.ready : copy.handoff.waiting}
          </div>
          <p className="mt-1 text-xs text-slate-500">
            {currentHandoff
              ? copy.handoff.localExport
              : copy.handoff.waiting}
          </p>
        </div>
        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          <Link
            href="/rehabilitation/clinical"
            className="inline-flex h-10 items-center justify-center gap-2 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.07] px-4 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-300/[0.12]"
          >
            <BriefcaseMedical className="h-4 w-4" />
            {clinicalCopy[locale].workspace.title}
          </Link>
          <Link
            href="/rehabilitation/progress"
            className="inline-flex h-10 items-center justify-center gap-2 rounded-xl border border-emerald-300/15 bg-emerald-300/[0.055] px-4 text-xs font-semibold text-emerald-100 transition hover:bg-emerald-300/[0.1]"
          >
            <History className="h-4 w-4" />
            {copy.handoff.progress}
          </Link>
          <button
            type="button"
            disabled={!currentHandoff}
            onClick={(event) =>
              openHandoff("presentation", event.currentTarget)
            }
            className="inline-flex h-10 items-center justify-center gap-2 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.07] px-4 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-300/[0.12] disabled:cursor-not-allowed disabled:border-white/[0.06] disabled:bg-white/[0.02] disabled:text-slate-600"
          >
            <MonitorPlay className="h-4 w-4" />
            {copy.handoff.presentation}
          </button>
          <button
            type="button"
            disabled={!currentHandoff}
            onClick={(event) => openHandoff("report", event.currentTarget)}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.035] px-4 text-xs font-semibold text-slate-200 transition hover:border-cyan-300/25 hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:border-white/[0.06] disabled:bg-white/[0.02] disabled:text-slate-600"
          >
            <FileText className="h-4 w-4" />
            {copy.handoff.report}
          </button>
        </div>
      </section>

      {mode === "demo" ? (
        <>
          <RehabInsightRail frame={frame} locale={locale} />
          <RehabEvidencePanel locale={locale} />
        </>
      ) : null}

      <details className="group rounded-xl border border-amber-300/10 bg-amber-300/[0.025]">
        <summary className="flex cursor-pointer list-none items-center gap-3 px-4 py-3 text-xs font-medium text-amber-100/75">
          <ShieldAlert className="h-4 w-4 text-amber-300/80" />
          <span className="flex-1">{copy.limitationTitle}</span>
          <ChevronDown className="h-4 w-4 transition group-open:rotate-180" />
        </summary>
        <p className="border-t border-amber-300/10 px-4 py-3 text-xs leading-relaxed text-amber-100/55">
          {copy.limitationBody}
        </p>
      </details>

      <p className="pb-2 text-center text-[11px] leading-relaxed text-slate-600">
        {copy.footer}
      </p>

      {handoffOverlay === "presentation" && currentHandoff ? (
        <RehabPresentationMode
          handoff={currentHandoff}
          onClose={closeHandoff}
          onOpenReport={() => setHandoffOverlay("report")}
          onReplayDemo={
            currentHandoff.source === "demo"
              ? () => {
                  runDemo();
                  closeHandoff();
                }
              : undefined
          }
        />
      ) : null}

      {handoffOverlay === "report" && currentHandoff ? (
        <ClinicalReport handoff={currentHandoff} onClose={closeHandoff} />
      ) : null}
    </div>
  );
}
