"use client";

import {
  ArrowLeft,
  Camera,
  Check,
  ChevronRight,
  FileUp,
  Languages,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  useCallback,
  useEffect,
  useReducer,
  useState,
} from "react";

import { LiveRehabWorkspace } from "@/components/rehabilitation/LiveRehabWorkspace";
import { RehabUploader } from "@/components/rehabilitation/RehabUploader";
import { CaptureReadinessPanel } from "@/components/rehabilitation/clinical/CaptureReadinessPanel";
import { VisitReview } from "@/components/rehabilitation/clinical/VisitReview";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import {
  evaluateCaptureReadiness,
  type CaptureReadinessResult,
} from "@/lib/captureReadiness";
import {
  clinicalApi,
  type CaptureQuality,
  type CaptureSource,
  type ClinicalVisit,
  type EpisodeDetail,
} from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import {
  REHAB_LOCALE_STORAGE_KEY,
  rehabCopy,
  type RehabLocale,
} from "@/lib/rehabCopy";
import type { LiveRehabUpdate } from "@/lib/rehabilitation";

import type { RehabAnalysisSnapshot } from "@/components/rehabilitation/rehabHandoff";

type VisitStep =
  | "context"
  | "readiness"
  | "analysis"
  | "review"
  | "summary";

type VisitState = {
  step: VisitStep;
  source: CaptureSource;
  preSessionNote: string;
  visit: ClinicalVisit | null;
  readiness: CaptureReadinessResult | null;
  warningAcknowledged: boolean;
  snapshot: RehabAnalysisSnapshot | null;
  trainingSessionId: number | null;
  specialistObservation: string;
  quality: CaptureQuality;
  qualityDetails: string;
  submitting: boolean;
  error: boolean;
};

type VisitAction =
  | { type: "source"; value: CaptureSource }
  | { type: "note"; value: string }
  | { type: "draft"; visit: ClinicalVisit }
  | { type: "step"; step: VisitStep }
  | { type: "readiness"; value: CaptureReadinessResult | null }
  | { type: "acknowledged"; value: boolean }
  | { type: "snapshot"; value: RehabAnalysisSnapshot | null }
  | { type: "saved"; sessionId: number }
  | { type: "observation"; value: string }
  | { type: "quality"; value: CaptureQuality }
  | { type: "qualityDetails"; value: string }
  | { type: "submitting"; value: boolean }
  | { type: "error"; value: boolean }
  | { type: "finalized"; visit: ClinicalVisit };

const initialState: VisitState = {
  step: "context",
  source: "live",
  preSessionNote: "",
  visit: null,
  readiness: null,
  warningAcknowledged: false,
  snapshot: null,
  trainingSessionId: null,
  specialistObservation: "",
  quality: "acceptable",
  qualityDetails: "",
  submitting: false,
  error: false,
};

function reducer(state: VisitState, action: VisitAction): VisitState {
  switch (action.type) {
    case "source":
      return {
        ...state,
        source: action.value,
        readiness: null,
        warningAcknowledged: false,
      };
    case "note":
      return { ...state, preSessionNote: action.value };
    case "draft":
      return { ...state, visit: action.visit, step: "readiness", error: false };
    case "step":
      return { ...state, step: action.step, error: false };
    case "readiness":
      return {
        ...state,
        readiness: action.value,
        warningAcknowledged:
          action.value?.state === "warning"
            ? state.warningAcknowledged
            : false,
      };
    case "acknowledged":
      return { ...state, warningAcknowledged: action.value };
    case "snapshot":
      return { ...state, snapshot: action.value };
    case "saved":
      return { ...state, trainingSessionId: action.sessionId };
    case "observation":
      return { ...state, specialistObservation: action.value };
    case "quality":
      return {
        ...state,
        quality: action.value,
        warningAcknowledged:
          action.value === "accepted_with_warning"
            ? state.warningAcknowledged
            : false,
      };
    case "qualityDetails":
      return { ...state, qualityDetails: action.value };
    case "submitting":
      return { ...state, submitting: action.value };
    case "error":
      return { ...state, error: action.value };
    case "finalized":
      return {
        ...state,
        visit: action.visit,
        submitting: false,
        error: false,
        step: "summary",
      };
  }
}

const uploadReady: CaptureReadinessResult = {
  state: "ready",
  code: "upload_ready",
  issues: [],
};

export default function NewClinicalVisitPage() {
  const params = useParams<{ episodeId: string }>();
  const episodeId = params.episodeId;
  const [locale, setLocale] = useState<RehabLocale>("uk");
  const [episode, setEpisode] = useState<EpisodeDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [state, dispatch] = useReducer(reducer, initialState);
  const copy = clinicalCopy[locale];
  const steps: VisitStep[] = [
    "context",
    "readiness",
    "analysis",
    "review",
    "summary",
  ];
  const activeStep = steps.indexOf(state.step);

  useEffect(() => {
    const stored = window.localStorage.getItem(REHAB_LOCALE_STORAGE_KEY);
    if (stored === "uk" || stored === "en") setLocale(stored);
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setLoadError(false);
    clinicalApi
      .getEpisode(episodeId)
      .then((value) => {
        if (active) setEpisode(value);
      })
      .catch(() => {
        if (active) setLoadError(true);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [episodeId]);

  const changeLocale = (value: RehabLocale) => {
    setLocale(value);
    window.localStorage.setItem(REHAB_LOCALE_STORAGE_KEY, value);
  };

  const createDraft = async () => {
    if (!episode || state.submitting) return;
    if (state.visit) {
      dispatch({ type: "step", step: "readiness" });
      return;
    }
    dispatch({ type: "submitting", value: true });
    dispatch({ type: "error", value: false });
    try {
      const visit = await clinicalApi.createVisit(episode.id, {
        captureSource: state.source,
        preSessionNote: state.preSessionNote.trim(),
      });
      dispatch({ type: "draft", visit });
    } catch {
      dispatch({ type: "error", value: true });
    } finally {
      dispatch({ type: "submitting", value: false });
    }
  };

  const onLiveUpdate = useCallback(
    (update: LiveRehabUpdate | null) => {
      if (!episode || !update) {
        dispatch({ type: "readiness", value: null });
        return;
      }
      dispatch({
        type: "readiness",
        value: evaluateCaptureReadiness({
          protocol: episode.protocol,
          poseDetected: update.pose_detected,
          landmarks: update.landmarks,
          cameraLevel: update.camera_level,
          confidence: update.camera_level.confidence,
        }),
      });
    },
    [episode],
  );

  const onSessionSaved = useCallback(
    (sessionId: number) => {
      dispatch({ type: "saved", sessionId });
      const visitId = state.visit?.id;
      if (visitId) {
        void clinicalApi
          .updateVisit(visitId, { trainingSessionId: sessionId })
          .catch(() => dispatch({ type: "error", value: true }));
      }
    },
    [state.visit?.id],
  );

  const startAnalysis = () => {
    const result = state.source === "upload" ? uploadReady : state.readiness;
    if (
      !result ||
      result.state === "blocked" ||
      (result.state === "warning" && !state.warningAcknowledged)
    ) {
      return;
    }
    if (result.state === "warning") {
      dispatch({ type: "quality", value: "accepted_with_warning" });
    }
    dispatch({ type: "step", step: "analysis" });
  };

  const finalize = async () => {
    if (
      !state.visit ||
      !state.trainingSessionId ||
      !state.specialistObservation.trim()
    ) {
      return;
    }
    dispatch({ type: "submitting", value: true });
    dispatch({ type: "error", value: false });
    try {
      await clinicalApi.updateVisit(state.visit.id, {
        trainingSessionId: state.trainingSessionId,
        specialistObservation: state.specialistObservation.trim(),
        captureQuality: state.quality,
        captureQualityDetails: state.qualityDetails.trim(),
        warningAcknowledged: state.warningAcknowledged,
      });
      dispatch({
        type: "finalized",
        visit: await clinicalApi.finalizeVisit(state.visit.id),
      });
    } catch {
      dispatch({ type: "submitting", value: false });
      dispatch({ type: "error", value: true });
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-[420px] items-center justify-center gap-3 text-sm text-slate-400">
        <RefreshCw className="h-5 w-5 animate-spin text-cyan-300" />
        {copy.loading}
      </div>
    );
  }

  if (loadError || !episode) {
    return (
      <div className="rounded-2xl border border-rose-400/20 bg-rose-400/[0.05] p-8 text-center text-rose-200">
        <ShieldAlert className="mx-auto h-7 w-7" />
        <p className="mt-3 text-sm">{copy.visit.saveError}</p>
      </div>
    );
  }

  const currentReadiness =
    state.source === "upload" ? uploadReady : state.readiness;
  const canStart =
    currentReadiness?.state === "ready" ||
    (currentReadiness?.state === "warning" &&
      state.warningAcknowledged);
  const captureActive =
    state.step === "readiness" || state.step === "analysis";

  return (
    <div className="animate-slide-up space-y-5 pb-10" lang={locale}>
      <header className="rounded-[28px] border border-white/[0.07] bg-[radial-gradient(circle_at_85%_0%,rgba(34,211,238,.12),transparent_34%),#080f16] p-5 md:p-7">
        <div className="flex flex-col justify-between gap-5 xl:flex-row xl:items-start">
          <div>
            <Link
              href={`/rehabilitation/clinical/patients/${episode.patient.id}`}
              className="inline-flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-cyan-200"
            >
              <ArrowLeft className="h-4 w-4" />
              {episode.patient.displayName}
            </Link>
            <p className="mt-5 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300">
              {copy.visit.eyebrow}
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-50 md:text-4xl">
              {copy.visit.title}
            </h1>
            <p className="mt-3 text-sm text-slate-400">
              {episode.title} · {rehabCopy[locale].protocols[episode.protocol]}
            </p>
          </div>
          <div>
            <span className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.13em] text-slate-500">
              <Languages className="h-3 w-3" />
              {copy.language}
            </span>
            <SegmentedControl
              value={locale}
              onChange={changeLocale}
              options={[
                { value: "uk", label: "Українська" },
                { value: "en", label: "English" },
              ]}
            />
          </div>
        </div>
        <ol className="mt-6 grid grid-cols-5 gap-1">
          {steps.map((step, index) => (
            <li key={step} className="min-w-0">
              <div
                className={`h-1 rounded-full ${
                  index <= activeStep ? "bg-cyan-300" : "bg-white/[0.07]"
                }`}
              />
              <p
                className={`mt-2 truncate text-[10px] font-semibold ${
                  index === activeStep ? "text-cyan-200" : "text-slate-600"
                }`}
              >
                {copy.visit.steps[step]}
              </p>
            </li>
          ))}
        </ol>
      </header>

      {state.step === "context" ? (
        <section className="rounded-2xl border border-white/[0.07] bg-[#091017] p-5 md:p-7">
          <div className="grid gap-3 sm:grid-cols-2">
            {(["live", "upload"] as const).map((source) => {
              const Icon = source === "live" ? Camera : FileUp;
              return (
                <button
                  key={source}
                  type="button"
                  onClick={() => dispatch({ type: "source", value: source })}
                  className={`rounded-2xl border p-5 text-left transition ${
                    state.source === source
                      ? "border-cyan-300/30 bg-cyan-300/[0.07]"
                      : "border-white/[0.07] bg-black/10"
                  }`}
                >
                  <Icon className="h-5 w-5 text-cyan-300" />
                  <p className="mt-3 text-sm font-semibold text-slate-100">
                    {source === "live" ? copy.visit.live : copy.visit.upload}
                  </p>
                </button>
              );
            })}
          </div>
          <label className="mt-5 block text-xs font-semibold text-slate-300">
            {copy.visit.preSessionNote}
            <textarea
              aria-label={copy.visit.preSessionNote}
              rows={4}
              value={state.preSessionNote}
              onChange={(event) =>
                dispatch({ type: "note", value: event.target.value })
              }
              className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-[#0c141d] px-3 py-3 text-sm text-slate-100 outline-none focus:border-cyan-300/40"
            />
          </label>
          <button
            type="button"
            disabled={state.submitting}
            onClick={() => void createDraft()}
            className="mt-5 inline-flex h-11 items-center gap-2 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] disabled:opacity-40"
          >
            {copy.visit.continue}
            <ChevronRight className="h-4 w-4" />
          </button>
        </section>
      ) : null}

      {captureActive ? (
        <section className="space-y-4">
          {state.source === "live" ? (
            <LiveRehabWorkspace
              protocol={episode.protocol}
              locale={locale}
              saveTarget={{
                athleteId: episode.athlete.id,
                athleteName: episode.athlete.name,
              }}
              onLiveUpdate={onLiveUpdate}
              onAnalysisChange={(value) =>
                dispatch({ type: "snapshot", value })
              }
              onSessionSaved={onSessionSaved}
            />
          ) : state.step === "analysis" ? (
            <RehabUploader
              protocol={episode.protocol}
              locale={locale}
              saveTarget={{
                athleteId: episode.athlete.id,
                athleteName: episode.athlete.name,
              }}
              onAnalysisChange={(value) =>
                dispatch({ type: "snapshot", value })
              }
              onSessionSaved={onSessionSaved}
            />
          ) : null}

          {state.step === "readiness" ? (
            <>
              <CaptureReadinessPanel
                result={currentReadiness}
                locale={locale}
                acknowledged={state.warningAcknowledged}
                onAcknowledgedChange={(value) =>
                  dispatch({ type: "acknowledged", value })
                }
              />
              <button
                type="button"
                disabled={!canStart}
                onClick={startAnalysis}
                className="inline-flex h-11 items-center gap-2 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] disabled:cursor-not-allowed disabled:opacity-35"
              >
                {copy.visit.startAnalysis}
                <ChevronRight className="h-4 w-4" />
              </button>
            </>
          ) : (
            <button
              type="button"
              disabled={!state.snapshot || !state.trainingSessionId}
              onClick={() => dispatch({ type: "step", step: "review" })}
              className="inline-flex h-11 items-center gap-2 rounded-xl bg-cyan-300 px-5 text-sm font-semibold text-[#031016] disabled:cursor-not-allowed disabled:opacity-35"
            >
              {copy.visit.reviewVisit}
              <ChevronRight className="h-4 w-4" />
            </button>
          )}
        </section>
      ) : null}

      {state.step === "review" && state.snapshot ? (
        <VisitReview
          episode={episode}
          baseline={episode.progress.at(-1) ?? null}
          snapshot={state.snapshot}
          locale={locale}
          observation={state.specialistObservation}
          quality={state.quality}
          qualityDetails={state.qualityDetails}
          warningAcknowledged={state.warningAcknowledged}
          submitting={state.submitting}
          onObservationChange={(value) =>
            dispatch({ type: "observation", value })
          }
          onQualityChange={(value) => dispatch({ type: "quality", value })}
          onQualityDetailsChange={(value) =>
            dispatch({ type: "qualityDetails", value })
          }
          onWarningAcknowledgedChange={(value) =>
            dispatch({ type: "acknowledged", value })
          }
          onFinalize={() => void finalize()}
        />
      ) : null}

      {state.step === "summary" ? (
        <section className="rounded-2xl border border-emerald-300/20 bg-[radial-gradient(circle_at_50%_0%,rgba(52,211,153,.12),transparent_45%),#091017] p-8 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-300 text-[#031016]">
            <Check className="h-7 w-7" />
          </div>
          <h2 className="mt-5 text-2xl font-semibold text-slate-50">
            {copy.visit.finalized}
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            {episode.patient.displayName} · {episode.title}
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            <Link
              href={`/rehabilitation/clinical/patients/${episode.patient.id}`}
              className="rounded-xl bg-cyan-300 px-5 py-3 text-sm font-semibold text-[#031016]"
            >
              {copy.visit.patientRecord}
            </Link>
          </div>
        </section>
      ) : null}

      {state.error ? (
        <p className="rounded-xl border border-rose-400/20 bg-rose-400/[0.06] px-4 py-3 text-xs text-rose-200">
          {copy.visit.saveError}
        </p>
      ) : null}
    </div>
  );
}
