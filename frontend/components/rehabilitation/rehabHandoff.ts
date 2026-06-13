import type { CaptureQuality } from "@/lib/clinical";
import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";
import type { RehabProtocol, RehabReport } from "@/lib/rehabilitation";

import { demoFrames } from "./demoSession";

export type RehabHandoffSource = "demo" | "live" | "upload";

export type RehabDelta = {
  leftRom: number;
  rightRom: number;
  symmetry: number;
  repetitions: number;
  completionScore: number;
};

export type ClinicalHandoffContext = {
  patientName: string;
  episodeTitle: string;
  functionalGoal: string;
  captureSource: "live" | "upload";
  captureQuality: CaptureQuality;
  qualityDetails: string;
  specialistObservation: string;
  baselineDelta: RehabDelta | null;
  previousDelta: RehabDelta | null;
};

export type RehabHandoff = {
  source: RehabHandoffSource;
  locale: RehabLocale;
  protocol: RehabProtocol;
  recordedAt: string;
  leftRom: number;
  rightRom: number;
  symmetry: number;
  repetitions: number;
  completionScore: number;
  confidence: number | null;
  poseCoverage: number | null;
  evidenceTimestamp: string | null;
  findingTitle: string;
  findingBody: string;
  disclaimer: string;
  clinical?: ClinicalHandoffContext;
};

export type RehabAnalysisSnapshot = {
  report: RehabReport;
  confidence?: number | null;
  poseCoverage?: number | null;
};

type ReportHandoffInput = RehabAnalysisSnapshot & {
  source: Exclude<RehabHandoffSource, "demo">;
  locale: RehabLocale;
  protocol: RehabProtocol;
  recordedAt?: string;
  clinical?: ClinicalHandoffContext;
};

function normalizePercentage(value: number | null | undefined): number | null {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return null;
  }
  const percentage = value >= 0 && value <= 1 ? value * 100 : value;
  return Math.round(Math.max(0, Math.min(100, percentage)));
}

function realFinding(
  locale: RehabLocale,
  leftRom: number,
  rightRom: number,
): { title: string; body: string } {
  const difference = Math.round(Math.abs(leftRom - rightRom));
  const rightIsLower = rightRom < leftRom;

  if (locale === "uk") {
    return {
      title: rightIsLower
        ? "Праворуч зафіксовано меншу амплітуду"
        : "Ліворуч зафіксовано меншу амплітуду",
      body: `У завершеній сесії ${
        rightIsLower ? "праворуч" : "ліворуч"
      } зафіксовано на ${difference}° менший ROM. Це спостереження для професійного обговорення, а не діагноз.`,
    };
  }

  return {
    title: rightIsLower
      ? "Lower range observed on the right side"
      : "Lower range observed on the left side",
    body: `The completed session shows ${difference}° less ROM on the ${
      rightIsLower ? "right" : "left"
    } side. This is an observation for professional review, not a diagnosis.`,
  };
}

export function createDemoHandoff(
  locale: RehabLocale,
  protocol: RehabProtocol,
  recordedAt = new Date().toISOString(),
): RehabHandoff {
  const frame = demoFrames[demoFrames.length - 1];
  const copy = rehabCopy[locale];

  return {
    source: "demo",
    locale,
    protocol,
    recordedAt,
    leftRom: frame.leftRom,
    rightRom: frame.rightRom,
    symmetry: frame.symmetry,
    repetitions: frame.repetitions,
    completionScore: 86,
    confidence: frame.confidence,
    poseCoverage: 100,
    evidenceTimestamp: "0:03.2",
    findingTitle: copy.demo.findingTitle,
    findingBody: copy.demo.findingBody,
    disclaimer: copy.limitationBody,
  };
}

export function createReportHandoff({
  source,
  locale,
  protocol,
  report,
  confidence = null,
  poseCoverage = null,
  recordedAt = new Date().toISOString(),
  clinical,
}: ReportHandoffInput): RehabHandoff {
  const leftRom = report.target_metrics.left.rom;
  const rightRom = report.target_metrics.right.rom;
  const finding = realFinding(locale, leftRom, rightRom);

  return {
    source,
    locale,
    protocol,
    recordedAt,
    leftRom,
    rightRom,
    symmetry: report.symmetry.score,
    repetitions: report.total_correct_reps,
    completionScore: report.completion_score,
    confidence: normalizePercentage(confidence),
    poseCoverage: normalizePercentage(poseCoverage),
    evidenceTimestamp: null,
    findingTitle: finding.title,
    findingBody: finding.body,
    disclaimer: rehabCopy[locale].limitationBody,
    clinical,
  };
}
