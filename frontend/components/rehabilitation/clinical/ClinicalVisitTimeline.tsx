import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Clock3,
  Upload,
} from "lucide-react";

import { StatusBadge } from "@/components/ui/StatusBadge";
import type { ClinicalVisit } from "@/lib/clinical";
import { clinicalCopy } from "@/lib/clinicalCopy";
import type { RehabLocale } from "@/lib/rehabCopy";

function formatDate(value: string, locale: RehabLocale): string {
  return new Intl.DateTimeFormat(locale === "uk" ? "uk-UA" : "en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function ClinicalVisitTimeline({
  visits,
  locale,
}: {
  visits: ClinicalVisit[];
  locale: RehabLocale;
}) {
  const copy = clinicalCopy[locale];

  return (
    <div className="space-y-2">
      {visits.toReversed().map((visit) => (
        <article
          key={visit.id}
          className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-4"
        >
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-cyan-300/10 text-cyan-300">
                {visit.captureSource === "live" ? (
                  <Camera className="h-4 w-4" />
                ) : (
                  <Upload className="h-4 w-4" />
                )}
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-100">
                  {formatDate(visit.visitedAt, locale)}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  {visit.specialistObservation ||
                    visit.preSessionNote ||
                    copy.workspace.noVisits}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              <StatusBadge
                variant={visit.status === "finalized" ? "success" : "neutral"}
                icon={visit.status === "finalized" ? CheckCircle2 : Clock3}
              >
                {visit.status === "finalized"
                  ? copy.visit.finalized
                  : copy.visit.steps.review}
              </StatusBadge>
              {visit.captureQuality ? (
                <StatusBadge
                  variant={
                    visit.captureQuality === "acceptable"
                      ? "success"
                      : visit.captureQuality === "repeat_required"
                        ? "danger"
                        : "warn"
                  }
                  icon={
                    visit.captureQuality === "acceptable"
                      ? CheckCircle2
                      : AlertTriangle
                  }
                >
                  {copy.quality[visit.captureQuality]}
                </StatusBadge>
              ) : null}
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
