"use client";

import { Activity, Camera, CheckCircle2, Dumbbell, Ruler, Target } from "lucide-react";

import { DrylandUploader } from "@/components/sports/DrylandUploader";
import { SportLanding } from "@/components/sports/SportLanding";
import {
  toSportLandingData,
  useSportOverview,
} from "@/lib/sportOverview";

export default function DrylandPage() {
  const { overview, loading, error, retry } = useSportOverview("dryland");
  const data = overview
    ? toSportLandingData(overview)
    : { metrics: [], sessions: [], insights: [] };
  const metrics =
    data.metrics.length > 0
      ? data.metrics
      : [
          {
            label: "Input",
            value: "Side view",
            icon: Camera,
            accent: true,
            hint: "Full body visible",
          },
          {
            label: "Evidence",
            value: "Full reps",
            icon: Dumbbell,
            hint: "Ready → effort → ready",
          },
          {
            label: "Measurement",
            value: "Tempo + ROM",
            icon: Activity,
            hint: "Per-rep evidence table",
          },
          {
            label: "Quality",
            value: "Gated",
            icon: Ruler,
            hint: "Pose and metric-ready frames",
          },
        ];

  return (
    <SportLanding
      uploader={<DrylandUploader />}
      title="Dryland · Strength evidence"
      subtitle="Exercise-specific repetition evidence for squats, lunges and push-ups with annotated video, tempo, ROM and movement consistency."
      badges={[
        { icon: Dumbbell, label: "Squat · Lunge · Push-up" },
        {
          variant: "success",
          icon: CheckCircle2,
          label: "Quality-gated analysis",
        },
        { variant: "info", label: "Fixed side view" },
      ]}
      hint="Choose the exercise first, then upload a fixed side-view clip. SPRINT only scores confirmed full repetitions."
      accentRgb="139,92,246"
      metrics={metrics}
      sessions={data.sessions}
      insights={data.insights}
      dataState={loading ? "loading" : error ? "error" : "ready"}
      onRetry={retry}
      uploadSubtitle="Squat, lunge, push-up · fixed side view"
      secondaryPanel={{
        title: "Capture checklist",
        subtitle: "Quality gates before scoring",
        content: (
          <div className="space-y-3">
            {[
              {
                icon: Camera,
                title: "Lock the camera",
                detail: "Use one fixed side view; avoid walking around the athlete.",
              },
              {
                icon: Target,
                title: "Keep joints visible",
                detail: "Do not crop shoulders, hips, knees, ankles, hands or feet.",
              },
              {
                icon: Activity,
                title: "Complete the cycle",
                detail: "Start ready, move through effort, then return to ready.",
              },
            ].map(({ icon: Icon, title, detail }) => (
              <div
                key={title}
                className="flex gap-3 rounded-lg border border-white/[0.05] bg-white/[0.02] p-3"
              >
                <Icon className="mt-0.5 h-4 w-4 shrink-0 text-violet-300" />
                <div>
                  <p className="text-sm font-medium text-slate-200">{title}</p>
                  <p className="mt-0.5 text-xs leading-relaxed text-slate-500">
                    {detail}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ),
      }}
    />
  );
}
