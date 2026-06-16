"use client";

import { Camera, Footprints, Route, Timer, TrendingUp } from "lucide-react";

import { RunningUploader } from "@/components/sports/RunningUploader";
import { SportLanding } from "@/components/sports/SportLanding";
import {
  toSportLandingData,
  useSportOverview,
} from "@/lib/sportOverview";

export default function RunningPage() {
  const { overview, loading, error, retry } = useSportOverview("running");
  const data = overview
    ? toSportLandingData(overview)
    : { metrics: [], sessions: [], insights: [] };

  return (
    <SportLanding
      uploader={<RunningUploader />}
      title="Running · Cadence, gait, form"
      subtitle="Per-step mechanics, foot strike classification, vertical oscillation, and arm-swing symmetry — locked onto your athlete across the whole run."
      badges={[
        { icon: Footprints, label: "Track · Trail · Treadmill" },
        { variant: "success", icon: TrendingUp, label: "Persisted athlete history" },
        { variant: "info", label: "Evidence-based metrics" },
      ]}
      hint="The pipeline locks onto the runner once detected and follows them across all zoom changes — no more lost analyses when the camera pulls back."
      accentRgb="34,211,238"
      metrics={data.metrics}
      sessions={data.sessions}
      insights={data.insights}
      dataState={loading ? "loading" : error ? "error" : "ready"}
      onRetry={retry}
      uploadSubtitle="Track, treadmill, trail · side-on view"
      secondaryPanel={{
        title: "Capture checklist",
        subtitle: "A clean clip makes the gait metrics comparable",
        content: (
          <div className="space-y-3">
            {[
              {
                icon: Camera,
                title: "Lock the camera",
                detail: "Use a stable side view; avoid panning during the pass.",
              },
              {
                icon: Route,
                title: "Keep one runner visible",
                detail: "Give the model a clear lane and avoid crossing traffic.",
              },
              {
                icon: Timer,
                title: "Record a full stride sequence",
                detail: "Capture 10-20 meters or 15-30 seconds at steady pace.",
              },
            ].map(({ icon: Icon, title, detail }) => (
              <div
                key={title}
                className="flex gap-3 rounded-lg border border-white/[0.05] bg-white/[0.02] p-3"
              >
                <Icon className="mt-0.5 h-4 w-4 shrink-0 text-cyan-300" />
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
