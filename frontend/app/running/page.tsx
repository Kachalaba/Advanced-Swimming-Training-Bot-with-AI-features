"use client";

import { Footprints, TrendingUp } from "lucide-react";

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
    />
  );
}
