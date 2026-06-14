"use client";

import { Eye, Target, Timer, Waves } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";
import {
  SwimmingFilmingGuide,
  SwimmingUploader,
} from "@/components/swimming/SwimmingUploader";
import {
  toSportLandingData,
  useSportOverview,
} from "@/lib/sportOverview";

export default function SwimmingPage() {
  const { overview, loading, error, retry } = useSportOverview("swimming");
  const saved = overview
    ? toSportLandingData(overview)
    : { metrics: [], sessions: [], insights: [] };

  return (
    <SportLanding
      title="Swimming · Waterline-aware analysis"
      subtitle="Evidence-first freestyle analysis for side-on pool video. See the main issue, its exact moment, and what to train next."
      badges={[
        { icon: Waves, label: "Freestyle · Side view" },
        { variant: "success", label: "Confidence-aware" },
        { variant: "info", label: "Waterline tracking" },
      ]}
      hint="SPRINT separates above-water and underwater evidence, selects the clearest complete cycles, and hides metrics it cannot support."
      accentRgb="34,211,238"
      metrics={[
        {
          label: "Technique zones",
          value: "5",
          icon: Target,
          accent: true,
          hint: "Body · rotation · catch · breathing · kick",
        },
        {
          label: "Selected cycles",
          value: "3–5",
          icon: Timer,
          hint: "Only the clearest complete stroke cycles",
        },
        {
          label: "Confidence",
          value: "Strict",
          icon: Eye,
          hint: "Unreliable segments and metrics stay hidden",
        },
        {
          label: "Result",
          value: "Action",
          icon: Waves,
          accent: true,
          hint: "Exact moment · drill · next-workout mini-set",
        },
      ]}
      sessions={saved.sessions}
      insights={saved.insights.length > 0 ? saved.insights : [
        {
          tag: "Evidence",
          variant: "success",
          title: "Every conclusion links to video",
          detail: "Jump directly to the cycle and timestamp that supports the finding.",
        },
        {
          tag: "Confidence",
          variant: "warn",
          title: "Missing evidence is not a zero",
          detail: "The result reports zone coverage and hides unsupported scores.",
        },
        {
          tag: "Coaching",
          variant: "info",
          title: "One issue, one next action",
          detail: "The main repeated issue becomes a drill and a concrete mini-set.",
        },
      ]}
      dataState={loading ? "loading" : error ? "error" : "ready"}
      onRetry={retry}
      uploader={<SwimmingUploader />}
      uploadSubtitle="Side-on freestyle · MP4, MOV, AVI or MKV"
      secondaryPanel={{
        title: "How to film",
        subtitle: "A reliable result starts with a reliable side view",
        content: <SwimmingFilmingGuide />,
      }}
    />
  );
}
