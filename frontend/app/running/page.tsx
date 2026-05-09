"use client";

import { Award, Footprints, Target, TrendingUp, Zap } from "lucide-react";

import { RunningUploader } from "@/components/sports/RunningUploader";
import { SportLanding } from "@/components/sports/SportLanding";

export default function RunningPage() {
  return (
    <SportLanding
      uploader={<RunningUploader />}
      title="Running · Cadence, gait, form"
      subtitle="Per-step mechanics, foot strike classification, vertical oscillation, and arm-swing symmetry — locked onto your athlete across the whole run."
      badges={[
        { icon: Footprints, label: "Track · Trail · Treadmill" },
        { variant: "success", icon: TrendingUp, label: "Form trending up" },
      ]}
      hint="The pipeline locks onto the runner once detected and follows them across all zoom changes — no more lost analyses when the camera pulls back."
      accentRgb="34,211,238"
      metrics={[
        { label: "5K Personal Record", value: "18:42", icon: Award, change: 2.1, accent: true, hint: "Track · 4 days ago" },
        { label: "Cadence", value: "172", unit: "spm", icon: Zap, change: -1, hint: "Target 178 for tempo pace" },
        { label: "Foot strike", value: "Midfoot", icon: Target, change: 0, hint: "84% of contacts mid/forefoot" },
        { label: "Vertical osc.", value: "7.2", unit: "cm", icon: Target, change: -4, accent: true, hint: "Inside ≤8cm target" },
      ]}
      sessions={[
        { id: 1, title: "Tempo run · Track", duration: "42:18", date: "2h ago", score: 87, thumb: "from-orange-500/30 to-rose-500/20" },
        { id: 2, title: "Easy 10K · Park trail", duration: "54:02", date: "Yesterday", score: 79, thumb: "from-emerald-500/30 to-cyan-500/20" },
        { id: 3, title: "VO2 intervals 6×800m", duration: "38:40", date: "3 days ago", score: 90, thumb: "from-rose-500/30 to-orange-500/20" },
      ]}
      insights={[
        { tag: "Cadence", variant: "warn", title: "Cadence drops 8 spm under fatigue", detail: "Last 3 reps below 170 spm. Consider 180-spm metronome drills." },
        { tag: "Strike", variant: "success", title: "Mid-foot strike consistent", detail: "84% of contacts mid/forefoot — heel-strike risk low." },
        { tag: "Symmetry", variant: "info", title: "Right arm crosses midline 8°", detail: "Adds rotational load. Cue: thumb to hip pocket." },
      ]}
    />
  );
}
