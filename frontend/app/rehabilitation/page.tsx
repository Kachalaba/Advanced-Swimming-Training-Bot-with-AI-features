"use client";

import { Activity, Gauge, HeartPulse, Scale, Target } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";

export default function RehabilitationPage() {
  return (
    <SportLanding
      title="Rehabilitation · Range of motion"
      subtitle="Clinical ROM analysis for kinesiotherapy: bilateral joint angles, left/right symmetry, target-ROM deficit, and correct-rep counting across guided protocols."
      badges={[
        { icon: HeartPulse, label: "Kinesiotherapy" },
        { variant: "info", label: "Bilateral ROM · symmetry" },
      ]}
      hint="Pick a protocol (shoulder flexion/abduction, elbow flexion, knee extension, hip abduction) and face the recording plane: sagittal for flexion, frontal for abduction. SPRINT measures each side independently and flags ROM deficit vs the clinical target."
      accentRgb="16,185,129"
      metrics={[
        { label: "Range of motion", value: "142", unit: "°", icon: Gauge, change: 6, accent: true, hint: "Shoulder flexion, left side" },
        { label: "L/R symmetry", value: "88", unit: "%", icon: Scale, change: 4, hint: "Asymmetry index 12%" },
        { label: "Correct reps", value: "9", unit: "/12", icon: Target, hint: "Reps reaching ≥80% of target ROM" },
        { label: "ROM deficit", value: "8", unit: "°", icon: Activity, change: -5, hint: "Vs 150° target — improving" },
      ]}
      sessions={[
        { id: 1, title: "Shoulder flexion · post-op", duration: "12:30", date: "2 days ago", score: 84, thumb: "from-emerald-500/30 to-teal-500/20" },
        { id: 2, title: "Knee extension · ACL", duration: "15:48", date: "4 days ago", score: 79, thumb: "from-teal-500/30 to-cyan-500/20" },
        { id: 3, title: "Hip abduction · mobility", duration: "10:05", date: "Last week", score: 86, thumb: "from-emerald-500/30 to-green-500/20" },
      ]}
      insights={[
        { tag: "ROM", variant: "warn", title: "Right shoulder lags 12°", detail: "Left reaches 142°, right 130°. Asymmetry index 12% — add unilateral mobility work on the right." },
        { tag: "Symmetry", variant: "info", title: "Asymmetry trending down", detail: "Down from 19% to 12% over three sessions — keep the current protocol." },
        { tag: "Target", variant: "success", title: "9 of 12 reps on target", detail: "Reps reaching ≥80% of the 150° flexion target, up from 6 last session." },
      ]}
    />
  );
}
