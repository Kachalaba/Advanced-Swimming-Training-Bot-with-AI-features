"use client";

import { Award, Target, Waves, Zap } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";

export default function SwimmingPage() {
  return (
    <SportLanding
      title="Swimming · Stroke intelligence"
      subtitle="Frame-perfect analysis of stroke mechanics, body roll, breathing, and underwater phase. 40+ metrics per session."
      badges={[
        { icon: Waves, label: "Pool · Open water" },
        { variant: "success", label: "Detection rate 96%" },
        { variant: "warn", label: "Demo metrics" },
      ]}
      hint="Upload a side-on or overhead video. SPRINT detects stroke phases (catch, pull, recovery), measures symmetry per arm, and flags timing breaks at 1/30s resolution."
      accentRgb="34,211,238"
      metrics={[
        { label: "100m PR (Free)", value: "1:02.4", icon: Award, change: 1.4, accent: true, hint: "Set 6 days ago · 25m pool" },
        { label: "Stroke rate", value: "42", unit: "/min", icon: Zap, change: 3, hint: "Avg over last 4 sessions" },
        { label: "Distance per stroke", value: "1.94", unit: "m", icon: Target, change: 5, hint: "Up from 1.85 m last block" },
        { label: "Symmetry", value: "92", unit: "%", icon: Target, change: 2, accent: true, hint: "Left/right pull balance" },
      ]}
      sessions={[
        { id: 1, title: "200m freestyle drills", duration: "1:08:42", date: "Yesterday", score: 92, thumb: "from-cyan-500/30 to-blue-500/20" },
        { id: 2, title: "Catch-up + 6/3/6 set", duration: "44:10", date: "3 days ago", score: 88, thumb: "from-blue-500/30 to-indigo-500/20" },
        { id: 3, title: "Open-water sighting", duration: "52:25", date: "Last week", score: 84, thumb: "from-cyan-500/30 to-emerald-500/20" },
      ]}
      insights={[
        { tag: "Catch", variant: "success", title: "Catch phase efficiency up 6%", detail: "Earlier vertical-forearm position over 14 days." },
        { tag: "Body roll", variant: "warn", title: "Right roll under-rotated", detail: "Avg 28° vs 35° on left. Add unilateral kick drills." },
        { tag: "Breathing", variant: "info", title: "Bilateral pattern stable", detail: "3-stroke breathing held for 86% of session." },
      ]}
    />
  );
}
