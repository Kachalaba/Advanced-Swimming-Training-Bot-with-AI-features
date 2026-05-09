import { Award, Bike, Flame, Target, Zap } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";

export default function CyclingPage() {
  return (
    <SportLanding
      title="Cycling · Power, posture, pedalling"
      subtitle="Bike fit and pedal-stroke analysis from a single side-on video. Knee tracking, hip drop, saddle height heuristics, and torque distribution."
      badges={[
        { icon: Bike, label: "Road · TT · Indoor" },
        { variant: "info", label: "35+ metrics per session" },
      ]}
      hint="Best results with a 30s clip in a steady aero or seated position. SPRINT will compute knee-extension angle, hip drop, and pedal-stroke smoothness."
      accentRgb="16,185,129"
      metrics={[
        { label: "FTP", value: "312", unit: "W", icon: Flame, change: 4, accent: true, hint: "Last test 11 days ago" },
        { label: "Cadence", value: "94", unit: "rpm", icon: Zap, change: 1, hint: "Sweet-spot range 88–98" },
        { label: "Knee extension", value: "146", unit: "°", icon: Target, change: 0, hint: "Target window 145–150°" },
        { label: "Posture score", value: "81", unit: "/100", icon: Award, change: 6, accent: true, hint: "Hip drop reduced 2°" },
      ]}
      sessions={[
        { id: 1, title: "Sweet spot intervals", duration: "1:24:05", date: "2 days ago", score: 81, thumb: "from-emerald-500/30 to-cyan-500/20" },
        { id: 2, title: "Aero hold · Indoor", duration: "48:30", date: "5 days ago", score: 88, thumb: "from-cyan-500/30 to-blue-500/20" },
        { id: 3, title: "Endurance ride", duration: "2:48:12", date: "Last week", score: 74, thumb: "from-amber-500/30 to-orange-500/20" },
      ]}
      insights={[
        { tag: "Fit", variant: "success", title: "Knee angle inside target window", detail: "146° avg — saddle height looks correct for current setup." },
        { tag: "Posture", variant: "warn", title: "Right hip drops 4° under load", detail: "Visible above 280W. Add single-leg stability work." },
        { tag: "Pedal", variant: "info", title: "Smoothness up 7% vs last block", detail: "Dead-spot at 1 o'clock improving consistently." },
      ]}
    />
  );
}
