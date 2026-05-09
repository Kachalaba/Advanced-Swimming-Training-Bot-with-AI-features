import { Activity, Dumbbell, Flame, Target } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";

export default function DrylandPage() {
  return (
    <SportLanding
      title="Dryland · Strength & mobility"
      subtitle="Counts, tempo, range-of-motion, and form for the core dryland set: squats, lunges, push-ups, planks, and rotational drills."
      badges={[
        { icon: Dumbbell, label: "Bodyweight · Loaded" },
        { variant: "info", label: "AI exercise detection" },
      ]}
      hint="Place the camera at hip height, capture the full body in frame. SPRINT auto-classifies the exercise and counts reps with eccentric/concentric tempo."
      accentRgb="139,92,246"
      metrics={[
        { label: "Weekly volume", value: "1,240", unit: "reps", icon: Flame, change: 18, accent: true, hint: "Across 4 dryland sessions" },
        { label: "Avg tempo", value: "2-1-2", icon: Activity, hint: "Ecc-pause-conc, target 3-1-1" },
        { label: "Range of motion", value: "94", unit: "%", icon: Target, change: 3, hint: "Squat depth consistent" },
        { label: "Form score", value: "76", unit: "/100", icon: Target, change: -2, hint: "Watch knee valgus on lunges" },
      ]}
      sessions={[
        { id: 1, title: "Core + mobility", duration: "38:50", date: "3 days ago", score: 76, thumb: "from-violet-500/30 to-fuchsia-500/20" },
        { id: 2, title: "Lower-body strength", duration: "52:12", date: "5 days ago", score: 82, thumb: "from-fuchsia-500/30 to-rose-500/20" },
        { id: 3, title: "Push/pull circuit", duration: "44:08", date: "Last week", score: 79, thumb: "from-violet-500/30 to-indigo-500/20" },
      ]}
      insights={[
        { tag: "Form", variant: "warn", title: "Knee valgus on right lunge", detail: "Knee tracks inside ankle on 4 of 12 reps. Add band-resisted abductions." },
        { tag: "Tempo", variant: "info", title: "Eccentric phase too quick", detail: "Avg 1.2s eccentric vs 3s target. Slow the descent for better stim." },
        { tag: "ROM", variant: "success", title: "Squat depth consistent", detail: "Hip below knee on 11 of 12 reps." },
      ]}
    />
  );
}
