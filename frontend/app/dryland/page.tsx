"use client";

import { Activity, Camera, Dumbbell, Target } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";

export default function DrylandPage() {
  return (
    <SportLanding
      title="Dryland · Strength & mobility"
      subtitle="Counts, tempo, range-of-motion, and form for the core dryland set: squats, lunges, push-ups, planks, and rotational drills."
      badges={[
        { icon: Dumbbell, label: "Bodyweight · Loaded" },
        { variant: "success", label: "Exercise analyzer available" },
        { variant: "warn", label: "Web workflow planned" },
      ]}
      hint="Place the camera at hip height, capture the full body in frame. SPRINT auto-classifies the exercise and counts reps with eccentric/concentric tempo."
      accentRgb="139,92,246"
      metrics={[
        { label: "Input", value: "Full body", icon: Camera, accent: true, hint: "Camera at hip height" },
        { label: "Detection", value: "Exercise type", icon: Dumbbell, hint: "Squat, lunge, push-up, plank and mobility" },
        { label: "Measurement", value: "Reps + tempo", icon: Activity, hint: "Phase timing and range of motion" },
        { label: "Web workflow", value: "Planned", icon: Target, hint: "Live/upload capture and persisted history" },
      ]}
      sessions={[]}
      insights={[
        { tag: "Available", variant: "success", title: "Dryland analysis is available in the shared core", detail: "The web workflow will reuse the capture-quality and athlete-history contracts." },
        { tag: "Evidence", variant: "info", title: "No athlete conclusions before measurement", detail: "This page intentionally stays empty until a real session is captured and saved." },
      ]}
      uploadAvailable={false}
      uploader={
        <p className="py-10 text-center text-sm text-slate-400">
          Dryland capture will open after exercise selection and readiness checks are connected.
        </p>
      }
    />
  );
}
