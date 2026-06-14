"use client";

import { Bike, Camera, Gauge, Target } from "lucide-react";

import { SportLanding } from "@/components/sports/SportLanding";

export default function CyclingPage() {
  return (
    <SportLanding
      title="Cycling · Power, posture, pedalling"
      subtitle="Bike fit and pedal-stroke analysis from a single side-on video. Knee tracking, hip drop, saddle height heuristics, and torque distribution."
      badges={[
        { icon: Bike, label: "Road · TT · Indoor" },
        { variant: "success", label: "Analyzer core available" },
        { variant: "warn", label: "Web analysis adapter planned" },
      ]}
      hint="Best results with a 30s clip in a steady aero or seated position. SPRINT will compute knee-extension angle, hip drop, and pedal-stroke smoothness."
      accentRgb="16,185,129"
      metrics={[
        { label: "Input", value: "Side view", icon: Camera, accent: true, hint: "Steady seated or aero position" },
        { label: "Fit", value: "Knee angles", icon: Target, hint: "Top and bottom pedal positions" },
        { label: "Pedalling", value: "Cycle phases", icon: Gauge, hint: "Cadence, smoothness and dead spots" },
        { label: "Web workflow", value: "Planned", icon: Bike, hint: "Upload, evidence video and persisted history" },
      ]}
      sessions={[]}
      insights={[
        { tag: "Available", variant: "success", title: "Cycling analyzer exists in the shared core", detail: "The next product step is a FastAPI adapter with quality gates and annotated output." },
        { tag: "Boundary", variant: "info", title: "Power is not estimated from video", detail: "Video supports posture and movement analysis; power requires a connected sensor or imported activity." },
      ]}
      uploadAvailable={false}
      uploader={
        <p className="py-10 text-center text-sm text-slate-400">
          Cycling upload will open after the web adapter and confidence checks are connected.
        </p>
      }
    />
  );
}
