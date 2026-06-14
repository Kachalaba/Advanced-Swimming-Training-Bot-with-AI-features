"use client";

import { Bike, Camera, CheckCircle2, Gauge, Target } from "lucide-react";

import { CyclingUploader } from "@/components/sports/CyclingUploader";
import { SportLanding } from "@/components/sports/SportLanding";
import {
  toSportLandingData,
  useSportOverview,
} from "@/lib/sportOverview";

export default function CyclingPage() {
  const { overview, loading, error, retry } = useSportOverview("cycling");
  const data = overview
    ? toSportLandingData(overview)
    : { metrics: [], sessions: [], insights: [] };

  return (
    <SportLanding
      uploader={<CyclingUploader />}
      title="Cycling · Position, posture, pedalling"
      subtitle="Side-view bike-fit screening with knee extension, hip position, upper-body stability, cadence, and pedal-cycle evidence."
      badges={[
        { icon: Bike, label: "Road · TT · Indoor trainer" },
        {
          variant: "success",
          icon: CheckCircle2,
          label: "Persisted athlete history",
        },
        { variant: "info", label: "Quality-gated evidence" },
      ]}
      hint="Video supports movement and position screening, not measured power. Compare clips recorded from the same fixed side view and riding position."
      accentRgb="16,185,129"
      metrics={data.metrics}
      sessions={data.sessions}
      insights={data.insights}
      dataState={loading ? "loading" : error ? "error" : "ready"}
      onRetry={retry}
      uploadSubtitle="Fixed side view · Full rider and bike visible"
      secondaryPanel={{
        title: "Capture checklist",
        subtitle: "A comparable clip produces a useful baseline",
        content: (
          <div className="space-y-3">
            {[
              {
                icon: Camera,
                title: "Lock the camera",
                detail: "Place it perpendicular to the bike at crank height.",
              },
              {
                icon: Target,
                title: "Show the complete system",
                detail: "Keep the rider, crank and both wheels inside frame.",
              },
              {
                icon: Gauge,
                title: "Hold one position",
                detail: "Ride seated at a steady cadence for 15–30 seconds.",
              },
            ].map(({ icon: Icon, title, detail }) => (
              <div
                key={title}
                className="flex gap-3 rounded-lg border border-white/[0.05] bg-white/[0.02] p-3"
              >
                <Icon className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
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
