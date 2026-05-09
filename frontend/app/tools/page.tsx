import {
  ArrowUpRight,
  Crop,
  Download,
  FileVideo,
  Layers,
  ScanLine,
  Scissors,
  Sparkles,
  Upload,
  Wrench,
  type LucideIcon,
} from "lucide-react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { FileDropZone } from "@/components/ui/FileDropZone";
import { StatusBadge } from "@/components/ui/StatusBadge";

type Tool = {
  icon: LucideIcon;
  title: string;
  description: string;
  status: "ready" | "beta" | "soon";
};

const tools: Tool[] = [
  {
    icon: Scissors,
    title: "Trim & cut",
    description:
      "Snip the relevant rep, lap, or interval from a long video before analysis.",
    status: "ready",
  },
  {
    icon: ScanLine,
    title: "Frame extractor",
    description:
      "Pull every Nth frame as a high-quality JPEG for manual review or thumbnails.",
    status: "ready",
  },
  {
    icon: Crop,
    title: "Stabilise & crop",
    description:
      "Gyroscope-style stabilisation plus auto-crop around the detected athlete.",
    status: "beta",
  },
  {
    icon: Layers,
    title: "Multi-angle merger",
    description:
      "Sync side + front cameras into one timeline for richer biomechanics.",
    status: "beta",
  },
  {
    icon: Sparkles,
    title: "Slow-motion remaster",
    description:
      "Frame-interpolation up to 240fps so subtle technique details stay readable.",
    status: "soon",
  },
  {
    icon: Download,
    title: "Bulk export",
    description:
      "Package all sessions in a date range as a single zip with reports + clips.",
    status: "soon",
  },
];

const statusStyles: Record<Tool["status"], { variant: "success" | "info" | "neutral"; label: string }> = {
  ready: { variant: "success", label: "Ready" },
  beta: { variant: "info", label: "Beta" },
  soon: { variant: "neutral", label: "Soon" },
};

export default function ToolsPage() {
  return (
    <div className="animate-slide-up space-y-8">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <StatusBadge variant="info" icon={Wrench}>
            Utilities
          </StatusBadge>
          <StatusBadge variant="neutral">No quotas</StatusBadge>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
          Video tools
        </h1>
        <p className="text-sm text-slate-400 mt-1.5 max-w-xl">
          Pre-process clips before analysis or export polished assets for review
          sessions with athletes.
        </p>
      </div>

      <ChartContainer
        title="Drop a video"
        subtitle="Pick a tool below — we apply it to whatever you upload here"
      >
        <FileDropZone />
      </ChartContainer>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {tools.map((t, i) => {
          const Icon = t.icon;
          const cfg = statusStyles[t.status];
          const interactive = t.status !== "soon";
          return (
            <button
              key={i}
              disabled={!interactive}
              className={`text-left bg-surface border border-white/[0.06] rounded-xl p-5 transition-all duration-200 group ${
                interactive
                  ? "hover:border-white/[0.12] hover:bg-elevated active:scale-[0.99]"
                  : "opacity-70 cursor-not-allowed"
              }`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-10 h-10 rounded-xl bg-white/5 group-hover:bg-cyan-400/10 group-hover:text-cyan-400 text-slate-300 flex items-center justify-center transition-colors">
                  <Icon className="w-5 h-5" />
                </div>
                <StatusBadge variant={cfg.variant}>{cfg.label}</StatusBadge>
              </div>
              <h3 className="text-sm font-semibold text-slate-100 mb-1">
                {t.title}
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                {t.description}
              </p>
              {interactive ? (
                <div className="mt-4 flex items-center gap-1 text-xs text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity">
                  Open <ArrowUpRight className="w-3 h-3" />
                </div>
              ) : null}
            </button>
          );
        })}
      </div>

      <ChartContainer
        title="Recent exports"
        subtitle="Last 24 hours"
      >
        <div className="space-y-2">
          {[
            { name: "tempo-run-may9_annotated.mp4", size: "248 MB", time: "2h ago" },
            { name: "freestyle-drills-may8.zip", size: "1.4 GB", time: "Yesterday" },
            { name: "frame-export-cycling.zip", size: "84 MB", time: "2 days ago" },
          ].map((f) => (
            <div
              key={f.name}
              className="flex items-center justify-between p-3 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:border-white/[0.08] transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-white/5 text-slate-300 flex items-center justify-center shrink-0">
                  <FileVideo className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <p className="text-sm text-slate-100 truncate">{f.name}</p>
                  <p className="text-[11px] text-slate-500">
                    {f.size} · {f.time}
                  </p>
                </div>
              </div>
              <button className="flex items-center gap-1 px-2.5 h-7 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] rounded-lg text-xs text-slate-200 transition-colors">
                <Download className="w-3 h-3" /> Download
              </button>
            </div>
          ))}
        </div>
      </ChartContainer>
    </div>
  );
}
