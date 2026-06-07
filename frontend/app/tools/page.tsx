import {
  Crop,
  Download,
  Layers,
  ScanLine,
  Scissors,
  Sparkles,
  Wrench,
  type LucideIcon,
} from "lucide-react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { VideoToolsWorkspace } from "@/components/tools/VideoToolsWorkspace";

type Tool = {
  icon: LucideIcon;
  title: string;
  description: string;
  status: "ready" | "prototype";
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
    status: "prototype",
  },
  {
    icon: Layers,
    title: "Multi-angle merger",
    description:
      "Sync side + front cameras into one timeline for richer biomechanics.",
    status: "prototype",
  },
  {
    icon: Sparkles,
    title: "Slow-motion remaster",
    description:
      "Frame-interpolation up to 240fps so subtle technique details stay readable.",
    status: "prototype",
  },
  {
    icon: Download,
    title: "Bulk export",
    description:
      "Package all sessions in a date range as a single zip with reports + clips.",
    status: "prototype",
  },
];

const statusStyles: Record<Tool["status"], { variant: "success" | "neutral"; label: string }> = {
  ready: { variant: "success", label: "Ready" },
  prototype: { variant: "neutral", label: "Prototype" },
};

export default function ToolsPage() {
  return (
    <div className="animate-slide-up space-y-8">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <StatusBadge variant="info" icon={Wrench}>
            Utilities
          </StatusBadge>
          <StatusBadge variant="success">2 tools ready · local processing</StatusBadge>
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
        title="Process a video"
        subtitle="Choose a utility, configure it, then download or save the result"
      >
        <VideoToolsWorkspace />
      </ChartContainer>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {tools.map((t, i) => {
          const Icon = t.icon;
          const cfg = statusStyles[t.status];
          const interactive = t.status === "ready";
          return (
            <div
              key={i}
              className={`text-left bg-surface border border-white/[0.06] rounded-xl p-5 transition-all duration-200 group ${
                interactive
                  ? "hover:border-white/[0.12] hover:bg-elevated"
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
            </div>
          );
        })}
      </div>

      <ChartContainer
        title="Processing notes"
        subtitle="Artifacts stay local until you explicitly save them"
      >
        <div className="grid gap-3 text-xs leading-relaxed text-slate-400 sm:grid-cols-3">
          <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-4">
            <p className="font-semibold text-slate-200">Temporary by default</p>
            <p className="mt-1">Download immediately or save the artifact to History.</p>
          </div>
          <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-4">
            <p className="font-semibold text-slate-200">Presentation-safe video</p>
            <p className="mt-1">Trimmed clips use H.264, AAC and fast-start metadata.</p>
          </div>
          <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] p-4">
            <p className="font-semibold text-slate-200">Auditable frame exports</p>
            <p className="mt-1">Every ZIP includes timestamps in a JSON manifest.</p>
          </div>
        </div>
      </ChartContainer>
    </div>
  );
}
