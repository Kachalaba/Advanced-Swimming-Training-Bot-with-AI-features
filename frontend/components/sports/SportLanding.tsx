"use client";

import {
  ArrowUpRight,
  Filter,
  Play,
  Plus,
  Sparkles,
  Target,
  Upload,
  type LucideIcon,
} from "lucide-react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { EmptyState } from "@/components/ui/EmptyState";
import { FileDropZone } from "@/components/ui/FileDropZone";
import { MetricCard } from "@/components/ui/MetricCard";
import { StatusBadge } from "@/components/ui/StatusBadge";

export type SportMetric = {
  label: string;
  value: string;
  unit?: string;
  change?: number;
  icon?: LucideIcon;
  accent?: boolean;
  hint?: string;
};

export type SportSession = {
  id: string | number;
  title: string;
  duration: string;
  date: string;
  score: number;
  thumb: string;
};

export type SportLandingProps = {
  title: string;
  subtitle: string;
  badges: { icon?: LucideIcon; label: string; variant?: "info" | "success" | "warn" | "danger" | "neutral" }[];
  hint: string;
  accentRgb: string;
  metrics: SportMetric[];
  sessions: SportSession[];
  insights: { tag: string; variant: "success" | "warn" | "info"; title: string; detail: string }[];
};

export function SportLanding({
  title,
  subtitle,
  badges,
  hint,
  accentRgb,
  metrics,
  sessions,
  insights,
}: SportLandingProps) {
  return (
    <div className="animate-slide-up space-y-8">
      <div className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-br from-surface to-bg p-6 md:p-8">
        <div
          className="absolute inset-0 opacity-[0.15] pointer-events-none"
          style={{
            backgroundImage: `linear-gradient(rgba(${accentRgb},0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(${accentRgb},0.3) 1px, transparent 1px)`,
            backgroundSize: "32px 32px",
            maskImage:
              "radial-gradient(ellipse at top right, black 0%, transparent 70%)",
            WebkitMaskImage:
              "radial-gradient(ellipse at top right, black 0%, transparent 70%)",
          }}
        />
        <div
          className="absolute -top-20 -right-20 w-80 h-80 rounded-full blur-3xl pointer-events-none"
          style={{ backgroundColor: `rgba(${accentRgb},0.10)` }}
        />

        <div className="relative flex items-start justify-between mb-6 flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              {badges.map((b, i) => (
                <StatusBadge key={i} variant={b.variant ?? "info"} icon={b.icon}>
                  {b.label}
                </StatusBadge>
              ))}
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-50">
              {title}
            </h1>
            <p className="text-sm text-slate-400 mt-1.5 max-w-xl">{subtitle}</p>
          </div>
          <div className="flex items-center gap-2">
            <button className="flex items-center gap-2 px-3 h-8 bg-cyan-400 hover:bg-cyan-300 text-slate-900 text-xs font-semibold rounded-lg transition-all duration-200 active:scale-[0.98]">
              <Upload className="w-3.5 h-3.5" />
              Upload session
            </button>
          </div>
        </div>

        <div className="relative grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {metrics.map((m, i) => (
            <MetricCard
              key={i}
              label={m.label}
              value={m.value}
              unit={m.unit}
              change={m.change}
              icon={m.icon}
              accent={m.accent}
            >
              {m.hint ? (
                <div className="text-xs text-slate-500">{m.hint}</div>
              ) : null}
            </MetricCard>
          ))}
        </div>

        <p className="relative text-xs text-slate-500 mt-5 max-w-2xl">{hint}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ChartContainer
            title="Recent sessions"
            subtitle="Click to open the analysis view"
            action={
              <button className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
                <Filter className="w-3 h-3" /> Filter
              </button>
            }
          >
            {sessions.length === 0 ? (
              <EmptyState
                icon={Target}
                title="No sessions yet"
                message="Upload your first video to start building a baseline for this discipline."
                action={
                  <button className="flex items-center gap-1.5 px-3 h-8 bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.06] text-xs font-medium text-slate-200 rounded-lg transition-colors">
                    <Plus className="w-3 h-3" /> Upload first session
                  </button>
                }
              />
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {sessions.map((s) => (
                  <button
                    key={s.id}
                    className="text-left bg-bg border border-white/[0.06] rounded-xl overflow-hidden hover:border-white/[0.12] hover:bg-elevated transition-all duration-200 group active:scale-[0.99]"
                  >
                    <div
                      className={`relative aspect-video bg-gradient-to-br ${s.thumb} flex items-center justify-center`}
                    >
                      <div className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
                        <Play className="w-4 h-4 text-white fill-white ml-0.5" />
                      </div>
                      <div className="absolute bottom-2 right-2 px-1.5 py-0.5 bg-black/60 backdrop-blur-sm rounded text-[10px] font-mono text-white tnum">
                        {s.duration}
                      </div>
                    </div>
                    <div className="p-3">
                      <p className="text-sm font-medium text-slate-100 truncate">
                        {s.title}
                      </p>
                      <div className="flex items-center justify-between mt-1.5">
                        <span className="text-[11px] text-slate-500">
                          {s.date}
                        </span>
                        <div className="flex items-center gap-1">
                          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                            Score
                          </span>
                          <span className="text-xs font-bold text-cyan-400 tnum">
                            {s.score}
                          </span>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </ChartContainer>
        </div>

        <ChartContainer
          title="AI Insights"
          subtitle="Pattern detection · 14d"
          action={<Sparkles className="w-3.5 h-3.5 text-cyan-400" />}
        >
          <div className="space-y-3">
            {insights.map((insight, i) => (
              <div
                key={i}
                className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:border-white/[0.08] transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-2 mb-1.5">
                  <StatusBadge variant={insight.variant}>
                    {insight.tag}
                  </StatusBadge>
                </div>
                <p className="text-sm font-medium text-slate-100">
                  {insight.title}
                </p>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  {insight.detail}
                </p>
              </div>
            ))}
          </div>
        </ChartContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartContainer
          title="Upload new session"
          subtitle="Multi-angle video supported"
        >
          <FileDropZone />
        </ChartContainer>
        <ChartContainer title="Drills library" subtitle="Discipline-specific cues">
          <EmptyState
            icon={Target}
            title="Coming soon"
            message="A curated library of drills tied to detected weaknesses in your video analysis."
            action={
              <button className="flex items-center gap-1.5 px-3 h-8 bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.06] text-xs font-medium text-slate-200 rounded-lg transition-colors">
                <ArrowUpRight className="w-3 h-3" /> Notify me
              </button>
            }
          />
        </ChartContainer>
      </div>
    </div>
  );
}
