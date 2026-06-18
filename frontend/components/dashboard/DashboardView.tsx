"use client";

import {
  Activity,
  ArrowUpRight,
  Award,
  Bike,
  ChevronDown,
  Clock,
  Dumbbell,
  Filter,
  Flame,
  Footprints,
  HeartPulse,
  Play,
  Plus,
  Sparkles,
  Target,
  TrendingUp,
  Upload,
  Waves,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { EmptyState } from "@/components/ui/EmptyState";
import { LiveSparkline } from "@/components/ui/LiveSparkline";
import { MetricCard } from "@/components/ui/MetricCard";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";

const recentSessions = [
  {
    id: 1,
    type: "Running",
    title: "Tempo run · Track",
    duration: "42:18",
    date: "2h ago",
    score: 87,
    thumb: "from-orange-500/30 to-rose-500/20",
  },
  {
    id: 2,
    type: "Swimming",
    title: "200m freestyle drills",
    duration: "1:08:42",
    date: "Yesterday",
    score: 92,
    thumb: "from-cyan-500/30 to-blue-500/20",
  },
  {
    id: 3,
    type: "Cycling",
    title: "Sweet spot intervals",
    duration: "1:24:05",
    date: "2 days ago",
    score: 81,
    thumb: "from-emerald-500/30 to-cyan-500/20",
  },
  {
    id: 4,
    type: "Dryland",
    title: "Core + mobility",
    duration: "38:50",
    date: "3 days ago",
    score: 76,
    thumb: "from-violet-500/30 to-fuchsia-500/20",
  },
];

const timelineDays = [
  { day: "Mon", load: 45, type: "swim" as const },
  { day: "Tue", load: 78, type: "run" as const },
  { day: "Wed", load: 62, type: "bike" as const },
  { day: "Thu", load: 30, type: "rest" as const },
  { day: "Fri", load: 88, type: "run" as const },
  { day: "Sat", load: 95, type: "bike" as const },
  { day: "Sun", load: 55, type: "swim" as const },
];

const phaseColors: Record<(typeof timelineDays)[number]["type"], string> = {
  swim: "from-cyan-400 to-cyan-600",
  run: "from-orange-400 to-rose-500",
  bike: "from-emerald-400 to-cyan-500",
  rest: "from-slate-600 to-slate-700",
};

type Period = "7d" | "30d" | "90d";

const analysisRoutes = [
  {
    href: "/swimming",
    label: "Swimming",
    detail: "Waterline-aware freestyle",
    icon: Waves,
    color: "text-cyan-300",
  },
  {
    href: "/running",
    label: "Running",
    detail: "Cadence and gait",
    icon: Footprints,
    color: "text-orange-300",
  },
  {
    href: "/cycling",
    label: "Cycling",
    detail: "Bike fit and posture",
    icon: Bike,
    color: "text-emerald-300",
  },
  {
    href: "/dryland",
    label: "Dryland",
    detail: "Strength movement evidence",
    icon: Dumbbell,
    color: "text-violet-300",
  },
  {
    href: "/rehabilitation",
    label: "Rehabilitation",
    detail: "Live ROM or uploaded video",
    icon: HeartPulse,
    color: "text-rose-300",
  },
] as const;

function AnalysisQuickStart() {
  return (
    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
      {analysisRoutes.map(({ href, label, detail, icon: Icon, color }) => (
        <Link
          key={href}
          href={href}
          className="group flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.025] p-3 transition hover:border-cyan-400/20 hover:bg-white/[0.045]"
        >
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-white/[0.06] bg-black/20">
            <Icon className={`h-4 w-4 ${color}`} />
          </span>
          <span className="min-w-0">
            <span className="block text-sm font-medium text-slate-200 transition group-hover:text-white">
              {label}
            </span>
            <span className="block truncate text-[11px] text-slate-500">
              {detail}
            </span>
          </span>
          <ArrowUpRight className="ml-auto h-3.5 w-3.5 text-slate-600 transition group-hover:text-cyan-300" />
        </Link>
      ))}
    </div>
  );
}

export function DashboardView({ athleteName }: { athleteName: string }) {
  const [period, setPeriod] = useState<Period>("7d");
  const [analysisMenuOpen, setAnalysisMenuOpen] = useState(false);

  return (
    <div className="animate-slide-up space-y-8">
      <div className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-br from-surface to-bg p-6 md:p-8">
        <div
          className="absolute inset-0 opacity-[0.15] pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(34,211,238,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.3) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
            maskImage:
              "radial-gradient(ellipse at top right, black 0%, transparent 70%)",
            WebkitMaskImage:
              "radial-gradient(ellipse at top right, black 0%, transparent 70%)",
          }}
        />
        <div className="absolute -top-20 -right-20 w-80 h-80 bg-cyan-400/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative flex items-start justify-between mb-6 flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <StatusBadge variant="info" icon={Activity}>
                Demo dashboard
              </StatusBadge>
              <StatusBadge variant="warn" icon={TrendingUp}>
                Sample metrics
              </StatusBadge>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-50">
              Good morning, {athleteName.split(" ")[0]}.
            </h1>
            <p className="text-sm text-slate-400 mt-1.5 max-w-xl">
              This overview uses sample planning data. Open Running,
              Rehabilitation, or History for live analysis and persisted
              sessions.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <SegmentedControl<Period>
              options={[
                { value: "7d", label: "7d" },
                { value: "30d", label: "30d" },
                { value: "90d", label: "90d" },
              ]}
              value={period}
              onChange={setPeriod}
            />
            <div className="relative">
              <button
                type="button"
                aria-haspopup="menu"
                aria-expanded={analysisMenuOpen}
                onClick={() => setAnalysisMenuOpen((open) => !open)}
                className="flex h-8 items-center gap-2 rounded-lg bg-cyan-400 px-3 text-xs font-semibold text-slate-900 transition-all duration-200 hover:bg-cyan-300 active:scale-[0.98]"
              >
                <Upload className="h-3.5 w-3.5" />
                Upload session
                <ChevronDown className="h-3 w-3" />
              </button>
              {analysisMenuOpen ? (
                <div
                  role="menu"
                  aria-label="Choose analysis workflow"
                  className="absolute right-0 top-10 z-30 w-72 rounded-xl border border-white/[0.08] bg-elevated p-2 shadow-2xl shadow-black/50"
                >
                  {analysisRoutes.map(
                    ({ href, label, detail, icon: Icon, color }) => (
                      <Link
                        key={href}
                        href={href}
                        role="menuitem"
                        onClick={() => setAnalysisMenuOpen(false)}
                        className="flex items-center gap-3 rounded-lg px-3 py-2.5 transition hover:bg-white/[0.05]"
                      >
                        <Icon className={`h-4 w-4 ${color}`} />
                        <span>
                          <span className="block text-xs font-semibold text-slate-200">
                            {label}
                          </span>
                          <span className="block text-[10px] text-slate-500">
                            {detail}
                          </span>
                        </span>
                      </Link>
                    ),
                  )}
                </div>
              ) : null}
            </div>
          </div>
        </div>

        <div className="relative grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard
            label="5K Personal Record"
            value="18:42"
            icon={Award}
            change={2.1}
            accent
          >
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <Clock className="w-3 h-3" />
              Set 4 days ago · Track
            </div>
          </MetricCard>

          <MetricCard
            label="Weekly Load"
            value="847"
            unit="TSS"
            icon={Flame}
            change={12}
          >
            <div className="flex gap-0.5 h-1.5 mt-1">
              {timelineDays.map((d, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-sm bg-cyan-400/40"
                  style={{ height: "100%", opacity: 0.3 + (d.load / 100) * 0.7 }}
                />
              ))}
            </div>
          </MetricCard>

          <MetricCard
            label="Fatigue Index"
            value="32"
            unit="/100"
            icon={Zap}
            change={-8}
          >
            <div className="flex items-center gap-1.5 mt-1">
              <span className="text-xs text-emerald-400 font-medium">
                Optimal range
              </span>
            </div>
          </MetricCard>

          <MetricCard
            label="Technique Score"
            value="87"
            unit="/100"
            icon={Target}
            change={8}
            accent
          >
            <div className="-mx-1 -mb-1">
              <LiveSparkline />
            </div>
          </MetricCard>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ChartContainer
            title="Training timeline"
            subtitle="Last 7 days · Load distribution by sport"
            action={
              <button className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
                <Filter className="w-3 h-3" /> Filter
              </button>
            }
          >
            <div className="relative h-56 flex items-end gap-2 pt-4">
              {timelineDays.map((d, i) => (
                <div
                  key={i}
                  className="flex-1 flex flex-col items-center gap-2 group"
                >
                  <div className="w-full flex-1 flex items-end relative">
                    <div
                      className={`w-full rounded-md bg-gradient-to-t ${phaseColors[d.type]} opacity-70 group-hover:opacity-100 transition-opacity duration-200 relative`}
                      style={{ height: `${d.load}%` }}
                    >
                      <div className="absolute -top-7 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-elevated border border-white/[0.08] rounded text-[10px] font-semibold text-slate-100 opacity-0 group-hover:opacity-100 transition-opacity tnum">
                        {d.load}
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">
                    {d.day}
                  </span>
                </div>
              ))}
            </div>
            <div className="flex items-center gap-4 mt-4 pt-4 border-t border-white/[0.06]">
              {[
                { label: "Swim", color: "bg-cyan-400" },
                { label: "Run", color: "bg-orange-400" },
                { label: "Bike", color: "bg-emerald-400" },
                { label: "Rest", color: "bg-slate-600" },
              ].map((l) => (
                <div key={l.label} className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-sm ${l.color}`} />
                  <span className="text-xs text-slate-400">{l.label}</span>
                </div>
              ))}
            </div>
          </ChartContainer>
        </div>

        <ChartContainer
          title="AI Insights"
          subtitle="Pattern detection · 24h"
          action={<Sparkles className="w-3.5 h-3.5 text-cyan-400" />}
        >
          <div className="space-y-3">
            {[
              {
                tag: "Recovery",
                variant: "success" as const,
                title: "HRV trending up 4 days",
                detail: "Sleep consistency is the driver. Hold current load.",
              },
              {
                tag: "Run",
                variant: "warn" as const,
                title: "Cadence dropping under fatigue",
                detail: "Consider 180-spm metronome drills mid-week.",
              },
              {
                tag: "Swim",
                variant: "info" as const,
                title: "Stroke rate stable, DPS improving",
                detail: "Catch phase efficiency up 6% over 14 days.",
              },
            ].map((insight, i) => (
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

      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Recent sessions
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any session to open the analysis view
            </p>
          </div>
          <Link
            href="/history"
            className="flex items-center gap-1 text-xs text-slate-400 transition-colors hover:text-slate-200"
          >
            View all <ArrowUpRight className="w-3 h-3" />
          </Link>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {recentSessions.map((s) => (
            <button
              key={s.id}
              className="text-left bg-surface border border-white/[0.06] rounded-xl overflow-hidden hover:border-white/[0.12] hover:bg-elevated transition-all duration-200 group active:scale-[0.99]"
            >
              <div
                className={`relative aspect-video bg-gradient-to-br ${s.thumb} flex items-center justify-center`}
              >
                <div className="w-10 h-10 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center group-hover:scale-110 transition-transform duration-200">
                  <Play className="w-4 h-4 text-white fill-white ml-0.5" />
                </div>
                <div className="absolute top-2 left-2">
                  <StatusBadge variant="neutral">{s.type}</StatusBadge>
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
                  <span className="text-[11px] text-slate-500">{s.date}</span>
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
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartContainer
          title="Start a new analysis"
          subtitle="Choose the workflow before uploading"
        >
          <AnalysisQuickStart />
        </ChartContainer>
        <ChartContainer title="Goals" subtitle="Set your race targets">
          <EmptyState
            icon={Target}
            title="No goals set yet"
            message="Define your A-race and key benchmark workouts to unlock progression tracking."
            action={
              <button className="flex items-center gap-1.5 px-3 h-8 bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.06] text-xs font-medium text-slate-200 rounded-lg transition-colors">
                <Plus className="w-3 h-3" /> Add goal
              </button>
            }
          />
        </ChartContainer>
      </div>
    </div>
  );
}
