"use client";

import {
  ArrowUpRight,
  Bike,
  Calendar,
  Dumbbell,
  Filter,
  Footprints,
  History as HistoryIcon,
  Search,
  Waves,
  type LucideIcon,
} from "lucide-react";
import { useState } from "react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";

type Sport = "all" | "swimming" | "running" | "cycling" | "dryland";

type HistoryRow = {
  id: number;
  sport: Exclude<Sport, "all">;
  title: string;
  date: string;
  duration: string;
  score: number;
  load: number;
  notes: string;
};

const sportIcon: Record<Exclude<Sport, "all">, LucideIcon> = {
  swimming: Waves,
  running: Footprints,
  cycling: Bike,
  dryland: Dumbbell,
};

const rows: HistoryRow[] = [
  { id: 1, sport: "running", title: "Tempo run · Track", date: "May 9 · 09:14", duration: "42:18", score: 87, load: 88, notes: "Cadence dropped after 3km" },
  { id: 2, sport: "swimming", title: "200m freestyle drills", date: "May 8 · 18:42", duration: "1:08:42", score: 92, load: 65, notes: "DPS up 5%" },
  { id: 3, sport: "cycling", title: "Sweet spot intervals", date: "May 7 · 06:30", duration: "1:24:05", score: 81, load: 95, notes: "Right hip drop above 280W" },
  { id: 4, sport: "dryland", title: "Core + mobility", date: "May 6 · 19:55", duration: "38:50", score: 76, load: 32, notes: "Lunge form needs work" },
  { id: 5, sport: "running", title: "Easy 10K · Trail", date: "May 5 · 07:10", duration: "54:02", score: 79, load: 58, notes: "Heart rate stable Z2" },
  { id: 6, sport: "swimming", title: "Catch-up + 6/3/6", date: "May 5 · 17:12", duration: "44:10", score: 88, load: 50, notes: "Right roll under-rotated" },
  { id: 7, sport: "cycling", title: "Aero hold · Indoor", date: "May 4 · 06:45", duration: "48:30", score: 88, load: 78, notes: "Aero position held 28min" },
  { id: 8, sport: "running", title: "VO2 6×800m", date: "May 3 · 17:00", duration: "38:40", score: 90, load: 92, notes: "Sub-3min reps consistent" },
];

const totals: { sport: Exclude<Sport, "all">; sessions: number; volumeLabel: string; volume: string; trend: number }[] = [
  { sport: "running", sessions: 9, volumeLabel: "Distance", volume: "84.2 km", trend: 6 },
  { sport: "swimming", sessions: 6, volumeLabel: "Distance", volume: "12.4 km", trend: 4 },
  { sport: "cycling", sessions: 4, volumeLabel: "Time", volume: "08:42", trend: 12 },
  { sport: "dryland", sessions: 5, volumeLabel: "Volume", volume: "1,240 reps", trend: 18 },
];

export default function HistoryPage() {
  const [sport, setSport] = useState<Sport>("all");
  const [period, setPeriod] = useState<"7d" | "30d" | "90d">("30d");
  const filtered = sport === "all" ? rows : rows.filter((r) => r.sport === sport);

  return (
    <div className="animate-slide-up space-y-8">
      <div className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <StatusBadge variant="info" icon={HistoryIcon}>
              All disciplines
            </StatusBadge>
            <StatusBadge variant="neutral" icon={Calendar}>
              Last 30 days
            </StatusBadge>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
            Training history
          </h1>
          <p className="text-sm text-slate-400 mt-1.5 max-w-xl">
            Every analysed session in one place. Filter by discipline, jump
            into any session for the full breakdown.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <SegmentedControl
            options={[
              { value: "7d", label: "7d" },
              { value: "30d", label: "30d" },
              { value: "90d", label: "90d" },
            ]}
            value={period}
            onChange={setPeriod}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {totals.map((t) => {
          const Icon = sportIcon[t.sport];
          return (
            <div
              key={t.sport}
              className="bg-surface border border-white/[0.06] rounded-xl p-5 hover:border-white/[0.12] transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-white/5 text-slate-300 flex items-center justify-center">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                    {t.sport}
                  </span>
                </div>
                <span className="text-xs font-medium text-emerald-400">
                  +{t.trend}%
                </span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-100 tnum">
                  {t.volume}
                </span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                {t.sessions} sessions · {t.volumeLabel}
              </div>
            </div>
          );
        })}
      </div>

      <ChartContainer
        title="Sessions"
        subtitle={`${filtered.length} session${filtered.length === 1 ? "" : "s"}`}
        action={
          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center gap-2 px-2.5 h-7 bg-white/[0.04] border border-white/[0.06] rounded-lg text-xs text-slate-400">
              <Search className="w-3 h-3" />
              <input
                type="text"
                placeholder="Search…"
                className="bg-transparent outline-none placeholder:text-slate-500 w-32"
              />
            </div>
            <button className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
              <Filter className="w-3 h-3" /> Filter
            </button>
          </div>
        }
      >
        <div className="flex items-center gap-1 mb-4">
          {([
            { value: "all", label: "All" },
            { value: "swimming", label: "Swimming" },
            { value: "running", label: "Running" },
            { value: "cycling", label: "Cycling" },
            { value: "dryland", label: "Dryland" },
          ] as const).map((opt) => {
            const active = sport === opt.value;
            return (
              <button
                key={opt.value}
                onClick={() => setSport(opt.value)}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                  active
                    ? "bg-cyan-400/10 text-cyan-300 border border-cyan-400/20"
                    : "text-slate-400 hover:text-slate-200 border border-transparent"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>

        <div className="overflow-x-auto -mx-2">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                <th className="text-left px-2 py-2">Session</th>
                <th className="text-left px-2 py-2">Date</th>
                <th className="text-right px-2 py-2">Duration</th>
                <th className="text-right px-2 py-2">Load</th>
                <th className="text-right px-2 py-2">Score</th>
                <th className="text-right px-2 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => {
                const Icon = sportIcon[r.sport];
                return (
                  <tr
                    key={r.id}
                    className="border-t border-white/[0.04] hover:bg-white/[0.02] transition-colors group"
                  >
                    <td className="px-2 py-3">
                      <div className="flex items-center gap-3">
                        <div className="w-7 h-7 rounded-lg bg-white/5 text-slate-300 flex items-center justify-center shrink-0">
                          <Icon className="w-3.5 h-3.5" />
                        </div>
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-slate-100 truncate">
                            {r.title}
                          </p>
                          <p className="text-[11px] text-slate-500 truncate">
                            {r.notes}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="px-2 py-3 text-xs text-slate-400 whitespace-nowrap">
                      {r.date}
                    </td>
                    <td className="px-2 py-3 text-right tnum text-sm text-slate-200">
                      {r.duration}
                    </td>
                    <td className="px-2 py-3 text-right">
                      <div className="inline-flex items-center gap-2">
                        <span className="text-xs text-slate-400 tnum">
                          {r.load}
                        </span>
                        <div className="w-12 h-1 bg-white/5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-cyan-400/60"
                            style={{ width: `${r.load}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-2 py-3 text-right">
                      <span className="text-sm font-bold text-cyan-400 tnum">
                        {r.score}
                      </span>
                    </td>
                    <td className="px-2 py-3 text-right">
                      <button className="opacity-0 group-hover:opacity-100 transition-opacity text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 ml-auto">
                        Open <ArrowUpRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartContainer>
    </div>
  );
}
