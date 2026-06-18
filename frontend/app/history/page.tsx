"use client";

import {
  Bike,
  Calendar,
  Download,
  Dumbbell,
  Footprints,
  HeartPulse,
  History as HistoryIcon,
  Waves,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { api, backendAssetUrl, type SessionSummary } from "@/lib/api";

type Sport =
  | "all"
  | "swimming"
  | "running"
  | "cycling"
  | "dryland"
  | "rehab"
  | "tool";
type SessionSport = Exclude<Sport, "all">;

const sportIcon: Record<SessionSport, LucideIcon> = {
  swimming: Waves,
  running: Footprints,
  cycling: Bike,
  dryland: Dumbbell,
  rehab: HeartPulse,
  tool: Wrench,
};

const sportLabels: Record<SessionSport, string> = {
  swimming: "Swimming",
  running: "Running",
  cycling: "Cycling",
  dryland: "Dryland",
  rehab: "Rehabilitation",
  tool: "Video tools",
};

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

function formatDuration(seconds: number): string {
  if (!seconds) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = Math.round(seconds % 60);
  return hours
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${minutes}:${String(rest).padStart(2, "0")}`;
}

function sessionTitle(session: SessionSummary): string {
  if (session.session_type === "rehab") {
    return session.exercise_type.replaceAll("_", " ") || "Rehabilitation session";
  }
  if (session.session_type === "tool") {
    return session.exercise_type === "frame_extractor"
      ? "Extracted video frames"
      : "Trimmed video";
  }
  return `${sportLabels[session.session_type as SessionSport] ?? session.session_type} analysis`;
}

export default function HistoryPage() {
  const [sport, setSport] = useState<Sport>("all");
  const [period, setPeriod] = useState<"7d" | "30d" | "90d">("30d");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [athleteName, setAthleteName] = useState("Athlete");
  const [error, setError] = useState<string | null>(null);
  const [referenceTime] = useState(() => Date.now());

  useEffect(() => {
    api
      .me()
      .then(async (athlete) => {
        setAthleteName(athlete.name);
        setSessions(await api.listSessions(athlete.id));
      })
      .catch((cause) =>
        setError(cause instanceof Error ? cause.message : "Could not load history"),
      );
  }, []);

  const filtered = useMemo(() => {
    const days = Number(period.slice(0, -1));
    const cutoff = referenceTime - days * 24 * 60 * 60 * 1000;
    return sessions.filter((session) => {
      const matchesSport = sport === "all" || session.session_type === sport;
      const date = new Date(session.date).getTime();
      return matchesSport && (!Number.isFinite(date) || date >= cutoff);
    });
  }, [period, referenceTime, sessions, sport]);

  const totals = (Object.keys(sportLabels) as SessionSport[]).map((item) => {
    const rows = sessions.filter((session) => session.session_type === item);
    return {
      sport: item,
      sessions: rows.length,
      reps: rows.reduce((sum, session) => sum + session.reps, 0),
      minutes: Math.round(
        rows.reduce((sum, session) => sum + session.duration_sec, 0) / 60,
      ),
    };
  });

  return (
    <div className="animate-slide-up space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <StatusBadge variant="info" icon={HistoryIcon}>
              Persisted sessions
            </StatusBadge>
            <StatusBadge variant="neutral" icon={Calendar}>
              {athleteName}
            </StatusBadge>
          </div>
          <h1 className="text-2xl font-bold tracking-tight md:text-3xl">
            Training history
          </h1>
          <p className="mt-1.5 text-sm text-slate-400">
            Sessions saved by the analysis workflows, read directly from SQLite.
          </p>
        </div>
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

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        {totals.map((total) => {
          const Icon = sportIcon[total.sport];
          return (
            <div key={total.sport} className="rounded-xl border border-white/[0.06] bg-surface p-4">
              <Icon className="h-4 w-4 text-cyan-300" />
              <p className="mt-3 text-xs uppercase tracking-wider text-slate-500">
                {sportLabels[total.sport]}
              </p>
              <p className="mt-1 text-2xl font-bold text-slate-100">{total.sessions}</p>
              <p className="text-[11px] text-slate-500">
                sessions · {total.reps ? `${total.reps} reps` : `${total.minutes} min`}
              </p>
            </div>
          );
        })}
      </div>

      <ChartContainer
        title="Sessions"
        subtitle={error ?? `${filtered.length} persisted session${filtered.length === 1 ? "" : "s"}`}
      >
        <div className="mb-4 flex flex-wrap items-center gap-1">
          {(["all", ...Object.keys(sportLabels)] as Sport[]).map((value) => (
            <button
              key={value}
              onClick={() => setSport(value)}
              className={`rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${
                sport === value
                  ? "border-cyan-400/20 bg-cyan-400/10 text-cyan-300"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {value === "all" ? "All" : sportLabels[value]}
            </button>
          ))}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-[10px] uppercase tracking-wider text-slate-500">
                <th className="px-2 py-2 text-left">Session</th>
                <th className="px-2 py-2 text-left">Date</th>
                <th className="px-2 py-2 text-right">Duration</th>
                <th className="px-2 py-2 text-right">Reps</th>
                <th className="px-2 py-2 text-right">Score</th>
                <th className="px-2 py-2 text-right">Artifact</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((session) => {
                const item = session.session_type as SessionSport;
                const Icon = sportIcon[item] ?? HistoryIcon;
                return (
                  <tr key={session.id} className="border-t border-white/[0.04]">
                    <td className="px-2 py-3">
                      <div className="flex items-center gap-3">
                        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-white/5">
                          <Icon className="h-3.5 w-3.5 text-slate-300" />
                        </div>
                        <div>
                          <p className="font-medium capitalize text-slate-100">
                            {sessionTitle(session)}
                          </p>
                          <p className="text-[11px] text-slate-500">
                            {session.summary || `Session #${session.id}`}
                            {session.has_video ? " · video saved" : ""}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-2 py-3 text-xs text-slate-400">
                      {formatDate(session.date)}
                    </td>
                    <td className="px-2 py-3 text-right text-slate-300">
                      {formatDuration(session.duration_sec)}
                    </td>
                    <td className="px-2 py-3 text-right text-slate-300">
                      {session.reps || "—"}
                    </td>
                    <td className="px-2 py-3 text-right font-bold text-cyan-300">
                      {session.score ? session.score.toFixed(0) : "—"}
                    </td>
                    <td className="px-2 py-3 text-right">
                      {session.artifact_download_url ? (
                        <a
                          href={backendAssetUrl(session.artifact_download_url)}
                          download
                          className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 text-xs font-semibold text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
                        >
                          <Download className="h-3.5 w-3.5" />
                          Download
                        </a>
                      ) : (
                        <span className="text-slate-600">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!filtered.length && !error ? (
            <p className="py-10 text-center text-sm text-slate-500">
              No saved sessions in this period.
            </p>
          ) : null}
        </div>
      </ChartContainer>
    </div>
  );
}
