"use client";

import { useCallback, useEffect, useState } from "react";

import type {
  SportName,
  SportOverview,
  SportOverviewInsight,
} from "./api";
import { api } from "./api";

type InsightVariant = "success" | "warn" | "info";

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatDuration(seconds: number): string {
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remainingSeconds = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function insightVariant(insight: SportOverviewInsight): InsightVariant {
  if (insight.level === "warning" || insight.level === "danger") return "warn";
  if (insight.level === "success") return "success";
  return "info";
}

function insightTag(insight: SportOverviewInsight): string {
  if (insight.level === "warning" || insight.level === "danger") {
    return "Movement flag";
  }
  if (insight.level === "success") return "Baseline";
  return "Context";
}

export function toSportLandingData(overview: SportOverview) {
  return {
    metrics: Object.values(overview.headline_metrics)
      .slice(0, 4)
      .map((metric) => ({
        label: metric.label,
        value: formatNumber(metric.value),
        unit: metric.unit || undefined,
      })),
    sessions: overview.sessions.map((session) => ({
      id: session.id,
      title: session.summary || `${overview.sport} analysis`,
      duration: formatDuration(session.duration_sec),
      date: session.date.slice(0, 10),
      score: session.score,
      hasVideo: session.has_video,
      thumb:
        overview.sport === "swimming"
          ? "from-cyan-500/30 to-blue-500/20"
          : "from-orange-500/30 to-rose-500/20",
    })),
    insights: overview.insights.map((insight) => ({
      tag: insightTag(insight),
      variant: insightVariant(insight),
      title: insight.title,
      detail: insight.detail,
    })),
  };
}

export function useSportOverview(sport: SportName) {
  const [overview, setOverview] = useState<SportOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestKey, setRequestKey] = useState(0);

  const retry = useCallback(() => setRequestKey((value) => value + 1), []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    api
      .me()
      .then((athlete) => api.sportOverview(athlete.id, sport))
      .then((result) => {
        if (active) setOverview(result);
      })
      .catch((loadError: unknown) => {
        if (active) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Could not load saved sessions",
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [requestKey, sport]);

  return { overview, loading, error, retry };
}
