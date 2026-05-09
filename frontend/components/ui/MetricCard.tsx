import type { LucideIcon } from "lucide-react";
import { TrendingDown, TrendingUp } from "lucide-react";

export function MetricCard({
  label,
  value,
  unit,
  change,
  icon: Icon,
  accent,
  loading,
  children,
}: {
  label: string;
  value: string | number;
  unit?: string;
  change?: number;
  icon?: LucideIcon;
  accent?: boolean;
  loading?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="relative overflow-hidden bg-surface border border-white/[0.06] rounded-xl p-5 transition-all duration-200 hover:border-white/[0.12] hover:bg-elevated group">
      {loading ? (
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.04] to-transparent -translate-x-full animate-shimmer" />
      ) : null}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {Icon ? (
            <div
              className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                accent
                  ? "bg-cyan-400/10 text-cyan-400"
                  : "bg-white/5 text-slate-400"
              }`}
            >
              <Icon className="w-4 h-4" />
            </div>
          ) : null}
          <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
            {label}
          </span>
        </div>
        {change !== undefined ? (
          <div
            className={`flex items-center gap-0.5 text-xs font-medium ${
              change >= 0 ? "text-emerald-400" : "text-rose-400"
            }`}
          >
            {change >= 0 ? (
              <TrendingUp className="w-3 h-3" />
            ) : (
              <TrendingDown className="w-3 h-3" />
            )}
            {Math.abs(change)}%
          </div>
        ) : null}
      </div>
      <div className="flex items-baseline gap-1.5 mb-2">
        <span className="text-3xl font-bold text-slate-100 tracking-tight tnum">
          {value}
        </span>
        {unit ? (
          <span className="text-sm text-slate-500 font-medium">{unit}</span>
        ) : null}
      </div>
      {children}
    </div>
  );
}
