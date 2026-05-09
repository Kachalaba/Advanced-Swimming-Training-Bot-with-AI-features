import type { LucideIcon } from "lucide-react";

type Variant = "success" | "warn" | "danger" | "info" | "neutral";

const variants: Record<Variant, string> = {
  success: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  warn: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  danger: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  info: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  neutral: "bg-white/5 text-slate-300 border-white/10",
};

export function StatusBadge({
  variant = "neutral",
  icon: Icon,
  children,
}: {
  variant?: Variant;
  icon?: LucideIcon;
  children: React.ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium border ${variants[variant]}`}
    >
      {Icon ? <Icon className="w-3 h-3" /> : null}
      {children}
    </span>
  );
}
