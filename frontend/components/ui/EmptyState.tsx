import type { LucideIcon } from "lucide-react";

export function EmptyState({
  icon: Icon,
  title,
  message,
  action,
}: {
  icon: LucideIcon;
  title: string;
  message: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-4">
        <Icon className="w-6 h-6 text-slate-500" />
      </div>
      <h3 className="text-sm font-semibold text-slate-200 mb-1">{title}</h3>
      <p className="text-xs text-slate-500 mb-4 max-w-xs">{message}</p>
      {action}
    </div>
  );
}
