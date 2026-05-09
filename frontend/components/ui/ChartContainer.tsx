export function ChartContainer({
  title,
  subtitle,
  action,
  children,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-surface border border-white/[0.06] rounded-xl p-5">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
          {subtitle ? (
            <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
          ) : null}
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}
