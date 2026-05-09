"use client";

type Option<T extends string> = { value: T; label: string };

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
}: {
  options: Option<T>[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex p-0.5 bg-white/5 border border-white/[0.06] rounded-lg">
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-200 ${
              active
                ? "bg-white/[0.08] text-slate-100 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
