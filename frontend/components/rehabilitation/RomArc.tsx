export function RomArc({
  value,
  side,
}: {
  value: number;
  side: "left" | "right";
}) {
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.max(0, Math.min(1, value / 180));

  return (
    <div
      data-testid={`${side}-rom-arc`}
      className={`absolute top-16 w-20 md:top-6 md:w-24 ${
        side === "left" ? "left-2 md:left-5" : "right-2 md:right-5"
      }`}
    >
      <svg viewBox="0 0 100 100" className="-rotate-90">
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke="rgba(148,163,184,.12)"
          strokeWidth="5"
        />
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="none"
          stroke={side === "left" ? "#67e8f9" : "#34d399"}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - progress)}
          className="transition-[stroke-dashoffset] duration-500 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-500">
          {side === "left" ? "L" : "R"}
        </span>
        <span className="font-mono text-lg font-bold text-slate-50 tnum">
          {value}°
        </span>
      </div>
    </div>
  );
}
