"use client";

import type { RehabProgressSession } from "@/lib/rehabProgress";

const SERIES = {
  leftRom: { color: "#22d3ee", label: { uk: "ROM ліворуч", en: "Left ROM" } },
  rightRom: {
    color: "#818cf8",
    label: { uk: "ROM праворуч", en: "Right ROM" },
  },
  symmetry: {
    color: "#34d399",
    label: { uk: "Симетрія", en: "Symmetry" },
  },
} as const;

function dateLabel(value: string, locale: "uk" | "en"): string {
  return new Intl.DateTimeFormat(locale === "uk" ? "uk-UA" : "en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function RehabProgressChart({
  sessions,
  locale,
}: {
  sessions: RehabProgressSession[];
  locale: "uk" | "en";
}) {
  const width = 720;
  const height = 280;
  const padding = { top: 26, right: 30, bottom: 48, left: 36 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (index: number) =>
    padding.left +
    (sessions.length === 1 ? plotWidth / 2 : (index / (sessions.length - 1)) * plotWidth);
  const romValues = sessions.flatMap((session) => [
    session.leftRom,
    session.rightRom,
  ]);
  const romMin = Math.max(0, Math.floor((Math.min(...romValues) - 10) / 10) * 10);
  const romMax = Math.ceil((Math.max(...romValues) + 10) / 10) * 10;
  const romRange = romMax - romMin || 1;
  const romY = (value: number) =>
    padding.top + plotHeight - ((value - romMin) / romRange) * plotHeight;
  const symmetryY = (value: number) =>
    padding.top + plotHeight - (Math.max(0, Math.min(100, value)) / 100) * plotHeight;
  const pointString = (
    key: "leftRom" | "rightRom" | "symmetry",
  ): string =>
    sessions
      .map((session, index) => {
        const value = session[key];
        const y = key === "symmetry" ? symmetryY(value) : romY(value);
        return `${x(index)},${y}`;
      })
      .join(" ");
  const sessionWord =
    locale === "uk"
      ? `${sessions.length} реабілітаційні сесії`
      : `${sessions.length === 2 ? "Two" : sessions.length} rehabilitation sessions`;

  return (
    <div>
      <div className="mb-5 flex flex-wrap gap-4">
        {(Object.keys(SERIES) as (keyof typeof SERIES)[]).map((key) => (
          <div
            key={key}
            className="flex items-center gap-2 text-xs font-medium text-slate-300"
          >
            <span
              className="h-2 w-6 rounded-full"
              style={{ backgroundColor: SERIES[key].color }}
            />
            {SERIES[key].label[locale]}
          </div>
        ))}
      </div>

      <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#080e14] p-2">
        <svg
          role="img"
          aria-label={`${sessionWord}: ${SERIES.leftRom.label[locale]}, ${SERIES.rightRom.label[locale]}, ${SERIES.symmetry.label[locale]}.`}
          viewBox={`0 0 ${width} ${height}`}
          className="h-[280px] w-full"
          preserveAspectRatio="none"
        >
          {[0, 0.5, 1].map((ratio) => {
            const y = padding.top + plotHeight * ratio;
            return (
              <line
                key={ratio}
                x1={padding.left}
                x2={width - padding.right}
                y1={y}
                y2={y}
                stroke="rgba(148,163,184,0.12)"
                strokeDasharray="4 6"
              />
            );
          })}

          {(Object.keys(SERIES) as (keyof typeof SERIES)[]).map((key) => (
            <polyline
              key={key}
              points={pointString(key)}
              fill="none"
              stroke={SERIES[key].color}
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          ))}

          {sessions.map((session, index) => (
            <g key={session.id}>
              {(Object.keys(SERIES) as (keyof typeof SERIES)[]).map((key) => {
                const value = session[key];
                const y = key === "symmetry" ? symmetryY(value) : romY(value);
                return (
                  <circle
                    key={key}
                    cx={x(index)}
                    cy={y}
                    r="5"
                    fill={SERIES[key].color}
                    stroke="#080e14"
                    strokeWidth="3"
                    vectorEffect="non-scaling-stroke"
                  />
                );
              })}
              <text
                x={x(index)}
                y={height - 17}
                textAnchor="middle"
                fill="#64748b"
                fontSize="12"
              >
                {dateLabel(session.date, locale)}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {sessions.map((session) => (
          <div
            key={session.id}
            className="rounded-lg border border-white/[0.05] bg-white/[0.025] px-3 py-2"
          >
            <p className="text-[10px] uppercase tracking-[0.12em] text-slate-500">
              {dateLabel(session.date, locale)}
            </p>
            <div className="mt-1 flex gap-3 text-xs font-semibold tabular-nums">
              <span className="text-cyan-300">{session.leftRom}°</span>
              <span className="text-indigo-300">{session.rightRom}°</span>
              <span className="text-emerald-300">{session.symmetry}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
