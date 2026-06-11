import type { RehabDemoFrame } from "./demoSession";

function armPoints(rom: number, side: "left" | "right") {
  const direction = side === "left" ? -1 : 1;
  const progress = Math.max(0, Math.min(1, (rom - 65) / 90));
  const shoulderX = side === "left" ? 252 : 348;
  return {
    shoulder: { x: shoulderX, y: 178 },
    elbow: {
      x: shoulderX + direction * (50 - progress * 14),
      y: 236 - progress * 118,
    },
    wrist: {
      x: shoulderX + direction * (82 - progress * 22),
      y: 302 - progress * 224,
    },
  };
}

export function RehabBodyFigure({ frame }: { frame: RehabDemoFrame }) {
  const left = armPoints(frame.leftRom, "left");
  const right = armPoints(frame.rightRom, "right");
  const skeleton = [
    [left.shoulder, right.shoulder],
    [left.shoulder, left.elbow],
    [left.elbow, left.wrist],
    [right.shoulder, right.elbow],
    [right.elbow, right.wrist],
    [{ x: 252, y: 178 }, { x: 270, y: 320 }],
    [{ x: 348, y: 178 }, { x: 330, y: 320 }],
    [{ x: 270, y: 320 }, { x: 330, y: 320 }],
    [{ x: 270, y: 320 }, { x: 250, y: 440 }],
    [{ x: 330, y: 320 }, { x: 350, y: 440 }],
  ];

  return (
    <svg
      data-testid="rehab-body-figure"
      viewBox="0 0 600 500"
      className="h-full w-full"
      aria-label="Analyzed movement figure"
    >
      <defs>
        <radialGradient id="bodyAura">
          <stop offset="0" stopColor="#22d3ee" stopOpacity=".18" />
          <stop offset=".6" stopColor="#22d3ee" stopOpacity=".04" />
          <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="bodySurface" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#263647" />
          <stop offset=".55" stopColor="#111a24" />
          <stop offset="1" stopColor="#071018" />
        </linearGradient>
        <filter id="clinicalGlow">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <ellipse cx="300" cy="250" rx="230" ry="230" fill="url(#bodyAura)" />
      <g opacity=".16" stroke="#67e8f9" strokeWidth="1">
        {[120, 210, 300, 390, 480].map((x) => (
          <line key={`v-${x}`} x1={x} x2={x} y1="30" y2="470" />
        ))}
        {[80, 160, 240, 320, 400].map((y) => (
          <line key={`h-${y}`} x1="70" x2="530" y1={y} y2={y} />
        ))}
      </g>
      <line
        x1="300"
        x2="300"
        y1="28"
        y2="470"
        stroke="#67e8f9"
        strokeDasharray="7 9"
        opacity=".45"
      />

      <g fill="none" stroke="#67e8f9" strokeWidth="8" opacity=".08">
        <path d="M252 178 Q192 145 160 80" />
        <path d="M348 178 Q408 145 440 80" />
      </g>

      <g fill="url(#bodySurface)" stroke="rgba(148,163,184,.24)" strokeWidth="2">
        <circle cx="300" cy="102" r="40" />
        <path d="M260 150 Q300 132 340 150 L365 305 Q330 338 300 336 Q270 338 235 305Z" />
        <path d="M272 320 L246 452 L282 454 L303 335Z" />
        <path d="M328 320 L297 335 L318 454 L354 452Z" />
      </g>

      <g
        fill="none"
        stroke="#67e8f9"
        strokeWidth="4"
        strokeLinecap="round"
        filter="url(#clinicalGlow)"
      >
        {skeleton.map(([start, end], index) => (
          <line
            key={index}
            x1={start.x}
            y1={start.y}
            x2={end.x}
            y2={end.y}
          />
        ))}
      </g>
      <g fill="#071018" stroke="#a5f3fc" strokeWidth="3">
        {[
          left.shoulder,
          left.elbow,
          left.wrist,
          right.shoulder,
          right.elbow,
          right.wrist,
          { x: 270, y: 320 },
          { x: 330, y: 320 },
        ].map((point, index) => (
          <circle key={index} cx={point.x} cy={point.y} r="7" />
        ))}
      </g>
    </svg>
  );
}
