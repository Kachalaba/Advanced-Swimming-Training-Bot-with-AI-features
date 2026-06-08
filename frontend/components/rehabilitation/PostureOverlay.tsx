"use client";

import type { NormalizedPoint, PostureUpdate } from "@/lib/rehabilitation";
import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";

type Props = {
  visible: boolean;
  posture: PostureUpdate | null;
  mirrored?: boolean;
  locale?: RehabLocale;
};

const connections = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
] as const;

function signed(value?: number) {
  if (value === undefined) return "—";
  if (value > 0) return `+${value.toFixed(1)}°`;
  if (value < 0) return `−${Math.abs(value).toFixed(1)}°`;
  return "0.0°";
}

function severityColor(severity?: string) {
  if (severity === "warning") return "#fb7185";
  if (severity === "moderate") return "#fbbf24";
  return "#67e8f9";
}

export function PostureOverlay({
  visible,
  posture,
  mirrored = true,
  locale = "uk",
}: Props) {
  if (!visible) return null;
  const copy = rehabCopy[locale].live;
  const points = posture?.points ?? {};
  const mapPoint = (point?: NormalizedPoint) =>
    point
      ? {
          x: (mirrored ? 1 - point.x : point.x) * 1000,
          y: point.y * 1000,
        }
      : null;
  const shoulderMid = mapPoint(points.shoulder_mid);
  const hipMid = mapPoint(points.hip_mid);

  return (
    <svg
      data-testid="posture-overlay"
      aria-label={copy.overlayAria}
      className="absolute inset-0 h-full w-full pointer-events-none"
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
    >
      <defs>
        <filter id="postureGlow">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <linearGradient id="plumbFade" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#67e8f9" stopOpacity="0" />
          <stop offset="0.12" stopColor="#67e8f9" stopOpacity="0.58" />
          <stop offset="0.88" stopColor="#67e8f9" stopOpacity="0.58" />
          <stop offset="1" stopColor="#67e8f9" stopOpacity="0" />
        </linearGradient>
      </defs>

      <g opacity="0.23">
        {Array.from({ length: 9 }).map((_, index) => (
          <line
            key={`grid-v-${index}`}
            x1={(index + 1) * 100}
            y1="0"
            x2={(index + 1) * 100}
            y2="1000"
            stroke="#67e8f9"
            strokeWidth="1"
          />
        ))}
        {Array.from({ length: 9 }).map((_, index) => (
          <line
            key={`grid-h-${index}`}
            x1="0"
            y1={(index + 1) * 100}
            x2="1000"
            y2={(index + 1) * 100}
            stroke="#67e8f9"
            strokeWidth="1"
          />
        ))}
      </g>

      <line
        data-testid="posture-plumb-axis"
        x1="500"
        y1="40"
        x2="500"
        y2="960"
        stroke="url(#plumbFade)"
        strokeWidth="2"
        strokeDasharray="11 9"
      />
      <line
        x1="40"
        y1="500"
        x2="960"
        y2="500"
        stroke="#67e8f9"
        strokeOpacity="0.22"
        strokeWidth="1.5"
      />

      {posture?.available ? (
        <g filter="url(#postureGlow)">
          {connections.map(([from, to]) => {
            const first = mapPoint(points[from]);
            const second = mapPoint(points[to]);
            if (!first || !second) return null;
            return (
              <line
                key={`${from}-${to}`}
                x1={first.x}
                y1={first.y}
                x2={second.x}
                y2={second.y}
                stroke="#67e8f9"
                strokeOpacity="0.72"
                strokeWidth="3"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}

          {Object.entries(points).map(([name, rawPoint]) => {
            if (name.endsWith("_mid")) return null;
            const point = mapPoint(rawPoint);
            return point ? (
              <circle
                key={name}
                cx={point.x}
                cy={point.y}
                r="7"
                fill="#0a0e14"
                stroke="#a5f3fc"
                strokeWidth="3"
                vectorEffect="non-scaling-stroke"
              />
            ) : null;
          })}

          {shoulderMid && points.left_shoulder && points.right_shoulder ? (
            <line
              x1={mapPoint(points.left_shoulder)?.x}
              y1={mapPoint(points.left_shoulder)?.y}
              x2={mapPoint(points.right_shoulder)?.x}
              y2={mapPoint(points.right_shoulder)?.y}
              stroke={severityColor(posture.shoulder?.severity)}
              strokeWidth="5"
              vectorEffect="non-scaling-stroke"
            />
          ) : null}
          {hipMid && points.left_hip && points.right_hip ? (
            <line
              x1={mapPoint(points.left_hip)?.x}
              y1={mapPoint(points.left_hip)?.y}
              x2={mapPoint(points.right_hip)?.x}
              y2={mapPoint(points.right_hip)?.y}
              stroke={severityColor(posture.pelvis?.severity)}
              strokeWidth="5"
              vectorEffect="non-scaling-stroke"
            />
          ) : null}
          {shoulderMid && hipMid ? (
            <line
              x1={shoulderMid.x}
              y1={shoulderMid.y}
              x2={hipMid.x}
              y2={hipMid.y}
              stroke={severityColor(posture.trunk?.severity)}
              strokeWidth="5"
              vectorEffect="non-scaling-stroke"
            />
          ) : null}
        </g>
      ) : null}

      {posture?.available ? (
        <g className="font-mono text-[24px] font-bold">
          <text x="42" y="78" fill={severityColor(posture.shoulder?.severity)}>
            {copy.shoulders} {signed(posture.shoulder_angle_deg)}
          </text>
          <text x="42" y="116" fill={severityColor(posture.pelvis?.severity)}>
            {copy.pelvis} {signed(posture.pelvis_angle_deg)}
          </text>
          <text x="42" y="154" fill={severityColor(posture.trunk?.severity)}>
            {copy.trunk} {signed(posture.trunk_lean_deg)}
          </text>
        </g>
      ) : null}
    </svg>
  );
}
