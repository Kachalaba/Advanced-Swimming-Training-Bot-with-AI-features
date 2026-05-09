"use client";

import { useEffect, useState } from "react";

export function LiveSparkline() {
  const [data, setData] = useState<number[]>([
    62, 65, 64, 68, 71, 70, 73, 75, 78, 82, 80, 84,
  ]);

  useEffect(() => {
    const id = setInterval(() => {
      setData((prev) => {
        const last = prev[prev.length - 1];
        const next = Math.max(60, Math.min(95, last + (Math.random() - 0.4) * 6));
        return [...prev.slice(1), Math.round(next)];
      });
    }, 2000);
    return () => clearInterval(id);
  }, []);

  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const points = data
    .map(
      (v, i) =>
        `${(i / (data.length - 1)) * 100},${
          100 - ((v - min) / range) * 80 - 10
        }`,
    )
    .join(" ");
  const lastY = 100 - ((data[data.length - 1] - min) / range) * 80 - 10;

  return (
    <svg className="w-full h-12" viewBox="0 0 100 100" preserveAspectRatio="none">
      <defs>
        <linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22D3EE" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#22D3EE" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={`0,100 ${points} 100,100`}
        fill="url(#spark-fill)"
        stroke="none"
        style={{ transition: "all 1.8s ease-out" }}
      />
      <polyline
        points={points}
        fill="none"
        stroke="#22D3EE"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ transition: "all 1.8s ease-out" }}
      />
      <circle
        cx="100"
        cy={lastY}
        r="1.8"
        fill="#22D3EE"
        style={{ transition: "all 1.8s ease-out" }}
      >
        <animate
          attributeName="r"
          values="1.8;3;1.8"
          dur="2s"
          repeatCount="indefinite"
        />
      </circle>
    </svg>
  );
}
