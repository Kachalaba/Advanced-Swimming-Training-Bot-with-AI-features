import { Activity, RefreshCcw, Repeat2, Scale } from "lucide-react";

import type {
  CameraLevelUpdate,
  RehabReport,
} from "@/lib/rehabilitation";
import { rehabCopy, type RehabLocale } from "@/lib/rehabCopy";

function Metric({
  label,
  value,
  accent = "cyan",
  icon: Icon,
}: {
  label: string;
  value: string;
  accent?: "cyan" | "green" | "amber" | "rose";
  icon: typeof Activity;
}) {
  const colors = {
    cyan: "text-cyan-300",
    green: "text-emerald-300",
    amber: "text-amber-300",
    rose: "text-rose-300",
  };
  return (
    <div className="min-w-0 rounded-xl border border-white/[0.09] bg-[#080c12]/80 px-3 py-2.5 backdrop-blur-xl shadow-xl shadow-black/20 sm:min-w-[118px] sm:px-3.5 sm:py-3">
      <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <div className={`mt-1.5 font-mono text-lg font-bold tnum ${colors[accent]}`}>
        {value}
      </div>
    </div>
  );
}

export function LiveMetricRail({
  report,
  cameraLevel,
  poseDetected,
  locale = "uk",
}: {
  report: RehabReport | null;
  cameraLevel: CameraLevelUpdate | null;
  poseDetected: boolean | null;
  locale?: RehabLocale;
}) {
  const liveCopy = rehabCopy[locale].live;
  const copy = liveCopy.metrics;
  const left = report?.target_metrics.left.rom;
  const right = report?.target_metrics.right.rom;
  const asymmetry = report?.symmetry.asymmetry_index;
  const cameraAccent =
    cameraLevel?.status === "level"
      ? "green"
      : cameraLevel?.status === "adjust"
        ? "amber"
        : "cyan";
  const cameraValue =
    cameraLevel?.angle_deg === null || cameraLevel?.angle_deg === undefined
      ? liveCopy.uncalibrated
      : `${Math.abs(cameraLevel.angle_deg).toFixed(1)}°`;

  return (
    <div className="grid w-full grid-cols-2 gap-2 sm:flex sm:flex-wrap">
      <Metric
        label={copy.rom}
        value={
          left === undefined || right === undefined
            ? "— / —"
            : `${left.toFixed(0)}° / ${right.toFixed(0)}°`
        }
        icon={Activity}
      />
      <Metric
        label={copy.repetitions}
        value={String(report?.total_correct_reps ?? 0)}
        accent="green"
        icon={Repeat2}
      />
      <Metric
        label={copy.asymmetry}
        value={asymmetry === undefined ? "—" : `${asymmetry.toFixed(1)}%`}
        accent={asymmetry !== undefined && asymmetry > 10 ? "rose" : "cyan"}
        icon={Scale}
      />
      <Metric
        label={copy.camera}
        value={cameraValue}
        accent={cameraAccent}
        icon={RefreshCcw}
      />
      <Metric
        label={copy.pose}
        value={
          poseDetected === null
            ? copy.poseWaiting
            : poseDetected
              ? copy.poseDetected
              : copy.poseMissing
        }
        accent={poseDetected === null ? "cyan" : poseDetected ? "green" : "amber"}
        icon={Activity}
      />
    </div>
  );
}
