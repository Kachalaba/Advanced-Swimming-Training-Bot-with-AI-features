"use client";

import {
  CheckCircle2,
  Dumbbell,
  Loader2,
  Move3D,
  Ruler,
  Upload,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import {
  uploadDrylandVideo,
  type DrylandExerciseType,
} from "@/lib/analysis";

type ExerciseProfile = {
  label: string;
  shortLabel: string;
  description: string;
  guidance: string;
  trackedJoint: string;
  checks: string[];
};

const EXERCISES: Record<DrylandExerciseType, ExerciseProfile> = {
  squat: {
    label: "Squat",
    shortLabel: "Squat",
    description: "Best for knee ROM, depth consistency and lower-limb control.",
    guidance:
      "Fixed side view. Keep shoulder, hip, knee, ankle and both feet visible from ready to depth and back.",
    trackedJoint: "Knee angle",
    checks: ["Fixed side view", "Full body visible", "Start and finish tall"],
  },
  lunge: {
    label: "Lunge",
    shortLabel: "Lunge",
    description: "Best for single-leg control, front-knee tracking and symmetry.",
    guidance:
      "Fixed side view. Record repeated reps on one side and keep both feet, knees and hips visible.",
    trackedJoint: "Front knee angle",
    checks: ["One repeated side", "Both feet visible", "Torso not cropped"],
  },
  push_up: {
    label: "Push-up",
    shortLabel: "Push-up",
    description: "Best for elbow ROM, trunk line and repeatable upper-body tempo.",
    guidance:
      "Camera at torso height. Keep hands, shoulders, hips and feet visible through every rep.",
    trackedJoint: "Elbow angle",
    checks: ["Camera at torso height", "Hands visible", "Hips and feet visible"],
  },
};

const ORDER: DrylandExerciseType[] = ["squat", "lunge", "push_up"];

export function DrylandUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [exercise, setExercise] = useState<DrylandExerciseType>("squat");
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeExercise = EXERCISES[exercise];

  async function handleFile(file: File) {
    setError(null);
    setUploading(true);
    try {
      const { jobId } = await uploadDrylandVideo(file, exercise, 15);
      router.push(`/dryland/${jobId}`);
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed",
      );
      setUploading(false);
    }
  }

  return (
    <div className="space-y-5">
      <input
        aria-label="Dryland video file"
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      <div>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-violet-200/70">
              Exercise profile
            </p>
            <h3 className="mt-1 text-base font-semibold text-slate-100">
              Choose the movement before upload
            </h3>
          </div>
          <div className="rounded-full border border-violet-300/20 bg-violet-400/10 px-3 py-1 text-xs font-medium text-violet-100">
            Quality-gated · 15 fps
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {ORDER.map((id) => {
            const item = EXERCISES[id];
            return (
              <button
                key={id}
                type="button"
                aria-pressed={exercise === id}
                aria-label={`Select ${item.label}`}
                onClick={() => setExercise(id)}
                className={`rounded-2xl border p-4 text-left transition ${
                  exercise === id
                    ? "border-violet-300/55 bg-violet-400/10 shadow-[0_0_28px_rgba(139,92,246,0.14)]"
                    : "border-white/[0.08] bg-white/[0.03] hover:border-white/20"
                }`}
              >
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
                  <Dumbbell className="h-4 w-4 text-violet-300" />
                  {item.label}
                </div>
                <p className="text-xs leading-relaxed text-slate-400">
                  {item.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-2xl border border-violet-300/15 bg-violet-400/[0.06] p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-violet-100">
            <Move3D className="h-4 w-4" />
            Recommended capture
          </div>
          <p className="text-sm leading-relaxed text-slate-300">
            {activeExercise.guidance}
          </p>
        </div>
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-100">
            <Ruler className="h-4 w-4 text-violet-300" />
            Metric target · {activeExercise.trackedJoint}
          </div>
          <div className="grid gap-2">
            {activeExercise.checks.map((check) => (
              <div
                key={check}
                className="flex items-center gap-2 text-xs text-slate-300"
              >
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
                {check}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label={`Upload ${activeExercise.shortLabel} dryland video`}
        onKeyDown={(event) => {
          if (!uploading && (event.key === "Enter" || event.key === " ")) {
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-all duration-200 ${
          dragging
            ? "border-violet-400 bg-violet-400/5"
            : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
        } ${uploading ? "pointer-events-none opacity-70" : ""}`}
      >
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-white/5">
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin text-violet-400" />
          ) : (
            <Upload className="h-5 w-5 text-slate-400" />
          )}
        </div>
        <p className="text-sm font-medium text-slate-200">
          {uploading
            ? "Uploading…"
            : `Drop a ${activeExercise.shortLabel.toLowerCase()} clip here`}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          MP4, MOV, AVI, MKV up to 512 MB · Fixed side view recommended
        </p>
        {error ? (
          <p className="mt-3 text-xs text-rose-400">{error}</p>
        ) : null}
      </div>
    </div>
  );
}
