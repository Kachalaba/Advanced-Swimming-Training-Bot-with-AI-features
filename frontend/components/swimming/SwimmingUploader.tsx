"use client";

import { Camera, Check, Loader2, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { uploadSwimmingVideo } from "@/lib/swimming";

const FILMING_CUES = [
  "Record from the pool deck, directly side-on",
  "Keep the phone stable and the full body visible",
  "Capture at least four complete freestyle cycles",
  "Avoid people crossing between the camera and swimmer",
];

export function SwimmingFilmingGuide() {
  return (
    <div className="space-y-3">
      {FILMING_CUES.map((cue, index) => (
        <div
          key={cue}
          className="flex items-start gap-3 rounded-lg border border-white/[0.05] bg-white/[0.02] p-3"
        >
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-cyan-400/10 font-mono text-[10px] font-bold text-cyan-300">
            {index + 1}
          </span>
          <p className="pt-0.5 text-xs leading-relaxed text-slate-300">{cue}</p>
        </div>
      ))}
      <p className="text-[11px] leading-relaxed text-slate-500">
        The analyzer will reject an unreliable clip instead of inventing a
        skeleton or technique score.
      </p>
    </div>
  );
}

export function SwimmingUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setFileName(file.name);
    setUploading(true);
    try {
      const { jobId } = await uploadSwimmingVideo(file);
      router.push(`/swimming/${jobId}`);
    } catch (uploadError) {
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed",
      );
      setUploading(false);
    }
  }

  return (
    <div>
      <input
        ref={inputRef}
        aria-label="Choose freestyle video"
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleFile(file);
        }}
      />
      <button
        type="button"
        disabled={uploading}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          const file = event.dataTransfer.files?.[0];
          if (file) void handleFile(file);
        }}
        onClick={() => inputRef.current?.click()}
        className={`group relative flex min-h-64 w-full flex-col items-center justify-center overflow-hidden rounded-xl border border-dashed px-6 text-center transition ${
          dragging
            ? "border-cyan-400/50 bg-cyan-400/[0.06]"
            : "border-white/10 bg-black/10 hover:border-cyan-400/30 hover:bg-cyan-400/[0.025]"
        } disabled:cursor-wait disabled:opacity-75`}
      >
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_10%,rgba(34,211,238,.08),transparent_45%)]" />
        <div className="relative flex h-12 w-12 items-center justify-center rounded-xl border border-white/10 bg-white/[0.05]">
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin text-cyan-300" />
          ) : (
            <Camera className="h-5 w-5 text-slate-300 transition group-hover:text-cyan-300" />
          )}
        </div>
        <p className="relative mt-4 text-sm font-semibold text-slate-100">
          {uploading ? "Uploading side-view clip…" : "Drop a freestyle video here"}
        </p>
        <p className="relative mt-1 max-w-sm text-xs leading-relaxed text-slate-500">
          MP4, MOV, AVI or MKV up to 512 MB. Side-on pool-deck footage only in
          this version.
        </p>
        {fileName ? (
          <span className="relative mt-4 inline-flex items-center gap-1.5 rounded-md border border-white/[0.06] bg-white/[0.04] px-2 py-1 text-[11px] text-slate-300">
            {uploading ? (
              <Upload className="h-3 w-3 text-cyan-300" />
            ) : (
              <Check className="h-3 w-3 text-emerald-300" />
            )}
            {fileName}
          </span>
        ) : null}
      </button>
      {error ? (
        <p className="mt-3 rounded-lg border border-rose-400/20 bg-rose-400/[0.06] px-3 py-2 text-xs text-rose-200">
          {error}
        </p>
      ) : null}
    </div>
  );
}
