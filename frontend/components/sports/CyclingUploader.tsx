"use client";

import { Bike, Loader2, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { uploadCyclingVideo } from "@/lib/analysis";

export function CyclingUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setUploading(true);
    try {
      const { jobId } = await uploadCyclingVideo(file);
      router.push(`/cycling/${jobId}`);
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
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload cycling video"
        onKeyDown={(event) => {
          if (
            !uploading &&
            (event.key === "Enter" || event.key === " ")
          ) {
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
            ? "border-emerald-400 bg-emerald-400/5"
            : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
        } ${uploading ? "pointer-events-none opacity-70" : ""}`}
      >
        <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-white/5">
          {uploading ? (
            <Loader2 className="h-5 w-5 animate-spin text-emerald-400" />
          ) : (
            <Upload className="h-5 w-5 text-slate-400" />
          )}
        </div>
        <p className="text-sm font-medium text-slate-200">
          {uploading ? "Uploading…" : "Drop a side-view cycling video here"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          MP4, MOV, AVI, MKV up to 512 MB · 15–30 seconds recommended
        </p>
        <div className="mt-4 inline-flex items-center gap-1.5 text-[11px] text-slate-500">
          <Bike className="h-3 w-3" />
          Keep the full rider, crank and wheels in frame
        </div>
        {error ? (
          <p className="mt-3 text-xs text-rose-400">{error}</p>
        ) : null}
      </div>
    </div>
  );
}
