"use client";

import { Loader2, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { uploadRunningVideo } from "@/lib/analysis";

export function RunningUploader() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setUploading(true);
    try {
      const { jobId } = await uploadRunningVideo(file);
      router.push(`/running/${jobId}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
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
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer ${
          dragging
            ? "border-cyan-400 bg-cyan-400/5"
            : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
        } ${uploading ? "pointer-events-none opacity-70" : ""}`}
      >
        <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center mx-auto mb-3">
          {uploading ? (
            <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
          ) : (
            <Upload className="w-5 h-5 text-slate-400" />
          )}
        </div>
        <p className="text-sm font-medium text-slate-200">
          {uploading ? "Uploading…" : "Drop a running video here"}
        </p>
        <p className="text-xs text-slate-500 mt-1">
          MP4, MOV up to 2GB · Side-on view recommended
        </p>
        {error ? (
          <p className="text-xs text-rose-400 mt-3">{error}</p>
        ) : null}
      </div>
    </div>
  );
}
