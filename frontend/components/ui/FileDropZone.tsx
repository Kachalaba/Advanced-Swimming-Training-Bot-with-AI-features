"use client";

import { Upload } from "lucide-react";
import { useState } from "react";

export function FileDropZone() {
  const [dragging, setDragging] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
      }}
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
        dragging
          ? "border-cyan-400 bg-cyan-400/5"
          : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
      }`}
    >
      <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center mx-auto mb-3">
        <Upload className="w-5 h-5 text-slate-400" />
      </div>
      <p className="text-sm font-medium text-slate-200">Drop video files here</p>
      <p className="text-xs text-slate-500 mt-1">
        MP4, MOV up to 2GB · Multi-angle supported
      </p>
    </div>
  );
}
