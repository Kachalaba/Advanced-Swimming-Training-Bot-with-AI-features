"use client";

import {
  Activity,
  Camera,
  CircleGauge,
  HeartPulse,
  ShieldCheck,
  Upload,
} from "lucide-react";
import { useState } from "react";

import { LiveRehabWorkspace } from "@/components/rehabilitation/LiveRehabWorkspace";
import { RehabUploader } from "@/components/rehabilitation/RehabUploader";
import { SegmentedControl } from "@/components/ui/SegmentedControl";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  rehabProtocols,
  type RehabProtocol,
} from "@/lib/rehabilitation";

const protocolLabels: Record<RehabProtocol, string> = {
  shoulder_flexion: "Сгибание плеча",
  shoulder_abduction: "Отведение плеча",
  elbow_flexion: "Сгибание локтя",
  knee_extension: "Разгибание колена",
  hip_abduction: "Отведение бедра",
};

type InputMode = "live" | "upload";

export default function RehabilitationPage() {
  const [protocol, setProtocol] =
    useState<RehabProtocol>("shoulder_flexion");
  const [mode, setMode] = useState<InputMode>("live");

  return (
    <div className="animate-slide-up space-y-6">
      <section className="relative overflow-hidden rounded-2xl border border-white/[0.07] bg-gradient-to-br from-surface via-surface to-bg p-6 md:p-8">
        <div className="absolute -right-20 -top-28 h-80 w-80 rounded-full bg-cyan-400/[0.08] blur-3xl" />
        <div
          className="absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(34,211,238,.24) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.24) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
            maskImage:
              "radial-gradient(ellipse at top right, black, transparent 68%)",
          }}
        />
        <div className="relative flex flex-col justify-between gap-6 xl:flex-row xl:items-end">
          <div>
            <div className="mb-3 flex flex-wrap gap-2">
              <StatusBadge variant="info" icon={HeartPulse}>
                Rehabilitation intelligence
              </StatusBadge>
              <StatusBadge variant="success" icon={ShieldCheck}>
                Local processing
              </StatusBadge>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-50 md:text-4xl">
              Posture, ROM & recovery
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
              Live-кинезиотерапия с двусторонним ROM, картой плечевого и
              тазового перекоса, контролем корпуса и относительным уровнем
              камеры.
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            <label className="space-y-1.5">
              <span className="block text-[10px] font-semibold uppercase tracking-[0.13em] text-slate-500">
                Протокол
              </span>
              <select
                value={protocol}
                onChange={(event) =>
                  setProtocol(event.target.value as RehabProtocol)
                }
                className="h-9 min-w-[210px] rounded-lg border border-white/10 bg-[#0b1118] px-3 text-xs font-medium text-slate-200 outline-none transition focus:border-cyan-400/50"
              >
                {rehabProtocols.map((value) => (
                  <option key={value} value={value}>
                    {protocolLabels[value]}
                  </option>
                ))}
              </select>
            </label>
            <SegmentedControl<InputMode>
              value={mode}
              onChange={setMode}
              options={[
                { value: "live", label: "Live camera" },
                { value: "upload", label: "Upload video" },
              ]}
            />
          </div>
        </div>
      </section>

      {mode === "live" ? (
        <LiveRehabWorkspace key={protocol} protocol={protocol} />
      ) : (
        <section className="rounded-2xl border border-white/[0.07] bg-surface p-5 md:p-6">
          <RehabUploader protocol={protocol} />
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-3">
        {[
          {
            icon: CircleGauge,
            title: "Двусторонний ROM",
            text: "Амплитуда слева и справа, дефицит цели и качество повторов.",
          },
          {
            icon: Activity,
            title: "Постуральные оси",
            text: "Плечи, таз и линия корпуса накладываются прямо на live-видео.",
          },
          {
            icon: mode === "live" ? Camera : Upload,
            title: "Web сейчас, Mac потом",
            text: "Контракты камеры и анализа готовы к будущей AVFoundation-оболочке.",
          },
        ].map(({ icon: Icon, title, text }) => (
          <div
            key={title}
            className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"
          >
            <Icon className="h-4 w-4 text-cyan-300" />
            <h2 className="mt-3 text-sm font-semibold text-slate-100">{title}</h2>
            <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{text}</p>
          </div>
        ))}
      </section>

      <p className="text-center text-[11px] leading-relaxed text-slate-600">
        Видеоанализ является тренировочным инструментом и не заменяет оценку
        врача или физического терапевта. Угол камеры измеряется относительно
        последней оптической калибровки.
      </p>
    </div>
  );
}
