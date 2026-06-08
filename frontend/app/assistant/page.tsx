"use client";

import {
  ArrowUp,
  Brain,
  Footprints,
  Paperclip,
  Sparkles,
  TrendingUp,
  Waves,
  Zap,
} from "lucide-react";
import { useState } from "react";

import { ChartContainer } from "@/components/ui/ChartContainer";
import { StatusBadge } from "@/components/ui/StatusBadge";

type Message = {
  role: "user" | "assistant";
  content: string;
  meta?: string;
};

const seedConversation: Message[] = [
  {
    role: "assistant",
    content:
      "I've reviewed your last 14 days. Three patterns stand out — running cadence dropping under fatigue, swimming body-roll asymmetry on the right, and cycling hip drop above 280W. Want me to dig into one?",
    meta: "Pattern detection · 14d window",
  },
  {
    role: "user",
    content: "Tell me more about the cadence drop in running.",
  },
  {
    role: "assistant",
    content:
      "On Tuesday's tempo run cadence held at 178 spm for the first 3km, then dropped to 168 by km 6 even though pace was steady. That gap is too wide for a 6km tempo — it points to neuromuscular fatigue rather than aerobic. Two things I'd try: 180-spm metronome strides twice this week, and a single-leg hop test before tempo days to see if power asymmetry is the driver.",
    meta: "Sources: tempo-run-may7.mp4 · tempo-run-may3.mp4",
  },
];

const promptChips = [
  { icon: Footprints, label: "Why is my cadence dropping?" },
  { icon: Waves, label: "Compare last 4 swim sessions" },
  { icon: TrendingUp, label: "Build me a 4-week run block" },
  { icon: Zap, label: "What should I focus on this week?" },
];

export default function AssistantPage() {
  const [draft, setDraft] = useState("");
  const [messages] = useState<Message[]>(seedConversation);

  return (
    <div className="animate-slide-up grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6 h-[calc(100vh-12rem)]">
      <div className="flex flex-col bg-surface border border-white/[0.06] rounded-2xl overflow-hidden">
        <div className="flex items-center justify-between px-5 h-14 border-b border-white/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-cyan-400/10 text-cyan-400 flex items-center justify-center">
              <Sparkles className="w-4 h-4" />
            </div>
            <div className="leading-tight">
              <p className="text-sm font-semibold">SPRINT AI Coach</p>
              <p className="text-[11px] text-slate-500">
                Context: 14 sessions · 4 disciplines
              </p>
            </div>
          </div>
          <StatusBadge variant="warn" icon={Brain}>
            Prototype · no model call
          </StatusBadge>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-6 space-y-5">
          {messages.map((m, i) =>
            m.role === "assistant" ? (
              <div key={i} className="flex gap-3">
                <div className="shrink-0 w-7 h-7 rounded-lg bg-cyan-400/10 text-cyan-400 flex items-center justify-center mt-0.5">
                  <Sparkles className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 max-w-2xl">
                  <p className="text-sm text-slate-100 leading-relaxed whitespace-pre-line">
                    {m.content}
                  </p>
                  {m.meta ? (
                    <p className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mt-2">
                      {m.meta}
                    </p>
                  ) : null}
                </div>
              </div>
            ) : (
              <div key={i} className="flex justify-end">
                <div className="max-w-xl bg-white/[0.04] border border-white/[0.06] rounded-2xl rounded-tr-sm px-4 py-2.5">
                  <p className="text-sm text-slate-100 leading-relaxed">
                    {m.content}
                  </p>
                </div>
              </div>
            ),
          )}
        </div>

        <div className="p-3 border-t border-white/[0.06] bg-bg/40">
          <div className="flex items-end gap-2 bg-elevated border border-white/[0.06] rounded-xl p-2 focus-within:border-cyan-400/40 transition-colors">
            <button className="w-8 h-8 flex items-center justify-center text-slate-500 hover:text-slate-200 transition-colors">
              <Paperclip className="w-4 h-4" />
            </button>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              disabled
              rows={1}
              placeholder="AI chat is not connected in this build"
              className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-500 outline-none resize-none py-1.5 max-h-32"
            />
            <button
              disabled
              className="w-8 h-8 flex items-center justify-center bg-cyan-400 hover:bg-cyan-300 disabled:bg-white/[0.06] disabled:text-slate-500 text-slate-900 rounded-lg transition-all duration-200 active:scale-95 shrink-0"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          </div>
          <p className="text-[10px] text-slate-500 mt-2 px-1">
            Sample conversation only. No data is sent to an AI model in this
            build.
          </p>
        </div>
      </div>

      <div className="space-y-4">
        <ChartContainer title="Suggested prompts" subtitle="Tap to start">
          <div className="space-y-2">
            {promptChips.map((p, i) => {
              const Icon = p.icon;
              return (
                <button
                  key={i}
                  className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:border-white/[0.08] hover:bg-white/[0.04] text-left transition-colors group"
                >
                  <div className="w-7 h-7 rounded-lg bg-white/5 group-hover:bg-cyan-400/10 group-hover:text-cyan-400 text-slate-400 flex items-center justify-center transition-colors shrink-0">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs text-slate-200 leading-snug">
                    {p.label}
                  </span>
                </button>
              );
            })}
          </div>
        </ChartContainer>

        <ChartContainer title="Active context" subtitle="Used by the model">
          <div className="space-y-2 text-xs">
            {[
              { label: "Athlete profile", value: "Nikita K." },
              { label: "Sessions in scope", value: "14 (last 30 days)" },
              { label: "Disciplines", value: "Swim · Run · Bike · Dryland" },
              { label: "Privacy", value: "Anonymised" },
            ].map((row) => (
              <div
                key={row.label}
                className="flex items-center justify-between py-1.5 border-b border-white/[0.04] last:border-0"
              >
                <span className="text-slate-400">{row.label}</span>
                <span className="text-slate-200 font-medium">{row.value}</span>
              </div>
            ))}
          </div>
        </ChartContainer>
      </div>
    </div>
  );
}
