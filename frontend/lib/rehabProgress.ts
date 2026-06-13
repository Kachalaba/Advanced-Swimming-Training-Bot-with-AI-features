import type { RehabProtocol } from "./rehabilitation";

export type RehabProgressSession = {
  id: number;
  date: string;
  protocol: RehabProtocol;
  leftRom: number;
  rightRom: number;
  symmetry: number;
  repetitions: number;
  completionScore: number;
  validFrames: number | null;
  hasVideo: boolean;
};

export type RehabProgressComparison = {
  baseline: RehabProgressSession;
  current: RehabProgressSession;
  deltas: {
    leftRom: number;
    rightRom: number;
    symmetry: number;
    repetitions: number;
    completionScore: number;
  };
};

export type RehabProtocolSummary = {
  protocol: RehabProtocol;
  count: number;
  latestDate: string;
};

function round(value: number): number {
  return Math.round(value * 10) / 10;
}

export function sessionsForProtocol(
  sessions: RehabProgressSession[],
  protocol: RehabProtocol,
): RehabProgressSession[] {
  return sessions
    .filter((session) => session.protocol === protocol)
    .toSorted((left, right) => left.date.localeCompare(right.date));
}

export function compareRehabProgress(
  sessions: RehabProgressSession[],
  protocol: RehabProtocol,
): RehabProgressComparison | null {
  const compatible = sessionsForProtocol(sessions, protocol);
  if (compatible.length < 2) return null;

  const baseline = compatible[0];
  const current = compatible[compatible.length - 1];
  return {
    baseline,
    current,
    deltas: {
      leftRom: round(current.leftRom - baseline.leftRom),
      rightRom: round(current.rightRom - baseline.rightRom),
      symmetry: round(current.symmetry - baseline.symmetry),
      repetitions: current.repetitions - baseline.repetitions,
      completionScore: round(
        current.completionScore - baseline.completionScore,
      ),
    },
  };
}

export function summarizeRehabProtocols(
  sessions: RehabProgressSession[],
): RehabProtocolSummary[] {
  const summaries = new Map<RehabProtocol, RehabProtocolSummary>();
  for (const session of sessions) {
    const current = summaries.get(session.protocol);
    summaries.set(session.protocol, {
      protocol: session.protocol,
      count: (current?.count ?? 0) + 1,
      latestDate:
        !current || session.date > current.latestDate
          ? session.date
          : current.latestDate,
    });
  }
  return [...summaries.values()].toSorted((left, right) =>
    right.latestDate.localeCompare(left.latestDate),
  );
}

function signed(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

export function buildProgressObservation(
  comparison: RehabProgressComparison | null,
  locale: "uk" | "en",
): string {
  if (!comparison) {
    return locale === "uk"
      ? "Для оцінки зміни потрібні щонайменше дві сумісні сесії."
      : "At least two compatible sessions are required to describe change.";
  }

  const { deltas } = comparison;
  if (locale === "uk") {
    const leftDirection = deltas.leftRom >= 0 ? "збільшився" : "зменшився";
    const rightDirection =
      deltas.rightRom >= 0 ? "збільшився" : "зменшився";
    return [
      `Ліворуч ROM ${leftDirection} на ${Math.abs(deltas.leftRom)}°.`,
      `Праворуч ROM ${rightDirection} на ${Math.abs(deltas.rightRom)}°.`,
      `Симетрія змінилася на ${signed(deltas.symmetry)} в.п.`,
    ].join(" ");
  }

  const leftDirection = deltas.leftRom >= 0 ? "increased" : "decreased";
  const rightDirection = deltas.rightRom >= 0 ? "increased" : "decreased";
  return [
    `Left ROM ${leftDirection} by ${Math.abs(deltas.leftRom)}°.`,
    `Right ROM ${rightDirection} by ${Math.abs(deltas.rightRom)}°.`,
    `Symmetry changed by ${signed(deltas.symmetry)} pp.`,
  ].join(" ");
}
