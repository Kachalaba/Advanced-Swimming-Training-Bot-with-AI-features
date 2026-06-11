export type RehabDemoFrame = {
  timestamp: number;
  leftRom: number;
  rightRom: number;
  symmetry: number;
  confidence: number;
  repetitions: number;
  phase: "ready" | "raising" | "evidence" | "complete";
};

export const demoFrames: RehabDemoFrame[] = [
  {
    timestamp: 0,
    leftRom: 72,
    rightRom: 68,
    symmetry: 94,
    confidence: 89,
    repetitions: 0,
    phase: "ready",
  },
  {
    timestamp: 0.8,
    leftRom: 88,
    rightRom: 82,
    symmetry: 93,
    confidence: 92,
    repetitions: 0,
    phase: "raising",
  },
  {
    timestamp: 1.6,
    leftRom: 112,
    rightRom: 101,
    symmetry: 90,
    confidence: 95,
    repetitions: 0,
    phase: "raising",
  },
  {
    timestamp: 2.4,
    leftRom: 139,
    rightRom: 122,
    symmetry: 88,
    confidence: 96,
    repetitions: 0,
    phase: "evidence",
  },
  {
    timestamp: 3.2,
    leftRom: 154,
    rightRom: 136,
    symmetry: 88,
    confidence: 96,
    repetitions: 1,
    phase: "evidence",
  },
  {
    timestamp: 4,
    leftRom: 128,
    rightRom: 116,
    symmetry: 91,
    confidence: 95,
    repetitions: 1,
    phase: "raising",
  },
  {
    timestamp: 4.8,
    leftRom: 154,
    rightRom: 136,
    symmetry: 88,
    confidence: 96,
    repetitions: 2,
    phase: "complete",
  },
];

export function formatDemoTimestamp(value: number): string {
  return `0:${value.toFixed(1).padStart(4, "0")}`;
}
