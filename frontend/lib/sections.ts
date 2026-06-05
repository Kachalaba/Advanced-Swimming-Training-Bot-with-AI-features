import {
  Bike,
  Dumbbell,
  Footprints,
  HeartPulse,
  History,
  Sparkles,
  Waves,
  Wrench,
  type LucideIcon,
} from "lucide-react";

export type SectionId =
  | "swimming"
  | "running"
  | "cycling"
  | "dryland"
  | "rehabilitation"
  | "history"
  | "assistant"
  | "tools";

export type Section = {
  id: SectionId;
  label: string;
  href: string;
  icon: LucideIcon;
};

export const sections: Section[] = [
  { id: "swimming", label: "Swimming", href: "/swimming", icon: Waves },
  { id: "running", label: "Running", href: "/running", icon: Footprints },
  { id: "cycling", label: "Cycling", href: "/cycling", icon: Bike },
  { id: "dryland", label: "Dryland", href: "/dryland", icon: Dumbbell },
  {
    id: "rehabilitation",
    label: "Rehabilitation",
    href: "/rehabilitation",
    icon: HeartPulse,
  },
  { id: "history", label: "History", href: "/history", icon: History },
  { id: "assistant", label: "AI Assistant", href: "/assistant", icon: Sparkles },
  { id: "tools", label: "Tools", href: "/tools", icon: Wrench },
];
