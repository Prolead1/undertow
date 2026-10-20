import type { ReactNode } from "react";

import { cn } from "./cn";

export interface StatProps {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "accent" | "bull" | "bear" | "muted";
  className?: string;
}

const TONE: Record<NonNullable<StatProps["tone"]>, string> = {
  default: "text-text-primary",
  accent: "text-accent",
  bull: "text-bull",
  bear: "text-bear",
  muted: "text-text-muted",
};

/** Compact label/value pair used across the shell and all three views. */
export function Stat({ label, value, hint, tone = "default", className }: StatProps) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="text-[11px] font-medium uppercase tracking-wide text-text-muted">{label}</div>
      <div className={cn("tabular mt-1 truncate text-lg font-semibold", TONE[tone])}>{value}</div>
      {hint && <div className="mt-0.5 truncate text-xs text-text-muted">{hint}</div>}
    </div>
  );
}
