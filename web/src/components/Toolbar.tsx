import type { ReactNode } from "react";

import { cn } from "./cn";

export interface ToolbarProps {
  label: string;
  children: ReactNode;
  className?: string;
}

/** Keyboard-reachable control strip for view-level filters and actions. */
export function Toolbar({ label, children, className }: ToolbarProps) {
  return (
    <div
      role="toolbar"
      aria-label={label}
      className={cn(
        "flex flex-wrap items-center gap-2 rounded-control border border-border bg-bg-surface px-2 py-1.5",
        className,
      )}
    >
      {children}
    </div>
  );
}
