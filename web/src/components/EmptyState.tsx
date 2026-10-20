import type { ReactNode } from "react";

import { cn } from "./cn";

export interface EmptyStateProps {
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

/** Every panel renders this instead of a blank canvas (CONTRACTS §8). */
export function EmptyState({ title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex min-h-40 flex-col items-center justify-center gap-2 rounded-control border border-dashed border-border bg-bg-base/40 p-6 text-center",
        className,
      )}
    >
      <p className="text-sm font-medium text-text-primary">{title}</p>
      {description && <p className="max-w-prose text-xs text-text-muted">{description}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
