import type { ReactNode } from "react";

import { cn } from "./cn";

export interface PanelProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

/** The single surface primitive every view composes (CONTRACTS §9). */
export function Panel({ title, subtitle, actions, children, className, bodyClassName }: PanelProps) {
  return (
    <section
      className={cn(
        "flex min-w-0 flex-col rounded-panel border border-border bg-bg-surface shadow-[0_1px_0_rgba(255,255,255,0.03)]",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-border px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="truncate text-sm font-semibold text-text-primary">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn("min-h-0 flex-1 p-4", bodyClassName)}>{children}</div>
    </section>
  );
}
