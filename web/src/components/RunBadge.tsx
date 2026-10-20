import type { RunRecord } from "../api/client";
import { cn } from "./cn";

export interface RunBadgeProps {
  run: RunRecord;
  selected?: boolean;
  pinned?: boolean;
  onSelect?: (run: RunRecord) => void;
  onTogglePin?: (run: RunRecord) => void;
  className?: string;
}

const STATE_DOT: Record<RunRecord["state"], string> = {
  running: "bg-accent",
  done: "bg-bull",
  failed: "bg-bear",
  unknown: "bg-sideways",
};

/** Run identity + provenance (config hash) in a single keyboard-reachable row. */
export function RunBadge({
  run,
  selected = false,
  pinned = false,
  onSelect,
  onTogglePin,
  className,
}: RunBadgeProps) {
  const content = (
    <>
      <span className={cn("size-2 shrink-0 rounded-full", STATE_DOT[run.state])} aria-hidden />
      <span className="min-w-0 flex-1 text-left">
        <span className="block truncate text-xs font-medium text-text-primary">{run.run_id}</span>
        <span className="block truncate text-[11px] text-text-muted">
          seed {run.seed} · {run.state}
          {run.config_hash ? ` · ${run.config_hash.slice(0, 8)}` : ""}
        </span>
      </span>
    </>
  );

  return (
    <div
      className={cn(
        "flex items-center gap-1 rounded-control border px-2 py-1.5",
        selected
          ? "border-accent/60 bg-bg-elevated"
          : "border-transparent hover:border-border hover:bg-bg-elevated",
        className,
      )}
    >
      {onSelect ? (
        <button
          type="button"
          onClick={() => onSelect(run)}
          aria-pressed={selected}
          className="flex min-w-0 flex-1 items-center gap-2 rounded-control text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          {content}
        </button>
      ) : (
        <div className="flex min-w-0 flex-1 items-center gap-2">{content}</div>
      )}
      {onTogglePin && (
        <button
          type="button"
          onClick={() => onTogglePin(run)}
          aria-pressed={pinned}
          aria-label={`${pinned ? "Unpin" : "Pin"} ${run.run_id}`}
          className={cn(
            "rounded-control px-1.5 py-1 text-xs",
            pinned ? "text-accent-2" : "text-text-muted hover:text-text-primary",
          )}
        >
          {pinned ? "★" : "☆"}
        </button>
      )}
    </div>
  );
}
