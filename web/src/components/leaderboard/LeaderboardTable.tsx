import { Fragment, useEffect, useMemo, useState } from "react";

import type { RunRecord } from "../../api/client";
import { Sparkline } from "../Sparkline";
import { MAX_COMPARE_RUNS } from "./pinned";
import { rankRuns, sortRuns, type LeaderboardSortKey, type SortDirection } from "./ranking";
import { ProvenanceCard } from "./ProvenanceCard";

// ---------------------------------------------------------------------------
// Presentational helpers
// ---------------------------------------------------------------------------

function formatNumber(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().slice(0, 19).replace("T", " ");
}

interface Column {
  key: LeaderboardSortKey;
  label: string;
  tooltip: string;
  numeric: boolean;
}

const COLUMNS: readonly Column[] = [
  {
    key: "rank",
    label: "Rank",
    tooltip:
      "Default ranking (PLAN §7): held-out eval Sharpe ↓, then worst-regime PnL ↓, then best_reward ↓. Train-only runs always sort after evaluated runs.",
    numeric: true,
  },
  { key: "run_id", label: "Run", tooltip: "Manifest run id; click a header to sort.", numeric: false },
  {
    key: "experiment",
    label: "Experiment / seed",
    tooltip: "Parent experiment directory and training seed.",
    numeric: false,
  },
  { key: "state", label: "State", tooltip: "Training status from status.json.", numeric: false },
  {
    key: "best_reward",
    label: "Best reward",
    tooltip: "Best normalised episode reward seen during training.",
    numeric: true,
  },
  {
    key: "sharpe",
    label: "Eval Sharpe",
    tooltip: "Held-out eval Sharpe; the primary ranking key.",
    numeric: true,
  },
  { key: "total_pnl", label: "Total PnL", tooltip: "Held-out eval total PnL (USDC).", numeric: true },
  {
    key: "max_drawdown",
    label: "Max DD",
    tooltip: "Held-out eval maximum drawdown; closer to 0 is better.",
    numeric: true,
  },
  {
    key: "worst_regime_pnl",
    label: "Worst-regime PnL",
    tooltip: "Held-out eval PnL in the worst regime; the first tiebreak.",
    numeric: true,
  },
  {
    key: "config_hash",
    label: "Config",
    tooltip: "Short config hash. Expand a row for the full provenance.",
    numeric: false,
  },
  { key: "git_commit", label: "Commit", tooltip: "Short git commit of the training code.", numeric: false },
  {
    key: "updated_at",
    label: "Updated",
    tooltip: "Last artifact write time (UTC).",
    numeric: false,
  },
];

function shortHash(value: string | null | undefined): string {
  return value ? value.slice(0, 8) : "—";
}

export interface LeaderboardTableProps {
  runs: readonly RunRecord[];
  pinnedIds?: readonly string[];
  onTogglePin?: (runId: string) => void;
  maxPinned?: number;
  /** Learning-curve `mean` values keyed by run id, for the expanded sparkline. */
  curves?: Record<string, readonly number[]>;
  /** Called when a row is expanded so the view can lazily fetch its curve. */
  onExpand?: (runId: string) => void;
  /** Reports the currently visible (filtered + sorted) rows for CSV export. */
  onVisibleChange?: (runs: RunRecord[]) => void;
  className?: string;
}

/**
 * Sortable, filterable leaderboard. Default order follows the pinned ranking
 * rule; every row exposes provenance and an expandable learning-curve sparkline.
 */
export function LeaderboardTable({
  runs,
  pinnedIds = [],
  onTogglePin,
  maxPinned = MAX_COMPARE_RUNS,
  curves = {},
  onExpand,
  onVisibleChange,
  className,
}: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<LeaderboardSortKey>("rank");
  const [direction, setDirection] = useState<SortDirection>("asc");
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("all");
  const [experimentFilter, setExperimentFilter] = useState("all");
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());

  const experiments = useMemo(
    () => [...new Set(runs.map((run) => run.experiment))].sort((a, b) => a.localeCompare(b)),
    [runs],
  );

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = runs.filter((run) => {
      if (stateFilter !== "all" && run.state !== stateFilter) return false;
      if (experimentFilter !== "all" && run.experiment !== experimentFilter) return false;
      if (needle && !run.run_id.toLowerCase().includes(needle)) return false;
      return true;
    });
    return sortRuns(filtered, sortKey, direction);
  }, [runs, query, stateFilter, experimentFilter, sortKey, direction]);

  // The displayed rank is the pinned-rule position within the visible set, so
  // it agrees with the CSV export even when the table is sorted by another
  // column. `sortRuns(visible, "rank", "asc")` is already the pinned order when
  // `sortKey === "rank"`, but recomputing keeps the mapping independent of it.
  const rankByRunId = useMemo(() => {
    const map = new Map<string, number>();
    rankRuns(visible).forEach((run, position) => map.set(run.run_id, position + 1));
    return map;
  }, [visible]);

  const atLimit = pinnedIds.length >= maxPinned;

  useEffect(() => {
    onVisibleChange?.(visible);
  }, [visible, onVisibleChange]);

  function handleSort(key: LeaderboardSortKey): void {
    if (key === sortKey) {
      setDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    const column = COLUMNS.find((candidate) => candidate.key === key);
    setSortKey(key);
    // Rank is already best-first; numeric metrics are most useful high-to-low.
    setDirection(key === "rank" ? "asc" : column?.numeric ? "desc" : "asc");
  }

  function toggleExpand(runId: string): void {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(runId)) {
        next.delete(runId);
      } else {
        next.add(runId);
        onExpand?.(runId);
      }
      return next;
    });
  }

  return (
    <div className={className}>
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-text-muted">
          Search run id
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="ppo_s0…"
            className="w-48 rounded-control border border-border bg-bg-base px-2 py-1 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-muted">
          State
          <select
            value={stateFilter}
            onChange={(event) => setStateFilter(event.target.value)}
            className="rounded-control border border-border bg-bg-base px-2 py-1 text-sm text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <option value="all">All states</option>
            <option value="running">running</option>
            <option value="done">done</option>
            <option value="failed">failed</option>
            <option value="unknown">unknown</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-muted">
          Experiment
          <select
            value={experimentFilter}
            onChange={(event) => setExperimentFilter(event.target.value)}
            className="rounded-control border border-border bg-bg-base px-2 py-1 text-sm text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <option value="all">All experiments</option>
            {experiments.map((experiment) => (
              <option key={experiment} value={experiment}>
                {experiment}
              </option>
            ))}
          </select>
        </label>
        <p data-testid="leaderboard-count" className="tabular ml-auto text-xs text-text-muted">
          {visible.length} of {runs.length} runs
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[68rem] border-collapse text-sm">
          <caption className="sr-only">
            Ranked runs. Default order: evaluated runs by eval Sharpe, then
            worst-regime PnL, then best reward; train-only runs last.
          </caption>
          <thead>
            <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-text-muted">
              <th scope="col" className="px-2 py-2">
                <span className="sr-only">Details</span>
              </th>
              <th scope="col" className="px-2 py-2">
                <span className="sr-only">Compare</span>
              </th>
              {COLUMNS.map((column) => (
                <th
                  key={column.key}
                  scope="col"
                  title={column.tooltip}
                  aria-sort={
                    sortKey === column.key
                      ? direction === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                  className={column.numeric ? "px-2 py-2 text-right" : "px-2 py-2"}
                >
                  <button
                    type="button"
                    onClick={() => handleSort(column.key)}
                    title={column.tooltip}
                    aria-label={`${column.label}. ${column.tooltip}`}
                    className="inline-flex items-center gap-1 hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    {column.label}
                    <span aria-hidden>{sortKey === column.key ? (direction === "asc" ? "▲" : "▼") : ""}</span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.length === 0 ? (
              <tr>
                <td colSpan={COLUMNS.length + 2} className="px-2 py-6 text-center text-text-muted">
                  No runs match the current filters.
                </td>
              </tr>
            ) : (
              visible.map((run, index) => {
                const isExpanded = expanded.has(run.run_id);
                const isPinned = pinnedIds.includes(run.run_id);
                const trainOnly = run.eval == null;
                const curve = curves[run.run_id];
                return (
                  <Fragment key={run.run_id}>
                    <tr
                      data-testid="leaderboard-row"
                      data-run-id={run.run_id}
                      data-train-only={trainOnly ? "true" : "false"}
                      className="border-b border-border/60 hover:bg-bg-elevated/60"
                    >
                      <td className="px-2 py-2 align-top">
                        <button
                          type="button"
                          onClick={() => toggleExpand(run.run_id)}
                          aria-expanded={isExpanded}
                          aria-label={`${isExpanded ? "Collapse" : "Expand"} ${run.run_id}`}
                          className="rounded-control px-1.5 py-0.5 text-xs text-text-muted hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                        >
                          {isExpanded ? "▾" : "▸"}
                        </button>
                      </td>
                      <td className="px-2 py-2 align-top">
                        <input
                          type="checkbox"
                          checked={isPinned}
                          disabled={!isPinned && atLimit}
                          onChange={() => onTogglePin?.(run.run_id)}
                          aria-label={`${isPinned ? "Unpin" : "Pin"} ${run.run_id} for comparison`}
                          title={
                            !isPinned && atLimit
                              ? `Comparison is limited to ${maxPinned} runs`
                              : isPinned
                                ? "Remove from comparison"
                                : "Add to comparison"
                          }
                          className="size-4 accent-[var(--accent-2)]"
                        />
                      </td>
                      <td className="tabular px-2 py-2 align-top text-right text-text-muted">
                        {rankByRunId.get(run.run_id) ?? index + 1}
                      </td>
                      <td className="px-2 py-2 align-top">
                        <span className="block max-w-64 truncate font-medium text-text-primary">
                          {run.run_id}
                        </span>
                        {trainOnly && (
                          <span className="mt-0.5 inline-block rounded-control border border-sideways/50 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-sideways">
                            train-only
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-2 align-top text-text-muted">
                        {run.experiment} / seed {run.seed}
                      </td>
                      <td className="px-2 py-2 align-top">
                        <span className="text-text-primary">{run.state}</span>
                      </td>
                      <td className="tabular px-2 py-2 align-top text-right">
                        {formatNumber(run.best_reward)}
                      </td>
                      <td className="tabular px-2 py-2 align-top text-right">
                        {formatNumber(run.eval?.sharpe)}
                      </td>
                      <td className="tabular px-2 py-2 align-top text-right">
                        {formatNumber(run.eval?.total_pnl, 2)}
                      </td>
                      <td className="tabular px-2 py-2 align-top text-right">
                        {formatNumber(run.eval?.max_drawdown, 2)}
                      </td>
                      <td className="tabular px-2 py-2 align-top text-right">
                        {formatNumber(run.eval?.worst_regime_pnl, 2)}
                      </td>
                      <td className="px-2 py-2 align-top font-mono text-xs text-text-muted">
                        {shortHash(run.config_hash)}
                      </td>
                      <td className="px-2 py-2 align-top font-mono text-xs text-text-muted">
                        {shortHash(run.git_commit)}
                      </td>
                      <td className="tabular whitespace-nowrap px-2 py-2 align-top text-xs text-text-muted">
                        <time dateTime={run.updated_at}>{formatTimestamp(run.updated_at)}</time>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr data-testid="leaderboard-detail" className="border-b border-border bg-bg-base/60">
                        <td colSpan={COLUMNS.length + 2} className="px-4 py-3">
                          <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]">
                            <div className="rounded-control border border-border bg-bg-surface p-3">
                              <p className="mb-2 text-xs uppercase tracking-wide text-text-muted">
                                Learning curve
                              </p>
                              {curve && curve.length > 0 ? (
                                <Sparkline
                                  values={curve}
                                  width={220}
                                  height={48}
                                  label={`Learning curve for ${run.run_id}`}
                                />
                              ) : (
                                <p className="text-xs text-text-muted">No curve loaded yet.</p>
                              )}
                            </div>
                            <ProvenanceCard run={run} />
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
