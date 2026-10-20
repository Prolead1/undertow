/**
 * Leaderboard ranking/sorting (pure, no React).
 *
 * Mirrors `undertow.ui.registry.rank` (PLAN §7 / ADR-025): evaluated runs first,
 * ordered by eval Sharpe ↓, then worst-regime PnL ↓, then `best_reward` ↓,
 * finally by code-point `run_id`.
 */

import type { RunRecord } from "../../api/client";

export type LeaderboardSortKey =
  | "rank"
  | "run_id"
  | "experiment"
  | "state"
  | "best_reward"
  | "sharpe"
  | "total_pnl"
  | "max_drawdown"
  | "worst_regime_pnl"
  | "config_hash"
  | "git_commit"
  | "updated_at";

export type SortDirection = "asc" | "desc";

/** `Math.inf` sentinel order: finite values compare, missing sorts last. */
function finite(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Descending comparison with missing/NaN/inf last (ascending key helper). */
function descending(a: number | null | undefined, b: number | null | undefined): number {
  const fa = finite(a);
  const fb = finite(b);
  if (fa === null && fb === null) return 0;
  if (fa === null) return 1;
  if (fb === null) return -1;
  if (fa === fb) return 0;
  return fb - fa;
}

/** The pinned leaderboard comparator (evaluated group first, best-first). */
export function rankCompare(a: RunRecord, b: RunRecord): number {
  const evaluatedA = a.eval != null;
  const evaluatedB = b.eval != null;
  if (evaluatedA !== evaluatedB) return evaluatedA ? -1 : 1;

  let cmp = 0;
  if (a.eval && b.eval) {
    cmp = descending(a.eval.sharpe, b.eval.sharpe);
    if (cmp === 0) cmp = descending(a.eval.worst_regime_pnl, b.eval.worst_regime_pnl);
  }
  if (cmp === 0) cmp = descending(a.best_reward, b.best_reward);
  if (cmp === 0) {
    // Plain code-point ordering to match Python's `rank` (localeCompare varies
    // with the runtime locale).
    if (a.run_id < b.run_id) cmp = -1;
    else if (a.run_id > b.run_id) cmp = 1;
  }
  return cmp;
}

/** A ranked copy of `runs` following the pinned rule. */
export function rankRuns(runs: readonly RunRecord[]): RunRecord[] {
  return [...runs].sort(rankCompare);
}

type EvalMetricKey = "sharpe" | "total_pnl" | "annualized_return" | "max_drawdown" | "worst_regime_pnl";

function evalMetric(run: RunRecord, key: EvalMetricKey): number | null {
  return finite(run.eval?.[key] ?? null);
}

function sortValue(run: RunRecord, key: LeaderboardSortKey): number | string | null {
  switch (key) {
    case "run_id":
      return run.run_id;
    case "experiment":
      return `${run.experiment} / seed ${String(run.seed).padStart(3, "0")}`;
    case "state":
      return run.state;
    case "best_reward":
      return finite(run.best_reward);
    case "config_hash":
      return run.config_hash ?? null;
    case "git_commit":
      return run.git_commit ?? null;
    case "updated_at":
      return run.updated_at;
    case "sharpe":
      return evalMetric(run, "sharpe");
    case "total_pnl":
      return evalMetric(run, "total_pnl");
    case "max_drawdown":
      return evalMetric(run, "max_drawdown");
    case "worst_regime_pnl":
      return evalMetric(run, "worst_regime_pnl");
    case "rank":
      return null;
  }
}

/**
 * Sort `runs` by `key`. The default `rank` sort (ascending) follows the pinned
 * rule; every other column toggles asc/desc while missing values always sort
 * last and `rankCompare` breaks ties.
 */
export function sortRuns(
  runs: readonly RunRecord[],
  key: LeaderboardSortKey,
  direction: SortDirection,
): RunRecord[] {
  if (key === "rank") {
    const ranked = rankRuns(runs);
    return direction === "desc" ? ranked.reverse() : ranked;
  }
  return [...runs].sort((a, b) => {
    const av = sortValue(a, key);
    const bv = sortValue(b, key);
    const aMissing = av === null || av === "";
    const bMissing = bv === null || bv === "";
    if (aMissing && bMissing) return rankCompare(a, b);
    if (aMissing) return 1;
    if (bMissing) return -1;
    let cmp: number;
    if (typeof av === "number" && typeof bv === "number") {
      cmp = av - bv;
    } else {
      cmp = String(av).localeCompare(String(bv));
    }
    if (direction === "desc") cmp = -cmp;
    return cmp === 0 ? rankCompare(a, b) : cmp;
  });
}
