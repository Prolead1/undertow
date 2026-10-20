/**
 * Client-side CSV serialisation for the leaderboard export (pure, no React).
 * The server never sees an export.
 */

import type { RunRecord } from "../../api/client";
import { rankRuns } from "./ranking";

export const CSV_HEADER = [
  "rank",
  "run_id",
  "experiment",
  "seed",
  "state",
  "best_reward",
  "sharpe",
  "total_pnl",
  "max_drawdown",
  "worst_regime_pnl",
  "config_hash",
  "git_commit",
  "updated_at",
] as const;

function csvCell(value: string | number | null | undefined): string {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/**
 * Serialise a set of runs to CSV text.
 *
 * The `rank` column is the run's position under the pinned ranking rule
 * (`rankRuns`), *not* its row index in `runs`. The visible table can be sorted
 * by any column; the exported rank must still answer "which policy is best?".
 * Ranks are computed within the passed set, so when the leaderboard is filtered
 * they renumber within the visible subset, matching the on-screen Rank column.
 */
export function runsToCsv(runs: readonly RunRecord[]): string {
  const rankByRunId = new Map<string, number>();
  rankRuns(runs).forEach((run, index) => rankByRunId.set(run.run_id, index + 1));
  const rows = runs.map((run) => [
    rankByRunId.get(run.run_id) ?? "",
    run.run_id,
    run.experiment,
    run.seed,
    run.state,
    run.best_reward ?? "",
    run.eval?.sharpe ?? "",
    run.eval?.total_pnl ?? "",
    run.eval?.max_drawdown ?? "",
    run.eval?.worst_regime_pnl ?? "",
    run.config_hash ?? "",
    run.git_commit ?? "",
    run.updated_at,
  ]);
  return [CSV_HEADER, ...rows].map((row) => row.map(csvCell).join(",")).join("\n") + "\n";
}
