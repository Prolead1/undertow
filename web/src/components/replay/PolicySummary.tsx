/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import { useMemo } from "react";

import { Panel } from "../Panel";
import { Stat } from "../Stat";
import { hasPoolState, poolFeeUsdc, type RolloutRow } from "./rows";

/**
 * Episode-level answer to "what did this policy capture?".
 *
 * All money fields are USDC. `agentFeeSharePct` is `null` when the rollout has
 * no recorded pool fee state or the pool generated nothing, so callers render
 * "—" instead of dividing by zero.
 */
export interface PolicySummaryData {
  steps: number;
  rebalances: number;
  /** Share of steps with `in_range`, 0–100. */
  inRangePct: number;
  /** Sum of the agent's step fee income (`fees`), USDC. */
  agentFees: number;
  /** Sum of `poolFeeUsdc` across the episode, USDC. */
  poolFees: number;
  /** `agentFees / poolFees` as a percentage, or `null` when undefined. */
  agentFeeSharePct: number | null;
  /** False for pre-fix rollouts, which have no pool-state columns. */
  poolStateRecorded: boolean;
  finalEquity: number | null;
  /** `finalEquity - firstEquity`, or `null` when either leg is missing. */
  pnl: number | null;
  il: number;
  gas: number;
  slippage: number;
}

/** Pure end-of-episode arithmetic, exported so the totals are directly testable. */
export function policySummary(rows: readonly RolloutRow[]): PolicySummaryData {
  let rebalances = 0;
  let inRange = 0;
  let agentFees = 0;
  let poolFees = 0;
  let il = 0;
  let gas = 0;
  let slippage = 0;
  for (const row of rows) {
    if (row.actionType === "rebalance") rebalances += 1;
    if (row.inRange) inRange += 1;
    agentFees += row.fees;
    poolFees += poolFeeUsdc(row);
    il += row.il;
    gas += row.gas;
    slippage += row.slippage;
  }

  const firstEquity = rows[0]?.equity ?? null;
  const finalEquity = rows[rows.length - 1]?.equity ?? null;
  return {
    steps: rows.length,
    rebalances,
    inRangePct: rows.length > 0 ? (inRange / rows.length) * 100 : 0,
    agentFees,
    poolFees,
    // Guard the divide: a pre-fix rollout values to 0 pool fees, and an empty
    // pool would make the agent's share undefined rather than 0%.
    agentFeeSharePct: poolFees > 0 ? (agentFees / poolFees) * 100 : null,
    poolStateRecorded: hasPoolState(rows),
    finalEquity,
    pnl: firstEquity !== null && finalEquity !== null ? finalEquity - firstEquity : null,
    il,
    gas,
    slippage,
  };
}

function formatUsdc(value: number | null): string {
  return value === null ? "—" : value.toFixed(2);
}

function formatPct(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)}%`;
}

function formatSignedUsdc(value: number | null): string {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

export interface PolicySummaryProps {
  rows: readonly RolloutRow[];
}

/** Compact episode scorecard: range discipline, fee capture, PnL and costs. */
export function PolicySummary({ rows }: PolicySummaryProps) {
  const summary = useMemo(() => policySummary(rows), [rows]);
  const pnlTone = summary.pnl === null ? "muted" : summary.pnl >= 0 ? "bull" : "bear";

  return (
    <Panel
      title="What this policy captured"
      subtitle="Episode totals: range discipline, fee capture and the costs paid"
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-5">
        <Stat
          label="Rebalances"
          value={summary.rebalances}
          hint={`of ${summary.steps} steps`}
          tone="accent"
        />
        <Stat label="In range" value={formatPct(summary.inRangePct)} hint="of steps" />
        <Stat
          label="Agent fees"
          value={formatUsdc(summary.agentFees)}
          hint="USDC captured"
          tone="bull"
        />
        <Stat
          label="Pool fees"
          value={summary.poolStateRecorded ? formatUsdc(summary.poolFees) : "—"}
          hint={summary.poolStateRecorded ? "USDC generated" : "not recorded"}
        />
        <Stat
          label="Agent share"
          value={formatPct(summary.agentFeeSharePct)}
          hint="of pool fees"
          tone="accent"
        />
        <Stat label="Final equity" value={formatUsdc(summary.finalEquity)} hint="USDC" />
        <Stat
          label="PnL"
          value={formatSignedUsdc(summary.pnl)}
          hint="vs first equity"
          tone={pnlTone}
        />
        <Stat label="IL" value={formatSignedUsdc(summary.il)} hint="USDC" tone="muted" />
        <Stat label="Gas" value={formatSignedUsdc(summary.gas)} hint="USDC" tone="muted" />
        <Stat
          label="Slippage"
          value={formatSignedUsdc(summary.slippage)}
          hint="USDC"
          tone="muted"
        />
      </div>
    </Panel>
  );
}
