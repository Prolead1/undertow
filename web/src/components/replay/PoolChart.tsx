/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import { useMemo } from "react";

import type { FigureSpec } from "../../api/client";
import { EmptyState } from "../EmptyState";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";
import { hasPoolState, poolFeeUsdc, withCursor, type RolloutRow } from "./rows";
import { DARK_LAYOUT, TOKEN } from "./theme";

/**
 * Pure spec builder, exported so trace names/values are directly testable.
 *
 * The pool's raw `L` and the agent's raw position `L` differ by one to two
 * orders of magnitude, so both liquidity series share a **log** primary axis;
 * the per-step pool fees are USDC-valued on a secondary linear axis.
 */
export function buildPoolSpec(rows: readonly RolloutRow[]): FigureSpec {
  const steps = rows.map((row) => row.step);
  const data: Array<Record<string, unknown>> = [
    {
      type: "scatter",
      x: steps,
      y: rows.map((row) => row.poolActiveLiquidity),
      mode: "lines",
      name: "pool active liquidity",
      line: { color: TOKEN.sideways, width: 1.5 },
      hovertemplate: "%{y:.4s} L<extra>pool active liquidity</extra>",
    },
    {
      type: "scatter",
      x: steps,
      y: rows.map((row) => row.positionLiquidity),
      mode: "lines",
      name: "agent position liquidity",
      line: { color: TOKEN.accent, width: 2 },
      hovertemplate: "%{y:.4s} L<extra>agent position liquidity</extra>",
    },
    {
      type: "bar",
      x: steps,
      y: rows.map(poolFeeUsdc),
      name: "pool fees (USDC)",
      marker: { color: TOKEN.bull },
      opacity: 0.55,
      yaxis: "y2",
      hovertemplate: "%{y:.4f} USDC<extra>pool fees</extra>",
    },
  ];

  return {
    data,
    layout: {
      ...DARK_LAYOUT,
      hovermode: "x unified",
      xaxis: { title: "step", gridcolor: TOKEN.grid, zeroline: false },
      yaxis: {
        title: "liquidity (raw L)",
        type: "log",
        gridcolor: TOKEN.grid,
        zeroline: false,
      },
      yaxis2: {
        title: "pool fees (USDC)",
        overlaying: "y",
        side: "right",
        showgrid: false,
        zeroline: false,
        tickfont: { color: TOKEN.textMuted },
      },
      margin: { l: 72, r: 72, t: 24, b: 44 },
      showlegend: true,
      legend: { orientation: "h", y: 1.08 },
    },
    config: { responsive: true, displayModeBar: false },
  };
}

export interface PoolChartProps {
  rows: readonly RolloutRow[];
  cursorIndex: number;
}

/**
 * Pool over time: the external pool's raw active liquidity against the agent's
 * own position liquidity (log axis), with the USDC value of the fees the pool
 * generated each step on a secondary axis. Shares the replay time cursor.
 *
 * Pre-fix rollouts have no pool columns (every field parses to `0`); showing
 * them as a flat line would be a lie, so the panel renders an explainable empty
 * state instead.
 */
export function PoolChart({ rows, cursorIndex }: PoolChartProps) {
  const cursorStep = rows[cursorIndex]?.step ?? null;
  const cursorSpec = useMemo(() => withCursor(buildPoolSpec(rows), cursorStep), [rows, cursorStep]);

  return (
    <Panel
      title="Pool over time"
      subtitle="Pool vs agent liquidity (log) with the pool fees generated each step"
    >
      {hasPoolState(rows) ? (
        <PlotlyFigure
          spec={cursorSpec}
          className="h-80"
          ariaLabel="Pool active liquidity and the agent position liquidity on a shared log axis, with the USDC value of pool fees per step on a secondary axis."
        />
      ) : (
        <EmptyState
          title="Pool state not recorded for this run"
          description="This rollout predates the pool-state columns. Re-record it with the current training code to chart the pool and the agent's share of its fees."
        />
      )}
    </Panel>
  );
}
