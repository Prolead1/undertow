/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import { useMemo } from "react";

import type { FigureSpec } from "../../api/client";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";
import { withCursor, type RolloutRow } from "./rows";
import { DARK_LAYOUT, TOKEN } from "./theme";

export interface CostBreakdown {
  fees: number;
  gas: number;
  slippage: number;
  il: number;
  riskPenalty: number;
  net: number;
}

/**
 * End-of-episode cost decomposition. `net` follows CONTRACTS §0:
 * `fees + gas + slippage + il + risk_penalty`. `fees` is positive income; every
 * other component is a signed cost (`<= 0` for gas/slippage/risk).
 */
export function costBreakdown(rows: readonly RolloutRow[]): CostBreakdown {
  let fees = 0;
  let gas = 0;
  let slippage = 0;
  let il = 0;
  let riskPenalty = 0;
  for (const row of rows) {
    fees += row.fees;
    gas += row.gas;
    slippage += row.slippage;
    il += row.il;
    riskPenalty += row.riskPenalty;
  }
  return { fees, gas, slippage, il, riskPenalty, net: fees + gas + slippage + il + riskPenalty };
}

function cumulative(rows: readonly RolloutRow[], select: (row: RolloutRow) => number): number[] {
  let running = 0;
  return rows.map((row) => {
    running += select(row);
    return running;
  });
}

/** Waterfall of the final decomposition (relative steps then the net total). */
export function buildWaterfallSpec(rows: readonly RolloutRow[]): FigureSpec {
  const totals = costBreakdown(rows);
  const trace = {
    type: "waterfall",
    orientation: "v",
    measure: ["relative", "relative", "relative", "relative", "relative", "total"],
    x: ["fees", "gas", "slippage", "IL", "risk", "net"],
    y: [totals.fees, totals.gas, totals.slippage, totals.il, totals.riskPenalty, totals.net],
    connector: { line: { color: TOKEN.textPrimary, width: 1, dash: "dot" } },
    increasing: { marker: { color: TOKEN.bull } },
    decreasing: { marker: { color: TOKEN.bear } },
    totals: { marker: { color: TOKEN.rebalance } },
    hovertemplate: "%{x}: %{y:.2f} USDC<extra></extra>",
  };
  return {
    data: [trace as unknown as Record<string, unknown>],
    layout: {
      ...DARK_LAYOUT,
      xaxis: { gridcolor: TOKEN.grid, zeroline: false },
      yaxis: { title: "USDC", gridcolor: TOKEN.grid, zeroline: true, zerolinecolor: TOKEN.grid },
      margin: { l: 56, r: 24, t: 16, b: 40 },
      showlegend: false,
    },
    config: { responsive: true, displayModeBar: false },
  };
}

/** Cumulative per-step components as a stacked area. */
export function buildStackedAreaSpec(rows: readonly RolloutRow[]): FigureSpec {
  const steps = rows.map((row) => row.step);
  const components: Array<[string, (row: RolloutRow) => number, string]> = [
    ["fees", (row) => row.fees, TOKEN.bull],
    ["gas", (row) => row.gas, TOKEN.bear],
    ["slippage", (row) => row.slippage, TOKEN.rebalance],
    ["IL", (row) => row.il, TOKEN.accent2],
    ["risk", (row) => row.riskPenalty, TOKEN.highVol],
  ];
  const data = components.map(([name, select, color]) => ({
    type: "scatter",
    x: steps,
    y: cumulative(rows, select),
    mode: "lines",
    stackgroup: "cost",
    name,
    line: { color, width: 1.5 },
  }));
  return {
    data,
    layout: {
      ...DARK_LAYOUT,
      hovermode: "x unified",
      xaxis: { title: "step", gridcolor: TOKEN.grid, zeroline: false },
      yaxis: { title: "cumulative USDC", gridcolor: TOKEN.grid, zeroline: true, zerolinecolor: TOKEN.grid },
      margin: { l: 56, r: 24, t: 16, b: 40 },
      showlegend: true,
      legend: { orientation: "h", y: 1.12 },
    },
    config: { responsive: true, displayModeBar: false },
  };
}

export interface CostWaterfallProps {
  rows: readonly RolloutRow[];
  cursorIndex: number;
}

/** End-of-episode cost decomposition plus a per-step stacked area. */
export function CostWaterfall({ rows, cursorIndex }: CostWaterfallProps) {
  const cursorStep = rows[cursorIndex]?.step ?? null;
  const totals = useMemo(() => costBreakdown(rows), [rows]);
  const waterfall = useMemo(
    () => withCursor(buildWaterfallSpec(rows), cursorStep),
    [rows, cursorStep],
  );
  const stacked = useMemo(
    () => withCursor(buildStackedAreaSpec(rows), cursorStep),
    [rows, cursorStep],
  );

  return (
    <Panel
      title="Cost waterfall"
      subtitle={`fees ${totals.fees.toFixed(2)} · gas ${totals.gas.toFixed(2)} · slippage ${totals.slippage.toFixed(
        2,
      )} · IL ${totals.il.toFixed(2)} · risk ${totals.riskPenalty.toFixed(2)} · net ${totals.net.toFixed(2)}`}
    >
      <div className="grid gap-2">
        <PlotlyFigure
          spec={waterfall}
          className="h-44"
          ariaLabel="Waterfall of end-of-episode fees, gas, slippage, IL and risk summing to net."
        />
        <PlotlyFigure
          spec={stacked}
          className="h-44"
          ariaLabel="Stacked area of cumulative per-step cost components."
        />
      </div>
    </Panel>
  );
}