/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import { useMemo } from "react";

import type { FigureSpec } from "../../api/client";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";
import { withCursor, type RolloutRow } from "./rows";
import { DARK_LAYOUT, TOKEN, withAlpha } from "./theme";

/**
 * HODL counterfactual: hold the initial capital in the risky asset.
 *
 * `wealth(t) = initialCapital * price(t) / price(0)`. When `initialCapital` is
 * omitted the episode's first equity is used (the agent's starting wealth).
 */
export function hodlBaseline(rows: readonly RolloutRow[], initialCapital?: number): number[] {
  if (rows.length === 0) return [];
  const first = rows[0]!;
  const base = initialCapital ?? first.equity ?? 0;
  const basePrice = first.price;
  if (!Number.isFinite(basePrice) || basePrice <= 0) {
    return rows.map(() => base);
  }
  return rows.map((row) => (base * row.price) / basePrice);
}

/** Running-peak drawdown (`<= 0`) for an equity series. */
export function drawdownSeries(values: readonly number[]): number[] {
  let peak = Number.NEGATIVE_INFINITY;
  return values.map((value) => {
    peak = Math.max(peak, value);
    return value - peak;
  });
}

/** Pure spec builder, exported so the baseline/drawdown math is directly testable. */
export function buildEquitySpec(rows: readonly RolloutRow[]): FigureSpec {
  const steps = rows.map((row) => row.step);
  const equity = rows.map((row) => row.equity ?? 0);
  const hodl = hodlBaseline(rows);
  const drawdown = drawdownSeries(equity);

  const data: Array<Record<string, unknown>> = [
    {
      type: "scatter",
      x: steps,
      y: equity,
      mode: "lines",
      name: "equity",
      line: { color: TOKEN.accent, width: 2 },
    },
    {
      type: "scatter",
      x: steps,
      y: hodl,
      mode: "lines",
      name: "HODL",
      line: { color: TOKEN.textMuted, width: 1.5, dash: "dot" },
    },
    {
      type: "scatter",
      x: steps,
      y: drawdown,
      mode: "lines",
      name: "drawdown",
      fill: "tozeroy",
      fillcolor: withAlpha(TOKEN.bear, 0.22),
      line: { color: TOKEN.bear, width: 1 },
      yaxis: "y2",
    },
  ];

  return {
    data,
    layout: {
      ...DARK_LAYOUT,
      hovermode: "x unified",
      xaxis: { title: "step", gridcolor: TOKEN.grid, zeroline: false },
      yaxis: { title: "equity (USDC)", gridcolor: TOKEN.grid, zeroline: false },
      yaxis2: {
        title: "drawdown",
        overlaying: "y",
        side: "right",
        showgrid: false,
        tickfont: { color: TOKEN.textMuted },
      },
      margin: { l: 64, r: 64, t: 24, b: 44 },
      showlegend: true,
      legend: { orientation: "h", y: 1.08 },
    },
    config: { responsive: true, displayModeBar: false },
  };
}

export interface EquityVsHodlProps {
  rows: readonly RolloutRow[];
  cursorIndex: number;
}

/** Equity against a client-computed HODL baseline, with drawdown fill. */
export function EquityVsHodl({ rows, cursorIndex }: EquityVsHodlProps) {
  const cursorStep = rows[cursorIndex]?.step ?? null;
  const spec = useMemo(() => withCursor(buildEquitySpec(rows), cursorStep), [rows, cursorStep]);

  return (
    <Panel title="Equity vs HODL" subtitle="Agent equity against holding the initial capital">
      <PlotlyFigure
        spec={spec}
        className="h-72"
        ariaLabel="Equity curve against a buy-and-hold baseline, with drawdown fill below."
      />
    </Panel>
  );
}