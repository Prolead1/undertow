/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import { useEffect, useMemo, useState } from "react";

import { getFigure, type FigureSpec } from "../../api/client";
import { EmptyState } from "../EmptyState";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";
import { withCursor, type RolloutRow } from "./rows";
import { DARK_LAYOUT, TOKEN, withAlpha } from "./theme";

// §9 regime fills, derived from the shared tokens so the band colours cannot
// drift from the solid regime strokes used elsewhere in the replay view.
const REGIME_FILL: Record<string, string> = {
  bull: withAlpha(TOKEN.bull, 0.1),
  bear: withAlpha(TOKEN.bear, 0.1),
  sideways: withAlpha(TOKEN.sideways, 0.08),
  high_vol: withAlpha(TOKEN.highVol, 0.1),
};

interface Segment {
  start: number;
  end: number;
  value: boolean;
}

/** Contiguous `[start, end)` runs of equal boolean values. */
export function inRangeSegments(values: readonly boolean[]): Segment[] {
  if (values.length === 0) return [];
  const segments: Segment[] = [];
  let start = 0;
  for (let index = 1; index <= values.length; index += 1) {
    if (index === values.length || values[index] !== values[start]) {
      segments.push({ start, end: index, value: values[start] === true });
      start = index;
    }
  }
  return segments;
}

/**
 * Locally-built ribbon spec, used when the backend `ribbon` figure is
 * unavailable (offline / older server). The preferred path is the server-side
 * figure (`GET /api/runs/{id}/figures/ribbon?seed=&episode=`), which keeps the
 * Plotly styling authoritative in `undertow.ui.figures.ribbon`.
 */
export function buildRibbonSpec(rows: readonly RolloutRow[]): FigureSpec {
  const steps = rows.map((row) => row.step);
  const prices = rows.map((row) => row.price);
  const first = rows[0];
  const baseEquity = first?.equity ?? null;
  const basePrice = first?.price ?? 0;
  const hodl =
    baseEquity !== null && basePrice > 0
      ? rows.map((row) => (baseEquity * row.price) / basePrice)
      : null;

  const data: Array<Record<string, unknown>> = [
    {
      type: "scatter",
      x: steps,
      y: prices,
      mode: "lines",
      name: "price",
      line: { color: TOKEN.accent, width: 2 },
    },
  ];
  if (hodl) {
    data.push({
      type: "scatter",
      x: steps,
      y: hodl,
      mode: "lines",
      name: "HODL",
      line: { color: TOKEN.textMuted, width: 1.5, dash: "dot" },
    });
  }

  const segments = inRangeSegments(rows.map((row) => row.inRange));
  let bandLegendShown = false;
  for (const segment of segments) {
    const slice = rows.slice(segment.start, segment.end);
    const lower = slice.map((row) => row.tickLower);
    const upper = slice.map((row) => row.tickUpper);
    if (upper.every((value) => value === null) && lower.every((value) => value === null)) continue;
    const color = segment.value ? TOKEN.bull : TOKEN.bear;
    data.push({
      type: "scatter",
      x: slice.map((row) => row.step),
      y: lower,
      mode: "lines",
      name: "tick_lower",
      yaxis: "y2",
      line: { width: 0, color: "rgba(0,0,0,0)" },
      hoverinfo: "skip",
      showlegend: false,
    });
    data.push({
      type: "scatter",
      x: slice.map((row) => row.step),
      y: upper,
      mode: "lines",
      name: "tick_upper",
      yaxis: "y2",
      fill: "tonexty",
      fillcolor: segment.value
        ? withAlpha(TOKEN.bull, 0.2)
        : withAlpha(TOKEN.bear, 0.2),
      line: { width: 1, color },
      showlegend: !bandLegendShown,
    });
    bandLegendShown = true;
  }

  const rebalances = rows.filter((row) => row.actionType === "rebalance");
  if (rebalances.length > 0) {
    data.push({
      type: "scatter",
      x: rebalances.map((row) => row.step),
      y: rebalances.map((row) => row.price),
      mode: "markers",
      name: "rebalance",
      marker: {
        symbol: "triangle-down",
        size: rebalances.map((row) => 7 + 6 * Math.sqrt(Math.abs(row.gas))),
        color: TOKEN.rebalance,
        line: { color: TOKEN.bgBase, width: 1 },
      },
    });
  }

  const shapes = regimeShapes(steps, rows.map((row) => row.regime));
  return {
    data,
    layout: {
      ...DARK_LAYOUT,
      hovermode: "x unified",
      xaxis: { title: "step", gridcolor: TOKEN.grid, zeroline: false },
      yaxis: {
        title: "price (USDC/WETH)",
        type: "log",
        gridcolor: TOKEN.grid,
        zeroline: false,
      },
      yaxis2: {
        title: "tick",
        overlaying: "y",
        side: "right",
        showgrid: false,
        tickfont: { color: TOKEN.textMuted },
      },
      shapes,
      margin: { l: 64, r: 64, t: 24, b: 44 },
      showlegend: true,
      legend: { orientation: "h", y: 1.08 },
    },
    config: { responsive: true, displayModeBar: false },
  };
}

/** One translucent paper-height band per contiguous regime run. */
function regimeShapes(steps: readonly number[], regimes: readonly string[]): Array<Record<string, unknown>> {
  const shapes: Array<Record<string, unknown>> = [];
  if (steps.length === 0) return shapes;
  const runs: Array<{ start: number; end: number; regime: string }> = [];
  let start = 0;
  for (let index = 1; index <= regimes.length; index += 1) {
    if (index === regimes.length || regimes[index] !== regimes[start]) {
      runs.push({ start, end: index, regime: regimes[start] ?? "unknown" });
      start = index;
    }
  }
  const fallbackWidth = steps.length > 1 ? steps[1]! - steps[0]! : 1;
  for (const run of runs) {
    if (run.regime === "unknown") continue;
    const x0 = steps[run.start] ?? 0;
    const x1 = run.end < steps.length ? steps[run.end]! : (steps[steps.length - 1] ?? x0) + fallbackWidth;
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0,
      x1,
      y0: 0,
      y1: 1,
      fillcolor: REGIME_FILL[run.regime] ?? withAlpha(TOKEN.textMuted, 0.08),
      line: { width: 0 },
      layer: "below",
    });
  }
  return shapes;
}

export interface RibbonChartProps {
  runId: string;
  seed: number;
  episode: number;
  rows: readonly RolloutRow[];
  cursorIndex: number;
}

/**
 * The hero chart: price on a log axis with the active `[tick_lower, tick_upper]`
 * band, rebalance markers sized by gas, HODL counterfactual and regime bands.
 *
 * Prefers the backend `ribbon` figure (which owns the styling and the HODL
 * series); falls back to a locally-built spec from the rollout rows.
 */
export function RibbonChart({ runId, seed, episode, rows, cursorIndex }: RibbonChartProps) {
  const [backendSpec, setBackendSpec] = useState<FigureSpec | null>(null);
  const [useFallback, setUseFallback] = useState(false);

  useEffect(() => {
    let active = true;
    setBackendSpec(null);
    setUseFallback(false);
    getFigure(runId, "ribbon", { seed, episode })
      .then((spec) => {
        if (!active) return;
        if (spec && Array.isArray(spec.data) && spec.data.length > 0) {
          setBackendSpec(spec);
        } else {
          setUseFallback(true);
        }
      })
      .catch(() => {
        if (active) setUseFallback(true);
      });
    return () => {
      active = false;
    };
  }, [runId, seed, episode]);

  const cursorStep = rows[cursorIndex]?.step ?? null;
  const spec = useMemo(() => {
    if (backendSpec) return withCursor(backendSpec, cursorStep);
    if (useFallback) return withCursor(buildRibbonSpec(rows), cursorStep);
    return null;
  }, [backendSpec, useFallback, rows, cursorStep]);

  return (
    <Panel
      title="Position ribbon"
      subtitle="Price (log) over the active tick range; markers are rebalances"
    >
      {spec === null ? (
        <EmptyState title="Loading ribbon…" description="Fetching the position range figure." />
      ) : (
        <PlotlyFigure
          spec={spec}
          className="h-96"
          ariaLabel="Price on a log axis with the position tick range band, in/out-of-range colouring, gas-sized rebalance markers and regime bands."
        />
      )}
    </Panel>
  );
}