import { useEffect, useMemo, useRef } from "react";
import type { Config, Data, Layout } from "plotly.js";
import Plotly from "plotly.js-dist-min";

import type { FigureSpec } from "../api/client";
import { EmptyState } from "./EmptyState";
import { cn } from "./cn";

export interface PlotlyFigureProps {
  spec: FigureSpec;
  ariaLabel?: string;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * The only Plotly entry point in the app (CONTRACTS §8/PLAN §7; ADR-028).
 *
 * Plotly is driven imperatively with `Plotly.react` instead of pulling in
 * `react-plotly.js`. The data/layout/config objects are memoised by their JSON
 * signature so a parent re-render does not re-plot, and `Plotly.react`
 * preserves view state (zoom/pan) via the layout's `uirevision`.
 */
export function PlotlyFigure({ spec, ariaLabel, className, style }: PlotlyFigureProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  // Signatures (not the objects) are the memo dependencies so a fresh object
  // with the same content does not trigger a re-plot.
  const dataSignature = JSON.stringify(spec.data);
  const layoutSignature = JSON.stringify(spec.layout);
  const configSignature = JSON.stringify(spec.config ?? {});

  const data = useMemo(() => spec.data as unknown as Data[], [dataSignature]);
  const layout = useMemo(() => spec.layout as Partial<Layout>, [layoutSignature]);
  const config = useMemo(
    () => ({ responsive: true, displayModeBar: false, ...spec.config }) as Partial<Config>,
    [configSignature],
  );

  const hasData = data.length > 0;

  useEffect(() => {
    const node = chartRef.current;
    if (node !== null) void Plotly.react(node, data, layout, config);
  }, [data, layout, config]);

  // Purge when the chart node goes away (data becomes empty) or on unmount.
  // The node is captured here because React detaches refs *before* running
  // effect cleanups, so reading `chartRef.current` in the cleanup would be null.
  useEffect(() => {
    const node = chartRef.current;
    if (node === null) return;
    return () => {
      Plotly.purge(node);
    };
  }, [hasData]);

  if (!hasData) {
    return (
      <EmptyState
        className={className}
        title="No figure data"
        description={ariaLabel ?? "This figure has no traces yet."}
      />
    );
  }

  return (
    <div className={cn("min-h-48 w-full", className)} style={style}>
      {ariaLabel && <p className="sr-only">{ariaLabel}</p>}
      <div ref={chartRef} className="h-full w-full" />
    </div>
  );
}
