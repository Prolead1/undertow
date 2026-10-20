import { useEffect, useMemo, useRef, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

import type { FigureSpec } from "../api/client";
import { EmptyState } from "./EmptyState";
import { cn } from "./cn";

const Plot = createPlotlyComponent(Plotly);

export interface PlotlyFigureProps {
  spec: FigureSpec;
  ariaLabel?: string;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * The only Plotly entry point in the app (CONTRACTS §8/PLAN §7).
 *
 * `revision` is bumped only when the *data* payload changes, so view-state
 * changes (zoom, selection, a new layout hint) do not reset Plotly's internal
 * chart state.
 */
export function PlotlyFigure({ spec, ariaLabel, className, style }: PlotlyFigureProps) {
  const data = useMemo(() => spec.data as unknown as Data[], [spec.data]);
  const layout = useMemo(() => spec.layout as Partial<Layout>, [spec.layout]);
  const config = useMemo(
    () => ({ responsive: true, displayModeBar: false, ...spec.config }) as Partial<Config>,
    [spec.config],
  );

  const dataSignature = useMemo(() => JSON.stringify(spec.data), [spec.data]);
  const previousSignature = useRef(dataSignature);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    if (previousSignature.current === dataSignature) return;
    previousSignature.current = dataSignature;
    setRevision((value) => value + 1);
  }, [dataSignature]);

  if (data.length === 0) {
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
      <Plot
        data={data}
        layout={layout}
        config={config}
        revision={revision}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        className="h-full w-full"
      />
    </div>
  );
}
