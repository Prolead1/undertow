import { useEffect, useMemo, useState } from "react";

import { getFigure, type FigureSpec, type TelemetryPoint } from "../../api/client";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";

const DIAGNOSTIC_METRICS = [
  "policy_loss",
  "value_loss",
  "entropy_loss",
  "approx_kl",
  "clip_fraction",
  "explained_variance",
] as const;

export type DiagnosticMetric = (typeof DIAGNOSTIC_METRICS)[number];

export interface DiagnosticsGridProps {
  runId: string;
  points: readonly TelemetryPoint[];
}

function finite(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

function metricLabel(metric: string): string {
  return metric.replace(/_/g, " ");
}

function pointMetric(point: TelemetryPoint, metric: DiagnosticMetric): number | null {
  const train = point.train as Record<string, unknown> | undefined;
  return finite(train?.[metric]);
}

/**
 * Append live diagnostics to the REST figure snapshot. The x-axis is the
 * timestep, so a repeated frame updates (never duplicates) a sample.
 */
function mergeDiagnosticSpec(
  base: FigureSpec | null,
  points: readonly TelemetryPoint[],
  metric: DiagnosticMetric,
  runId: string,
): FigureSpec | null {
  const samples = new Map<number, number>();
  const baseData = base?.data ?? [];
  const baseTrace = baseData.find(
    (trace) => trace["type"] === "scatter" || Array.isArray(trace["x"]),
  ) as Record<string, unknown> | undefined;

  const baseX = Array.isArray(baseTrace?.["x"]) ? (baseTrace?.["x"] as unknown[]) : [];
  const baseY = Array.isArray(baseTrace?.["y"]) ? (baseTrace?.["y"] as unknown[]) : [];
  for (let index = 0; index < Math.min(baseX.length, baseY.length); index += 1) {
    const x = finite(baseX[index]);
    const y = finite(baseY[index]);
    if (x !== null && y !== null) samples.set(x, y);
  }

  for (const point of points) {
    const x = finite(point.timestep);
    const y = pointMetric(point, metric);
    if (x !== null && y !== null) samples.set(x, y);
  }

  if (samples.size === 0) return null;
  const ordered = [...samples.entries()].sort(([left], [right]) => left - right);
  const label = metricLabel(metric);
  const trace: Record<string, unknown> = {
    type: "scatter",
    mode: "lines",
    name: label,
    ...(baseTrace ?? {}),
    x: ordered.map(([x]) => x),
    y: ordered.map(([, y]) => y),
  };
  const baseLayout = (base?.layout ?? {}) as Record<string, unknown>;
  return {
    data: [trace],
    layout: {
      ...baseLayout,
      title: undefined,
      // Keep a user's zoom while live points extend the series.
      uirevision: `diagnostics:${runId}:${metric}`,
      showlegend: false,
      margin: { l: 56, r: 16, t: 20, b: 36 },
      xaxis: { ...(baseLayout["xaxis"] as object | undefined), title: "timestep" },
      yaxis: { ...(baseLayout["yaxis"] as object | undefined), title: label },
    },
    config: { displayModeBar: false, responsive: true },
  };
}

function metricClass(metric: DiagnosticMetric): string {
  return `diagnostic-${metric}`;
}

/** Small multiples for PPO health metrics, each fed by the diagnostics figure route. */
export function DiagnosticsGrid({ runId, points }: DiagnosticsGridProps) {
  const [specs, setSpecs] = useState<Record<string, FigureSpec | null>>({});

  useEffect(() => {
    let cancelled = false;
    setSpecs({});
    void Promise.all(
      DIAGNOSTIC_METRICS.map(async (metric) => {
        try {
          const spec = await getFigure(runId, "diagnostics", { metric });
          return [metric, spec] as const;
        } catch {
          return [metric, null] as const;
        }
      }),
    ).then((entries) => {
      if (!cancelled) setSpecs(Object.fromEntries(entries));
    });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const specsByMetric = useMemo(
    () =>
      Object.fromEntries(
        DIAGNOSTIC_METRICS.map((metric) => [
          metric,
          mergeDiagnosticSpec(specs[metric] ?? null, points, metric, runId),
        ]),
      ) as Record<DiagnosticMetric, FigureSpec | null>,
    [specs, points, runId],
  );

  return (
    <section aria-label="Training diagnostics" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {DIAGNOSTIC_METRICS.map((metric) => {
        const spec = specsByMetric[metric];
        return (
          <Panel
            key={metric}
            title={<span className="capitalize">{metricLabel(metric)}</span>}
            subtitle="train/*"
            bodyClassName="p-2"
          >
            <div data-testid={metricClass(metric)}>
              {spec ? (
                <PlotlyFigure
                  spec={spec}
                  ariaLabel={`${metricLabel(metric)} over timesteps`}
                  className="h-44"
                />
              ) : (
                <p className="flex min-h-44 items-center justify-center text-xs text-text-muted">
                  not reported
                </p>
              )}
            </div>
          </Panel>
        );
      })}
    </section>
  );
}
