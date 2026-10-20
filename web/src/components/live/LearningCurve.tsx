import { useEffect, useMemo, useState } from "react";

import { getCurve, type LearningCurve as LearningCurvePayload, type TelemetryPoint } from "../../api/client";
import { EmptyState } from "../EmptyState";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";
import { Toolbar } from "../Toolbar";
import { cn } from "../cn";

const ACCENT = "#22D3EE";
const ACCENT_2 = "#A78BFA";
const MUTED = "#8B93A7";
const GRID = "#1A2130";

export interface LearningCurveProps {
  runId: string;
  points: readonly TelemetryPoint[];
}

interface MergedCurve {
  timesteps: number[];
  mean: number[];
  std: number[];
  perSeed: Record<string, Array<number | null>>;
}

function finite(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

/**
 * Merge the REST curve snapshot with live telemetry without duplicating a
 * `(seed, timestep)` point. A repeated telemetry frame is idempotent, so a
 * reconnect/fallback overlap can never add a second dot.
 */
function mergeLiveCurve(
  base: LearningCurvePayload | null,
  points: readonly TelemetryPoint[],
): MergedCurve | null {
  const valuesByTimestep = new Map<number, Map<string, number>>();
  const baseStats = new Map<number, { mean: number; std: number }>();

  if (base) {
    for (let index = 0; index < base.timesteps.length; index += 1) {
      const timestep = finite(base.timesteps[index]);
      const mean = finite(base.mean[index]);
      if (timestep === null || mean === null) continue;
      const std = finite(base.std[index]) ?? 0;
      baseStats.set(timestep, { mean, std });
    }
    for (const [seed, series] of Object.entries(base.per_seed)) {
      for (let index = 0; index < series.length; index += 1) {
        const timestep = finite(base.timesteps[index]);
        const value = finite(series[index]);
        if (timestep === null || value === null) continue;
        const bucket = valuesByTimestep.get(timestep) ?? new Map<string, number>();
        bucket.set(seed, value);
        valuesByTimestep.set(timestep, bucket);
      }
    }
  }

  for (const point of points) {
    const timestep = finite(point.timestep);
    const value = finite(point.ep_rew_mean);
    if (timestep === null || value === null) continue;
    const bucket = valuesByTimestep.get(timestep) ?? new Map<string, number>();
    bucket.set(String(point.seed), value);
    valuesByTimestep.set(timestep, bucket);
  }

  const timesteps = [...new Set([...baseStats.keys(), ...valuesByTimestep.keys()])].sort(
    (a, b) => a - b,
  );
  if (timesteps.length === 0) return null;

  const seeds = new Set<string>();
  for (const bucket of valuesByTimestep.values()) {
    for (const seed of bucket.keys()) seeds.add(seed);
  }

  const mean: number[] = [];
  const std: number[] = [];
  for (const timestep of timesteps) {
    const seedValues = [...(valuesByTimestep.get(timestep)?.values() ?? [])];
    if (seedValues.length > 0) {
      const average = seedValues.reduce((sum, value) => sum + value, 0) / seedValues.length;
      const variance =
        seedValues.reduce((sum, value) => sum + (value - average) ** 2, 0) / seedValues.length;
      mean.push(average);
      std.push(Math.sqrt(variance));
    } else {
      const fallback = baseStats.get(timestep);
      mean.push(fallback?.mean ?? Number.NaN);
      std.push(fallback?.std ?? 0);
    }
  }

  const perSeed: Record<string, Array<number | null>> = {};
  for (const seed of [...seeds].sort()) {
    perSeed[seed] = timesteps.map((timestep) => valuesByTimestep.get(timestep)?.get(seed) ?? null);
  }

  return { timesteps, mean, std, perSeed };
}

function buildSpec(
  curve: MergedCurve,
  runId: string,
  logY: boolean,
  showSeeds: boolean,
): { data: Array<Record<string, unknown>>; layout: Record<string, unknown>; config: Record<string, unknown> } {
  const { timesteps, mean, std } = curve;
  const upper = mean.map((value, index) => value + (std[index] ?? 0));
  const lower = mean.map((value, index) => value - (std[index] ?? 0));
  const data: Array<Record<string, unknown>> = [
    {
      type: "scatter",
      x: [...timesteps, ...timesteps.slice().reverse()],
      y: [...upper, ...lower.slice().reverse()],
      mode: "lines",
      name: "±1 std",
      legendgroup: "mean",
      fill: "toself",
      fillcolor: "rgba(34,211,238,0.16)",
      line: { width: 0, color: "rgba(34,211,238,0)" },
      hoverinfo: "skip",
    },
  ];

  if (showSeeds) {
    for (const [seed, values] of Object.entries(curve.perSeed)) {
      const xs: number[] = [];
      const ys: number[] = [];
      values.forEach((value, index) => {
        if (value === null) return;
        xs.push(timesteps[index] as number);
        ys.push(value);
      });
      if (ys.length === 0) continue;
      data.push({
        type: "scatter",
        x: xs,
        y: ys,
        mode: "lines",
        name: `seed ${seed}`,
        legendgroup: "seeds",
        line: { width: 1, color: MUTED },
        opacity: 0.6,
      });
    }
  }

  data.push({
    type: "scatter",
    x: timesteps,
    y: mean.map((value) => (Number.isFinite(value) ? value : null)),
    mode: "lines",
    name: "mean",
    legendgroup: "mean",
    line: { width: 2.5, color: ACCENT },
  });

  let bestIndex = -1;
  for (let index = 0; index < mean.length; index += 1) {
    if (!Number.isFinite(mean[index] as number)) continue;
    if (bestIndex === -1 || (mean[index] as number) > (mean[bestIndex] as number)) bestIndex = index;
  }
  if (bestIndex >= 0) {
    data.push({
      type: "scatter",
      x: [timesteps[bestIndex]],
      y: [mean[bestIndex]],
      mode: "markers",
      name: "best mean",
      marker: {
        symbol: "diamond",
        size: 9,
        color: ACCENT_2,
        line: { color: "#0B0E14", width: 1 },
      },
    });
  }

  return {
    data,
    layout: {
      title: { text: runId },
      hovermode: "x unified",
      // A stable uirevision keeps the user's x-range/zoom while new points arrive.
      uirevision: `learning:${runId}`,
      xaxis: { title: "timestep", gridcolor: GRID },
      yaxis: { title: "return", gridcolor: GRID, ...(logY ? { type: "log" } : {}) },
      legend: { orientation: "h", y: 1.02, x: 0 },
      margin: { l: 56, r: 24, t: 48, b: 44 },
    },
    config: { displayModeBar: false, responsive: true },
  };
}

/** Learning curve: REST snapshot + live append, with per-seed and log-y controls. */
export function LearningCurve({ runId, points }: LearningCurveProps) {
  const [curve, setCurve] = useState<LearningCurvePayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [logY, setLogY] = useState(false);
  const [showSeeds, setShowSeeds] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setCurve(null);
    setLoadError(null);
    getCurve(runId)
      .then((payload) => {
        if (!cancelled) setCurve(payload);
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoadError(String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const merged = useMemo(() => mergeLiveCurve(curve, points), [curve, points]);
  const spec = useMemo(
    () => (merged ? buildSpec(merged, runId, logY, showSeeds) : null),
    [merged, runId, logY, showSeeds],
  );

  const controls = (
    <Toolbar label="Learning curve controls">
      <button
        type="button"
        aria-pressed={logY}
        onClick={() => setLogY((value) => !value)}
        className={cn(
          "rounded-control border px-2 py-1 text-xs",
          logY ? "border-accent/60 bg-bg-elevated text-accent" : "border-border text-text-muted",
        )}
      >
        Log y
      </button>
      <button
        type="button"
        aria-pressed={showSeeds}
        onClick={() => setShowSeeds((value) => !value)}
        className={cn(
          "rounded-control border px-2 py-1 text-xs",
          showSeeds
            ? "border-accent-2/60 bg-bg-elevated text-accent-2"
            : "border-border text-text-muted",
        )}
      >
        Per-seed
      </button>
    </Toolbar>
  );

  const pointCount = merged?.timesteps.length ?? 0;
  return (
    <Panel
      title="Learning curve"
      subtitle={
        merged
          ? `${pointCount.toLocaleString()} points · ${Object.keys(merged.perSeed).length} seed(s)`
          : "Mean ± std across seeds"
      }
      actions={controls}
    >
      {spec ? (
        <PlotlyFigure
          spec={spec}
          ariaLabel={`Learning curve with ${pointCount} points`}
          className="h-80"
        />
      ) : (
        <EmptyState
          title="No learning curve yet"
          description={
            loadError
              ? `The curve endpoint failed for ${runId}: ${loadError}`
              : "Waiting for the first episode return. Start training with `uv run undertow-sim train --config configs/sim_default.toml --output runs/experiment_1/`."
          }
        />
      )}
    </Panel>
  );
}
