import type { FigureSpec, RolloutSeries } from "../../api/client";
import { TOKEN } from "./theme";

/** One rollout-parquet row, normalised to camelCase and finite defaults. */
export interface RolloutRow {
  step: number;
  price: number;
  tick: number;
  tickLower: number | null;
  tickUpper: number | null;
  inRange: boolean;
  positionValue: number | null;
  equity: number | null;
  actionType: "hold" | "rebalance";
  actionIndex: number | null;
  centerOffset: number;
  width: number;
  fees: number;
  gas: number;
  slippage: number;
  il: number;
  riskPenalty: number;
  reward: number | null;
  regime: string;
  gasPriceWei: number | null;
  observation: number[] | null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringValue(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function toActionType(value: unknown, centerOffset: number, width: number): "hold" | "rebalance" {
  if (value === "rebalance" || value === "hold") return value;
  return centerOffset !== 0 || width !== 0 ? "rebalance" : "hold";
}

/**
 * Normalise the observation vector without changing its length: a non-finite
 * entry becomes `0`, so observation dimension indices stay aligned with their
 * labels even when a single feature is missing.
 */
function toObservation(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  if (value.length === 0) return null;
  return value.map((item) => finiteNumber(item) ?? 0);
}

/**
 * Most frequent regime label; ties break alphabetically (matches the backend).
 *
 * A blank label counts as `unknown`, and an empty input is `unknown` too, so
 * the filmstrip always has a colour and a label to render.
 */
export function dominantRegime(labels: readonly string[]): string {
  const counts = new Map<string, number>();
  for (const label of labels) {
    const key = label || "unknown";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  if (counts.size === 0) return "unknown";
  let best = "unknown";
  let bestCount = -1;
  for (const [label, count] of counts) {
    if (count > bestCount || (count === bestCount && label < best)) {
      best = label;
      bestCount = count;
    }
  }
  return best;
}

/**
 * Normalise the rollout wire rows into typed rows sorted by step.
 *
 * Rows without a finite `step`/`price` are dropped (the parquet reader should
 * never emit them, but the view must not crash on a partial frame).
 */
export function parseRolloutRows(rows: RolloutSeries["rows"]): RolloutRow[] {
  const parsed: RolloutRow[] = [];
  rows.forEach((raw, index) => {
    const step = finiteNumber(raw.step) ?? index;
    const price = finiteNumber(raw.price);
    if (price === null) return;
    const centerOffset = finiteNumber(raw.center_offset) ?? 0;
    const width = finiteNumber(raw.width) ?? 0;
    parsed.push({
      step,
      price,
      tick: finiteNumber(raw.tick) ?? 0,
      tickLower: finiteNumber(raw.tick_lower),
      tickUpper: finiteNumber(raw.tick_upper),
      inRange: raw.in_range === true,
      positionValue: finiteNumber(raw.position_value),
      equity: finiteNumber(raw.equity),
      actionType: toActionType(raw.action_type, centerOffset, width),
      actionIndex: finiteNumber(raw.action_index),
      centerOffset,
      width,
      fees: finiteNumber(raw.fees) ?? 0,
      gas: finiteNumber(raw.gas) ?? 0,
      slippage: finiteNumber(raw.slippage) ?? 0,
      il: finiteNumber(raw.il) ?? 0,
      riskPenalty: finiteNumber(raw.risk_penalty) ?? 0,
      reward: finiteNumber(raw.reward),
      regime: stringValue(raw.regime, "unknown"),
      gasPriceWei: finiteNumber(raw.gas_price_wei),
      observation: toObservation(raw.observation),
    });
  });
  parsed.sort((a, b) => a.step - b.step);
  return parsed;
}

/** Append a vertical cursor line to a spec without mutating it. */
export function withCursor(spec: FigureSpec, step: number | null): FigureSpec {
  if (step === null || !Number.isFinite(step)) return spec;
  const layout = spec.layout as Record<string, unknown>;
  const existing = Array.isArray(layout.shapes)
    ? (layout.shapes as Array<Record<string, unknown>>)
    : [];
  const shapes = [
    ...existing,
    {
      type: "line",
      xref: "x",
      yref: "paper",
      x0: step,
      x1: step,
      y0: 0,
      y1: 1,
      line: { color: TOKEN.textPrimary, width: 1, dash: "dot" },
      layer: "above",
    },
  ];
  return { ...spec, layout: { ...layout, shapes } };
}
