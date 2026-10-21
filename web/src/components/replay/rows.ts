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
  /** Raw pool active liquidity `L`; `0` for rollouts predating §2.1's additive columns. */
  poolActiveLiquidity: number;
  /** Raw agent-position liquidity `L` (comparable to the pool's, per ADR-009). */
  positionLiquidity: number;
  /** Raw token0 (USDC) fees the whole pool generated this step. */
  poolFeeEarned0: number;
  /** Raw token1 (WETH) fees the whole pool generated this step. */
  poolFeeEarned1: number;
  observation: number[] | null;
}

// ADR-009: the replay rollouts are for the pinned USDC (token0, 6dp) /
// WETH (token1, 18dp) pool, and CONTRACTS §2.1 records the pool fee legs in
// raw base units.
const USDC_DECIMALS = 6;
const WETH_DECIMALS = 18;

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
      poolActiveLiquidity: finiteNumber(raw.pool_active_liquidity) ?? 0,
      positionLiquidity: finiteNumber(raw.position_liquidity) ?? 0,
      poolFeeEarned0: finiteNumber(raw.pool_fee_earned_0) ?? 0,
      poolFeeEarned1: finiteNumber(raw.pool_fee_earned_1) ?? 0,
      observation: toObservation(raw.observation),
    });
  });
  parsed.sort((a, b) => a.step - b.step);
  return parsed;
}

/**
 * USDC value of the fees the whole pool generated in one step.
 *
 * CONTRACTS §2.1 records both fee legs in **raw base units** while `price` is
 * human USDC/WETH (ADR-009), so each leg is first humanised with its token
 * decimals: `fee0 / 10**6 + (fee1 / 10**18) * price`. Both legs default to
 * `0`, so a pre-fix rollout values to `0` rather than `NaN`.
 */
export function poolFeeUsdc(row: RolloutRow): number {
  const token0Usdc = row.poolFeeEarned0 / 10 ** USDC_DECIMALS;
  const token1Weth = row.poolFeeEarned1 / 10 ** WETH_DECIMALS;
  return token0Usdc + token1Weth * row.price;
}

/**
 * True when the rollout carries at least one non-zero pool-state value.
 *
 * Rollouts recorded before the §2.1 additive columns parse every pool field to
 * `0`; the replay panels use this to show "not recorded" instead of a flat
 * zero line (and a divide-by-zero fee share).
 */
export function hasPoolState(rows: readonly RolloutRow[]): boolean {
  return rows.some(
    (row) =>
      row.poolActiveLiquidity !== 0 ||
      row.positionLiquidity !== 0 ||
      row.poolFeeEarned0 !== 0 ||
      row.poolFeeEarned1 !== 0,
  );
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
