import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { FigureSpec, RolloutSeries } from "../api/client";
import { PoolChart, buildPoolSpec } from "../components/replay/PoolChart";
import { PolicySummary, policySummary } from "../components/replay/PolicySummary";
import {
  hasPoolState,
  parseRolloutRows,
  poolFeeUsdc,
  type RolloutRow,
} from "../components/replay/rows";

const mocks = vi.hoisted(() => ({ plotlySpec: vi.fn() }));

vi.mock("../components/PlotlyFigure", () => ({
  PlotlyFigure: (props: { spec: unknown; ariaLabel?: string }) => {
    mocks.plotlySpec(props.spec);
    return <div data-testid="plotly-figure" aria-label={props.ariaLabel} />;
  },
}));

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function rawRow(step: number, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    step,
    price: 100 + step * 100,
    tick: 1000,
    tick_lower: 1000,
    tick_upper: 2000,
    in_range: step % 2 === 0,
    position_value: 1000,
    equity: 1000 + step * 10,
    action_index: step,
    action_type: step % 2 === 0 ? "hold" : "rebalance",
    center_offset: step % 2 === 0 ? 0 : 2,
    width: step % 2 === 0 ? 0 : 5,
    fees: 0.5,
    gas: -0.5,
    slippage: -0.25,
    il: -0.1,
    risk_penalty: -0.05,
    reward: 0.1,
    regime: "bull",
    pool_active_liquidity: 500_000 + step,
    position_liquidity: 5_000 + step,
    pool_fee_earned_0: 0,
    pool_fee_earned_1: 0,
    ...overrides,
  };
}

/**
 * Four steps with hand-computed totals:
 * agent fees 1 + 1 + 0.5 + 0.5 = 3 USDC;
 * pool fees 2.1 + 2 + 0.6 + 2 = 6.7 USDC, where each leg is recorded raw
 * (2 USDC = 2_000_000 raw; 0.001 WETH = 1e15 raw) and token1 is valued at
 * that row's human price;
 * 2 rebalances, 2 in-range steps, equity 1000 -> 1050.
 */
const SUMMARY_ROWS: Array<Record<string, unknown>> = [
  rawRow(0, {
    price: 100,
    equity: 1000,
    action_type: "hold",
    in_range: true,
    fees: 1,
    pool_fee_earned_0: 2_000_000,
    pool_fee_earned_1: 1_000_000_000_000_000,
  }),
  rawRow(1, {
    price: 200,
    equity: 1020,
    action_type: "rebalance",
    in_range: false,
    fees: 1,
    pool_fee_earned_0: 2_000_000,
    pool_fee_earned_1: 0,
  }),
  rawRow(2, {
    price: 300,
    equity: 1040,
    action_type: "hold",
    in_range: true,
    fees: 0.5,
    pool_fee_earned_0: 0,
    pool_fee_earned_1: 2_000_000_000_000_000,
  }),
  rawRow(3, {
    price: 400,
    equity: 1050,
    action_type: "rebalance",
    in_range: false,
    fees: 0.5,
    pool_fee_earned_0: 2_000_000,
    pool_fee_earned_1: 0,
  }),
];

/** Same shape as `rawRow` but with none of the §2.1 pool columns. */
function legacyRow(step: number, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const row = rawRow(step, overrides);
  delete row.pool_active_liquidity;
  delete row.position_liquidity;
  delete row.pool_fee_earned_0;
  delete row.pool_fee_earned_1;
  return row;
}

function parse(raw: Array<Record<string, unknown>>): RolloutRow[] {
  return parseRolloutRows(raw as RolloutSeries["rows"]);
}

afterEach(() => {
  cleanup();
});

// ---------------------------------------------------------------------------
// 1. Row parsing
// ---------------------------------------------------------------------------
describe("parseRolloutRows pool fields", () => {
  it("parses the four pool-state columns into camelCase fields", () => {
    const rows = parse([
      rawRow(0, {
        pool_active_liquidity: 7,
        position_liquidity: 3,
        pool_fee_earned_0: 1.5,
        pool_fee_earned_1: 0.25,
      }),
    ]);

    expect(rows[0]).toMatchObject({
      poolActiveLiquidity: 7,
      positionLiquidity: 3,
      poolFeeEarned0: 1.5,
      poolFeeEarned1: 0.25,
    });
  });

  it("defaults absent and non-finite pool fields to 0 so legacy rollouts still parse", () => {
    const rows = parse([
      legacyRow(0),
      rawRow(1, {
        pool_active_liquidity: null,
        position_liquidity: "not-a-number",
        pool_fee_earned_0: Number.NaN,
        pool_fee_earned_1: undefined,
      }),
    ]);

    expect(
      rows.map((row) => [
        row.poolActiveLiquidity,
        row.positionLiquidity,
        row.poolFeeEarned0,
        row.poolFeeEarned1,
      ]),
    ).toEqual([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// 2. Valuation and presence helpers
// ---------------------------------------------------------------------------
describe("pool fee valuation", () => {
  it("humanises the raw token units before valuing them at the row price", () => {
    const row = parse([
      rawRow(0, { price: 2000, pool_fee_earned_0: 1_500_000, pool_fee_earned_1: 400_000_000_000_000 }),
    ])[0]!;

    expect(poolFeeUsdc(row)).toBeCloseTo(1.5 + 0.0004 * 2000, 10);
  });

  it("converts the documented raw scale (1e6 / 1e18) to USDC", () => {
    // 10_000_000 raw USDC = 10 USDC; 3e15 raw WETH = 0.003 WETH = 9 USDC at $3000.
    const row = parse([
      rawRow(0, {
        price: 3000,
        pool_fee_earned_0: 10_000_000,
        pool_fee_earned_1: 3_000_000_000_000_000,
      }),
    ])[0]!;

    expect(poolFeeUsdc(row)).toBeCloseTo(19, 10);
  });

  it("treats a zero-valued fee as zero and flags pool state presence", () => {
    const recorded = parse([rawRow(0), rawRow(1)]);
    expect(hasPoolState(recorded)).toBe(true);

    const legacy = parse([legacyRow(0), legacyRow(1)]);
    expect(hasPoolState(legacy)).toBe(false);
    expect(hasPoolState([])).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 3. Pool spec
// ---------------------------------------------------------------------------
describe("buildPoolSpec", () => {
  const rows = parse(SUMMARY_ROWS);

  it("names the three traces and maps each liquidity series to its column", () => {
    const traces = buildPoolSpec(rows).data as Array<Record<string, unknown>>;
    const byName = new Map(traces.map((trace) => [trace.name, trace]));

    expect([...byName.keys()]).toEqual([
      "pool active liquidity",
      "agent position liquidity",
      "pool fees (USDC)",
    ]);
    expect(byName.get("pool active liquidity")!.y).toEqual(
      rows.map((row) => row.poolActiveLiquidity),
    );
    expect(byName.get("agent position liquidity")!.y).toEqual(
      rows.map((row) => row.positionLiquidity),
    );
  });

  it("puts the USDC-valued pool fees on the secondary axis", () => {
    const traces = buildPoolSpec(rows).data as Array<Record<string, unknown>>;
    const fees = traces.find((trace) => trace.name === "pool fees (USDC)")!;

    expect(fees.yaxis).toBe("y2");
    expect(fees.x).toEqual([0, 1, 2, 3]);
    const valued = fees.y as number[];
    expect(valued).toHaveLength(4);
    expect(valued[0]).toBeCloseTo(2.1, 10);
    expect(valued[1]).toBeCloseTo(2, 10);
    expect(valued[2]).toBeCloseTo(0.6, 10);
    expect(valued[3]).toBeCloseTo(2, 10);
    expect(valued).toEqual(rows.map(poolFeeUsdc));
  });

  it("uses a log liquidity axis, a linear fee axis and the inline dark theme", () => {
    const layout = buildPoolSpec(rows).layout as Record<string, unknown>;

    expect((layout.yaxis as Record<string, unknown>).type).toBe("log");
    expect((layout.yaxis2 as Record<string, unknown>).overlaying).toBe("y");
    expect((layout.yaxis2 as Record<string, unknown>).side).toBe("right");
    expect(layout.paper_bgcolor).toBe("#11161F");
    expect(layout.plot_bgcolor).toBe("#11161F");
    expect(layout.showlegend).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 4. PoolChart component
// ---------------------------------------------------------------------------
describe("PoolChart", () => {
  it("renders the chart and appends the shared time cursor to the spec", () => {
    const rows = parse(SUMMARY_ROWS);
    render(<PoolChart rows={rows} cursorIndex={2} />);

    const spec = mocks.plotlySpec.mock.calls.at(-1)?.[0] as FigureSpec;
    const shapes = spec.layout.shapes as Array<Record<string, unknown>>;
    expect(screen.getByTestId("plotly-figure")).toBeTruthy();
    expect(shapes).toHaveLength(1);
    expect(shapes[0]!.type).toBe("line");
    expect(shapes[0]!.x0).toBe(2);
  });

  it("shows an explainable empty state for a rollout without pool columns", () => {
    const legacyRows = parse([legacyRow(0), legacyRow(1), legacyRow(2)]);
    render(<PoolChart rows={legacyRows} cursorIndex={0} />);

    expect(screen.getByText("Pool state not recorded for this run")).toBeTruthy();
    expect(screen.queryByTestId("plotly-figure")).toBeNull();
    expect(mocks.plotlySpec).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 5. Policy summary arithmetic
// ---------------------------------------------------------------------------
describe("policySummary", () => {
  it("computes the hand-checked episode totals", () => {
    const summary = policySummary(parse(SUMMARY_ROWS));

    expect(summary.steps).toBe(4);
    expect(summary.rebalances).toBe(2);
    expect(summary.inRangePct).toBeCloseTo(50, 10);
    expect(summary.agentFees).toBeCloseTo(3, 10);
    expect(summary.poolFees).toBeCloseTo(6.7, 10);
    expect(summary.agentFeeSharePct).toBeCloseTo((3 / 6.7) * 100, 10);
    expect(summary.poolStateRecorded).toBe(true);
    expect(summary.finalEquity).toBe(1050);
    expect(summary.pnl).toBe(50);
    expect(summary.il).toBeCloseTo(-0.4, 10);
    expect(summary.gas).toBeCloseTo(-2, 10);
    expect(summary.slippage).toBeCloseTo(-1, 10);
  });

  it("guards the fee-share divide by zero and reports no pool state", () => {
    const legacy = policySummary(parse([legacyRow(0), legacyRow(1)]));
    expect(legacy.poolFees).toBe(0);
    expect(legacy.agentFeeSharePct).toBeNull();
    expect(legacy.poolStateRecorded).toBe(false);

    const empty = policySummary([]);
    expect(empty.steps).toBe(0);
    expect(empty.inRangePct).toBe(0);
    expect(empty.agentFees).toBe(0);
    expect(empty.poolFees).toBe(0);
    expect(empty.agentFeeSharePct).toBeNull();
    expect(empty.finalEquity).toBeNull();
    expect(empty.pnl).toBeNull();
  });

  it("reports a real 0% share when the pool generated fees but the agent captured none", () => {
    const rows = parse([
      rawRow(0, { fees: 0, pool_fee_earned_0: 5_000_000, pool_fee_earned_1: 0 }),
      rawRow(1, { fees: 0, pool_fee_earned_0: 0, pool_fee_earned_1: 1_000_000_000_000_000, price: 100 }),
    ]);
    const summary = policySummary(rows);
    expect(summary.poolFees).toBeCloseTo(5.1, 10);
    expect(summary.agentFeeSharePct).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 6. Policy summary card
// ---------------------------------------------------------------------------
describe("PolicySummary", () => {
  it("renders the formatted scorecard values", () => {
    render(<PolicySummary rows={parse(SUMMARY_ROWS)} />);

    expect(screen.getByText("What this policy captured")).toBeTruthy();
    expect(screen.getByText("Rebalances")).toBeTruthy();
    expect(screen.getByText("50.0%")).toBeTruthy();
    expect(screen.getByText("6.70")).toBeTruthy();
    expect(screen.getByText("+50.00")).toBeTruthy();
  });

  it("renders missing pool values as em-dashes for a legacy rollout", () => {
    render(<PolicySummary rows={parse([legacyRow(0), legacyRow(1)])} />);

    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("not recorded")).toBeTruthy();
    expect(screen.getByText("1.00")).toBeTruthy();
  });
});
