import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { MemoryRouter, useLocation } from "../router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { RunRecord, RolloutSeries, RolloutSummary } from "../api/client";
import {
  CostWaterfall,
  buildStackedAreaSpec,
  buildWaterfallSpec,
  costBreakdown,
} from "../components/replay/CostWaterfall";
import { REGIME_COLORS } from "../components/replay/EpisodeFilmstrip";
import { buildEquitySpec, drawdownSeries, hodlBaseline } from "../components/replay/EquityVsHodl";
import { RibbonChart } from "../components/replay/RibbonChart";
import { dominantRegime, parseRolloutRows } from "../components/replay/rows";
import { StateInspector } from "../components/replay/StateInspector";
import { TOKEN, withAlpha } from "../components/replay/theme";
import PolicyReplay from "../views/PolicyReplay";

const mocks = vi.hoisted(() => ({
  getRuns: vi.fn(),
  listRollouts: vi.fn(),
  getRollout: vi.fn(),
  getFigure: vi.fn(),
  plotlyRender: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getRuns: mocks.getRuns,
    listRollouts: mocks.listRollouts,
    getRollout: mocks.getRollout,
    getFigure: mocks.getFigure,
  };
});

vi.mock("plotly.js-dist-min", () => ({
  default: {
    react: (_node: unknown, data: unknown, layout: unknown, config: unknown) => {
      mocks.plotlyRender({ data, layout, config });
    },
    purge: vi.fn(),
  },
}));

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function rawRow(step: number, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: 1,
    run_id: "run-1",
    seed: 0,
    episode: 0,
    step,
    price: 100 + step * 10,
    tick: 1000 + step,
    tick_lower: 1000 + step,
    tick_upper: 2000 + step,
    in_range: step % 2 === 0,
    position_value: 1000,
    equity: 1000 + step,
    action_index: step,
    action_type: "hold",
    center_offset: 0,
    width: 0,
    fees: 1,
    gas: -0.5,
    slippage: -0.25,
    il: -0.1,
    risk_penalty: -0.05,
    reward: 0.1,
    regime: "bull",
    gas_price_wei: 20,
    ...overrides,
  };
}

function rawSeries(rows: Array<Record<string, unknown>>, seed = 0, episode = 0): RolloutSeries {
  return {
    run_id: "run-1",
    seed,
    episode,
    columns: [],
    rows: rows as RolloutSeries["rows"],
    downsampled: false,
  };
}

function summary(seed: number, episode: number, regime = "bull"): RolloutSummary {
  return {
    seed,
    episode,
    n_steps: 5,
    regime,
    start: "2024-01-01T00:00:00Z",
    end: "2024-01-01T01:00:00Z",
  };
}

const RUN: RunRecord = {
  run_id: "run-1",
  experiment: "experiment_1",
  seed: 0,
  state: "done",
  updated_at: "2024-01-01T00:00:00Z",
  total_timesteps: 1000,
  latest_timestep: 1000,
  best_reward: 1,
  config_hash: "abc",
  git_commit: "deadbeef",
  manifest: null,
  eval: null,
  has_telemetry: true,
  has_rollouts: true,
  telemetry_points: 10,
};

function rowsFor(seed: number, episode: number): Array<Record<string, unknown>> {
  const count = 4 + episode;
  const rows: Array<Record<string, unknown>> = [];
  for (let step = 0; step < count; step += 1) {
    rows.push(
      rawRow(step, {
        seed,
        episode,
        action_type: step % 3 === 0 ? "rebalance" : "hold",
        center_offset: step % 3 === 0 ? 2 : 0,
        width: step % 3 === 0 ? 5 : 0,
      }),
    );
  }
  return rows;
}

function LocationProbe() {
  const location = useLocation();
  return <span data-testid="location">{`${location.pathname}${location.search}`}</span>;
}

function renderView(initialEntry = "/replay?run_id=run-1") {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <PolicyReplay />
      <LocationProbe />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  mocks.getRuns.mockReset();
  mocks.listRollouts.mockReset();
  mocks.getRollout.mockReset();
  mocks.getFigure.mockReset();
  mocks.plotlyRender.mockClear();

  mocks.getRuns.mockResolvedValue([RUN]);
  mocks.listRollouts.mockResolvedValue([summary(0, 0), summary(0, 1), summary(1, 0, "bear")]);
  mocks.getRollout.mockImplementation((_runId: string, seed: number, episode: number) =>
    Promise.resolve(rawSeries(rowsFor(seed, episode), seed, episode)),
  );
  mocks.getFigure.mockResolvedValue({
    data: [{ type: "scatter", name: "price", x: [0, 1], y: [100, 110] }],
    layout: {},
    config: {},
  });
});

afterEach(() => {
  cleanup();
});

// ---------------------------------------------------------------------------
// 1. Selection in the URL
// ---------------------------------------------------------------------------
describe("PolicyReplay selection", () => {
  it("fetches the selected seed/episode and writes the selection into the URL", async () => {
    renderView("/replay?run_id=run-1&seed=0&episode=0");
    await screen.findByLabelText("Episode");

    fireEvent.change(screen.getByLabelText("Episode"), { target: { value: "1" } });

    await waitFor(() => {
      expect(mocks.getRollout).toHaveBeenCalledWith("run-1", 0, 1);
    });
    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toContain("episode=1");
    });
    expect(screen.getByTestId("location").textContent).toContain("run_id=run-1");
  });

  it("defaults to the first seed/episode and rewrites the URL to make it shareable", async () => {
    renderView("/replay?run_id=run-1");
    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toContain("seed=0");
    });
    expect(screen.getByTestId("location").textContent).toContain("episode=0");
    expect(mocks.getRollout).toHaveBeenCalledWith("run-1", 0, 0);
  });
});

// ---------------------------------------------------------------------------
// 2. Ribbon: tick range series and exactly the rebalance rows
// ---------------------------------------------------------------------------
describe("RibbonChart", () => {
  it("prefers the backend ribbon figure, requesting the selected seed/episode", async () => {
    const backend = {
      data: [{ type: "scatter", name: "price", x: [0, 1], y: [100, 110] }],
      layout: { title: { text: "backend" } },
      config: {},
    };
    mocks.getFigure.mockResolvedValue(backend);

    render(
      <RibbonChart runId="run-1" seed={2} episode={7} rows={[]} cursorIndex={0} />,
    );

    await waitFor(() => {
      expect(mocks.getFigure).toHaveBeenCalledWith("run-1", "ribbon", { seed: 2, episode: 7 });
    });
    await waitFor(() => {
      expect(mocks.plotlyRender).toHaveBeenCalled();
    });
    const last = mocks.plotlyRender.mock.calls.at(-1)?.[0] as {
      data: Array<Record<string, unknown>>;
    };
    expect(last.data).toEqual(backend.data);
  });

  it("falls back to a local spec with the right tick_lower/tick_upper series and rebalance rows", async () => {
    mocks.getFigure.mockRejectedValue(new Error("backend unavailable"));

    const raw = [
      rawRow(0, { action_type: "rebalance", tick_lower: 900, tick_upper: 1100 }),
      rawRow(1, { action_type: "hold", tick_lower: 910, tick_upper: 1110 }),
      rawRow(2, { action_type: "hold", tick_lower: 920, tick_upper: 1120 }),
      rawRow(3, { action_type: "rebalance", tick_lower: 930, tick_upper: 1130 }),
      rawRow(4, { action_type: "hold", tick_lower: 940, tick_upper: 1140 }),
    ];
    const rows = parseRolloutRows(raw as RolloutSeries["rows"]);

    render(<RibbonChart runId="run-1" seed={0} episode={0} rows={rows} cursorIndex={0} />);

    await waitFor(() => {
      expect(mocks.plotlyRender).toHaveBeenCalled();
    });
    const props = mocks.plotlyRender.mock.calls.at(-1)?.[0] as {
      data: Array<Record<string, unknown>>;
      layout: Record<string, unknown>;
    };
    // The client-built specs must inline the dark theme (the backend template
    // is not registered in the browser bundle).
    expect(props.layout.paper_bgcolor).toBe("#11161F");
    const lower = props.data
      .filter((trace) => trace.name === "tick_lower")
      .flatMap((trace) => trace.y as number[]);
    const upper = props.data
      .filter((trace) => trace.name === "tick_upper")
      .flatMap((trace) => trace.y as number[]);
    expect(lower).toEqual([900, 910, 920, 930, 940]);
    expect(upper).toEqual([1100, 1110, 1120, 1130, 1140]);

    const rebalance = props.data.find((trace) => trace.name === "rebalance");
    expect(rebalance?.x).toEqual([0, 3]);
    expect(props.data.some((trace) => trace.name === "HODL")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// 3. HODL baseline math
// ---------------------------------------------------------------------------
describe("HODL baseline", () => {
  it("scales the initial capital by the price series", () => {
    const rows = parseRolloutRows([
      rawRow(0, { price: 100, equity: 1000 }),
      rawRow(1, { price: 110, equity: 900 }),
      rawRow(2, { price: 90, equity: 950 }),
    ] as RolloutSeries["rows"]);

    expect(hodlBaseline(rows)).toEqual([1000, 1100, 900]);
    expect(hodlBaseline(rows, 2000)).toEqual([2000, 2200, 1800]);
  });

  it("falls back to a flat baseline when the first price is unusable", () => {
    const rows = parseRolloutRows([rawRow(0, { price: 0, equity: 500 })] as RolloutSeries["rows"]);
    expect(hodlBaseline(rows)).toEqual([500]);
  });

  it("computes running-peak drawdown against the previous high", () => {
    expect(drawdownSeries([100, 120, 90, 130, 65])).toEqual([0, 0, -30, 0, -65]);
    expect(drawdownSeries([])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 4. Cost waterfall components sum to net
// ---------------------------------------------------------------------------
describe("CostWaterfall", () => {
  it("decomposes end-of-episode costs and sums the components to net", () => {
    const rows = parseRolloutRows([
      rawRow(0, { fees: 2, gas: -1, slippage: -0.5, il: -0.25, risk_penalty: -0.1 }),
      rawRow(1, { fees: 1, gas: -0.5, slippage: -0.25, il: -0.5, risk_penalty: -0.2 }),
    ] as RolloutSeries["rows"]);

    const totals = costBreakdown(rows);
    expect(totals.fees).toBeCloseTo(3);
    expect(totals.gas).toBeCloseTo(-1.5);
    expect(totals.slippage).toBeCloseTo(-0.75);
    expect(totals.il).toBeCloseTo(-0.75);
    expect(totals.riskPenalty).toBeCloseTo(-0.3);
    expect(totals.net).toBeCloseTo(
      totals.fees + totals.gas + totals.slippage + totals.il + totals.riskPenalty,
    );
  });

  it("renders without crashing and exposes the net in the subtitle", () => {
    const rows = parseRolloutRows(rowsFor(0, 0) as RolloutSeries["rows"]);
    render(<CostWaterfall rows={rows} cursorIndex={0} />);
    expect(screen.getByText(/net 0\.40/)).toBeTruthy();
  });

  it("inlines the dark theme in every client-built spec", () => {
    const rows = parseRolloutRows(rowsFor(0, 0) as RolloutSeries["rows"]);
    const specs = [buildEquitySpec(rows), buildWaterfallSpec(rows), buildStackedAreaSpec(rows)];
    for (const spec of specs) {
      expect(spec.layout.paper_bgcolor).toBe("#11161F");
      expect(spec.layout.plot_bgcolor).toBe("#11161F");
    }
  });
});

// ---------------------------------------------------------------------------
// 4b. Pool over time + policy summary wiring
// ---------------------------------------------------------------------------
describe("PolicyReplay pool panels", () => {
  it("mounts the pool chart and policy summary below the replay panels", async () => {
    renderView("/replay?run_id=run-1&seed=0&episode=0");

    expect(await screen.findByText("Pool over time")).toBeTruthy();
    // The fixture predates the pool-state columns, so the chart explains itself
    // instead of drawing a flat zero line.
    expect(screen.getByText("Pool state not recorded for this run")).toBeTruthy();
    expect(screen.getByText("What this policy captured")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// 5. Filmstrip
// ---------------------------------------------------------------------------
describe("EpisodeFilmstrip", () => {
  it("renders one tile per episode and clicking a tile changes the selection", async () => {
    renderView("/replay?run_id=run-1&seed=0&episode=0");
    const tile = await screen.findByLabelText("Episode 1, dominant regime bull");
    expect(screen.getByLabelText("Episode 0, dominant regime bull")).toBeTruthy();

    fireEvent.click(tile);
    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toContain("episode=1");
    });
    expect(mocks.getRollout).toHaveBeenCalledWith("run-1", 0, 1);
  });

  it("clears the previous seed's tiles before the new ones resolve", async () => {
    renderView("/replay?run_id=run-1&seed=0&episode=0");
    // Seed 0 has episodes 0 and 1; wait until both tiles have rendered.
    expect(await screen.findByLabelText("Episode 1, dominant regime bull")).toBeTruthy();

    // Seed 1 has a single episode, so exactly two fetches are issued after the
    // switch: the *main* series first (resolves immediately, so the filmstrip
    // remounts), then the filmstrip tile fetch (held open). Without the
    // `setTiles([])` reset the seed-0 tiles would survive that remount and
    // remain on screen until the tile fetch resolves.
    let resolveTile!: (series: RolloutSeries) => void;
    const tilePending = new Promise<RolloutSeries>((resolve) => {
      resolveTile = resolve;
    });
    let seedOneCalls = 0;
    mocks.getRollout.mockImplementation((_runId: string, seed: number, episode: number) => {
      if (seed !== 1) return Promise.resolve(rawSeries(rowsFor(seed, episode), seed, episode));
      seedOneCalls += 1;
      return seedOneCalls === 1
        ? Promise.resolve(rawSeries(rowsFor(1, 0), 1, 0))
        : tilePending;
    });

    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "1" } });

    await waitFor(() => {
      expect(mocks.getRollout).toHaveBeenCalledWith("run-1", 1, 0);
    });
    // Wait for the main series to load so the filmstrip is mounted again.
    await screen.findByText("Episodes");

    expect(screen.queryByLabelText("Episode 1, dominant regime bull")).toBeNull();
    expect(screen.queryByLabelText("Episode 0, dominant regime bull")).toBeNull();

    const bearRows = rowsFor(1, 0).map((row) => ({ ...row, regime: "bear" }));
    resolveTile(rawSeries(bearRows, 1, 0));

    expect(await screen.findByLabelText("Episode 0, dominant regime bear")).toBeTruthy();
    expect(screen.queryByLabelText("Episode 1, dominant regime bull")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 6. Missing rollouts
// ---------------------------------------------------------------------------
describe("missing rollouts", () => {
  it("shows an EmptyState that names training.record_rollouts", async () => {
    mocks.listRollouts.mockResolvedValue([]);
    renderView("/replay?run_id=run-1");

    expect(await screen.findByText("No recorded rollouts")).toBeTruthy();
    expect(screen.getByText("training.record_rollouts = true")).toBeTruthy();
  });

  it("shows a selection prompt when no run is selected", async () => {
    renderView("/replay");
    expect(screen.getByText("No run selected")).toBeTruthy();
    await waitFor(() => {
      expect(mocks.getRuns).toHaveBeenCalled();
    });
    expect(mocks.listRollouts).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// 7. Row parsing edge cases
// ---------------------------------------------------------------------------
describe("parseRolloutRows", () => {
  it("drops non-finite rows and sorts by step", () => {
    const rows = parseRolloutRows([
      rawRow(2),
      rawRow(0),
      { ...rawRow(1), price: null },
      { ...rawRow(3), price: Number.NaN },
    ] as RolloutSeries["rows"]);
    expect(rows.map((row) => row.step)).toEqual([0, 2]);
  });

  it("maps null, empty and non-array observations to null", () => {
    const rows = parseRolloutRows([
      rawRow(0, { observation: null }),
      rawRow(1, { observation: [] }),
      rawRow(2, { observation: "not-an-array" }),
    ] as RolloutSeries["rows"]);

    expect(rows.map((row) => row.observation)).toEqual([null, null, null]);
  });

  it("preserves observation length and dimension labels for non-finite entries", () => {
    const rows = parseRolloutRows([
      rawRow(0, { observation: [1, Number.NaN, 3] }),
    ] as RolloutSeries["rows"]);

    expect(rows[0]!.observation).toEqual([1, 0, 3]);
    expect(rows[0]!.observation).toHaveLength(3);

    render(<StateInspector rows={rows} cursorIndex={0} />);
    expect(screen.getByText("Observation (3 dims)")).toBeTruthy();
    expect(screen.getByText("obs 0")).toBeTruthy();
    expect(screen.getByText("obs 1")).toBeTruthy();
    expect(screen.getByText("obs 2")).toBeTruthy();
    expect(screen.getByLabelText("observation dimension 2")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// 8. Dominant regime
// ---------------------------------------------------------------------------
describe("dominantRegime", () => {
  it("returns the most frequent label, breaking ties alphabetically", () => {
    expect(dominantRegime(["bull", "bear", "bull"])).toBe("bull");
    expect(dominantRegime(["bear", "bull", "bear", "bull"])).toBe("bear");
    expect(dominantRegime([])).toBe("unknown");
  });
});

// ---------------------------------------------------------------------------
// 9. Theme derivation
// ---------------------------------------------------------------------------
describe("theme tokens", () => {
  it("derives every regime colour from the shared CONTRACTS §9 tokens", () => {
    expect(REGIME_COLORS).toEqual({
      bull: TOKEN.bull,
      bear: TOKEN.bear,
      sideways: TOKEN.sideways,
      high_vol: TOKEN.highVol,
      unknown: TOKEN.textMuted,
    });
  });

  it("translates a hex token into an rgba string without changing the channels", () => {
    expect(withAlpha(TOKEN.bull, 0.1)).toBe("rgba(52, 211, 153, 0.1)");
    expect(withAlpha(TOKEN.bear, 0.2)).toBe("rgba(248, 113, 113, 0.2)");
    expect(withAlpha(TOKEN.textMuted, 0.08)).toBe("rgba(139, 147, 167, 0.08)");
  });
});