import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import type { RunRecord } from "../api/client";
import { CSV_HEADER, runsToCsv } from "../components/leaderboard/csv";
import { normalizeValues } from "../components/leaderboard/fingerprint";
import { LeaderboardTable } from "../components/leaderboard/LeaderboardTable";
import {
  MAX_COMPARE_RUNS,
  parsePinnedIds,
  serializePinnedIds,
  togglePinnedId,
} from "../components/leaderboard/pinned";
import { PolicyFingerprint } from "../components/leaderboard/PolicyFingerprint";
import { sortRuns } from "../components/leaderboard/ranking";
import { ProvenanceCard } from "../components/leaderboard/ProvenanceCard";
import Compare from "../views/Compare";
import Leaderboard from "../views/Leaderboard";

const mocks = vi.hoisted(() => ({
  getRuns: vi.fn(),
  getCurve: vi.fn(),
  compare: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getRuns: mocks.getRuns,
    getCurve: mocks.getCurve,
    compare: mocks.compare,
  };
});

// Plotly renders a canvas in jsdom; the view logic under test never needs it.
vi.mock("../components/PlotlyFigure", () => ({
  PlotlyFigure: () => <div data-testid="plotly" />,
}));

type Eval = NonNullable<RunRecord["eval"]>;

function makeEval(overrides: Partial<Eval>): Eval {
  return {
    sharpe: null,
    total_pnl: null,
    annualized_return: null,
    max_drawdown: null,
    worst_regime_pnl: null,
    per_regime: {},
    source: "results/summary.json",
    ...overrides,
  };
}

function makeRun(overrides: Partial<RunRecord> & { run_id: string }): RunRecord {
  return {
    experiment: "experiment_1",
    seed: 0,
    state: "done",
    updated_at: "2026-01-01T00:00:00Z",
    total_timesteps: 1000,
    latest_timestep: 1000,
    best_reward: 0,
    config_hash: null,
    git_commit: null,
    manifest: null,
    eval: null,
    has_telemetry: true,
    has_rollouts: false,
    telemetry_points: 10,
    ...overrides,
  };
}

const EVALUATED: RunRecord[] = [
  makeRun({ run_id: "low-sharpe", best_reward: 1, eval: makeEval({ sharpe: 0.5, worst_regime_pnl: -10 }) }),
  makeRun({ run_id: "high-sharpe", best_reward: 1, eval: makeEval({ sharpe: 2, worst_regime_pnl: -5 }) }),
  makeRun({ run_id: "tie-worse-regime", best_reward: 1, eval: makeEval({ sharpe: 2, worst_regime_pnl: -8 }) }),
];
const TRAIN_ONLY = makeRun({ run_id: "train-only-run", best_reward: 999 });

beforeEach(() => {
  mocks.getRuns.mockReset();
  mocks.getRuns.mockResolvedValue([]);
  mocks.getCurve.mockReset();
  mocks.getCurve.mockResolvedValue({ run_id: "x", timesteps: [], mean: [], std: [], per_seed: {}, downsampled: false });
  mocks.compare.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

function rowIds(): (string | null)[] {
  return screen.getAllByTestId("leaderboard-row").map((row) => row.getAttribute("data-run-id"));
}

describe("leaderboard ranking", () => {
  it("defaults to the pinned rule on a fixture with mixed eval coverage", () => {
    const shuffled = [TRAIN_ONLY, EVALUATED[0]!, EVALUATED[2]!, EVALUATED[1]!];
    render(<LeaderboardTable runs={shuffled} />);

    expect(rowIds()).toEqual(["high-sharpe", "tie-worse-regime", "low-sharpe", "train-only-run"]);
  });

  it("sorts train-only runs last and marks them", () => {
    render(<LeaderboardTable runs={[TRAIN_ONLY, ...EVALUATED]} />);

    expect(rowIds().at(-1)).toBe("train-only-run");
    const trainRow = screen.getByText("train-only").closest("tr");
    expect(trainRow?.getAttribute("data-train-only")).toBe("true");
    expect(screen.getAllByText("train-only")).toHaveLength(1);
  });

  it("places missing values last even when sorting a metric descending", () => {
    const runs = [
      makeRun({ run_id: "a", eval: makeEval({ sharpe: 1 }) }),
      makeRun({ run_id: "b", eval: makeEval({ sharpe: 3 }) }),
      makeRun({ run_id: "c", eval: null }),
    ];
    expect(sortRuns(runs, "sharpe", "desc").map((run) => run.run_id)).toEqual(["b", "a", "c"]);
    expect(sortRuns(runs, "sharpe", "asc").map((run) => run.run_id)).toEqual(["a", "b", "c"]);
  });

  it("shows the pinned-rule rank independently of the active sort", () => {
    const runs = [
      makeRun({ run_id: "low", best_reward: 5, eval: makeEval({ sharpe: 1 }) }),
      makeRun({ run_id: "high", best_reward: 1, eval: makeEval({ sharpe: 2 }) }),
    ];
    render(<LeaderboardTable runs={runs} />);

    fireEvent.click(screen.getByRole("button", { name: /^Best reward/ }));

    const rows = screen.getAllByTestId("leaderboard-row") as HTMLTableRowElement[];
    expect(rows[0]!.getAttribute("data-run-id")).toBe("low");
    // The rank cell still reflects the pinned rule (high = #1), not row order.
    expect(rows[0]!.cells[2]!.textContent).toBe("2");
    expect(rows[1]!.cells[2]!.textContent).toBe("1");
  });
});

describe("leaderboard filtering", () => {
  const runs = [
    makeRun({ run_id: "ppo_s0_alpha", experiment: "experiment_1", state: "done", eval: makeEval({ sharpe: 1 }) }),
    makeRun({ run_id: "ppo_s1_beta", experiment: "experiment_2", state: "failed", eval: makeEval({ sharpe: 2 }) }),
  ];

  it("narrows rows by run-id search", () => {
    render(<LeaderboardTable runs={runs} />);
    expect(rowIds()).toHaveLength(2);

    fireEvent.change(screen.getByLabelText("Search run id"), { target: { value: "beta" } });
    expect(rowIds()).toEqual(["ppo_s1_beta"]);
  });

  it("narrows rows by state and experiment filters", () => {
    render(<LeaderboardTable runs={runs} />);

    fireEvent.change(screen.getByLabelText("State"), { target: { value: "failed" } });
    expect(rowIds()).toEqual(["ppo_s1_beta"]);

    fireEvent.change(screen.getByLabelText("State"), { target: { value: "all" } });
    fireEvent.change(screen.getByLabelText("Experiment"), { target: { value: "experiment_1" } });
    expect(rowIds()).toEqual(["ppo_s0_alpha"]);
  });
});

describe("pinned comparison set", () => {
  it("allows six runs and prevents a seventh client-side", () => {
    let ids: string[] = [];
    for (let index = 0; index < MAX_COMPARE_RUNS; index += 1) {
      ids = togglePinnedId(ids, `run-${index}`);
    }
    expect(ids).toHaveLength(6);

    const seventh = togglePinnedId(ids, "run-6");
    expect(seventh).toHaveLength(6);
    expect(seventh).not.toContain("run-6");

    // Unpinning one makes room again.
    const removed = togglePinnedId(ids, "run-0");
    expect(togglePinnedId(removed, "run-6")).toContain("run-6");
  });

  it("disables the pin control at the limit with an explanatory tooltip", () => {
    const runs = ["r1", "r2", "r3", "r4", "r5", "r6", "r7"].map((run_id) => makeRun({ run_id }));
    render(
      <LeaderboardTable
        runs={runs}
        pinnedIds={["r1", "r2", "r3", "r4", "r5", "r6"]}
        onTogglePin={() => undefined}
      />,
    );

    const seventhPin = screen.getByLabelText("Pin r7 for comparison") as HTMLInputElement;
    expect(seventhPin.disabled).toBe(true);
    expect(seventhPin.getAttribute("title")).toContain("limited to 6");
  });

  it("round-trips the pinned ids through the URL query", () => {
    expect(parsePinnedIds("?run_ids=a,b,c")).toEqual(["a", "b", "c"]);
    expect(parsePinnedIds("?run_ids=a,,a,b")).toEqual(["a", "b"]);
    expect(serializePinnedIds("?state=done", ["a", "b"])).toBe("state=done&run_ids=a%2Cb");
    expect(serializePinnedIds("?run_ids=a&state=done", [])).toBe("state=done");
  });
});

describe("ProvenanceCard", () => {
  it("shows the full (untruncated) config hash and commit", () => {
    const configHash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const gitCommit = "abcdef0123456789abcdef0123456789abcdef01";
    const run = makeRun({
      run_id: "full-provenance",
      config_hash: configHash,
      git_commit: gitCommit,
      manifest: {
        run_id: "full-provenance",
        seed: 0,
        config_hash: configHash,
        git_commit: gitCommit,
        algorithm: "PPO",
        total_timesteps: 1000,
        best_reward: 1,
        checkpoint_path: "runs/experiment_1/seed_0/checkpoints/final.zip",
        library_versions: { "stable-baselines3": "2.3.0", numpy: "2.0.0" },
      },
    });

    render(<ProvenanceCard run={run} />);

    expect(screen.getByText(configHash)).toBeTruthy();
    expect(screen.getByText(gitCommit)).toBeTruthy();
    expect(screen.getByText(/stable-baselines3 2.3.0/)).toBeTruthy();
    expect(screen.getByText("runs/experiment_1/seed_0/checkpoints/final.zip")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Copy Config hash" })).toBeTruthy();
  });
});

describe("CSV export", () => {
  it("contains a header plus one row per visible run", () => {
    const runs = [
      makeRun({ run_id: "a", eval: makeEval({ sharpe: 1.5 }) }),
      makeRun({ run_id: "b", eval: makeEval({ sharpe: 0.5 }) }),
      makeRun({ run_id: "c,with-comma" }),
    ];
    const csv = runsToCsv(runs);
    const lines = csv.trimEnd().split("\n");

    expect(lines[0]).toBe(CSV_HEADER.join(","));
    expect(lines).toHaveLength(runs.length + 1);
    expect(lines[1]).toContain("a,experiment_1,0,done");
    // A comma in a run id is quoted so the CSV stays machine-readable.
    expect(lines[3]).toContain('"c,with-comma"');
  });

  it("labels each row with its pinned-rule rank, not the visible row index", () => {
    const runs = [
      makeRun({ run_id: "train-only" }),
      makeRun({ run_id: "low", eval: makeEval({ sharpe: 1 }) }),
      makeRun({ run_id: "high", eval: makeEval({ sharpe: 2 }) }),
    ];
    const lines = runsToCsv(runs).trimEnd().split("\n");

    // The visible order is preserved; the rank column reflects the pinned rule.
    expect(lines[1]).toMatch(/^3,train-only/);
    expect(lines[2]).toMatch(/^2,low/);
    expect(lines[3]).toMatch(/^1,high/);
  });
});

describe("PolicyFingerprint", () => {
  it("normalises each metric across the selected set", () => {
    expect(normalizeValues([0, 5, 10])).toEqual([0, 0.5, 1]);
    expect(normalizeValues([3, 3])).toEqual([0.5, 0.5]);
    expect(normalizeValues([1, null, 3])).toEqual([0, null, 1]);
    expect(normalizeValues([null])).toEqual([null]);
  });

  it("renders one axis per metric and a legend entry per run", () => {
    render(
      <PolicyFingerprint
        metrics={["Sharpe", "Total PnL"]}
        series={[
          { label: "run-a", values: [1, 100] },
          { label: "run-b", values: [2, 200] },
        ]}
      />,
    );

    expect(screen.getAllByTestId("fingerprint-metric")).toHaveLength(2);
    const legend = screen.getByLabelText("Fingerprint legend");
    expect(within(legend).getByText("run-a")).toBeTruthy();
    expect(within(legend).getByText("run-b")).toBeTruthy();
  });
});

describe("Leaderboard view", () => {
  it("loads runs, exports the visible ranking and lazily fetches a curve", async () => {
    mocks.getRuns.mockResolvedValue([
      makeRun({ run_id: "run-a", eval: makeEval({ sharpe: 2 }) }),
      makeRun({ run_id: "run-b", eval: makeEval({ sharpe: 1 }) }),
    ]);
    mocks.getCurve.mockResolvedValue({
      run_id: "run-a",
      timesteps: [1, 2],
      mean: [0.1, 0.2],
      std: [0, 0],
      per_seed: {},
      downsampled: false,
    });

    render(
      <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Leaderboard />
      </MemoryRouter>,
    );

    await screen.findByText("run-a");
    expect(rowIds()).toEqual(["run-a", "run-b"]);
    expect(screen.getByRole("button", { name: "Download CSV" })).toBeTruthy();

    fireEvent.click(screen.getByLabelText("Expand run-a"));
    await waitFor(() => expect(mocks.getCurve).toHaveBeenCalledWith("run-a"));
    await screen.findByTestId("provenance-card");
  });

  it("exports a header-only CSV when the visible set is empty", async () => {
    mocks.getRuns.mockResolvedValue([makeRun({ run_id: "run-a", eval: makeEval({ sharpe: 2 }) })]);
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    const RealBlob = globalThis.Blob;
    let parts: BlobPart[] | undefined;
    class CapturingBlob extends RealBlob {
      constructor(blobParts?: BlobPart[], options?: BlobPropertyBag) {
        super(blobParts, options);
        parts = blobParts;
      }
    }
    URL.createObjectURL = vi.fn(() => "blob:test") as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn() as unknown as typeof URL.revokeObjectURL;
    vi.stubGlobal("Blob", CapturingBlob);
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);

    try {
      render(
        <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Leaderboard />
        </MemoryRouter>,
      );
      await screen.findByText("run-a");

      fireEvent.change(screen.getByLabelText("Search run id"), { target: { value: "no-match" } });
      await waitFor(() =>
        expect(screen.getByTestId("leaderboard-count").textContent).toContain("0 of 1"),
      );
      fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));

      const text = String(parts?.[0] ?? "");
      expect(text).toBe(runsToCsv([]));
      expect(text).toBe(`${CSV_HEADER.join(",")}\n`);
      await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:test"));
    } finally {
      clickSpy.mockRestore();
      vi.stubGlobal("Blob", RealBlob);
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    }
  });
});

describe("Compare view", () => {
  it("calls the API with the pinned ids and renders one legend entry per run", async () => {
    const r1 = makeRun({ run_id: "run-a", eval: makeEval({ sharpe: 2, total_pnl: 10 }) });
    const r2 = makeRun({ run_id: "run-b", eval: makeEval({ sharpe: 1, total_pnl: 5 }) });
    mocks.getRuns.mockResolvedValue([r1, r2]);
    mocks.compare.mockResolvedValue({
      figures: { compare: { data: [{ x: [1], y: [1], name: "run-a" }], layout: {}, config: {} } },
      records: [r2, r1],
    });

    render(
      <MemoryRouter
        initialEntries={["/compare?run_ids=run-a,run-b"]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Compare />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(mocks.compare).toHaveBeenCalledWith({ run_ids: ["run-a", "run-b"], metric: "sharpe" }),
    );

    const legend = await screen.findByTestId("compare-legend");
    expect(within(legend).getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByTestId("plotly")).toBeTruthy();
    expect(screen.getAllByTestId("compare-metric-row")).toHaveLength(2);
  });

  it("clamps a URL with more than six pinned runs and reports the overflow", async () => {
    mocks.getRuns.mockResolvedValue([]);
    mocks.compare.mockResolvedValue({ figures: {}, records: [] });
    render(
      <MemoryRouter
        initialEntries={["/compare?run_ids=a,b,c,d,e,f,g"]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Compare />
      </MemoryRouter>,
    );

    await waitFor(() =>
      expect(mocks.compare).toHaveBeenCalledWith({
        run_ids: ["a", "b", "c", "d", "e", "f"],
        metric: "sharpe",
      }),
    );
    expect(screen.getByText(/Only the first 6 pinned runs/)).toBeTruthy();
  });

  it("uses the Rebalances axis when a record carries the optional extension", async () => {
    const r1 = makeRun({ run_id: "run-a", eval: makeEval({ sharpe: 1 }) }) as RunRecord & {
      n_rebalances: number;
    };
    r1.n_rebalances = 42;
    const r2 = makeRun({ run_id: "run-b", eval: makeEval({ sharpe: 2 }) });
    mocks.getRuns.mockResolvedValue([r1, r2]);
    mocks.compare.mockResolvedValue({ figures: {}, records: [r1, r2] });

    render(
      <MemoryRouter
        initialEntries={["/compare?run_ids=run-a,run-b"]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Compare />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Rebalances")).toBeTruthy();
    expect(screen.queryByText("Annualised return")).toBeNull();
  });

  it("reads a nested eval n_rebalances extension too", async () => {
    const r1 = makeRun({ run_id: "run-a", eval: makeEval({ sharpe: 1 }) }) as RunRecord & {
      eval: NonNullable<RunRecord["eval"]> & { n_rebalances: number };
    };
    r1.eval.n_rebalances = 7;
    const r2 = makeRun({ run_id: "run-b", eval: makeEval({ sharpe: 2 }) });
    mocks.getRuns.mockResolvedValue([r1, r2]);
    mocks.compare.mockResolvedValue({ figures: {}, records: [r1, r2] });

    render(
      <MemoryRouter
        initialEntries={["/compare?run_ids=run-a,run-b"]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Compare />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Rebalances")).toBeTruthy();
    expect(screen.queryByText("Annualised return")).toBeNull();
  });

  it("explains how to add more when fewer than two runs are pinned", async () => {
    mocks.getRuns.mockResolvedValue([]);
    render(
      <MemoryRouter
        initialEntries={["/compare?run_ids=only-one"]}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Compare />
      </MemoryRouter>,
    );

    expect(await screen.findByText("Pin one more run to compare")).toBeTruthy();
    expect(mocks.compare).not.toHaveBeenCalled();
  });
});
