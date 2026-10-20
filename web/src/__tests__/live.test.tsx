import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "../router";

import type {
  FigureSpec,
  LearningCurve as LearningCurvePayload,
  RunRecord,
  TelemetryPoint,
} from "../api/client";
import LiveCockpit from "../views/LiveCockpit";

const mocks = vi.hoisted(() => ({
  getRuns: vi.fn(),
  getTelemetry: vi.fn(),
  getCurve: vi.fn(),
  getFigure: vi.fn(),
  plotlySpec: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getRuns: mocks.getRuns,
    getTelemetry: mocks.getTelemetry,
    getCurve: mocks.getCurve,
    getFigure: mocks.getFigure,
  };
});

vi.mock("../components/PlotlyFigure", () => ({
  PlotlyFigure: (props: { spec: unknown; ariaLabel?: string }) => {
    mocks.plotlySpec(props.spec);
    return (
      <div
        data-testid="plotly-figure"
        data-spec={JSON.stringify(props.spec)}
        aria-label={props.ariaLabel}
      />
    );
  },
}));

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  open(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  send(type: string, data: unknown): void {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify({ type, run_id: "run-1", data }) }),
    );
  }

  close(): void {
    if (this.readyState === MockWebSocket.CLOSED) return;
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent("close"));
  }
}

const RUN: RunRecord = {
  run_id: "run-1",
  experiment: "experiment_1",
  seed: 0,
  state: "running",
  updated_at: "2026-10-19T18:12:00Z",
  total_timesteps: 1_000,
  latest_timestep: 100,
  config_hash: "abcdef0123456789",
  git_commit: "1234567890abcdef",
  manifest: null,
  eval: null,
  has_telemetry: true,
  has_rollouts: false,
  telemetry_points: 2,
};

const CURVE: LearningCurvePayload = {
  run_id: "run-1",
  timesteps: [0, 100],
  mean: [1, 2],
  std: [0, 0.1],
  per_seed: { "0": [1, 2] },
  downsampled: false,
};

const EMPTY_FIGURE: FigureSpec = { data: [], layout: {}, config: {} };

function diagnosticFigure(metric: string): FigureSpec {
  return {
    data: [
      {
        type: "scatter",
        mode: "lines",
        name: metric.replace(/_/g, " "),
        x: [0, 100],
        y: metric === "value_loss" ? [4, 3] : [0.1, 0.2],
      },
    ],
    layout: { xaxis: { title: "timestep" }, yaxis: { title: metric } },
    config: {},
  };
}

const ACTIONS_FIGURE: FigureSpec = {
  data: [{ type: "heatmap", x: [1, 2], y: [-1, 0, 1], z: [[1, 0], [0, 2], [1, 1]] }],
  layout: {},
  config: {},
};

function telemetry(timestep: number, reward: number): TelemetryPoint {
  return {
    schema_version: 1,
    run_id: "run-1",
    seed: 0,
    timestep,
    ep_rew_mean: reward,
    train: {
      policy_loss: 0.1,
      value_loss: 3,
      entropy_loss: -1,
      approx_kl: 0.01,
      clip_fraction: 0.1,
      explained_variance: 0.4,
    },
  };
}

function statusFrame(
  state: "running" | "done" | "failed",
  timestep: number,
  error: string | null = null,
): Record<string, unknown> {
  return {
    schema_version: 1,
    run_id: "run-1",
    seed: 0,
    state,
    timestep,
    total_timesteps: 1_000,
    updated_at: "2026-10-19T18:20:00Z",
    pid: 42,
    error,
  };
}

function renderCockpit(entry = "/live?run_id=run-1"): ReturnType<typeof render> {
  return render(
    <MemoryRouter
      initialEntries={[entry]}
    >
      <LiveCockpit />
    </MemoryRouter>,
  );
}

async function openStream(): Promise<MockWebSocket> {
  await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(0));
  const socket = MockWebSocket.instances[0];
  if (!socket) throw new Error("no WebSocket was opened");
  act(() => socket.open());
  return socket;
}

function learningSpec(): Record<string, unknown> {
  const element = screen.getByLabelText(/Learning curve with \d+ points/);
  return JSON.parse(element.getAttribute("data-spec") ?? "{}") as Record<string, unknown>;
}

function diagnosticSpec(label: string): Record<string, unknown> {
  const element = screen.getByLabelText(`${label} over timesteps`);
  return JSON.parse(element.getAttribute("data-spec") ?? "{}") as Record<string, unknown>;
}

function firstTrace(spec: Record<string, unknown>): Record<string, unknown> {
  const data = spec["data"] as Array<Record<string, unknown>>;
  const first = data[0];
  if (!first) throw new Error("missing first trace");
  return first;
}

function trace(spec: Record<string, unknown>, name: string): Record<string, unknown> {
  const data = spec["data"] as Array<Record<string, unknown>>;
  const found = data.find((entry) => entry["name"] === name);
  if (!found) throw new Error(`missing trace ${name}`);
  return found;
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
  MockWebSocket.instances = [];

  mocks.getRuns.mockReset();
  mocks.getRuns.mockResolvedValue([RUN]);
  mocks.getTelemetry.mockReset();
  mocks.getTelemetry.mockResolvedValue({
    run_id: "run-1",
    status: null,
    points: [],
    next_offset: 0,
    done: false,
  });
  mocks.getCurve.mockReset();
  mocks.getCurve.mockResolvedValue(CURVE);
  mocks.getFigure.mockReset();
  mocks.getFigure.mockImplementation(
    async (_runId: string, name: string, params: { metric?: string } = {}) => {
      if (name === "diagnostics") return diagnosticFigure(params.metric ?? "policy_loss");
      if (name === "actions") return ACTIONS_FIGURE;
      return EMPTY_FIGURE;
    },
  );
  mocks.plotlySpec.mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("LiveCockpit", () => {
  it("renders the run state and flips to done when a status frame arrives", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByTestId("run-state").dataset.state).toBe("running"),
    );

    const socket = await openStream();
    act(() =>
      socket.send("status", {
        schema_version: 1,
        run_id: "run-1",
        seed: 0,
        state: "done",
        timestep: 1_000,
        total_timesteps: 1_000,
        updated_at: "2026-10-19T18:20:00Z",
        pid: 42,
        error: null,
      }),
    );

    await waitFor(() => {
      expect(screen.getByTestId("run-state").dataset.state).toBe("done");
      expect(screen.getByTestId("run-state").textContent).toContain("done");
    });
    expect(screen.getByText(/1,000 \/ 1,000/)).toBeTruthy();
  });

  it("appends a live learning-curve point and dedupes a repeated telemetry frame", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const socket = await openStream();
    act(() => socket.send("telemetry", telemetry(200, 3)));
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy(),
    );

    let mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100, 200]);
    expect(mean["y"]).toEqual([1, 2, 3]);

    act(() => socket.send("telemetry", telemetry(200, 3)));
    mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100, 200]);
    expect(mean["y"]).toEqual([1, 2, 3]);
    expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy();
  });

  it("renders 'not reported' for a null diagnostic metric instead of NaN", async () => {
    mocks.getFigure.mockImplementation(
      async (_runId: string, name: string, params: { metric?: string } = {}) => {
        if (name === "diagnostics" && params.metric === "policy_loss") return EMPTY_FIGURE;
        if (name === "diagnostics") return diagnosticFigure(params.metric ?? "value_loss");
        return ACTIONS_FIGURE;
      },
    );

    renderCockpit();
    const panel = await screen.findByTestId("diagnostic-policy_loss");
    await waitFor(() => expect(within(panel).getByText("not reported")).toBeTruthy());
    expect(within(panel).queryByText("NaN")).toBeNull();
  });

  it("shows an EmptyState with the training command when a run has no telemetry", async () => {
    mocks.getRuns.mockResolvedValue([
      { ...RUN, has_telemetry: false, telemetry_points: 0, latest_timestep: 0 },
    ]);

    renderCockpit();
    expect(await screen.findByText("No telemetry for this run")).toBeTruthy();
    expect(screen.getByTestId("training-command").textContent).toContain("undertow-sim train");
  });

  it("shows a degraded banner while retaining the points already received", async () => {
    const view = renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const socket = await openStream();
    act(() => socket.send("telemetry", telemetry(200, 3)));
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy(),
    );

    act(() => socket.close());
    expect(await screen.findByText(/Live stream degraded/i)).toBeTruthy();
    expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy();
    const mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100, 200]);
    view.unmount();
  });

  it("freezes the learning curve at a done status and discards later telemetry", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const socket = await openStream();
    act(() => socket.send("status", statusFrame("done", 1_000)));
    await waitFor(() => expect(screen.getByTestId("run-state").dataset.state).toBe("done"));

    // A frame inside the cutoff proves telemetry still flows after the terminal
    // status; the out-of-range frame must then be discarded.
    await act(async () => {
      socket.send("telemetry", telemetry(900, 5));
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy(),
    );

    await act(async () => {
      socket.send("telemetry", telemetry(2_000, 99));
    });

    // Pre-terminal point count is unchanged and timestep 2000 never lands.
    expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy();
    const mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100, 900]);
    expect(mean["x"]).not.toContain(2_000);
  });

  it("freezes the learning curve at a failed status and discards later telemetry", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const socket = await openStream();
    act(() => socket.send("status", statusFrame("failed", 500, "boom")));
    await waitFor(() => expect(screen.getByTestId("run-state").dataset.state).toBe("failed"));
    expect(screen.getByText("boom")).toBeTruthy();

    await act(async () => {
      socket.send("telemetry", telemetry(2_000, 99));
    });

    expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy();
    const mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100]);
    expect(mean["x"]).not.toContain(2_000);
  });

  it("filters telemetry against the registry latest_timestep when no status frame arrived", async () => {
    mocks.getRuns.mockResolvedValue([
      { ...RUN, state: "done", latest_timestep: 100, has_telemetry: true },
    ]);

    renderCockpit();
    await waitFor(() => expect(screen.getByTestId("run-state").dataset.state).toBe("done"));
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const socket = await openStream();
    await act(async () => {
      socket.send("telemetry", telemetry(2_000, 99));
    });

    expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy();
    const mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100]);
  });

  it("retroactively freezes the curve when a terminal status arrives after out-of-range telemetry", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const socket = await openStream();
    // Out-of-range telemetry lands before the terminal status is known...
    await act(async () => {
      socket.send("telemetry", telemetry(2_000, 99));
    });
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 3 points")).toBeTruthy(),
    );

    // ...then the terminal status proves it should never have been kept.
    act(() => socket.send("status", statusFrame("done", 1_000)));
    await waitFor(() => expect(screen.getByTestId("run-state").dataset.state).toBe("done"));

    expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy();
    const mean = trace(learningSpec(), "mean");
    expect(mean["x"]).toEqual([0, 100]);
  });

  it("renders the action heatmap with its legend when the figure has data", async () => {
    renderCockpit();
    expect(
      await screen.findByLabelText("Action distribution heatmap over center offset and width"),
    ).toBeTruthy();
    expect(screen.getByText(/Grid: center offset/)).toBeTruthy();
  });

  it("shows an EmptyState naming the run when the actions figure fails", async () => {
    mocks.getFigure.mockImplementation(
      async (_runId: string, name: string, params: { metric?: string } = {}) => {
        if (name === "actions") throw new Error("boom");
        if (name === "diagnostics") return diagnosticFigure(params.metric ?? "policy_loss");
        return EMPTY_FIGURE;
      },
    );

    renderCockpit();
    expect(await screen.findByText("No action distribution yet")).toBeTruthy();
    expect(screen.getByText(/actions figure failed for run-1/).textContent).toContain("run-1");
  });

  it("adds a per-seed trace and switches to log y when toggled", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );

    const best = trace(learningSpec(), "best mean");
    expect(best["x"]).toEqual([100]);
    const names = (learningSpec()["data"] as Array<Record<string, unknown>>).map(
      (entry) => entry["name"],
    );
    expect(names).toContain("best mean");
    expect(names).not.toContain("checkpoint");

    fireEvent.click(screen.getByRole("button", { name: "Per-seed" }));
    await waitFor(() => {
      const seedTrace = trace(learningSpec(), "seed 0");
      expect(seedTrace["x"]).toEqual([0, 100]);
    });

    fireEvent.click(screen.getByRole("button", { name: "Log y" }));
    await waitFor(() => {
      const layout = learningSpec()["layout"] as Record<string, unknown>;
      const yaxis = layout["yaxis"] as Record<string, unknown>;
      expect(yaxis["type"]).toBe("log");
    });
  });

  it("appends telemetry to the diagnostics trace without duplicating repeats", async () => {
    renderCockpit();
    await waitFor(() =>
      expect(screen.getByLabelText("Learning curve with 2 points")).toBeTruthy(),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("policy loss over timesteps")).toBeTruthy(),
    );

    const socket = await openStream();
    await act(async () => {
      socket.send("telemetry", telemetry(200, 3));
    });

    await waitFor(() => {
      const series = firstTrace(diagnosticSpec("policy loss"));
      expect(series["x"]).toEqual([0, 100, 200]);
    });

    await act(async () => {
      socket.send("telemetry", telemetry(200, 3));
    });
    const series = firstTrace(diagnosticSpec("policy loss"));
    expect(series["x"]).toEqual([0, 100, 200]);
    expect(series["y"]).toEqual([0.1, 0.2, 0.1]);
  });

  it("prompts the user to select a run when no run_id is present", async () => {
    renderCockpit("/live");
    await waitFor(() => expect(screen.getByText("Select a run")).toBeTruthy());
    // Let the initial run-list fetch settle inside act().
    await act(async () => {});
  });

  it("waits for run metadata when the run is not in the registry", async () => {
    mocks.getRuns.mockResolvedValue([]);
    renderCockpit();
    expect(await screen.findByText("Waiting for run metadata")).toBeTruthy();
    expect(screen.getByText(/run-1 is not in the registry/)).toBeTruthy();
  });
});
