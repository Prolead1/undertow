import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { AppRouter, MemoryRouter, useLocation } from "../router";

import {
  ApiError,
  apiUrl,
  compare,
  encodeRunId,
  getCurve,
  getFigure,
  getRollout,
  getRun,
  type TelemetryPoint,
} from "../api/client";
import {
  buildStreamUrl,
  mergeTelemetry,
  reconnectDelay,
  useLiveStream,
} from "../api/useLiveStream";
import { PlotlyFigure } from "../components/PlotlyFigure";
import { QueryLink } from "../components/QueryLink";
import { withPreservedSearch } from "../navigation";

const mocks = vi.hoisted(() => ({
  getRuns: vi.fn(),
  getTelemetry: vi.fn(),
  plotlyReact: vi.fn(),
  plotlyPurge: vi.fn(),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, getRuns: mocks.getRuns, getTelemetry: mocks.getTelemetry };
});

vi.mock("plotly.js-dist-min", () => ({
  default: { react: mocks.plotlyReact, purge: mocks.plotlyPurge },
}));

const HERE = dirname(fileURLToPath(import.meta.url));

function point(timestep: number, runId = "run-1"): TelemetryPoint {
  return { schema_version: 1, run_id: runId, seed: 0, timestep };
}

beforeEach(() => {
  mocks.getRuns.mockReset();
  mocks.getRuns.mockResolvedValue([]);
  mocks.getTelemetry.mockReset();
  mocks.plotlyReact.mockClear();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

// ---------------------------------------------------------------------------
// 1. API client: URL/params + typed errors
// ---------------------------------------------------------------------------
describe("api client", () => {
  it("builds URLs, preserves path aliases and serializes only provided params", () => {
    expect(apiUrl("/api/runs", { state: "running", refresh: true })).toBe(
      "/api/runs?state=running&refresh=true",
    );
    expect(apiUrl("/api/runs", { state: undefined, refresh: undefined })).toBe("/api/runs");
    expect(encodeRunId("experiment_1/seed_3")).toBe("experiment_1/seed_3");
    expect(encodeRunId("weird id?")).toBe("weird%20id%3F");
  });

  it("calls fetch with the expected URL and returns typed JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([]), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    // `useLiveStream` mocks `getRuns`; use the real module here so the URL/params
    // contract is still exercised end to end.
    const { getRuns: realGetRuns } =
      await vi.importActual<typeof import("../api/client")>("../api/client");
    const runs = await realGetRuns({ state: "running", refresh: true });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/runs?state=running&refresh=true");
    expect(runs).toEqual([]);
  });

  it("builds figure/curve/rollout URLs and the compare POST body", async () => {
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify({}), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await getCurve("experiment_1/seed_3");
    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/runs/experiment_1/seed_3/curve");

    await getFigure("run 1", "learning", { seed: 2, episode: 7, log_y: true, metric: "sharpe" });
    expect(fetchMock.mock.calls[1]?.[0]).toBe(
      "/api/runs/run%201/figures/learning?seed=2&episode=7&log_y=true&metric=sharpe",
    );

    await getFigure("run-1", "actions");
    expect(fetchMock.mock.calls[2]?.[0]).toBe("/api/runs/run-1/figures/actions");

    await getRollout("run/1", 3, 9, { max_points: 500 });
    expect(fetchMock.mock.calls[3]?.[0]).toBe("/api/rollouts/run/1/3/9?max_points=500");

    const body = { run_ids: ["a", "b"], metric: "sharpe" };
    await compare(body);
    expect(fetchMock.mock.calls[4]?.[0]).toBe("/api/compare");
    const init = fetchMock.mock.calls[4]?.[1] as RequestInit | undefined;
    expect(init).toMatchObject({ method: "POST", body: JSON.stringify(body) });
    expect(init?.headers).toMatchObject({
      Accept: "application/json",
      "Content-Type": "application/json",
    });
  });

  it("throws a typed ApiError on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() =>
        Promise.resolve(
          new Response(JSON.stringify({ detail: "unknown run 'nope'" }), {
            status: 404,
            headers: { "Content-Type": "application/json" },
          }),
        ),
      ),
    );

    await expect(getRun("nope")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      detail: "unknown run 'nope'",
    });
    await expect(getRun("nope")).rejects.toBeInstanceOf(ApiError);
  });

  it("converts network failures to a typed ApiError with status 0", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(getRun("run-1")).rejects.toMatchObject({ name: "ApiError", status: 0 });
  });
});

// ---------------------------------------------------------------------------
// 2. useLiveStream: merge/dedupe, degrade + REST fallback
// ---------------------------------------------------------------------------
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

  send(data: unknown): void {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(data) }));
  }

  close(): void {
    if (this.readyState === MockWebSocket.CLOSED) return;
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new CloseEvent("close"));
  }
}

describe("useLiveStream", () => {
  it("dedupes telemetry by (run_id, timestep) and sorts by timestep", () => {
    const merged = mergeTelemetry([point(2)], [point(1), point(2), point(3, "run-2"), point(1)]);
    expect(merged.map((p) => `${p.run_id}:${p.timestep}`)).toEqual([
      "run-1:1",
      "run-1:2",
      "run-2:3",
    ]);
  });

  it("caps reconnect backoff at 30 seconds", () => {
    expect(reconnectDelay(0)).toBe(1_000);
    expect(reconnectDelay(3)).toBe(8_000);
    expect(reconnectDelay(10)).toBe(30_000);
  });

  it("maps API base protocols to websocket and keeps wss", () => {
    vi.stubEnv("VITE_API_BASE", "https://api.example.com");
    expect(buildStreamUrl("run-1")).toBe("wss://api.example.com/api/stream?run_id=run-1");

    vi.stubEnv("VITE_API_BASE", "wss://api.example.com");
    expect(buildStreamUrl("run-1")).toBe("wss://api.example.com/api/stream?run_id=run-1");

    vi.stubEnv("VITE_API_BASE", "http://api.example.com/");
    expect(buildStreamUrl("run-1")).toBe("ws://api.example.com/api/stream?run_id=run-1");
  });

  it("merges live deltas, dedupes overlaps and enters degraded fallback on close", async () => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
    mocks.getTelemetry.mockResolvedValue({
      run_id: "run-1",
      status: null,
      points: [point(3)],
      next_offset: 3,
      done: false,
    });

    const { result, unmount } = renderHook(() => useLiveStream({ runId: "run-1" }));
    const socket = MockWebSocket.instances[0];
    expect(socket).toBeDefined();
    expect(socket?.url).toContain(buildStreamUrl("run-1"));
    expect(socket?.url).toContain("run_id=run-1");

    act(() => socket?.open());
    expect(result.current.status).toBe("live");

    act(() => {
      socket?.send({ type: "telemetry", run_id: "run-1", data: point(1) });
      socket?.send({ type: "telemetry", run_id: "run-1", data: point(1) });
      socket?.send({ type: "telemetry", run_id: "run-1", data: point(2) });
    });
    expect(result.current.telemetryByRun["run-1"]?.map((p) => p.timestep)).toEqual([1, 2]);

    await act(async () => {
      socket?.close();
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.status).toBe("degraded");
    expect(mocks.getTelemetry).toHaveBeenCalledWith(
      "run-1",
      expect.objectContaining({ since: 0, limit: 1_000, signal: expect.any(AbortSignal) }),
    );
    // The polled point is merged into the same buffer and the offset advances.
    expect(result.current.telemetryByRun["run-1"]?.map((p) => p.timestep)).toEqual([1, 2, 3]);

    // The fallback schedules a retry at telemetry_poll_seconds and tails from next_offset.
    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });
    expect(mocks.getTelemetry.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(mocks.getTelemetry.mock.calls[1]?.[0]).toBe("run-1");
    expect(mocks.getTelemetry.mock.calls[1]?.[1]).toMatchObject({ since: 3, limit: 1_000 });

    unmount();
  });

  it("stops REST polling when the socket reconnects mid-poll", async () => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);

    let resolveTail: ((value: unknown) => void) | undefined;
    let pollSignal: AbortSignal | undefined;
    mocks.getTelemetry.mockImplementationOnce(
      (_runId: string, params: { signal?: AbortSignal }) => {
        pollSignal = params.signal;
        return new Promise((resolve) => {
          resolveTail = resolve;
        });
      },
    );

    const { result, unmount } = renderHook(() => useLiveStream({ runId: "run-1" }));
    const first = MockWebSocket.instances[0];
    act(() => first?.open());
    act(() => first?.close());
    expect(result.current.status).toBe("degraded");

    // Reconnect fires while the REST poll is still in flight.
    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });
    const second = MockWebSocket.instances[1];
    expect(second).toBeDefined();
    act(() => second?.open());
    expect(result.current.status).toBe("live");
    // Reconnect aborts the in-flight fallback request so a hung fetch cannot
    // stall the next degraded cycle.
    expect(pollSignal?.aborted).toBe(true);

    await act(async () => {
      resolveTail?.({
        run_id: "run-1",
        status: null,
        points: [point(9)],
        next_offset: 9,
        done: false,
      });
      await Promise.resolve();
    });
    // A live socket owns the stream: the late poll must not reschedule.
    await act(async () => {
      vi.advanceTimersByTime(10_000);
      await Promise.resolve();
    });
    expect(mocks.getTelemetry).toHaveBeenCalledTimes(1);

    unmount();
  });

  it("keeps exactly one in-flight poll chain under sustained reconnect failure", async () => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);

    let inFlight = 0;
    let maxInFlight = 0;
    let resolveTail: ((value: unknown) => void) | undefined;
    mocks.getTelemetry.mockImplementation(() => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise((resolve) => {
        resolveTail = (value) => {
          inFlight -= 1;
          resolve(value);
        };
      });
    });

    const { result, unmount } = renderHook(() => useLiveStream({ runId: "run-1", pollMs: 1_000 }));
    const first = MockWebSocket.instances[0];
    act(() => first?.close());
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.status).toBe("degraded");
    expect(mocks.getTelemetry).toHaveBeenCalledTimes(1);

    // Sustained failure: every reconnect yields a socket that closes again while
    // the initial REST poll is still hanging.
    for (let attempt = 0; attempt < 6; attempt += 1) {
      await act(async () => {
        vi.advanceTimersByTime(30_000);
        await Promise.resolve();
      });
      const latest = MockWebSocket.instances[MockWebSocket.instances.length - 1];
      if (latest && latest.readyState !== MockWebSocket.CLOSED) {
        act(() => latest.close());
      }
    }

    expect(mocks.getTelemetry).toHaveBeenCalledTimes(1);
    expect(maxInFlight).toBe(1);

    // Releasing the hung poll must reschedule the single chain exactly once.
    await act(async () => {
      resolveTail?.({ run_id: "run-1", status: null, points: [], next_offset: 5, done: false });
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(inFlight).toBe(0);
    expect(mocks.getTelemetry).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("keeps tailing telemetry when the run-list poll fails", async () => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
    mocks.getRuns.mockRejectedValue(new Error("runs down"));
    mocks.getTelemetry.mockResolvedValue({
      run_id: "run-1",
      status: null,
      points: [point(4)],
      next_offset: 4,
      done: false,
    });

    const { result, unmount } = renderHook(() => useLiveStream({ runId: "run-1" }));
    const socket = MockWebSocket.instances[0];
    act(() => socket?.close());
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.status).toBe("degraded");
    expect(mocks.getTelemetry).toHaveBeenCalledTimes(1);
    expect(result.current.telemetryByRun["run-1"]?.map((p) => p.timestep)).toEqual([4]);

    unmount();
  });

  it("polls the run list without a selected run and never tails telemetry", async () => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);

    const { result, unmount } = renderHook(() => useLiveStream({ runId: null }));
    const socket = MockWebSocket.instances[0];
    expect(socket).toBeDefined();
    // A status-only stream must not carry a run_id query param.
    expect(socket?.url).not.toContain("run_id=");

    act(() => socket?.close());
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(result.current.status).toBe("degraded");

    const callsAfterClose = mocks.getRuns.mock.calls.length;
    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
      await Promise.resolve();
    });
    // The degraded fallback keeps the rail fresh even with no run selected…
    expect(mocks.getRuns.mock.calls.length).toBeGreaterThan(callsAfterClose);
    // …but there is no run to tail, so telemetry is never requested.
    expect(mocks.getTelemetry).not.toHaveBeenCalled();
    expect(result.current.telemetryByRun).toEqual({});

    unmount();
  });
});

// ---------------------------------------------------------------------------
// 3. Router: selection query params survive navigation
// ---------------------------------------------------------------------------
function RouterProbe() {
  const location = useLocation();
  return (
    <div>
      <span data-testid="location">{`${location.pathname}${location.search}`}</span>
      <QueryLink to="/replay">Replay</QueryLink>
    </div>
  );
}

describe("router", () => {
  it("preserves run_id/seed/episode across navigation", () => {
    render(
      <MemoryRouter initialEntries={["/live?run_id=run-1&seed=2&episode=7"]}>
        <RouterProbe />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByText("Replay"));
    expect(screen.getByTestId("location").textContent).toBe(
      "/replay?run_id=run-1&seed=2&episode=7",
    );
    expect(withPreservedSearch("/compare", "?run_id=r&seed=1&episode=0")).toBe(
      "/compare?run_id=r&seed=1&episode=0",
    );
  });

  it("renders the eager shell and resolves a lazily-loaded view", async () => {
    render(
      <MemoryRouter initialEntries={["/live"]}>
        <AppRouter />
      </MemoryRouter>,
    );

    // The App shell is eager: it renders before the route chunk resolves.
    expect(screen.getByText("Policy cockpit")).toBeTruthy();
    // The lazy `LiveCockpit` view resolves and renders its empty state.
    expect(await screen.findByText("Select a run")).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// 4. PlotlyFigure: empty spec is safe
// ---------------------------------------------------------------------------
describe("PlotlyFigure", () => {
  beforeEach(() => {
    mocks.plotlyReact.mockClear();
    mocks.plotlyPurge.mockClear();
  });

  it("renders an EmptyState (without throwing) for an empty spec", () => {
    render(<PlotlyFigure spec={{ data: [], layout: {}, config: {} }} ariaLabel="No traces" />);
    expect(screen.getByText("No figure data")).toBeTruthy();
    expect(screen.getByText("No traces")).toBeTruthy();
    expect(mocks.plotlyReact).not.toHaveBeenCalled();
  });

  it("re-plots only when the data payload changes", () => {
    const { rerender } = render(
      <PlotlyFigure
        spec={{ data: [{ x: [1], y: [2] }], layout: { title: "curve" }, config: {} }}
        ariaLabel="Learning curve"
      />,
    );
    expect(mocks.plotlyReact).toHaveBeenCalledTimes(1);

    // Same data content, fresh object/array identity: no re-plot.
    rerender(
      <PlotlyFigure
        spec={{ data: [{ x: [1], y: [2] }], layout: { title: "curve" }, config: {} }}
        ariaLabel="Learning curve"
      />,
    );
    expect(mocks.plotlyReact).toHaveBeenCalledTimes(1);

    // Real data change: one more plot with the new traces.
    rerender(
      <PlotlyFigure
        spec={{ data: [{ x: [1, 2], y: [2, 3] }], layout: { title: "curve" }, config: {} }}
        ariaLabel="Learning curve"
      />,
    );
    expect(mocks.plotlyReact).toHaveBeenCalledTimes(2);
    expect(mocks.plotlyReact).toHaveBeenLastCalledWith(
      expect.anything(),
      [{ x: [1, 2], y: [2, 3] }],
      expect.anything(),
      expect.anything(),
    );
    // Re-plotting must not purge (that would reset zoom/view state).
    expect(mocks.plotlyPurge).not.toHaveBeenCalled();
  });

  it("purges the Plotly instance on unmount", () => {
    const { unmount } = render(
      <PlotlyFigure spec={{ data: [{ x: [1], y: [2] }], layout: {}, config: {} }} />,
    );
    expect(mocks.plotlyPurge).not.toHaveBeenCalled();
    unmount();
    expect(mocks.plotlyPurge).toHaveBeenCalledTimes(1);
  });

  it("purges when the spec becomes empty", () => {
    const { rerender } = render(
      <PlotlyFigure spec={{ data: [{ x: [1], y: [2] }], layout: {}, config: {} }} />,
    );
    expect(mocks.plotlyPurge).not.toHaveBeenCalled();
    rerender(<PlotlyFigure spec={{ data: [], layout: {}, config: {} }} />);
    expect(screen.getByText("No figure data")).toBeTruthy();
    expect(mocks.plotlyPurge).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// 5. Design tokens: every CONTRACTS §9 token exists
// ---------------------------------------------------------------------------
describe("design tokens", () => {
  const TOKENS = [
    "bg-base",
    "bg-surface",
    "bg-elevated",
    "border",
    "text-primary",
    "text-muted",
    "grid",
    "accent",
    "accent-2",
    "bull",
    "bear",
    "sideways",
    "high-vol",
    "hold",
    "rebalance",
  ] as const;

  it("declares all §9 tokens in tokens.css", () => {
    const css = readFileSync(resolve(HERE, "../theme/tokens.css"), "utf8");
    for (const token of TOKENS) {
      expect(css, `missing --${token}`).toContain(`--${token}:`);
    }
    expect(css).toContain("color-scheme: dark");
  });
});
