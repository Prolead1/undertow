import { useEffect, useReducer, useRef } from "react";

import { getRuns, getTelemetry, type RunRecord, type RunStatus, type TelemetryPoint } from "./client";

export type ConnectionStatus = "connecting" | "live" | "degraded" | "closed";

export interface LiveState {
  status: ConnectionStatus;
  runs: RunRecord[];
  telemetryByRun: Record<string, TelemetryPoint[]>;
  lastStatus: RunStatus | null;
}

export interface LiveStreamOptions {
  /** Selected run; omitted means a status-only stream (run list + pings). */
  runId?: string | null;
  enabled?: boolean;
  /** REST-tail cadence while the WebSocket is down (CONTRACTS §7). */
  pollMs?: number;
  /** Exponential reconnect cap; the contract fixes this at 30 seconds. */
  maxBackoffMs?: number;
}

export const MAX_RECONNECT_MS = 30_000;
export const DEFAULT_POLL_MS = 1_000;
const RECONNECT_BASE_MS = 1_000;

interface StreamEnvelope {
  type: string;
  run_id: string | null;
  data: unknown;
  ts?: string;
}

const initialState: LiveState = {
  status: "connecting",
  runs: [],
  telemetryByRun: {},
  lastStatus: null,
};

type LiveAction =
  | { type: "connecting" }
  | { type: "live" }
  | { type: "degraded" }
  | { type: "closed" }
  | { type: "runs"; runs: RunRecord[] }
  | { type: "run_added"; run: RunRecord }
  | { type: "telemetry"; points: TelemetryPoint[] }
  | { type: "status"; status: RunStatus };

/** Stable identity for the CONTRACTS §8 live-merge rule. */
export function telemetryKey(point: Pick<TelemetryPoint, "run_id" | "timestep">): string {
  return `${point.run_id}\u0000${point.timestep}`;
}

/**
 * Append new telemetry, ignoring duplicate `(run_id, timestep)` points.
 * Returns a new sorted array so charts receive a monotonic timestep stream.
 */
export function mergeTelemetry(
  existing: readonly TelemetryPoint[],
  incoming: readonly TelemetryPoint[],
): TelemetryPoint[] {
  const seen = new Set(existing.map(telemetryKey));
  const merged = [...existing];
  for (const point of incoming) {
    const key = telemetryKey(point);
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(point);
  }
  merged.sort((a, b) => a.timestep - b.timestep || a.run_id.localeCompare(b.run_id));
  return merged;
}

/** Merge run lists by `run_id`, keeping the most recently received record. */
export function mergeRuns(existing: readonly RunRecord[], incoming: readonly RunRecord[]): RunRecord[] {
  const byId = new Map<string, RunRecord>();
  for (const run of existing) byId.set(run.run_id, run);
  for (const run of incoming) byId.set(run.run_id, run);
  return [...byId.values()];
}

function liveReducer(state: LiveState, action: LiveAction): LiveState {
  switch (action.type) {
    case "connecting":
      return { ...state, status: "connecting" };
    case "live":
      return { ...state, status: "live" };
    case "degraded":
      return { ...state, status: "degraded" };
    case "closed":
      return { ...state, status: "closed" };
    case "runs":
      return { ...state, runs: mergeRuns(state.runs, action.runs) };
    case "run_added":
      return { ...state, runs: mergeRuns(state.runs, [action.run]) };
    case "status":
      return { ...state, lastStatus: action.status };
    case "telemetry": {
      if (action.points.length === 0) return state;
      const grouped = new Map<string, TelemetryPoint[]>();
      for (const point of action.points) {
        const bucket = grouped.get(point.run_id);
        if (bucket) bucket.push(point);
        else grouped.set(point.run_id, [point]);
      }
      const telemetryByRun = { ...state.telemetryByRun };
      for (const [runId, points] of grouped) {
        telemetryByRun[runId] = mergeTelemetry(telemetryByRun[runId] ?? [], points);
      }
      return { ...state, telemetryByRun };
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isTelemetryPoint(value: unknown): value is TelemetryPoint {
  return (
    isRecord(value) &&
    typeof value.run_id === "string" &&
    typeof value.timestep === "number" &&
    typeof value.seed === "number"
  );
}

function isRunState(value: unknown): value is RunRecord["state"] {
  return value === "running" || value === "done" || value === "failed" || value === "unknown";
}

function isRunRecord(value: unknown): value is RunRecord {
  return (
    isRecord(value) &&
    typeof value.run_id === "string" &&
    typeof value.seed === "number" &&
    isRunState(value.state) &&
    typeof value.total_timesteps === "number" &&
    typeof value.latest_timestep === "number"
  );
}

function isRunStatus(value: unknown): value is RunStatus {
  return (
    isRecord(value) &&
    typeof value.run_id === "string" &&
    typeof value.seed === "number" &&
    typeof value.timestep === "number" &&
    typeof value.total_timesteps === "number" &&
    (value.state === "running" || value.state === "done" || value.state === "failed")
  );
}

function parseEnvelope(raw: string): StreamEnvelope | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || typeof parsed.type !== "string") return null;
    return {
      type: parsed.type,
      run_id: typeof parsed.run_id === "string" ? parsed.run_id : null,
      data: parsed.data,
    };
  } catch {
    return null;
  }
}

/** Delay before reconnect attempt `attempt` (0-based), capped by the contract. */
export function reconnectDelay(attempt: number, maxBackoffMs = MAX_RECONNECT_MS): number {
  return Math.min(RECONNECT_BASE_MS * 2 ** Math.max(0, attempt), maxBackoffMs);
}

/** Build the WebSocket URL, respecting `VITE_API_BASE` and http→ws conversion. */
export function buildStreamUrl(runId?: string | null): string {
  const configured = import.meta.env.VITE_API_BASE ?? "";
  const base = configured
    ? new URL(configured, typeof window === "undefined" ? "http://localhost" : window.location.href)
    : new URL(typeof window === "undefined" ? "http://localhost" : window.location.origin);
  base.protocol = base.protocol === "https:" || base.protocol === "wss:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/+$/, "")}/api/stream`;
  base.search = "";
  base.hash = "";
  if (runId) base.searchParams.set("run_id", runId);
  return base.toString();
}

function applyEnvelope(envelope: StreamEnvelope, dispatch: (action: LiveAction) => void): void {
  switch (envelope.type) {
    case "hello": {
      const data = envelope.data;
      if (isRecord(data) && Array.isArray(data.runs)) {
        dispatch({ type: "runs", runs: data.runs.filter(isRunRecord) });
      }
      break;
    }
    case "run_added":
      if (isRunRecord(envelope.data)) dispatch({ type: "run_added", run: envelope.data });
      break;
    case "telemetry":
      if (isTelemetryPoint(envelope.data)) {
        dispatch({ type: "telemetry", points: [envelope.data] });
      }
      break;
    case "status":
      if (isRunStatus(envelope.data)) dispatch({ type: "status", status: envelope.data });
      break;
    case "ping":
    case "error":
      // `error` leaves the last good state intact (CONTRACTS §5.2); the UI can
      // surface the message from its own toast channel in U08.
      break;
    default:
      break;
  }
}

/**
 * Own one WebSocket to `/api/stream` for the selected run.
 *
 * On close/error the hook marks the connection `degraded`, polls
 * `GET .../telemetry?since=` every `pollMs`, and reconnects with exponential
 * backoff (1 s, 2 s, 4 s … capped at 30 s). Telemetry is merged by
 * `(run_id, timestep)` so a reconnect/fallback overlap cannot duplicate points.
 */
export function useLiveStream(options: LiveStreamOptions = {}): LiveState {
  const {
    runId = null,
    enabled = true,
    pollMs = DEFAULT_POLL_MS,
    maxBackoffMs = MAX_RECONNECT_MS,
  } = options;
  const [state, dispatch] = useReducer(liveReducer, initialState);
  const sinceRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      dispatch({ type: "closed" });
      return;
    }

    let disposed = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let socketIsOpen = false;
    let attempt = 0;
    let pollInFlight = false;
    // Bumped whenever the polling chain is torn down (reconnect/unmount) so a
    // late `pollOnce` continuation can never reschedule a dead chain.
    let pollGeneration = 0;
    let pollController: AbortController | null = null;
    sinceRef.current = 0;

    const clearReconnectTimer = (): void => {
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const stopPolling = (): void => {
      pollGeneration += 1;
      pollInFlight = false;
      if (pollController !== null) {
        pollController.abort();
        pollController = null;
      }
      if (pollTimer !== null) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
    };

    /** One-shot run-list load so the rail works without a WebSocket `hello`. */
    const fetchRuns = async (signal: AbortSignal): Promise<void> => {
      try {
        const runs = await getRuns({ signal });
        if (disposed || signal.aborted) return;
        dispatch({ type: "runs", runs });
      } catch {
        // Non-fatal: the WS hello or the degraded poll chain will fill the rail.
      }
    };

    /**
     * One tick of the degraded fallback. `generation`/`signal` tie the tick to
     * the chain that started it while `pollInFlight` (checked by `startPolling`)
     * guarantees at most one outstanding request even under sustained reconnect
     * failures. `/api/runs` is polled even without a selected run, so the run
     * rail keeps working and the `degraded` badge is truthful.
     */
    const pollOnce = async (generation: number, signal: AbortSignal): Promise<void> => {
      pollTimer = null;
      if (disposed || generation !== pollGeneration || socketIsOpen) return;
      pollInFlight = true;
      let failed = false;
      const stale = (): boolean =>
        disposed || generation !== pollGeneration || socketIsOpen || signal.aborted;
      try {
        // The run list and the telemetry tail are polled independently: a
        // failing `/api/runs` must not stop telemetry (and vice versa).
        try {
          const runs = await getRuns({ signal });
          if (stale()) return;
          dispatch({ type: "runs", runs });
        } catch {
          failed = true;
        }
        if (runId && !stale()) {
          try {
            const tail = await getTelemetry(runId, {
              since: sinceRef.current,
              limit: 1_000,
              signal,
            });
            if (stale()) return;
            sinceRef.current = Math.max(sinceRef.current, tail.next_offset);
            if (tail.points.length > 0) dispatch({ type: "telemetry", points: tail.points });
            if (tail.status) dispatch({ type: "status", status: tail.status });
            if (tail.done) return;
          } catch {
            failed = true;
          }
        }
      } finally {
        if (generation === pollGeneration) pollInFlight = false;
      }
      if (failed && !stale()) dispatch({ type: "degraded" });
      if (!disposed && generation === pollGeneration && !socketIsOpen) {
        pollTimer = setTimeout(() => void pollOnce(generation, signal), pollMs);
      }
    };

    const startPolling = (): void => {
      dispatch({ type: "degraded" });
      // Single owner of the chain: no second tick while one is outstanding or
      // already scheduled, however many `close` events fire.
      if (disposed || socketIsOpen || pollInFlight || pollTimer !== null) return;
      pollGeneration += 1;
      pollController = new AbortController();
      void pollOnce(pollGeneration, pollController.signal);
    };

    const scheduleReconnect = (): void => {
      if (disposed || reconnectTimer !== null) return;
      const delay = reconnectDelay(attempt, maxBackoffMs);
      attempt += 1;
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, delay);
    };

    const connect = (): void => {
      if (disposed) return;
      dispatch({ type: "connecting" });
      let ws: WebSocket;
      try {
        ws = new WebSocket(buildStreamUrl(runId));
      } catch {
        startPolling();
        scheduleReconnect();
        return;
      }
      socket = ws;

      ws.onopen = (): void => {
        if (disposed) return;
        socketIsOpen = true;
        attempt = 0;
        stopPolling();
        dispatch({ type: "live" });
      };
      ws.onmessage = (event: MessageEvent): void => {
        if (disposed || typeof event.data !== "string") return;
        const envelope = parseEnvelope(event.data);
        if (envelope) applyEnvelope(envelope, dispatch);
      };
      ws.onerror = (): void => {
        if (disposed) return;
        // onclose follows in real browsers; closing here also covers mocks that
        // only emit an error before the next reconnect.
        if (ws.readyState !== WebSocket.CLOSED) ws.close();
      };
      ws.onclose = (): void => {
        if (disposed) return;
        if (socket === ws) socket = null;
        socketIsOpen = false;
        startPolling();
        scheduleReconnect();
      };
    };

    // Populate the run rail even if the socket never connects (CONTRACTS §7).
    const initialController = new AbortController();
    void fetchRuns(initialController.signal);

    dispatch({ type: "connecting" });
    connect();

    return () => {
      disposed = true;
      initialController.abort();
      clearReconnectTimer();
      stopPolling();
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
        socket = null;
      }
    };
  }, [runId, enabled, pollMs, maxBackoffMs]);

  return state;
}
