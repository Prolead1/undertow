import type { components } from "./types";

/** Typed aliases over the OpenAPI-generated wire models. */
export type RunRecord = components["schemas"]["RunRecord"];
export type RunStatus = components["schemas"]["RunStatusModel"];
export type TelemetryPoint = components["schemas"]["TelemetryPoint"];
export type TelemetryTail = components["schemas"]["TelemetryTail"];
export type LearningCurve = components["schemas"]["LearningCurve"];
export type FigureSpec = components["schemas"]["FigureSpecModel"];
export type RolloutSummary = components["schemas"]["RolloutSummary"];
export type RolloutSeries = components["schemas"]["RolloutSeries"];
export type CompareRequest = components["schemas"]["CompareRequest"];
export type CompareResponse = components["schemas"]["CompareResponse"];

type QueryValue = string | number | boolean | null | undefined;
type QueryParams = Record<string, QueryValue>;

export interface RequestOptions {
  query?: QueryParams;
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
}

/** Typed error raised for every non-2xx response (and network failures). */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  readonly url: string;

  constructor(message: string, options: { status: number; detail: unknown; url: string }) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.detail = options.detail;
    this.url = options.url;
  }
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

/** Build an API URL; exported so the URL contract is directly testable. */
export function apiUrl(path: string, query?: QueryParams): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${API_BASE}${normalizedPath}`;
  if (!query) return url;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) continue;
    search.append(key, String(value));
  }
  const serialized = search.toString();
  return serialized ? `${url}?${serialized}` : url;
}

/**
 * A run id may be a manifest id or the CONTRACTS §7 `<experiment>/seed_<n>`
 * alias. Encode each segment so an alias stays a path (not a query delimiter)
 * while a literal slash cannot break the route.
 */
export function encodeRunId(runId: string): string {
  return runId
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = apiUrl(path, options.query);
  const headers: Record<string, string> = { Accept: "application/json" };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";

  let response: Response;
  try {
    response = await fetch(url, {
      method: options.method ?? "GET",
      headers,
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options.signal,
    });
  } catch (cause) {
    throw new ApiError(`request failed: ${String(cause)}`, {
      status: 0,
      detail: cause,
      url,
    });
  }

  if (!response.ok) {
    let detail: unknown = response.statusText;
    try {
      const payload: unknown = await response.json();
      if (payload && typeof payload === "object" && "detail" in payload) {
        detail = (payload as { detail: unknown }).detail;
      } else {
        detail = payload;
      }
    } catch {
      // A non-JSON error body is still surfaced through the typed error.
    }
    const suffix = typeof detail === "string" ? `: ${detail}` : "";
    throw new ApiError(`request to ${url} failed (${response.status}${suffix})`, {
      status: response.status,
      detail,
      url,
    });
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export interface GetRunsParams {
  state?: RunRecord["state"];
  refresh?: boolean;
  signal?: AbortSignal;
}

export function getRuns(params: GetRunsParams = {}): Promise<RunRecord[]> {
  return request<RunRecord[]>("/api/runs", {
    query: { state: params.state, refresh: params.refresh },
    signal: params.signal,
  });
}

export function getRun(runId: string): Promise<RunRecord> {
  return request<RunRecord>(`/api/runs/${encodeRunId(runId)}`);
}

export interface TelemetryParams {
  since?: number;
  limit?: number;
  signal?: AbortSignal;
}

export function getTelemetry(runId: string, params: TelemetryParams = {}): Promise<TelemetryTail> {
  return request<TelemetryTail>(`/api/runs/${encodeRunId(runId)}/telemetry`, {
    query: { since: params.since, limit: params.limit },
    signal: params.signal,
  });
}

export function getCurve(runId: string): Promise<LearningCurve> {
  return request<LearningCurve>(`/api/runs/${encodeRunId(runId)}/curve`);
}

export interface FigureParams {
  seed?: number;
  episode?: number;
  log_y?: boolean;
  metric?: string;
}

export function getFigure(runId: string, name: string, params: FigureParams = {}): Promise<FigureSpec> {
  return request<FigureSpec>(`/api/runs/${encodeRunId(runId)}/figures/${encodeURIComponent(name)}`, {
    query: {
      seed: params.seed,
      episode: params.episode,
      log_y: params.log_y,
      metric: params.metric,
    },
  });
}

export function listRollouts(runId: string): Promise<RolloutSummary[]> {
  return request<RolloutSummary[]>(`/api/runs/${encodeRunId(runId)}/rollouts`);
}

export interface RolloutParams {
  max_points?: number;
}

export function getRollout(
  runId: string,
  seed: number,
  episode: number,
  params: RolloutParams = {},
): Promise<RolloutSeries> {
  return request<RolloutSeries>(
    `/api/rollouts/${encodeRunId(runId)}/${encodeURIComponent(String(seed))}/${encodeURIComponent(String(episode))}`,
    { query: { max_points: params.max_points } },
  );
}

export function compare(body: CompareRequest): Promise<CompareResponse> {
  return request<CompareResponse>("/api/compare", { method: "POST", body });
}
