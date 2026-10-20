import type { RunRecord, RunStatus, TelemetryPoint } from "../../api/client";
import type { ConnectionStatus } from "../../api/useLiveStream";
import { Panel } from "../Panel";
import { Stat } from "../Stat";
import { cn } from "../cn";

export interface RunHeaderProps {
  run: RunRecord;
  status: RunStatus | null;
  connection: ConnectionStatus;
  points: readonly TelemetryPoint[];
}

export type DisplayState = RunRecord["state"];

const STATE_STYLE: Record<DisplayState, { dot: string; text: string; label: string }> = {
  running: {
    dot: "bg-accent animate-pulse",
    text: "border-accent/50 text-accent",
    label: "running",
  },
  done: { dot: "bg-bull", text: "border-bull/50 text-bull", label: "done" },
  failed: { dot: "bg-bear", text: "border-bear/60 text-bear", label: "failed" },
  unknown: { dot: "bg-sideways", text: "border-border text-text-muted", label: "unknown" },
};

/** Prefer the live status frame over the registry snapshot while the stream is active. */
function displayState(
  run: Pick<RunRecord, "state">,
  status: RunStatus | null,
): DisplayState {
  if (!status) return run.state;
  return status.state;
}

function shortHash(value: string | null | undefined, length = 8): string {
  if (!value) return "—";
  return value.slice(0, length);
}

function formatTimesteps(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return Math.max(0, Math.trunc(value)).toLocaleString();
}

/** Compact wall-clock duration; sub-minute values keep one decimal. */
function formatDuration(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const whole = Math.round(seconds);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

function latestElapsed(points: readonly TelemetryPoint[]): number | null {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const point = points[index];
    const elapsed = point?.time?.time_elapsed ?? point?.wall_time_s;
    if (typeof elapsed === "number" && Number.isFinite(elapsed)) return elapsed;
  }
  return null;
}

/** Run identity, provenance, live state and throughput for the cockpit. */
export function RunHeader({ run, status, connection, points }: RunHeaderProps) {
  const state = displayState(run, status);
  const presentation = STATE_STYLE[state];
  const latestTimestep = status?.timestep ?? run.latest_timestep;
  const totalTimesteps = status?.total_timesteps ?? run.total_timesteps;
  const elapsed = latestElapsed(points);
  const latestPoint = points.length > 0 ? points[points.length - 1] : null;
  const total = totalTimesteps > 0 ? Math.min(1, latestTimestep / totalTimesteps) : 0;
  const configHash = run.config_hash ?? run.manifest?.config_hash ?? null;
  const gitCommit = run.git_commit ?? run.manifest?.git_commit ?? null;

  return (
    <Panel
      title={
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate">{run.run_id}</span>
          <span
            data-testid="run-state"
            data-state={state}
            role="status"
            aria-label={`Run state: ${presentation.label}`}
            className={cn(
              "inline-flex shrink-0 items-center gap-1.5 rounded-full border bg-bg-elevated px-2 py-0.5 text-[11px] font-medium capitalize",
              presentation.text,
            )}
          >
            <span className={cn("size-1.5 rounded-full", presentation.dot)} aria-hidden />
            {presentation.label}
          </span>
        </span>
      }
      subtitle={
        <span className="tabular">
          {run.experiment} · seed {run.seed} · {connection === "live" ? "streaming" : connection}
        </span>
      }
      bodyClassName="grid gap-4 sm:grid-cols-2 xl:grid-cols-5"
    >
      <Stat
        label="Timesteps"
        value={`${formatTimesteps(latestTimestep)} / ${formatTimesteps(totalTimesteps)}`}
        hint={`${(total * 100).toFixed(1)}% complete`}
        tone={state === "failed" ? "bear" : state === "done" ? "bull" : "accent"}
      />
      <Stat
        label="Elapsed"
        value={formatDuration(elapsed)}
        hint={latestPoint ? `last point @ ${formatTimesteps(latestPoint.timestep)}` : "waiting for telemetry"}
      />
      <Stat
        label="Config"
        value={<span className="font-mono text-base">{shortHash(configHash, 12)}</span>}
        hint={configHash ? "config_hash" : "not reported"}
      />
      <Stat
        label="Git"
        value={<span className="font-mono text-base">{shortHash(gitCommit)}</span>}
        hint={gitCommit ? "git_commit" : "not reported"}
      />
      <Stat
        label="Telemetry"
        value={points.length.toLocaleString()}
        hint="points received this session"
        tone="muted"
      />
      {state === "failed" && status?.error && (
        <p className="sm:col-span-2 xl:col-span-5 rounded-control border border-bear/40 bg-bear/10 px-3 py-2 text-xs text-bear">
          {status.error}
        </p>
      )}
    </Panel>
  );
}
