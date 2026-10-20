import { useMemo } from "react";
import { useLocation } from "../router";

import { useLiveStream } from "../api/useLiveStream";
import { EmptyState } from "../components/EmptyState";
import { ActionHeatmap } from "../components/live/ActionHeatmap";
import { DiagnosticsGrid } from "../components/live/DiagnosticsGrid";
import { LearningCurve } from "../components/live/LearningCurve";
import { RunHeader } from "../components/live/RunHeader";

const TRAINING_COMMAND =
  "uv run undertow-sim train --config configs/sim_default.toml --output runs/experiment_1/";

/** U08 — live telemetry cockpit for a single selected run. */
export default function LiveCockpit() {
  const location = useLocation();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const runId = searchParams.get("run_id");
  const { status, runs, telemetryByRun, lastStatus } = useLiveStream({ runId });

  const selectedRun = runs.find((run) => run.run_id === runId) ?? null;
  const statusForRun = lastStatus && lastStatus.run_id === runId ? lastStatus : null;
  const runState = statusForRun?.state ?? selectedRun?.state ?? "unknown";
  const terminal = runState === "done" || runState === "failed";
  const rawPoints = useMemo(
    () => (runId ? (telemetryByRun[runId] ?? []) : []),
    [runId, telemetryByRun],
  );

  // Once the server reports done/failed, no later telemetry may extend the
  // charts. Keep everything up to the final reported timestep and discard the rest.
  // Fall back to the registry's `latest_timestep` when no live `status` frame
  // has arrived yet (e.g. the run finished before the cockpit opened).
  const terminalTimestep =
    typeof statusForRun?.timestep === "number"
      ? statusForRun.timestep
      : (selectedRun?.latest_timestep ?? null);
  const points = useMemo(() => {
    if (!terminal || terminalTimestep === null) return rawPoints;
    return rawPoints.filter((point) => point.timestep <= terminalTimestep);
  }, [terminal, rawPoints, terminalTimestep]);

  if (!runId) {
    return (
      <EmptyState
        title="Select a run"
        description="Choose a run in the left rail to watch its learning curve, PPO diagnostics and action distribution stream live."
      />
    );
  }

  if (!selectedRun) {
    return (
      <EmptyState
        title="Waiting for run metadata"
        description={`The run ${runId} is not in the registry yet. The cockpit will fill in as soon as the server discovers it.`}
      />
    );
  }

  const hasTelemetry = points.length > 0 || selectedRun.has_telemetry;
  if (!hasTelemetry) {
    return (
      <EmptyState
        title="No telemetry for this run"
        description="This run has no telemetry.jsonl yet. Start a training run with telemetry enabled:"
        action={
          <code
            data-testid="training-command"
            className="rounded-control border border-border bg-bg-base px-3 py-2 text-xs text-accent"
          >
            {TRAINING_COMMAND}
          </code>
        }
      />
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      {status === "degraded" && (
        <div
          role="status"
          aria-live="polite"
          className="rounded-control border border-high-vol/50 bg-high-vol/10 px-3 py-2 text-xs text-high-vol"
        >
          Live stream degraded — showing the last received telemetry. Reconnecting automatically.
        </div>
      )}

      <RunHeader
        run={selectedRun}
        status={statusForRun}
        connection={status}
        points={points}
      />

      <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <LearningCurve runId={runId} points={points} />
        <ActionHeatmap runId={runId} />
      </div>

      <DiagnosticsGrid runId={runId} points={points} />
    </div>
  );
}
