import { useEffect, useState } from "react";

import { getFigure, type FigureSpec } from "../../api/client";
import { EmptyState } from "../EmptyState";
import { Panel } from "../Panel";
import { PlotlyFigure } from "../PlotlyFigure";

export interface ActionHeatmapProps {
  runId: string;
}

const LEGEND =
  "Grid: center offset (y) × width (x), in tick-spacings. hold = no rebalance (offset 0, width 0).";

/** Current action distribution from the server-owned actions figure. */
export function ActionHeatmap({ runId }: ActionHeatmapProps) {
  const [spec, setSpec] = useState<FigureSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSpec(null);
    getFigure(runId, "actions")
      .then((payload) => {
        if (!cancelled) setSpec(payload);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(String(cause));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, reloadToken]);

  return (
    <Panel
      title="Action distribution"
      subtitle="Counts from the latest recorded rollout episode"
      actions={
        <button
          type="button"
          onClick={() => setReloadToken((value) => value + 1)}
          className="rounded-control border border-border px-2 py-1 text-xs text-text-muted hover:border-accent/50 hover:text-text-primary"
        >
          Refresh
        </button>
      }
    >
      <div className="flex flex-col gap-2">
        {loading ? (
          <p className="flex min-h-64 items-center justify-center text-xs text-text-muted">
            Loading action distribution…
          </p>
        ) : spec && spec.data.length > 0 ? (
          <PlotlyFigure
            spec={spec}
            ariaLabel="Action distribution heatmap over center offset and width"
            className="h-64"
          />
        ) : (
          <EmptyState
            title="No action distribution yet"
            description={
              error
                ? `The actions figure failed for ${runId}: ${error}`
                : "The heatmap is built from recorded rollout episodes. Enable `training.record_rollouts = true` before training to populate it."
            }
          />
        )}
        <p className="text-[11px] text-text-muted">{LEGEND}</p>
      </div>
    </Panel>
  );
}
