import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { ApiError, getCurve, getRuns, type RunRecord } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import { runsToCsv } from "../components/leaderboard/csv";
import { LeaderboardTable } from "../components/leaderboard/LeaderboardTable";
import {
  MAX_COMPARE_RUNS,
  parsePinnedIds,
  serializePinnedIds,
  togglePinnedId,
} from "../components/leaderboard/pinned";
import { rankRuns } from "../components/leaderboard/ranking";

/** Trigger a client-side file download; no server endpoint is involved. */
function downloadCsv(filename: string, text: string): void {
  if (typeof document === "undefined" || typeof URL.createObjectURL !== "function") return;
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel the download in some browsers.
  window.setTimeout(() => {
    if (typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(url);
  }, 0);
}

/** Leaderboard view: ranked, provenance-visible runs with client-side CSV export. */
export default function Leaderboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const pinnedIds = useMemo(() => parsePinnedIds(searchParams.toString()), [searchParams]);

  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [curves, setCurves] = useState<Record<string, number[]>>({});
  const [requestedCurves, setRequestedCurves] = useState<ReadonlySet<string>>(new Set());
  const [visibleRuns, setVisibleRuns] = useState<RunRecord[] | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    getRuns({ signal: controller.signal })
      .then((records) => {
        setRuns(records);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof ApiError ? cause.message : String(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const handleTogglePin = useCallback(
    (runId: string) => {
      const next = togglePinnedId(pinnedIds, runId, MAX_COMPARE_RUNS);
      setSearchParams(new URLSearchParams(serializePinnedIds(searchParams.toString(), next)), {
        replace: true,
      });
    },
    [pinnedIds, searchParams, setSearchParams],
  );

  const handleExpand = useCallback(
    (runId: string) => {
      if (requestedCurves.has(runId)) return;
      setRequestedCurves((current) => new Set(current).add(runId));
      getCurve(runId)
        .then((curve) => setCurves((current) => ({ ...current, [runId]: curve.mean })))
        .catch(() => {
          // A missing curve is a normal state for an in-progress run; the
          // expanded row simply keeps its "No curve loaded yet." text.
        });
    },
    [requestedCurves],
  );

  const ranked = useMemo(() => rankRuns(runs), [runs]);
  const handleVisibleChange = useCallback((next: RunRecord[]) => setVisibleRuns(next), []);
  // `null` means the table has not reported its visible set yet; an empty array
  // is a real "no rows match" state and must export a header-only CSV.
  const csvRuns = visibleRuns ?? ranked;

  return (
    <Panel
      title="Leaderboard"
      subtitle="Ranked by held-out eval Sharpe, then worst-regime PnL, then best reward"
      actions={
        <button
          type="button"
          onClick={() => downloadCsv("undertow-leaderboard.csv", runsToCsv(csvRuns))}
          disabled={ranked.length === 0}
          className="rounded-control border border-border px-2.5 py-1 text-xs text-text-primary hover:border-accent disabled:opacity-50"
        >
          Download CSV
        </button>
      }
    >
      {loading && runs.length === 0 ? (
        <EmptyState title="Loading runs…" description="Scanning the run registry." />
      ) : error ? (
        <EmptyState title="Could not load runs" description={error} />
      ) : runs.length === 0 ? (
        <EmptyState
          title="No ranked policies yet"
          description="Completed runs with results are ranked here. Point the server at a populated runs directory."
        />
      ) : (
        <LeaderboardTable
          runs={ranked}
          pinnedIds={pinnedIds}
          onTogglePin={handleTogglePin}
          maxPinned={MAX_COMPARE_RUNS}
          curves={curves}
          onExpand={handleExpand}
          onVisibleChange={handleVisibleChange}
        />
      )}
    </Panel>
  );
}
