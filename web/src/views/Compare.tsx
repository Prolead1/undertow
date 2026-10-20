import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import {
  ApiError,
  compare,
  getRuns,
  type CompareResponse,
  type RunRecord,
} from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import { PlotlyFigure } from "../components/PlotlyFigure";
import { MAX_COMPARE_RUNS, parsePinnedIds, serializePinnedIds, togglePinnedId } from "../components/leaderboard/pinned";
import { PolicyFingerprint, SERIES_COLORS } from "../components/leaderboard/PolicyFingerprint";
import { rankRuns } from "../components/leaderboard/ranking";

/** Stable identity so memo dependencies do not churn while no compare is loaded. */
const EMPTY_RECORDS: RunRecord[] = [];

type EvalKey = keyof NonNullable<RunRecord["eval"]>;

interface FingerprintField {
  key: EvalKey | "n_rebalances";
  label: string;
}

/** Metrics shown in the compare table and fingerprints (CONTRACTS §4 EvalMetrics). */
const BASE_FINGERPRINT_FIELDS: readonly FingerprintField[] = [
  { key: "sharpe", label: "Sharpe" },
  { key: "total_pnl", label: "Total PnL" },
  { key: "max_drawdown", label: "Max DD (lower is better)" },
  { key: "worst_regime_pnl", label: "Worst-regime PnL" },
];

const FALLBACK_FIELD: FingerprintField = { key: "annualized_return", label: "Annualised return" };
const REBALANCE_FIELD: FingerprintField = { key: "n_rebalances", label: "Rebalances" };

/**
 * Read the optional `n_rebalances` extension defensively. It is not part of the
 * frozen `EvalMetrics` contract, so this returns `null` against today's API and
 * simply lights up if the field is ever added.
 */
function rebalanceCount(run: RunRecord): number | null {
  const extended = run as RunRecord & { n_rebalances?: unknown };
  const evalExtended = run.eval as (NonNullable<RunRecord["eval"]> & { n_rebalances?: unknown }) | null;
  const value = extended.n_rebalances ?? evalExtended?.n_rebalances;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function fingerprintFields(records: readonly RunRecord[]): FingerprintField[] {
  const hasRebalances = records.some((run) => rebalanceCount(run) !== null);
  return [...BASE_FINGERPRINT_FIELDS, hasRebalances ? REBALANCE_FIELD : FALLBACK_FIELD];
}

function metricValue(run: RunRecord, key: FingerprintField["key"]): number | null {
  if (key === "n_rebalances") return rebalanceCount(run);
  const value = run.eval?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function format(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function errorMessage(cause: unknown): string {
  if (cause instanceof ApiError) {
    if (cause.status === 422) {
      return "The compare request was rejected (422). Pin at least two known runs (and at most 6) and retry.";
    }
    return cause.message;
  }
  return String(cause);
}

/** Compare view: pin up to six runs and overlay their equity + metrics. */
export default function Compare() {
  const [searchParams, setSearchParams] = useSearchParams();
  const allPinned = useMemo(() => parsePinnedIds(searchParams.toString()), [searchParams]);
  const pinnedIds = useMemo(() => allPinned.slice(0, MAX_COMPARE_RUNS), [allPinned]);

  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [response, setResponse] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getRuns({ signal: controller.signal })
      .then(setRuns)
      .catch(() => {
        // The pinned runs still resolve through `/api/compare`; the picker
        // degrades to a static message rather than blanking the view.
      });
    return () => controller.abort();
  }, []);

  const pinnedKey = pinnedIds.join(",");

  useEffect(() => {
    if (pinnedIds.length < 2) {
      setResponse(null);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    compare({ run_ids: pinnedIds, metric: "sharpe" })
      .then((payload) => {
        if (cancelled) return;
        setResponse(payload);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setResponse(null);
        setError(errorMessage(cause));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // `pinnedKey` is the stable identity of the pinned set.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pinnedKey]);

  const setPinned = useCallback(
    (ids: readonly string[]) => {
      setSearchParams(new URLSearchParams(serializePinnedIds(searchParams.toString(), ids)), {
        replace: true,
      });
    },
    [searchParams, setSearchParams],
  );

  const handleTogglePin = useCallback(
    (runId: string) => setPinned(togglePinnedId(allPinned, runId, MAX_COMPARE_RUNS)),
    [allPinned, setPinned],
  );

  const records = response?.records ?? EMPTY_RECORDS;
  const fields = useMemo(() => fingerprintFields(records), [records]);
  // Colour by the pinned (URL) order so a run's chip, legend entry and bars all
  // agree even when the API re-ranks `records` before returning them.
  const colorFor = useMemo(() => {
    const map = new Map<string, string>();
    pinnedIds.forEach((runId, index) => {
      map.set(runId, SERIES_COLORS[index % SERIES_COLORS.length] ?? SERIES_COLORS[0]);
    });
    return map;
  }, [pinnedIds]);
  const fingerprintSeries = useMemo(
    () =>
      records.map((run) => ({
        label: run.run_id,
        values: fields.map((field) => metricValue(run, field.key)),
        color: colorFor.get(run.run_id),
      })),
    [records, fields, colorFor],
  );

  const available = useMemo(
    () => rankRuns(runs.filter((run) => !pinnedIds.includes(run.run_id))),
    [runs, pinnedIds],
  );
  const atLimit = pinnedIds.length >= MAX_COMPARE_RUNS;
  const overflow = allPinned.length > MAX_COMPARE_RUNS;

  return (
    <Panel
      title="Compare policies"
      subtitle="Overlay pinned runs, their metrics and normalised fingerprints"
    >
      <div className="flex flex-col gap-4">
        {overflow && (
          <p role="status" className="rounded-control border border-high-vol/50 bg-bg-elevated px-3 py-2 text-xs text-high-vol">
            Only the first {MAX_COMPARE_RUNS} pinned runs are compared (the API accepts at most{" "}
            {MAX_COMPARE_RUNS}).
          </p>
        )}

        {pinnedIds.length === 0 ? (
          <EmptyState
            title="No policies pinned"
            description="Pin runs from the leaderboard (or the picker below) to compare their equity, metrics and fingerprints. The pinned set lives in the URL, so a comparison link is shareable."
          />
        ) : (
          <div className="flex flex-wrap items-center gap-2" aria-label="Pinned runs">
            {pinnedIds.map((runId) => (
              <span
                key={runId}
                data-testid="pinned-chip"
                className="inline-flex items-center gap-2 rounded-control border border-border bg-bg-elevated px-2 py-1 text-xs"
              >
                <span
                  className="inline-block size-2.5 rounded-full"
                  style={{ backgroundColor: colorFor.get(runId) }}
                  aria-hidden
                />
                <span className="max-w-56 truncate font-mono text-text-primary">{runId}</span>
                <button
                  type="button"
                  onClick={() => handleTogglePin(runId)}
                  aria-label={`Unpin ${runId}`}
                  className="text-text-muted hover:text-text-primary"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}

        {pinnedIds.length === 1 && (
          <EmptyState
            title="Pin one more run to compare"
            description="Comparison needs at least two runs. Use the leaderboard checkboxes or the picker below to add another."
          />
        )}

        {pinnedIds.length >= 2 && loading && records.length === 0 ? (
          <EmptyState title="Comparing…" description="Requesting the equity overlay from the server." />
        ) : null}

        {pinnedIds.length >= 2 && error ? (
          <EmptyState title="Compare failed" description={error} />
        ) : null}

        {pinnedIds.length >= 2 && response && !error ? (
          <>
            <section aria-label="Equity overlay" className="rounded-control border border-border bg-bg-base/40 p-3">
              <PlotlyFigure
                spec={response.figures.compare ?? { data: [], layout: {}, config: {} }}
                ariaLabel={`Equity overlay for ${records.map((run) => run.run_id).join(", ")}`}
                className="min-h-72"
              />
            </section>

            <div className="grid gap-4 xl:grid-cols-2">
              <section aria-label="Metric comparison" className="rounded-control border border-border bg-bg-surface p-3">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                  Metrics
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-[11px] uppercase tracking-wide text-text-muted">
                        <th scope="col" className="py-2 pr-3">
                          Run
                        </th>
                        <th scope="col" className="py-2 pr-3">
                          State
                        </th>
                        <th scope="col" className="py-2 pr-3 text-right">
                          Sharpe
                        </th>
                        <th scope="col" className="py-2 pr-3 text-right">
                          Total PnL
                        </th>
                        <th scope="col" className="py-2 pr-3 text-right">
                          Max DD
                        </th>
                        <th scope="col" className="py-2 pr-3 text-right">
                          Worst regime
                        </th>
                        <th scope="col" className="py-2 text-right">
                          Best reward
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {records.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="py-4 text-center text-text-muted">
                            No records returned.
                          </td>
                        </tr>
                      ) : (
                        records.map((run, index) => (
                          <tr
                            key={`${run.run_id}-${index}`}
                            data-testid="compare-metric-row"
                            className="border-b border-border/60"
                          >
                            <td className="py-2 pr-3 font-mono text-xs text-text-primary">
                              {run.run_id}
                            </td>
                            <td className="py-2 pr-3 text-text-muted">{run.state}</td>
                            <td className="tabular py-2 pr-3 text-right">{format(run.eval?.sharpe)}</td>
                            <td className="tabular py-2 pr-3 text-right">
                              {format(run.eval?.total_pnl, 2)}
                            </td>
                            <td className="tabular py-2 pr-3 text-right">
                              {format(run.eval?.max_drawdown, 2)}
                            </td>
                            <td className="tabular py-2 pr-3 text-right">
                              {format(run.eval?.worst_regime_pnl, 2)}
                            </td>
                            <td className="tabular py-2 text-right">{format(run.best_reward)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <section
                aria-label="Metric fingerprints"
                className="rounded-control border border-border bg-bg-surface p-3"
              >
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
                  Fingerprints
                </h3>
                <PolicyFingerprint
                  metrics={fields.map((field) => field.label)}
                  series={fingerprintSeries}
                />
              </section>
            </div>

            <ul data-testid="compare-legend" className="sr-only">
              {records.map((run) => (
                <li key={run.run_id}>{run.run_id}</li>
              ))}
            </ul>
          </>
        ) : null}

        <section aria-label="Add runs to comparison" className="rounded-control border border-border bg-bg-surface p-3">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
            Add a run
          </h3>
          {available.length === 0 ? (
            <p className="text-xs text-text-muted">
              {runs.length === 0 ? "No runs discovered yet." : "Every discovered run is already pinned."}
            </p>
          ) : (
            <ul className="flex flex-wrap gap-2">
              {available.map((run) => (
                <li key={run.run_id}>
                  <button
                    type="button"
                    onClick={() => handleTogglePin(run.run_id)}
                    disabled={atLimit}
                    title={atLimit ? `Comparison is limited to ${MAX_COMPARE_RUNS} runs` : `Pin ${run.run_id}`}
                    className="rounded-control border border-border px-2 py-1 text-xs text-text-primary hover:border-accent-2 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    + {run.run_id}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </Panel>
  );
}
