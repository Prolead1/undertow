import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "./router";

import { useLiveStream, type ConnectionStatus } from "./api/useLiveStream";
import type { RunRecord } from "./api/client";
import { cn } from "./components/cn";
import { QueryLink } from "./components/QueryLink";
import { RunBadge } from "./components/RunBadge";
import { ToastRegion, type ToastMessage } from "./components/Toast";

const NAV = [
  { to: "/live", label: "Live cockpit" },
  { to: "/replay", label: "Policy replay" },
  { to: "/leaderboard", label: "Leaderboard" },
  { to: "/compare", label: "Compare" },
] as const;

const CONNECTION: Record<ConnectionStatus, { label: string; dot: string; text: string }> = {
  live: { label: "Live", dot: "bg-bull", text: "text-bull" },
  connecting: { label: "Connecting", dot: "bg-high-vol", text: "text-high-vol" },
  degraded: { label: "Degraded · polling", dot: "bg-bear", text: "text-bear" },
  closed: { label: "Offline", dot: "bg-sideways", text: "text-text-muted" },
};

function ConnectionBadge({ status, live = false }: { status: ConnectionStatus; live?: boolean }) {
  const presentation = CONNECTION[status];
  return (
    <span
      // Only one badge is a live region: two `role="status"` nodes would make
      // assistive tech announce every connection change twice.
      role={live ? "status" : undefined}
      aria-live={live ? "polite" : undefined}
      aria-hidden={live ? undefined : true}
      className={cn(
        "inline-flex items-center gap-2 rounded-control border border-border bg-bg-elevated px-2.5 py-1 text-xs font-medium",
        presentation.text,
      )}
    >
      <span className={cn("size-2 rounded-full", presentation.dot)} aria-hidden />
      {presentation.label}
    </span>
  );
}

export default function App({ children }: { children: ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const selectedRunId = searchParams.get("run_id");
  const { status, runs, lastStatus } = useLiveStream({ runId: selectedRunId });
  const [pinned, setPinned] = useState<string[]>([]);
  const [dismissedDegraded, setDismissedDegraded] = useState(false);

  useEffect(() => {
    if (status !== "degraded") setDismissedDegraded(false);
  }, [status]);

  const selectedRun = runs.find((run) => run.run_id === selectedRunId) ?? null;
  const pinnedRuns = pinned
    .map((runId) => runs.find((run) => run.run_id === runId))
    .filter((run): run is RunRecord => run !== undefined);

  const toasts: ToastMessage[] =
    status === "degraded" && !dismissedDegraded
      ? [
          {
            id: "degraded",
            message: "Live stream degraded — tailing REST telemetry until the socket reconnects.",
            tone: "error",
            // The header connection badge is the live region for this state;
            // this toast is the visual counterpart and must not announce twice.
            announce: false,
          },
        ]
      : [];

  const selectRun = (run: RunRecord): void => {
    const next = new URLSearchParams(location.search);
    next.set("run_id", run.run_id);
    navigate({ pathname: location.pathname, search: next.toString() });
  };

  const togglePin = (run: RunRecord): void => {
    setPinned((current) =>
      current.includes(run.run_id)
        ? current.filter((runId) => runId !== run.run_id)
        : [...current, run.run_id],
    );
  };

  const section = NAV.find((item) => item.to === location.pathname)?.label ?? "Live cockpit";

  return (
    <div className="min-h-screen bg-bg-base text-text-primary">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-control focus:bg-bg-elevated focus:px-3 focus:py-2"
      >
        Skip to content
      </a>

      <div className="grid min-h-screen lg:grid-cols-[18rem_minmax(0,1fr)]">
        <aside className="border-b border-border bg-bg-surface lg:border-b-0 lg:border-r">
          <div className="flex h-full flex-col gap-5 p-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">
                Undertow
              </p>
              <h1 className="mt-1 text-lg font-semibold">Policy cockpit</h1>
              <p className="mt-1 text-xs text-text-muted">
                Artifact-only view of the RL training pipeline.
              </p>
            </div>

            <nav aria-label="Primary" className="flex flex-col gap-1">
              {NAV.map((item) => {
                const active = location.pathname === item.to;
                return (
                  <QueryLink
                    key={item.to}
                    to={item.to}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "rounded-control border px-3 py-2 text-sm transition-colors",
                      active
                        ? "border-accent/50 bg-bg-elevated text-text-primary"
                        : "border-transparent text-text-muted hover:border-border hover:bg-bg-elevated hover:text-text-primary",
                    )}
                  >
                    {item.label}
                  </QueryLink>
                );
              })}
            </nav>

            <div className="flex min-h-0 flex-1 flex-col">
              <div className="mb-2 flex items-baseline justify-between gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                  Runs
                </h2>
                <span className="tabular text-xs text-text-muted">{runs.length}</span>
              </div>
              <div className="flex min-h-0 flex-col gap-1 overflow-y-auto pr-1">
                {runs.length === 0 ? (
                  <p className="rounded-control border border-dashed border-border p-3 text-xs text-text-muted">
                    No runs discovered yet. Start a training run or point the server at a runs
                    directory.
                  </p>
                ) : (
                  runs.map((run) => (
                    <RunBadge
                      key={run.run_id}
                      run={run}
                      selected={run.run_id === selectedRunId}
                      pinned={pinned.includes(run.run_id)}
                      onSelect={selectRun}
                      onTogglePin={togglePin}
                    />
                  ))
                )}
              </div>
            </div>

            <div className="border-t border-border pt-3 text-xs text-text-muted">
              <div className="flex items-center justify-between gap-2">
                <span>{status === "live" ? "Socket connected" : "Socket unavailable"}</span>
                <ConnectionBadge status={status} />
              </div>{" "}
              {lastStatus && (
                <p className="tabular mt-2 truncate">
                  {lastStatus.state} · {lastStatus.timestep.toLocaleString()} /{" "}
                  {lastStatus.total_timesteps.toLocaleString()} steps
                </p>
              )}
            </div>
          </div>
        </aside>

        <div className="flex min-w-0 flex-col">
          <header className="sticky top-0 z-20 flex flex-wrap items-center gap-3 border-b border-border bg-bg-base/95 px-4 py-3 backdrop-blur">
            <div className="min-w-0 flex-1">
              <p className="text-[11px] uppercase tracking-wide text-text-muted">{section}</p>
              <p className="truncate text-sm font-medium">
                {selectedRun ? selectedRun.run_id : "No run selected"}
                {selectedRun?.config_hash && (
                  <span className="ml-2 font-mono text-xs text-text-muted">
                    {selectedRun.config_hash.slice(0, 12)}
                  </span>
                )}
              </p>
            </div>

            {pinnedRuns.length > 0 && (
              <div
                className="flex max-w-full flex-wrap items-center gap-1"
                aria-label="Pinned runs"
              >
                <span className="text-[11px] uppercase tracking-wide text-text-muted">Pinned</span>
                {pinnedRuns.map((run) => (
                  <span
                    key={run.run_id}
                    className="inline-flex items-center gap-1 rounded-control border border-accent-2/40 bg-bg-elevated px-2 py-1 text-xs"
                  >
                    <button
                      type="button"
                      onClick={() => selectRun(run)}
                      className="max-w-48 truncate text-accent-2 hover:underline"
                    >
                      {run.run_id}
                    </button>
                    <button
                      type="button"
                      onClick={() => togglePin(run)}
                      aria-label={`Unpin ${run.run_id}`}
                      className="text-text-muted hover:text-text-primary"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            )}

            <ConnectionBadge status={status} live />
          </header>

          <main id="main" className="min-w-0 flex-1 p-4">
            {children}
          </main>
        </div>
      </div>

      <ToastRegion toasts={toasts} onDismiss={() => setDismissedDegraded(true)} />
    </div>
  );
}
