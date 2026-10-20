import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "../router";

import {
  ApiError,
  getRollout,
  getRuns,
  listRollouts,
  type RolloutSeries,
  type RolloutSummary,
  type RunRecord,
} from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";
import { CostWaterfall } from "../components/replay/CostWaterfall";
import { EpisodeFilmstrip, type FilmstripTile } from "../components/replay/EpisodeFilmstrip";
import { EquityVsHodl } from "../components/replay/EquityVsHodl";
import { PolicySummary } from "../components/replay/PolicySummary";
import { PoolChart } from "../components/replay/PoolChart";
import { RibbonChart } from "../components/replay/RibbonChart";
import { dominantRegime, parseRolloutRows } from "../components/replay/rows";
import { StateInspector } from "../components/replay/StateInspector";

function uniqueSorted(values: readonly number[]): number[] {
  return [...new Set(values)].sort((a, b) => a - b);
}

function episodesForSeed(summaries: readonly RolloutSummary[], seed: number | null): number[] {
  if (seed === null) return [];
  return uniqueSorted(
    summaries.filter((summary) => summary.seed === seed).map((summary) => summary.episode),
  );
}

function parseQueryInt(value: string | null): number | null {
  if (value === null || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

const FILMSTRIP_CAP = 24;

/**
 * Policy Replay view (U09): run -> seed -> episode selection is carried in the
 * URL (`run_id`/`seed`/`episode`) so any replay is shareable. The time cursor is
 * deliberately view-local (`cursorIndex`) and never leaves this component.
 */
export default function PolicyReplay() {
  const [searchParams, setSearchParams] = useSearchParams();
  const runId = searchParams.get("run_id");
  const seedParam = parseQueryInt(searchParams.get("seed"));
  const episodeParam = parseQueryInt(searchParams.get("episode"));

  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [summaries, setSummaries] = useState<RolloutSummary[] | null>(null);
  const [listError, setListError] = useState<unknown>(null);
  const [series, setSeries] = useState<RolloutSeries | null>(null);
  const [rolloutError, setRolloutError] = useState<unknown>(null);
  const [cursorIndex, setCursorIndex] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const [tiles, setTiles] = useState<FilmstripTile[]>([]);

  useEffect(() => {
    let active = true;
    getRuns()
      .then((next) => {
        if (active) setRuns(next);
      })
      .catch(() => {
        /* The left rail still drives selection; a failed list is non-fatal. */
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!runId) {
      setSummaries(null);
      setListError(null);
      return;
    }
    let active = true;
    setSummaries(null);
    setListError(null);
    listRollouts(runId)
      .then((next) => {
        if (active) setSummaries(next);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setSummaries([]);
        setListError(error);
      });
    return () => {
      active = false;
    };
  }, [runId]);

  const seeds = useMemo(
    () => uniqueSorted((summaries ?? []).map((summary) => summary.seed)),
    [summaries],
  );

  const seed = useMemo(() => {
    if (seedParam !== null && seeds.includes(seedParam)) return seedParam;
    return seeds[0] ?? null;
  }, [seedParam, seeds]);

  const episodes = useMemo(() => episodesForSeed(summaries ?? [], seed), [summaries, seed]);

  const episode = useMemo(() => {
    if (episodeParam !== null && episodes.includes(episodeParam)) return episodeParam;
    return episodes[0] ?? null;
  }, [episodeParam, episodes]);

  // Keep the URL authoritative: once the defaults resolve, write them back so a
  // copy-paste of the address bar restores the exact episode.
  useEffect(() => {
    if (runId === null || seed === null || episode === null) return;
    if (String(seed) === searchParams.get("seed") && String(episode) === searchParams.get("episode")) {
      return;
    }
    const next = new URLSearchParams(searchParams);
    next.set("seed", String(seed));
    next.set("episode", String(episode));
    setSearchParams(next, { replace: true });
  }, [runId, seed, episode, searchParams, setSearchParams]);

  useEffect(() => {
    if (!runId || seed === null || episode === null) {
      setSeries(null);
      setRolloutError(null);
      return;
    }
    let active = true;
    setSeries(null);
    setRolloutError(null);
    getRollout(runId, seed, episode)
      .then((next) => {
        if (active) setSeries(next);
      })
      .catch((error: unknown) => {
        if (active) setRolloutError(error);
      });
    return () => {
      active = false;
    };
  }, [runId, seed, episode]);

  const rows = useMemo(() => (series ? parseRolloutRows(series.rows) : []), [series]);

  useEffect(() => {
    setCursorIndex(0);
  }, [series]);

  const visibleEpisodes = useMemo(
    () => (showAll ? episodes : episodes.slice(0, FILMSTRIP_CAP)),
    [showAll, episodes],
  );

  useEffect(() => {
    setTiles([]);
    if (!runId || seed === null || visibleEpisodes.length === 0) {
      return;
    }
    let active = true;
    Promise.all(
      visibleEpisodes.map(async (episodeIndex): Promise<FilmstripTile | null> => {
        try {
          const tileSeries = await getRollout(runId, seed, episodeIndex);
          const tileRows = parseRolloutRows(tileSeries.rows);
          return {
            seed,
            episode: episodeIndex,
            equity: tileRows.map((row) => row.equity ?? 0),
            regime: dominantRegime(tileRows.map((row) => row.regime)),
          };
        } catch {
          return null;
        }
      }),
    ).then((next) => {
      if (!active) return;
      setTiles(next.filter((tile): tile is FilmstripTile => tile !== null));
    });
    return () => {
      active = false;
    };
  }, [runId, seed, visibleEpisodes]);

  const updateSelection = (nextSeed: number | null, nextEpisode: number | null): void => {
    const next = new URLSearchParams(searchParams);
    if (nextSeed !== null) next.set("seed", String(nextSeed));
    if (nextEpisode !== null) next.set("episode", String(nextEpisode));
    setSearchParams(next);
  };

  const onRunChange = (nextRunId: string): void => {
    const next = new URLSearchParams(searchParams);
    if (nextRunId) {
      next.set("run_id", nextRunId);
    } else {
      next.delete("run_id");
    }
    next.delete("seed");
    next.delete("episode");
    setSearchParams(next);
  };

  const onSeedChange = (nextSeed: number): void => {
    const firstEpisode = episodesForSeed(summaries ?? [], nextSeed)[0] ?? 0;
    updateSelection(nextSeed, firstEpisode);
  };

  const onEpisodeChange = (nextEpisode: number): void => {
    updateSelection(seed, nextEpisode);
  };

  const cursorStep = rows[cursorIndex]?.step ?? null;

  const toolbar = (
    <Panel title="Replay selection" subtitle="Shareable: the address bar carries run, seed and episode">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-text-muted">
          Run
          <select
            aria-label="Run"
            value={runId ?? ""}
            onChange={(event) => onRunChange(event.target.value)}
            className="min-w-48 rounded-control border border-border bg-bg-elevated px-2 py-1.5 text-sm text-text-primary"
          >
            <option value="">Select a run…</option>
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-text-muted">
          Seed
          <select
            aria-label="Seed"
            value={seed ?? ""}
            disabled={seeds.length === 0}
            onChange={(event) => onSeedChange(Number(event.target.value))}
            className="rounded-control border border-border bg-bg-elevated px-2 py-1.5 text-sm text-text-primary disabled:opacity-50"
          >
            {seeds.map((value) => (
              <option key={value} value={value}>
                seed {value}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-text-muted">
          Episode
          <select
            aria-label="Episode"
            value={episode ?? ""}
            disabled={episodes.length === 0}
            onChange={(event) => onEpisodeChange(Number(event.target.value))}
            className="rounded-control border border-border bg-bg-elevated px-2 py-1.5 text-sm text-text-primary disabled:opacity-50"
          >
            {episodes.map((value) => (
              <option key={value} value={value}>
                episode {value}
              </option>
            ))}
          </select>
        </label>

        <label className="flex min-w-56 flex-1 flex-col gap-1 text-xs text-text-muted">
          Time cursor{cursorStep !== null ? ` · step ${cursorStep}` : ""}
          <input
            type="range"
            aria-label="Time cursor"
            min={0}
            max={Math.max(rows.length - 1, 0)}
            value={Math.min(cursorIndex, Math.max(rows.length - 1, 0))}
            disabled={rows.length === 0}
            onChange={(event) => setCursorIndex(Number(event.target.value))}
            className="accent-accent"
          />
        </label>
      </div>
    </Panel>
  );

  let body;
  if (!runId) {
    body = (
      <EmptyState
        title="No run selected"
        description="Pick a run in the left rail, or use the selector above, to replay its recorded episodes."
      />
    );
  } else if (listError !== null) {
    body = (
      <EmptyState
        title="Rollouts unavailable"
        description={`Could not list rollouts${
          listError instanceof ApiError ? `: ${String(listError.detail)}` : ""
        }. Check the run id and that the API server is reachable.`}
      />
    );
  } else if (summaries === null) {
    body = <EmptyState title="Loading rollouts…" description="Reading the recorded episodes." />;
  } else if (summaries.length === 0) {
    body = (
      <EmptyState
        title="No recorded rollouts"
        description="This run has no replay episodes. Re-train with training.record_rollouts = true to record them."
        action={
          <code className="rounded-control border border-border bg-bg-elevated px-2 py-1 text-xs text-accent">
            training.record_rollouts = true
          </code>
        }
      />
    );
  } else if (rolloutError !== null) {
    body = (
      <EmptyState
        title="Episode failed to load"
        description={
          rolloutError instanceof ApiError
            ? String(rolloutError.detail)
            : "The rollout parquet could not be read."
        }
      />
    );
  } else if (series === null) {
    body = <EmptyState title="Loading episode…" description="Reading the rollout rows." />;
  } else if (rows.length === 0) {
    body = <EmptyState title="Empty episode" description="The selected rollout has no rows." />;
  } else {
    body = (
      <div className="grid gap-4">
        <RibbonChart
          runId={runId}
          seed={seed ?? 0}
          episode={episode ?? 0}
          rows={rows}
          cursorIndex={cursorIndex}
        />

        <div className="grid gap-4 xl:grid-cols-2">
          <EquityVsHodl rows={rows} cursorIndex={cursorIndex} />
          <CostWaterfall rows={rows} cursorIndex={cursorIndex} />
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <StateInspector rows={rows} cursorIndex={cursorIndex} />
          <EpisodeFilmstrip
            tiles={tiles}
            selectedSeed={seed ?? 0}
            selectedEpisode={episode ?? 0}
            totalEpisodes={episodes.length}
            showAll={showAll}
            onToggleShowAll={() => setShowAll((value) => !value)}
            onSelect={(nextSeed, nextEpisode) => updateSelection(nextSeed, nextEpisode)}
          />
        </div>

        <PoolChart rows={rows} cursorIndex={cursorIndex} />
        <PolicySummary rows={rows} />
      </div>
    );
  }

  return (
    <div className="grid gap-4">
      {toolbar}
      {body}
    </div>
  );
}

