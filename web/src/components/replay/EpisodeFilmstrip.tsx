/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import type { ReactNode } from "react";

import { cn } from "../cn";
import { EmptyState } from "../EmptyState";
import { Panel } from "../Panel";
import { TOKEN } from "./theme";

export interface FilmstripTile {
  seed: number;
  episode: number;
  equity: number[];
  regime: string;
}

/** Regime -> CONTRACTS §9 token colour (colour is paired with a text label). */
export const REGIME_COLORS: Record<string, string> = {
  bull: TOKEN.bull,
  bear: TOKEN.bear,
  sideways: TOKEN.sideways,
  high_vol: TOKEN.highVol,
  unknown: TOKEN.textMuted,
};

function regimeColor(regime: string): string {
  return REGIME_COLORS[regime] ?? REGIME_COLORS.unknown!;
}

function MiniEquity({ values, color }: { values: readonly number[]; color: string }): ReactNode {
  const finite = values.filter((value) => Number.isFinite(value));
  const width = 96;
  const height = 28;
  if (finite.length === 0) {
    return <svg width={width} height={height} aria-hidden="true" />;
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const range = max - min || 1;
  const points = finite
    .map((value, index) => {
      const x = finite.length === 1 ? width / 2 : (index / (finite.length - 1)) * width;
      const y = height - ((value - min) / range) * (height - 2) - 1;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
      focusable="false"
      className="w-full"
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export interface EpisodeFilmstripProps {
  tiles: readonly FilmstripTile[];
  selectedSeed: number;
  selectedEpisode: number;
  totalEpisodes: number;
  showAll: boolean;
  onToggleShowAll: () => void;
  onSelect: (seed: number, episode: number) => void;
  cap?: number;
}

/**
 * One mini equity tile per episode of the selected seed/run, coloured by the
 * episode's dominant regime. Clicking a tile loads it into the main panels.
 */
export function EpisodeFilmstrip({
  tiles,
  selectedSeed,
  selectedEpisode,
  totalEpisodes,
  showAll,
  onToggleShowAll,
  onSelect,
  cap = 24,
}: EpisodeFilmstripProps) {
  const hidden = totalEpisodes - tiles.length;
  if (totalEpisodes === 0) {
    return (
      <Panel title="Episodes" subtitle="Mini equity curves for every recorded episode">
        <EmptyState title="No episodes" description="This seed has no recorded rollouts." />
      </Panel>
    );
  }

  return (
    <Panel
      title="Episodes"
      subtitle={`${totalEpisodes} recorded episode${totalEpisodes === 1 ? "" : "s"} · coloured by dominant regime`}
      actions={
        (hidden > 0 || (showAll && totalEpisodes > cap)) && (
          <button
            type="button"
            onClick={onToggleShowAll}
            className="rounded-control border border-border bg-bg-elevated px-2 py-1 text-xs text-text-muted hover:text-text-primary"
          >
            {showAll ? "Show fewer" : `Show all (${totalEpisodes})`}
          </button>
        )
      }
    >
      <ul className="grid max-h-72 grid-cols-2 gap-2 overflow-y-auto pr-1 sm:grid-cols-3">
        {tiles.map((tile) => {
          const color = regimeColor(tile.regime);
          const selected = tile.seed === selectedSeed && tile.episode === selectedEpisode;
          return (
            <li key={`${tile.seed}-${tile.episode}`}>
              <button
                type="button"
                aria-pressed={selected}
                aria-label={`Episode ${tile.episode}, dominant regime ${tile.regime}`}
                onClick={() => onSelect(tile.seed, tile.episode)}
                className={cn(
                  "w-full rounded-control border bg-bg-elevated p-2 text-left transition-colors",
                  selected ? "border-accent" : "border-border hover:border-text-muted",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="tabular text-xs text-text-primary">ep {tile.episode}</span>
                  <span className="flex items-center gap-1 text-[11px] text-text-muted">
                    <span
                      className="size-2 rounded-full"
                      style={{ backgroundColor: color }}
                      aria-hidden="true"
                    />
                    {tile.regime}
                  </span>
                </div>
                <MiniEquity values={tile.equity} color={color} />
              </button>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}