import { cn } from "../cn";
import { normalizeValues, SERIES_COLORS } from "./fingerprint";

export interface FingerprintSeries {
  label: string;
  /** One value per metric, aligned to `metrics`; `null` renders as "no data". */
  values: Array<number | null>;
  /**
   * Stable colour shared with the compare legend/chips. Falls back to the
   * palette by position when omitted.
   */
  color?: string;
}

export interface PolicyFingerprintProps {
  /** Axis labels, in order. */
  metrics: readonly string[];
  /** One entry per selected run. */
  series: readonly FingerprintSeries[];
  className?: string;
}

function formatValue(value: number | null): string {
  return value === null ? "—" : Number.isInteger(value) ? String(value) : value.toFixed(3);
}

/**
 * Small-multiple normalised bars over a run's metrics. Each metric is normalised
 * independently across the selected set so shapes are comparable regardless of
 * units; colours are shared with the compare legend.
 */
export function PolicyFingerprint({ metrics, series, className }: PolicyFingerprintProps) {
  const normalizedByMetric = metrics.map((_, metricIndex) =>
    normalizeValues(series.map((entry) => entry.values[metricIndex] ?? null)),
  );

  return (
    <div className={cn("min-w-0", className)}>
      <ul className="mb-3 flex flex-wrap gap-3" aria-label="Fingerprint legend">
        {series.map((entry, index) => (
          <li key={entry.label} className="flex items-center gap-2 text-xs text-text-primary">
            <span
              className="inline-block size-2.5 rounded-full"
              style={{ backgroundColor: entry.color ?? SERIES_COLORS[index % SERIES_COLORS.length] }}
              aria-hidden
            />
            <span className="max-w-48 truncate">{entry.label}</span>
          </li>
        ))}
      </ul>

      <div className="grid gap-4 sm:grid-cols-2">
        {metrics.map((metric, metricIndex) => (
          <section
            key={metric}
            data-testid="fingerprint-metric"
            className="rounded-control border border-border bg-bg-surface p-3"
          >
            <h4 className="mb-2 text-[11px] uppercase tracking-wide text-text-muted">{metric}</h4>
            <ul className="flex flex-col gap-1.5">
              {series.map((entry, seriesIndex) => {
                const raw = entry.values[metricIndex] ?? null;
                const normalized = normalizedByMetric[metricIndex]?.[seriesIndex] ?? null;
                const width = normalized === null ? 0 : Math.max(2, normalized * 100);
                return (
                  <li key={entry.label} className="flex items-center gap-2">
                    <span className="w-24 shrink-0 truncate text-[11px] text-text-muted" title={entry.label}>
                      {entry.label}
                    </span>
                    <span className="h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-grid">
                      <span
                        className="block h-full rounded-full"
                        style={{
                          width: `${width}%`,
                          backgroundColor:
                            entry.color ?? SERIES_COLORS[seriesIndex % SERIES_COLORS.length],
                          opacity: normalized === null ? 0.25 : 1,
                        }}
                      />
                    </span>
                    <span className="tabular w-16 shrink-0 text-right text-[11px] text-text-primary">
                      {formatValue(raw)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
