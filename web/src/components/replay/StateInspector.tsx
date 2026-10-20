/* eslint-disable react-refresh/only-export-components -- U09 colocated pure helpers are exported for unit testing. */
import { EmptyState } from "../EmptyState";
import { Panel } from "../Panel";
import { Sparkline } from "../Sparkline";
import { Stat } from "../Stat";
import type { RolloutRow } from "./rows";

/** Human label for the discrete action at a row. */
export function actionLabel(row: RolloutRow): string {
  if (row.actionType !== "rebalance") return "hold";
  const sign = row.centerOffset >= 0 ? "+" : "";
  return `rebalance(offset=${sign}${row.centerOffset}, width=${row.width})`;
}

function observationWidth(rows: readonly RolloutRow[]): number {
  let width = 0;
  for (const row of rows) {
    if (row.observation) width = Math.max(width, row.observation.length);
  }
  return width;
}

export interface StateInspectorProps {
  rows: readonly RolloutRow[];
  cursorIndex: number;
}

/** Cursor-step readout: action, range status, position value, gas and regime. */
export function StateInspector({ rows, cursorIndex }: StateInspectorProps) {
  const row = rows[cursorIndex];
  const obsWidth = observationWidth(rows);

  return (
    <Panel title="State inspector" subtitle="Everything the policy saw at the cursor step">
      {row === undefined ? (
        <EmptyState title="No step selected" description="Move the time cursor to inspect a step." />
      ) : (
        <div className="grid gap-3">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Stat label="Step" value={row.step} />
            <Stat
              label="Action"
              value={actionLabel(row)}
              tone={row.actionType === "rebalance" ? "accent" : "muted"}
            />
            <Stat
              label="Range"
              value={row.inRange ? "in range" : "out of range"}
              tone={row.inRange ? "bull" : "bear"}
            />
            <Stat
              label="Position value"
              value={row.positionValue === null ? "—" : row.positionValue.toFixed(2)}
              hint="USDC"
            />
            <Stat
              label="Gas price"
              value={row.gasPriceWei === null ? "—" : `${row.gasPriceWei.toFixed(2)}`}
              hint="wei"
            />
            <Stat label="Regime" value={row.regime} />
          </div>

          {obsWidth > 0 && (
            <div>
              <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-text-muted">
                Observation ({obsWidth} dims)
              </p>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {Array.from({ length: obsWidth }, (_, dim) => (
                  <div key={dim} className="flex items-center gap-2">
                    <span className="tabular w-12 shrink-0 text-[11px] text-text-muted">
                      obs {dim}
                    </span>
                    <Sparkline
                      values={rows.map((entry) => entry.observation?.[dim] ?? 0)}
                      width={80}
                      height={20}
                      label={`observation dimension ${dim}`}
                      tone="accent-2"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}