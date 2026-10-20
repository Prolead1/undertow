import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";

/** Placeholder replaced by U08. */
export default function LiveCockpit() {
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Live cockpit" subtitle="Streaming telemetry, reward curve and policy health">
        <EmptyState
          title="Waiting for a live run"
          description="Select a run in the left rail. U08 replaces this placeholder with the live charts."
        />
      </Panel>
      <Panel title="Position range" subtitle="The hero ribbon: price against the active tick range">
        <EmptyState
          title="No telemetry selected"
          description="The concentrated-liquidity ribbon will render here once a run is streaming."
        />
      </Panel>
    </div>
  );
}
