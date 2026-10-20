import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";

/** Placeholder replaced by U10. */
export default function Leaderboard() {
  return (
    <Panel title="Leaderboard" subtitle="Ranked by held-out eval Sharpe, with provenance">
      <EmptyState
        title="No ranked policies yet"
        description="Completed runs with results are ranked here. U10 replaces this placeholder."
      />
    </Panel>
  );
}
