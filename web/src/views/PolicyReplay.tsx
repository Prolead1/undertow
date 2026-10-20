import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";

/** Placeholder replaced by U09. */
export default function PolicyReplay() {
  return (
    <Panel title="Policy replay" subtitle="Step through a recorded rollout episode">
      <EmptyState
        title="No rollout selected"
        description="Choose a run with recorded rollouts to replay its actions and position range. U09 replaces this placeholder."
      />
    </Panel>
  );
}
