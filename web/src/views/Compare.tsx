import { EmptyState } from "../components/EmptyState";
import { Panel } from "../components/Panel";

/** Placeholder replaced by U10. */
export default function Compare() {
  return (
    <Panel title="Compare policies" subtitle="Overlay pinned runs and compare their trade-offs">
      <EmptyState
        title="No policies pinned"
        description="Pin runs from the left rail or run cards, then compare their curves here. U10 replaces this placeholder."
      />
    </Panel>
  );
}
