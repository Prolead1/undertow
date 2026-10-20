/**
 * Pinned comparison-set helpers (pure, no React).
 *
 * The pinned set is carried in the URL query (`?run_ids=a,b,c`) so a comparison
 * is shareable and restorable without a cross-view store (`PLAN.md` §8, U10
 * task).  These helpers are the single source of truth for parsing/serialising
 * it and for the client-side ≤6 guard.
 */

/** Client-side ceiling shared with `POST /api/compare` (`CONTRACTS.md` §7). */
export const MAX_COMPARE_RUNS = 6;

/** Parse `run_ids` from a query string, trimming blanks and de-duplicating. */
export function parsePinnedIds(search: string): string[] {
  const raw = new URLSearchParams(search).get("run_ids");
  if (!raw) return [];
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const part of raw.split(",")) {
    const id = part.trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    ids.push(id);
  }
  return ids;
}

/** Serialise `ids` back into `search`, removing the param when empty. */
export function serializePinnedIds(search: string, ids: readonly string[]): string {
  const params = new URLSearchParams(search);
  if (ids.length === 0) {
    params.delete("run_ids");
  } else {
    params.set("run_ids", ids.join(","));
  }
  return params.toString();
}

/**
 * Toggle a run id in the pinned set. Adding beyond `max` is a no-op so the UI
 * can never exceed the API's `max_length`; the API's 422 remains the backstop.
 */
export function togglePinnedId(
  ids: readonly string[],
  runId: string,
  max: number = MAX_COMPARE_RUNS,
): string[] {
  if (ids.includes(runId)) return ids.filter((id) => id !== runId);
  if (ids.length >= max) return [...ids];
  return [...ids, runId];
}
