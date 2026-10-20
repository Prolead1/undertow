/**
 * Metric normalisation for the policy fingerprint (pure, no React).
 */

/**
 * Normalise a metric across the selected set with min-max scaling.
 * Finite values map to `[0, 1]`; when every value is equal (including a single
 * run) they map to `0.5` so nothing is falsely crowned "best". Non-finite and
 * missing values stay `null`.
 */
export function normalizeValues(values: readonly (number | null | undefined)[]): Array<number | null> {
  const finite = values.map((value) =>
    typeof value === "number" && Number.isFinite(value) ? value : null,
  );
  const present = finite.filter((value): value is number => value !== null);
  if (present.length === 0) return finite.map(() => null);
  const min = Math.min(...present);
  const max = Math.max(...present);
  const span = max - min;
  return finite.map((value) => {
    if (value === null) return null;
    return span === 0 ? 0.5 : (value - min) / span;
  });
}
