/**
 * Merge `search` into a destination, without overriding params already on the
 * destination. Used to keep `?run_id=&seed=&episode=` shareable across nav.
 */
export function withPreservedSearch(to: string, search: string): string {
  if (!search) return to;
  const [pathname = "", toSearch = ""] = to.split("?", 2);
  const merged = new URLSearchParams(toSearch);
  const current = new URLSearchParams(search);
  for (const [key, value] of current) {
    if (!merged.has(key)) merged.set(key, value);
  }
  const serialized = merged.toString();
  return serialized ? `${pathname}?${serialized}` : pathname;
}
