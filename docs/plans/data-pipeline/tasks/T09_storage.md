# T09 — Parquet storage + dataset manifest

**Wave 3 · size M · depends on: T03 · blocks: T11, T14, T16**
**Branch:** `feature-data-storage`

## Why this task exists

The Week-1 deliverable is *"a validated, reproducible dataset loader (script + a parquet snapshot)"*
(§10.6), and §10.7's headline mitigation for the schedule risk is *"cache to parquet so you never
re-query."* This module is that snapshot layer. "Reproducible" is the operative word: the same config
over the same block range must produce the same bytes, and every dataset must carry enough provenance
that a number in the thesis can be traced to the query that produced it.

## Files you own

```
src/undertow/data/storage/__init__.py
src/undertow/data/storage/parquet.py
src/undertow/data/storage/manifest.py
tests/data/test_storage_parquet.py
tests/data/test_storage_manifest.py
```

## What to build

Implement `CONTRACTS.md` §7.

### 1. `parquet.py`

**Two stream families — read `CONTRACTS.md` §4.0 first.** Not every stream is pool-scoped and not every
stream is keyed by `(block_number, log_index)`. Partitioning `reference` (Binance klines) under a pool
address would be wrong and would duplicate the same bars under both pinned pools.

- Pool-scoped streams: `swap`, `mint`, `burn`, `collect`, `flash`, `fee_growth`, `event_tape`
  → `root/pool=<address>/stream=<stream>/block_bucket=<block_number // 100_000>/part.parquet`
- Global streams: `gas` (chain-wide), `reference`, `regime`
  → `root/stream=<stream>/...` with no `pool=` level. `gas` still buckets by block; `reference` and
  `regime` bucket by month (`YYYY-MM` from their timestamp column) since they have no block column.
- Take the classification from T03's `POOL_SCOPED` frozenset and the sort key from T03's `SORT_KEYS`
  — do **not** hardcode either here, and do not assume `(block_number, log_index)` exists on every
  stream.

Hive-style partitioning throughout so `read_stream` can push predicates down to whole directories.
- `write_stream` **must be deterministic**: validate against the schema, sort by that stream's
  `SORT_KEYS`, write with fixed options (`compression="zstd"`, fixed `compression_level`, fixed
  `row_group_size`, `write_statistics=True`), and no wall-clock, hostname, or dict-ordering leakage into
  the file. Note that Parquet writers may embed a `created_by` string — that is acceptable (it is
  metadata, not data), but `content_hash` must be computed over the **table**, not the file bytes, so it
  stays stable across pyarrow versions. Document that distinction; it is subtle and someone will
  otherwise "fix" it wrongly.
- `read_stream(root, stream, *, pool, start_block=None, end_block=None)` — predicate pushdown on
  `block_number` *and* bucket pruning; returns a schema-validated table sorted by the key. Reading a
  range that spans buckets must not duplicate or reorder rows.
- Idempotent re-write: writing the same stream twice leaves one logical dataset, not two part files with
  duplicated rows. Decide the mechanism (overwrite the bucket's part file) and test it — accidental row
  duplication on a re-run is exactly the bug that makes a "resumable" pipeline silently wrong.
- Partial-write safety: write to a temp path and `os.replace` into place, so an interrupted `pull` never
  leaves a half-written part that reads as valid.
- `content_hash(table)` — stable across runs and machines: hash the schema (names + types, in order) then
  the row data in sorted key order. Must **not** depend on row-group boundaries, compression, or the
  pyarrow version.

### 2. `manifest.py`
`DatasetManifest` / `StreamManifest` per the contract, serialized to `manifest.json` (pretty-printed,
sorted keys, so it diffs cleanly in git).

Per-stream: `row_count`, `min_block`, `max_block`, `content_hash`, `route`, `n_requests`,
`warnings`. Dataset-level: `schema_version`, `dataset_id`, `pool`, `window`, `created_at_utc`,
`git_commit`, `library_versions`.

- `dataset_id` = deterministic hash of `(pool address, window, sorted route set, schema_version)` —
  **not** of the wall clock. Same logical dataset ⇒ same id, so a re-pull is recognizable as the same
  thing.
- `git_commit` from `git rev-parse HEAD`, with a `"+dirty"` suffix when the working tree is dirty. A
  dirty tree producing a thesis number is something the human must be able to see afterwards. If git is
  unavailable, store `"unknown"` — never crash.
- `library_versions` for `pyarrow`, `polars`, `numpy`, `requests`, and `undertow` itself.
- **Secrets never enter the manifest.** Store the endpoint **host** only (`gateway.thegraph.com`), never
  the full URL, never a key. Test this with a URL containing a key.
- `read_manifest` must reject a manifest whose `schema_version` is incompatible with the current
  `SCHEMA_VERSION` (major-version mismatch ⇒ `ValidationError` telling the user to re-pull). This is how
  a schema change stops silently mixing old and new parquet.

## Tests you must write

1. Round-trip: write a synthetic table for each schema in `SCHEMA_REGISTRY`, read it back, assert
   equality including types and column order (parametrize over the registry — this is what forces you to
   handle the global streams, which have no `pool_address` and no `log_index`).
1b. Layout: a pool-scoped stream lands under a `pool=` directory; a global stream does **not**. Writing
   `reference` for two different pools produces **one** copy on disk, not two.
2. **Determinism:** write the same table twice to two roots → identical `content_hash`, and identical
   file bytes for the part files (if byte-identity proves brittle across pyarrow versions, assert hash
   identity and document why in the test's docstring — do not just delete the assertion).
3. Sorting: an input table in shuffled order is written sorted; `read_stream` returns it sorted.
4. Predicate pushdown: a range query spanning three buckets returns exactly the in-range rows, in order,
   with no duplicates; and a query fully inside one bucket touches only that bucket (assert via a
   filesystem-read spy or by asserting row counts on a dataset whose buckets have distinguishable data).
5. Idempotent re-write: writing the same stream twice → row count unchanged, hash unchanged.
6. Interrupted write: patch the writer to raise mid-write → no part file left behind; the prior contents
   (if any) survive intact.
7. Empty table: `write_stream(empty_table(SWAP_SCHEMA), ...)` then `read_stream` returns 0 rows,
   schema-valid, no exception.
8. Big-integer fidelity: a table containing `2**256 - 1` as a string-encoded column round-trips to the
   exact same integer after `decode_uint`. This is the whole reason for the string encoding — test it.
9. Manifest round-trip: write → read → equal dataclass, with `created_at_utc` UTC-aware.
10. `dataset_id` stability: same inputs ⇒ same id; a changed window ⇒ different id; a changed
    `created_at_utc` ⇒ **same** id.
11. Secret hygiene: a `graph_url` containing `abc123KEY` produces a manifest whose serialized JSON
    contains neither `abc123KEY` nor the full URL.
12. Schema-version guard: a manifest written with `"0.9.0"` read under `SCHEMA_VERSION == "1.0.0"`
    raises `ValidationError` mentioning re-pulling.
13. Dirty-tree marker: with a mocked git returning a dirty status, `git_commit` ends in `+dirty`.

## Acceptance criteria

- `uv run pytest tests/data/test_storage_*.py` green; `mypy` clean.
- No fetching, no domain math. This module moves tables to and from disk and records provenance.
- `manifest.json` is human-readable and diff-friendly.

## Handoff notes for your PR body

- The exact on-disk layout strings, both families (T14's `--out` and T16's loader depend on them).
- `content_hash`'s definition, precisely — T13 may re-verify it.
- Whether byte-identity or hash-identity was used for the determinism test, and why.

## Process (mandatory)

Branch → implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
