# U02 — Run registry, path discovery, schemas, results parsing

**Wave 1 · size M · depends on: U00, U01 · blocks: U04, U05, U08, U10**
**Branch:** `feature-ui-registry`

> If U00's PR is not merged yet, branch off U00's branch and mark this PR
> `Stacked on #<U00 PR> (branch feature-sim-telemetry). Merge that first.`

## Why this task exists

Every view needs to answer "what runs exist, and what is the state/provenance of each?". This task
turns a `runs/` directory tree into a deterministic in-memory registry of `RunRecord`s, and parses
optional `results/` evaluation artifacts into `EvalMetrics`. It is the backend's source of truth.

## Files you own

```
src/undertow/ui/schemas.py        (new — every pydantic wire model, CONTRACTS §4)
src/undertow/ui/paths.py          (new — run-directory layout discovery)
src/undertow/ui/registry.py       (new — RunRegistry: scan/refresh/lookup)
src/undertow/ui/results.py        (new — parse results/ markdown + CSV into EvalMetrics)
tests/ui/test_registry.py         (new)
tests/ui/test_results.py          (new)
tests/ui/conftest.py              (edit — add the shared `run_dir` factory fixture)
```

Do not touch `telemetry.py`, `streaming.py`, `figures/`, `app.py` (later tasks).

## What to build

Read `CONTRACTS.md` §4 first — the models are frozen.

1. **`schemas.py`.** All pydantic v2 models from §4 (`TrainMetrics`, `TimeMetrics`,
   `TelemetryPoint`, `EvalMetrics`, `RunManifestSummary`, `RunRecord`, `LearningCurve`,
   `RolloutSummary`, `RolloutSeries`), plus re-export of `RunStatus` from
   `undertow.sim.train.telemetry` (import the sim dataclass and adapt it — do **not** redefine the
   schema; if importing sim here is undesirable, build a `RunStatusModel` that validates the same
   fields and document it). Configure `model_config = ConfigDict(extra="allow")` only where forward
   compatibility requires it (telemetry metrics), otherwise forbid extras.

2. **`paths.py`.** Pure functions over a `runs_dir`:
   - `discover_run_dirs(root) -> list[RunDir]`, where a run dir is any directory containing
     `manifest.json` **or** `telemetry.jsonl` **or** `status.json`. Support both layouts:
     `runs/<experiment>/seed_<n>/` and a bare `runs/seed_<n>/`.
   - `parse_run_dir(path) -> RunDir` with `(experiment, seed, run_id, manifest_path, telemetry_path,
     status_path, rollout_dir)`; `run_id` prefers the manifest's value, else `status.json`, else
     `<experiment>/seed_<n>`.
   - Layout discovery must be deterministic and tolerate extra files.

3. **`registry.py`.** `RunRegistry(runs_dir, results_dir, max_points)`:
   - `scan() -> list[RunRecord]` reads each run dir: manifest (if present), `status.json` (if
     present), counts telemetry lines, lists rollout parquet files, attaches `EvalMetrics` when a
     matching results artifact exists.
   - `get(run_id) -> RunRecord | None`; `curve(run_id) -> LearningCurve | None` (aggregates per-seed
     curves from manifests; computes mean/std and downsamples to `max_points` with a **uniform
     stride that always keeps the first and last point**); `refresh()` re-scans.
   - Ranking helper `rank(records, key)` implementing the pinned rule (eval Sharpe → worst-regime PnL
     → `best_reward`), used by U10.
   - Corrupt/partial files must not crash the scan: a bad `manifest.json` yields a record with
     `state="unknown"` and null provenance, and is logged at `warning`. Never raise out of `scan()`.

4. **`results.py`.** Parse the S15/S16 artifacts into `EvalMetrics`: `regime_comparison.md` (or
   `regime_matrix_*.md`) for per-regime PnL and `worst_regime_pnl`, and `summary.json` /
   `results_summary.md` for Sharpe/PnL/MaxDD where available. Be defensive: unknown markdown shapes
   return `None` fields, not exceptions. Record the relative `source` path.

5. **`conftest.py`.** Add `run_dir(tmp_path)` factory producing a synthetic run tree (manifest +
   telemetry + status + one rollout parquet + optional results dir) reusable by U03/U04/U05/U06. Keep
   it small and documented; later tasks extend it only by adding new fixtures, not editing this one.

## Tests

`test_registry.py`:
1. Empty root → `[]`.
2. One well-formed run → fields populated, `state` from status, curve lengths correct.
3. Both layouts (`<exp>/seed_n`, bare `seed_n`) discovered.
4. Corrupt `manifest.json` → record present, `state="unknown"`, no exception.
5. Deterministic ordering (two records, assert order).
6. `curve()` mean/std match a hand-computed fixture; downsampling keeps first+last and respects
   `max_points`.
7. `rank()` orders by Sharpe, then worst-regime PnL, then `best_reward`; a missing eval falls back.

`test_results.py`:
8. Realistic `regime_comparison.md` → expected per-regime dict + `worst_regime_pnl`.
9. Unrecognised markdown → all-None metrics, no raise.
10. `summary.json` parsing maps keys correctly.

## Definition of done

- `uv run pytest tests/ui -q` green; full suite green at ≥90%.
- `ruff` + `mypy src/undertow` clean.
- `RunRecord`/`LearningCurve` match `CONTRACTS.md` §4 exactly.
- `code-reviewer` APPROVE; PR from `feature-ui-registry`; `STATUS.md` updated.
