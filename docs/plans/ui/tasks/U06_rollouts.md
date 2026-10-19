# U06 — Rollout recorder (sim) + reader (ui)

**Wave 1 · size M · depends on: U00, U01 · blocks: U05, U09**
**Branch:** `feature-ui-rollouts`

> This task edits `train/ppo_runner.py` and `train/__init__.py`, which U00 also edits. Branch off
> U00's branch and mark the PR `Stacked on #<U00 PR> (branch feature-sim-telemetry). Merge that first.`

## Why this task exists

The Policy Replay view needs the agent's actual trajectory: at each decision, where the position was,
whether it was in range, the action taken, and the cost breakdown. The environment already computes all
of it in `step()`'s `info`; this task persists it to parquet for a handful of deterministic evaluation
episodes, and provides the reader the API layer serves.

## Files you own

```
src/undertow/sim/train/rollouts.py       (new — RolloutRecorder)
src/undertow/sim/train/ppo_runner.py     (edit — optional recording pass after training)
src/undertow/sim/train/__init__.py       (edit — re-export RolloutRecorder)
tests/sim/test_rollouts.py               (new)
src/undertow/ui/rollouts.py              (new — read parquet → RolloutSeries)
tests/ui/test_rollouts.py                (new)
```

Do not touch the evaluation runner or `undertow.sim.evaluate.*`; record via a fresh env from the same
`env_fn` the trainer already has.

## What to build

Read `CONTRACTS.md` §2 first — the parquet schema is frozen.

1. **`sim/train/rollouts.py`.**
   - `ROLLOUT_DIRNAME`, `SCHEMA_VERSION`, `RolloutRecorder` exactly as §2.
   - `record_episode(episode_index, rows)` writes `rollouts/episode_<k>.parquet` via **polars**
     (already a core dep) with the exact column names/dtypes in §2.1. Validate against the schema
     before writing: unknown keys raise `ValueError`; missing keys raise with the column name; an
     empty row sequence raises.
   - A helper `rows_from_env(info_sequence)` (optional, private) that maps the env `info` dicts
     (`reward_breakdown`, `action`, `tick_lower/upper`, `equity`, `position_value`, regime, gas price)
     into the §2.1 rows — keep the mapping in one place and unit-test it.
   - `close()` is idempotent.

2. **Wiring in `ppo_runner.py`.** After `model.learn`, when `training.record_rollouts` is true:
   - build a fresh env from `env_fn()` (do **not** reuse the training vec env), reset with the run
     seed, and step `model.predict(obs, deterministic=True)` for up to
     `training.rollout_max_episodes` episodes or until terminated/truncated;
   - capture the observation vector + `info` each step and hand them to `RolloutRecorder`;
   - this pass must not touch the training RNG or change the saved model/manifest; guard it in a
     `try/except` that logs and continues (a recording failure must never fail a training run);
   - when `record_rollouts` is false, write nothing and keep behaviour identical to U00.

3. **`ui/rollouts.py`.**
   - `list_rollouts(rollout_dir) -> list[RolloutSummary]` — sorted by `(seed, episode)`; a missing dir
     returns `[]`; a corrupt parquet is skipped with a warning.
   - `read_rollout(path, *, max_points) -> RolloutSeries` — loads through polars, serialises rows to
     dicts with ISO timestamps, downsampling with the same uniform-stride-keep-ends rule as U02.
   - Never import `undertow.sim` for the reader (the schema is the contract); read parquet directly.

## Tests

`tests/sim/test_rollouts.py`:
1. Round-trip: write rows → read back with polars → exact dtypes and values.
2. Schema validation: unknown key raises; missing key raises with its name; empty raises.
3. `record_episode` writes the expected filename and is idempotent on `close()`.
4. Integration: a tiny trained model with `record_rollouts=True` produces at least one parquet whose
   `step` column is contiguous `0..n-1` and whose `in_range` agrees with `tick_lower <= tick < tick_upper`.
5. `record_rollouts=False` writes no `rollouts/` directory.
6. A recording error (monkeypatch the recorder to raise) does not fail training.

`tests/ui/test_rollouts.py`:
7. `list_rollouts` on a missing dir → `[]`; on a fixture → correct summaries, sorted.
8. `read_rollout` downsamples to `<= max_points` keeping first+last.
9. Corrupt parquet is skipped, not raised.

## Definition of done

- `uv run pytest tests/sim/test_rollouts.py tests/ui/test_rollouts.py -q` green; full suite green at
  ≥90%; `ruff` + `mypy src/undertow` clean.
- Parquet schema matches `CONTRACTS.md` §2.1 exactly.
- `code-reviewer` APPROVE; PR from `feature-ui-rollouts`; `STATUS.md` updated.
