# U00 — Sim telemetry instrumentation (`telemetry.jsonl` + `status.json`)

**Wave 0 · size M · depends on: — · blocks: U02, U03, U04, U05, U08, U09**
**Branch:** `feature-sim-telemetry`

## Why this task exists

The dashboard's Live Cockpit needs a machine-readable stream of training progress. Today
`_LearningCurveCallback` records only the mean episode return (periodically, in memory) and the
manifest is written once at the end. SB3 already computes policy/value loss, entropy, approx KL, clip
fraction, explained variance and FPS in `model.logger.name_to_value` — this task persists them so the
UI (and any later analysis) can read a run *while it is training*.

This is the one task in this plan that touches `undertow.sim`. It must not change training behaviour,
determinism, or any existing artifact format. It only adds sidecars.

## Files you own

```
src/undertow/sim/train/telemetry.py        (new — TelemetryCallback, RunStatus, write_status)
src/undertow/sim/train/ppo_runner.py       (edit: wire the callback + initial/final status)
src/undertow/sim/train/runs.py             (edit: pass run_id/log paths; write terminal status)
src/undertow/sim/train/__init__.py         (edit: re-export TelemetryCallback, RunStatus)
src/undertow/sim/config.py                 (edit: add telemetry flags — additive only)
tests/sim/test_telemetry.py                (new)
```

Do **not** touch `src/undertow/ui/**` (it does not exist yet) or any other sim module.

## What to build

Read `docs/plans/ui/CONTRACTS.md` §1 first — the sidecar schemas are frozen.

1. **Config.** Add to `TrainingConfig` (append fields, do not reorder existing ones):
   ```python
   telemetry: bool = True
   record_rollouts: bool = False     # consumed by U06; declare it here so the config is stable
   rollout_max_episodes: int = 5     # consumed by U06
   ```
   `record_rollouts=False` and `telemetry=True` must reproduce today's behaviour plus the sidecars.

2. **`telemetry.py`.**
   - `SCHEMA_VERSION`, `TELEMETRY_FILENAME`, `STATUS_FILENAME` constants.
   - `RunStatus` frozen dataclass exactly as `CONTRACTS.md` §1.
   - `write_status(path, status)` — atomic (`tempfile` in the same dir + `os.replace`), sorted keys,
     trailing newline.
   - `TelemetryCallback(BaseCallback)`:
     - appends one JSON object per `log_freq_steps` timesteps to `telemetry.jsonl` **in a single
       `write()` call** (open in append mode, write `json.dumps(point) + "\n"`);
     - reads SB3 metrics from `self.logger.name_to_value`, stripping `train/` and `time/` prefixes;
       unknown keys are preserved under `train`/`time`; missing keys are omitted (reader treats them
       as null);
     - coalesces multiple timed events into at most one point per `log_freq_steps` (the existing
       `_LearningCurveCallback` uses a `while` loop — reuse that pattern);
     - refreshes `status.json` with `state="running"` at the same cadence;
     - on `_on_training_end`, writes a terminal `state="done"` status at the final timestep.
   - Keep the mean-reward sourcing identical to `_LearningCurveCallback.mean_reward()`; do not write a
     separate evaluation environment.

3. **Wiring in `ppo_runner.py`.**
   - Build the callback only when `training.telemetry` is true; always register the existing
     `_LearningCurveCallback` (it feeds the manifest and must not regress).
   - Write an initial `status.json` (`state="running"`, timestep 0, `pid=os.getpid()`) **before**
     `model.learn`, and wrap the learn call so an exception writes `state="failed"` with the message,
     then re-raises. Use `try/except/finally` — a crash must not leave a stale `running` status.
   - `run_id` is already computed in `_train_single_seed`; pass it through.
   - Keep the existing temp-dir fallback for `log_dir=None` (sidecars land there too).

4. **`runs.py`.** No signature changes. Ensure the terminal `done` status is written after the
   manifest/checkpoint is saved (so `state=done` implies the artifacts exist).

## Tests (`tests/sim/test_telemetry.py`)

Use `tmp_path` and a tiny SB3 run (mirror `tests/sim/test_training.py` fixtures). Assert real values:

1. `write_status` is atomic and round-trips every field; a concurrent read never sees a partial file
   (write, then read and compare).
2. `TelemetryCallback` writes exactly `total_timesteps // log_freq_steps` lines for a config where
   they divide, each valid JSON with the required keys and increasing `timestep`.
3. `status.json` ends in `state="done"` with the final timestep after a normal train.
4. A callback forced to raise (monkeypatch `model.learn`) leaves `state="failed"` and a non-null
   `error`, and the exception propagates.
5. `telemetry=False` writes **no** sidecars (opt-out works).
6. Metric extraction: a fake `name_to_value` dict maps `train/policy_loss` → `train.policy_loss` and
   an unknown `train/foo` is preserved.
7. `RecordSimSnapshot`-style edge: an empty info buffer yields `ep_rew_mean=None`, not `0.0`
   (document the choice: null means "not yet known").

Keep the tiny run under the existing `@pytest.mark.slow` budget conventions.

## Definition of done

- `uv run pytest tests/sim/test_telemetry.py -q` green; `uv run pytest` green (no regression to S13).
- `uv run ruff check .` and `uv run mypy src/undertow` clean.
- Sidecar schemas match `CONTRACTS.md` §1 exactly.
- `code-reviewer` APPROVE; PR against `main` from `feature-sim-telemetry`; `STATUS.md` updated.
