# S17 — Episode window resampling on reset

**Wave 6 · size S · depends on: S03, S12 · blocks: S20, S23**
**Branch:** `bug-sim-episode-window-resampling`

## Why this task exists

Training runs `runs/experiment2` and `runs/experiment3` produced perfectly flat
learning curves because every episode replayed the **same** 30-day window.

Root cause: `LpEnvironment.reset(seed=None)` keeps the previous `self._seed`, and
`MarketView.build_episode(self._seed, …)` samples the window from that seed. SB3's
`VecEnv` auto-reset calls `env.reset()` **without** a seed (verified in the
installed `DummyVecEnv.step_wait`), so all 1M training steps see one window. The
agent can therefore never learn regime-conditional behaviour, which is the core
of the thesis (ADR-030 C2/C3).

## Files you own

```
src/undertow/sim/env/lp_env.py        (reset() + seed bookkeeping only)
tests/sim/test_env.py                 (new tests; extend TestReset)
```

Do **not** touch `_rebalance`, the reward, or `observations.py` — S18 and S20 own
those. Do not modify `MarketView.build_episode`.

## What to build

### 1. Seed bookkeeping in `LpEnvironment.__init__`

Replace the single `self._seed` with:

```python
self._base_seed: int = int(config.seed)
self._episode_index: int = -1
self._rng: np.random.Generator = np.random.default_rng(self._base_seed)
```

Keep `self._rng` as the process-wide generator used by `price_process.step`
(replay ignores it; calibrated uses it).

### 2. `reset()` semantics

```python
if seed is not None:
    self._base_seed = int(seed)
    self._episode_index = 0
else:
    self._episode_index += 1
window_seed = self._base_seed + self._episode_index
self._rng = np.random.default_rng(window_seed)
...
start_seq, end_seq = self.market_view.build_episode(
    window_seed, max_steps=config_steps + 1, episode_config=episode
)
```

Required behaviour (pin these exactly):

| call sequence | window seed |
|---|---|
| first `reset(seed=42)` | 42 |
| `reset(seed=42)` again | 42 (identical window and observation) |
| `reset(seed=42)` then `reset()` | 42 then 43 |
| first `reset()` (no prior seed) | `config.seed` (index −1 → 0) |
| successive `reset()` calls | `config.seed`, `config.seed+1`, … |

`reset(seed=k)` must always return to window `k` (reproducibility), regardless of
how many no-seed resets preceded it.

Update the class docstring "Determinism" paragraph: a seeded reset is
reproducible; a no-seed reset advances to the next window so a training loop sees
a new episode each time.

### 3. Tests (`tests/sim/test_env.py`, extend `TestReset`)

Use `_calibrated_env(tiny_market_view, …)` (it already exposes `_start_seq` and
has multiple candidate windows — `test_reset_seed_selects_episode_window` proves
seeds 42 and 7 give different starts).

1. `test_reset_without_seed_advances_window` — after `reset(seed=0)`, two
   successive `reset()` calls produce at least two distinct `_start_seq` values,
   and none equals the seeded window unless the view genuinely has one candidate
   (skip in that case with a clear reason).
2. `test_reset_without_seed_is_reproducible_from_a_seeded_start` — sequence
   `[reset(seed=3), reset(), reset()]` twice yields identical `_start_seq`
   sequences.
3. `test_reset_with_seed_returns_to_that_window` — after several no-seed resets,
   `reset(seed=3)` reproduces the first seeded `_start_seq`.
4. `test_first_reset_without_seed_uses_config_seed` — a fresh env's first
   `reset()` uses `config.seed` (assert `_base_seed + _episode_index ==
   config.seed`).
5. Keep all existing reset tests passing unchanged (they must not be edited to
   accommodate the new behaviour).

Assert real values (`_start_seq`, observation arrays), not just "runs".

## Acceptance criteria

- `uv run pytest tests/sim/test_env.py -q` green.
- `uv run pytest -q` green (full suite).
- No change to `build_episode`, the reward, or the observation vector.
- The diff is confined to `lp_env.py` seed bookkeeping + `reset()` and the new
  tests.

## Out of scope

- Slippage/rebalance cost (S18).
- Any change to `TrainingConfig` or the PPO runner — SB3's auto-reset is exactly
  the code path being fixed, so no harness change is needed.
