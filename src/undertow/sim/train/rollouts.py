"""Deterministic evaluation-rollout recording (U06, ``CONTRACTS.md`` §2).

The Policy Replay view needs the agent's actual trajectory: at every decision,
where the position was, whether it was in range, the action taken and the cost
breakdown.  The environment already computes all of it in ``step()``'s ``info``
(``LpEnvironment._make_info``) and this module persists a handful of
deterministic evaluation episodes to one parquet per episode under
``<seed_dir>/rollouts/episode_<k>.parquet``.

The parquet schema is fixed by ``CONTRACTS.md`` §2.1.  Column names, order
and dtypes are declared once in :data:`ROLLOUT_SCHEMA`; :class:`RolloutRecorder`
validates every row against it *before* writing, so a schema drift fails loudly
at the recording site instead of corrupting the UI's replay reader.  Additive
columns (e.g. the pool-state fields added for the pool-over-time chart) are
permitted when they are declared in §2.1 first, and do not bump
:data:`SCHEMA_VERSION`; readers select by name and tolerate absent columns.

The recorder never steps the training vector env: the caller
(``ppo_runner``) runs a separate, deterministic, offline pass over a fresh env
built from the same ``env_fn`` and hands the resulting rows here.  A recording
failure is the caller's concern to log-and-continue; this module simply raises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast

import polars as pl

from undertow.sim.types import Action

__all__ = [
    "ROLLOUT_DIRNAME",
    "ROLLOUT_SCHEMA",
    "SCHEMA_VERSION",
    "RolloutRecorder",
    "RolloutRow",
    "rows_from_env",
]

#: Sub-directory (under a seed directory) holding the episode parquet files.
ROLLOUT_DIRNAME: str = "rollouts"

#: Rollout artifact schema version (``CONTRACTS.md`` §0/§2).
SCHEMA_VERSION: int = 1

#: Exact §2.1 column names, order and dtypes.  A parquet written with any other
#: key set is invalid, so :meth:`RolloutRecorder.record_episode` checks against
#: this mapping before constructing the frame.
ROLLOUT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "schema_version": pl.Int16,
    "run_id": pl.Utf8,
    "seed": pl.Int64,
    "episode": pl.Int32,
    "step": pl.Int32,
    "timestamp": pl.Datetime("us", "UTC"),
    "price": pl.Float64,
    "tick": pl.Int64,
    "tick_lower": pl.Int64,
    "tick_upper": pl.Int64,
    "in_range": pl.Boolean,
    "position_value": pl.Float64,
    "equity": pl.Float64,
    "action_index": pl.Int32,
    "action_type": pl.Utf8,
    "center_offset": pl.Int32,
    "width": pl.Int32,
    "fees": pl.Float64,
    "gas": pl.Float64,
    "slippage": pl.Float64,
    "il": pl.Float64,
    "risk_penalty": pl.Float64,
    "reward": pl.Float64,
    "regime": pl.Utf8,
    "gas_price_wei": pl.Float64,
    # Additive §2.1 extension: how the external pool evolves over the episode
    # and how much of it the agent's position covers. `active_liquidity` is the
    # pool's raw L, `position_liquidity` the agent's; `pool_fee_earned_*` are the
    # fees the whole pool generated in the step (raw token units), before the
    # agent's share is taken.
    "pool_active_liquidity": pl.Float64,
    "position_liquidity": pl.Float64,
    "pool_fee_earned_0": pl.Float64,
    "pool_fee_earned_1": pl.Float64,
}


class RolloutRow(TypedDict):
    """One decision recorded in a rollout episode (``CONTRACTS.md`` §2.1)."""

    schema_version: int
    run_id: str
    seed: int
    episode: int
    step: int
    timestamp: datetime
    price: float
    tick: int
    tick_lower: int
    tick_upper: int
    in_range: bool
    position_value: float
    equity: float
    action_index: int
    action_type: str
    center_offset: int
    width: int
    fees: float
    gas: float
    slippage: float
    il: float
    risk_penalty: float
    reward: float
    regime: str
    gas_price_wei: float
    pool_active_liquidity: float
    position_liquidity: float
    pool_fee_earned_0: float
    pool_fee_earned_1: float


def rows_from_env(
    info_sequence: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    seed: int,
    episode: int,
) -> list[RolloutRow]:
    """Map a sequence of env ``info`` dicts to §2.1 rows.

    This is the *single* place the environment ``info`` layout is translated into
    the frozen parquet schema, so a change to either side is a one-line review.

    **Units.** The monetary cost columns (``fees``, ``gas``, ``slippage``, ``il``,
    ``risk_penalty``) are read from ``info["reward_breakdown_raw"]`` and are
    therefore USDC, matching ``equity``/``position_value`` (``CONTRACTS.md`` §0).
    Only ``reward`` is the agent's learning signal, so it is read from the
    capital-normalised ``info["reward_breakdown"]``. Mixing the two dicts would
    scale costs down by the initial capital (``~100,000x`` for the default
    capital) and break the UI's cost/equity arithmetic.

    ``in_range`` is recomputed from the recorded ticks rather than trusted, which
    keeps the parquet invariant ``tick_lower <= tick < tick_upper`` true by
    construction (the UI replay view relies on it).
    """
    rows: list[RolloutRow] = []
    for info in info_sequence:
        breakdown = cast(Mapping[str, object], info["reward_breakdown"])
        raw = cast(Mapping[str, object], info["reward_breakdown_raw"])
        action = cast(Action, info["action"])
        tick = int(cast(int, info["tick"]))
        tick_lower = int(cast(int, info["tick_lower"]))
        tick_upper = int(cast(int, info["tick_upper"]))
        row: RolloutRow = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "seed": int(seed),
            "episode": int(episode),
            "step": int(cast(int, info["step_in_episode"])),
            "timestamp": cast(datetime, info["timestamp"]),
            "price": float(cast(float, info["price"])),
            "tick": tick,
            "tick_lower": tick_lower,
            "tick_upper": tick_upper,
            "in_range": tick_lower <= tick < tick_upper,
            "position_value": float(cast(float, info["position_value"])),
            "equity": float(cast(float, info["equity"])),
            "action_index": int(cast(int, info["action_index"])),
            "action_type": str(info.get("action_type", action.action_type)),
            "center_offset": int(action.center_offset),
            "width": int(action.width),
            "fees": float(cast(float, raw["fees"])),
            "gas": float(cast(float, raw["gas"])),
            "slippage": float(cast(float, raw["slippage"])),
            "il": float(cast(float, raw["il_change"])),
            "risk_penalty": float(cast(float, raw["risk_penalty"])),
            "reward": float(cast(float, breakdown["reward"])),
            "regime": str(info["regime"]),
            "gas_price_wei": float(cast(float, info["gas_price_wei"])),
            "pool_active_liquidity": float(
                cast(float, info.get("active_liquidity") or 0.0)
            ),
            "position_liquidity": float(
                cast(float, info.get("position_liquidity") or 0.0)
            ),
            "pool_fee_earned_0": float(
                cast(float, info.get("pool_fee_earned_0") or 0.0)
            ),
            "pool_fee_earned_1": float(
                cast(float, info.get("pool_fee_earned_1") or 0.0)
            ),
        }
        rows.append(row)
    return rows


class RolloutRecorder:
    """Write deterministic evaluation episodes to ``rollouts/episode_<k>.parquet``.

    ``out_dir`` is the run/seed directory; the recorder creates ``out_dir /
    :data:`ROLLOUT_DIRNAME``` lazily on the first successful write, so opting out
    of recording leaves no empty directory behind.  The object holds no file
    handle: every episode is written and closed atomically by polars, making
    :meth:`close` a trivial idempotent flag flip.
    """

    def __init__(
        self, run_id: str, seed: int, out_dir: Path, max_episodes: int = 5
    ) -> None:
        self.run_id = str(run_id)
        self.seed = int(seed)
        self.out_dir = Path(out_dir) / ROLLOUT_DIRNAME
        self.max_episodes = int(max_episodes)
        self._closed = False

    def record_episode(
        self, episode_index: int, rows: Sequence[RolloutRow]
    ) -> Path:
        """Validate ``rows`` against §2.1 and write ``episode_<k>.parquet``.

        Raises :class:`ValueError` for an empty episode, an out-of-range episode
        index, an unknown row key or a missing column (the message names the
        offending column so a caller can fix the mapping without guessing).
        """
        if self._closed:
            raise ValueError("RolloutRecorder.record_episode called after close()")
        index = int(episode_index)
        if index < 0 or index >= self.max_episodes:
            raise ValueError(
                f"episode {index} is outside [0, {self.max_episodes}) "
                "for this recorder"
            )
        materialised = [dict(cast(Mapping[str, object], row)) for row in rows]
        if not materialised:
            raise ValueError("cannot record an empty rollout episode")

        declared = set(ROLLOUT_SCHEMA)
        for position, row in enumerate(materialised):
            keys = set(row)
            unknown = keys - declared
            if unknown:
                raise ValueError(
                    f"unknown rollout column(s) at row {position}: "
                    f"{', '.join(sorted(unknown))}"
                )
            missing = declared - keys
            if missing:
                raise ValueError(
                    f"missing rollout column(s) at row {position}: "
                    f"{', '.join(sorted(missing))}"
                )

        frame = pl.DataFrame(materialised, schema=ROLLOUT_SCHEMA, strict=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"episode_{index}.parquet"
        frame.write_parquet(path)
        return path

    def close(self) -> None:
        """Mark the recorder closed. Safe to call more than once."""
        self._closed = True
