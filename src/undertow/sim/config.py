"""Simulation configuration tree for ``undertow.sim``.

All knobs live in a frozen dataclass hierarchy. Implements `CONTRACTS.md` §2
exactly. No logic here — just types, defaults, validation, and the TOML loader.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

from undertow.sim.types import PriceMode, SimConfigError

# ---------------------------------------------------------------------------
# Gas units per action type (source: representative NFT-position-manager costs)
# ---------------------------------------------------------------------------
GAS_MINT: int = 460_000
GAS_BURN: int = 215_000
GAS_COLLECT: int = 130_000
GAS_REBALANCE_SWAP: int = 150_000  # approximate; sweepable

# -- Slippage defaults --
FIXED_IMPACT_BPS: float = 5.0  # basis points


# ---------------------------------------------------------------------------
# Config tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    duration_days: int = 30
    step_minutes: int = 10  # decision cadence
    agent_capital_usdc: float = 100_000.0
    marginal_agent: bool = True  # agent liquidity never moves the price path
    fee_tier_bps: int = 3000  # 0.30% default for WETH/USDC
    tick_spacing: int = 60


@dataclass(frozen=True, slots=True)
class ActionGridConfig:
    center_offsets: Sequence[int] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    widths: Sequence[int] = (1, 2, 5, 10, 25, 50)
    include_hold: bool = True

    def __post_init__(self) -> None:
        if len(set(self.center_offsets)) != len(self.center_offsets):
            raise SimConfigError("center_offsets must not contain duplicates")
        if len(set(self.widths)) != len(self.widths):
            raise SimConfigError("widths must not contain duplicates")


@dataclass(frozen=True, slots=True)
class GasConfig:
    mint_units: int = GAS_MINT
    burn_units: int = GAS_BURN
    collect_units: int = GAS_COLLECT
    rebalance_swap_units: int = GAS_REBALANCE_SWAP


@dataclass(frozen=True, slots=True)
class SlippageConfig:
    fixed_impact_bps: float = FIXED_IMPACT_BPS


@dataclass(frozen=True, slots=True)
class RewardConfig:
    risk_penalty_lambda: float = 0.0  # default off; ablatable
    risk_rolling_window_steps: int = 144  # 1 day at 10-min steps
    normalize_by_capital: bool = True  # reward = ΔPnL_net / W_0
    # Ablation flags (RQ1 instrument)
    fees_enabled: bool = True
    gas_enabled: bool = True
    slippage_enabled: bool = True
    il_enabled: bool = True
    risk_penalty_enabled: bool = False


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_start_utc: datetime = datetime(2022, 1, 1, tzinfo=UTC)
    train_end_utc: datetime = datetime(2023, 12, 31, tzinfo=UTC)
    eval_start_utc: datetime = datetime(2024, 1, 1, tzinfo=UTC)
    eval_end_utc: datetime = datetime(2024, 12, 31, tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.train_end_utc > self.eval_start_utc:
            raise SimConfigError(
                "train_end_utc must be <= eval_start_utc (splits must be disjoint)"
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    algorithm: Literal["ppo"] = "ppo"
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.15
    hidden_size: int = 256
    n_hidden: int = 2
    parallel_envs: int = 4
    total_timesteps: int = 1_000_000
    checkpoint_freq_steps: int = 100_000
    log_freq_steps: int = 10_000


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    compute_decomposition: bool = True
    log_decision_every_step: bool = False


@dataclass(frozen=True, slots=True)
class SimConfig:
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    action_grid: ActionGridConfig = field(default_factory=ActionGridConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    price_mode: PriceMode = PriceMode.REPLAY
    seed: int = 0
    output_dir: Path | None = None

    def rng(self) -> np.random.Generator:
        """Construct a seeded ``numpy.random.Generator`` from ``self.seed``."""
        return np.random.default_rng(self.seed)


def load_sim_config(path: str | Path) -> SimConfig:
    """Load a ``SimConfig`` from a TOML file.

    The file follows the structure of ``configs/sim_default.toml`` — one
    ``[section]`` per config subtree, with keys matching the dataclass fields.
    Omitted sections/keys fall back to defaults.

    Raises ``SimConfigError`` if the file contains unknown top-level sections.
    """
    import typing

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    _known_sections = {
        "episode",
        "action_grid",
        "gas",
        "slippage",
        "reward",
        "split",
        "training",
        "backtest",
        "price",
    }
    unknown = set(raw.keys()) - _known_sections
    if unknown:
        raise SimConfigError(
            f"Unknown config section(s): {', '.join(sorted(unknown))}"
        )

    def _populate(cls: type, data: dict | None) -> object:
        """Build a frozen dataclass from a dict, falling back to defaults."""
        data = data or {}
        type_hints = typing.get_type_hints(cls)
        kwargs: dict[str, object] = {}
        for f_name, f_type in type_hints.items():
            if f_name in data:
                val = data[f_name]
                # Convert TOML arrays to tuples for Sequence-typed fields
                if isinstance(val, list):
                    origin = typing.get_origin(f_type)
                    if origin is not None and issubclass(
                        origin, Sequence
                    ):
                        val = tuple(val)
                kwargs[f_name] = val
        return cls(**kwargs)

    price_mode_raw = raw.get("price", {}).get("mode", "replay")
    price_mode = (
        PriceMode(price_mode_raw)
        if isinstance(price_mode_raw, str)
        else PriceMode.REPLAY
    )

    return SimConfig(
        episode=_populate(EpisodeConfig, raw.get("episode")),
        action_grid=_populate(ActionGridConfig, raw.get("action_grid")),
        gas=_populate(GasConfig, raw.get("gas")),
        slippage=_populate(SlippageConfig, raw.get("slippage")),
        reward=_populate(RewardConfig, raw.get("reward")),
        split=_populate(SplitConfig, raw.get("split")),
        training=_populate(TrainingConfig, raw.get("training")),
        backtest=_populate(BacktestConfig, raw.get("backtest")),
        price_mode=price_mode,
    )