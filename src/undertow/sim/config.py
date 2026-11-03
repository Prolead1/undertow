"""Simulation configuration tree for ``undertow.sim``.

All knobs live in a frozen dataclass hierarchy. Implements `CONTRACTS.md` §2
exactly. No logic here — just types, defaults, validation, and the TOML loader.
"""

from __future__ import annotations

import math
import tomllib
import typing
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
    fee_tier_bps: int = 500  # 0.05% default: the primary USDC/WETH study pool
    tick_spacing: int = 10
    #: Minimum number of decision steps between rebalance *attempts* for the
    #: learned policy.  ``0`` disables the constraint and preserves historical
    #: behaviour exactly.  A positive value makes the environment coerce an
    #: earlier rebalance attempt to ``hold``, and the checkpoint evaluation
    #: wrapper (``_CheckpointPolicy``) gates identically so the simulator and
    #: the backtester agree (ADR-037).  Scripted baselines ignore this field,
    #: but the environment gate is policy-agnostic: keep it at ``0`` for
    #: baseline and parity runs, because the exact backtester has no gate.
    min_rebalance_interval_steps: int = 0

    def __post_init__(self) -> None:
        interval = float(self.min_rebalance_interval_steps)
        if not math.isfinite(interval) or not interval.is_integer():
            raise SimConfigError(
                "episode.min_rebalance_interval_steps must be a finite integer, "
                f"got {self.min_rebalance_interval_steps!r}"
            )
        if interval < 0:
            raise SimConfigError(
                "episode.min_rebalance_interval_steps must be >= 0, "
                f"got {self.min_rebalance_interval_steps!r}"
            )


@dataclass(frozen=True, slots=True)
class ActionGridConfig:
    center_offsets: Sequence[int] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    #: Widths are half-ranges in tick-spacing units.  The ladder matches the
    #: primary 0.05 % pool's spacing-10 grid (``configs/sim_default.toml``):
    #: larger values are needed than at the 0.30 % contrast pool's spacing 60 to
    #: cover the same price band.
    widths: Sequence[int] = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
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
    #: Gate the hedge PnL/funding/fee terms (S20, ADR-030).  ``True`` keeps the
    #: terms in the learning reward; ``False`` is the RQ1 "hedge off" ablation
    #: (the ungated `pnl_net` still records them).
    hedge_enabled: bool = True


@dataclass(frozen=True, slots=True)
class HedgeConfig:
    """Short-perp hedge overlay configuration (S20, ``CONTRACTS.md`` §2).

    Disabled by default: with ``enabled=False`` the environment, reward and
    observation are exactly the LP-only ones (the hedge terms are zero).

    ``ratios`` is the discrete hedge-ratio head S20b will expose; this task
    only consumes ``ratio`` (the deterministic overlay's target fraction of the
    LP's ETH delta).
    """

    enabled: bool = False
    fee_bps: float = 5.0  # perp taker fee, true basis points (S19)
    funding_enabled: bool = True
    ratio: float = 1.0  # deterministic overlay target = ratio * LP ETH delta
    ratios: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)  # S20b action head

    def __post_init__(self) -> None:
        fee_bps = float(self.fee_bps)
        if not math.isfinite(fee_bps) or fee_bps < 0.0:
            raise SimConfigError(f"hedge.fee_bps must be >= 0, got {self.fee_bps!r}")
        ratio = float(self.ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise SimConfigError(f"hedge.ratio must be in [0, 1], got {self.ratio!r}")
        for value in self.ratios:
            entry = float(value)
            if not math.isfinite(entry) or not 0.0 <= entry <= 1.0:
                raise SimConfigError(
                    f"hedge.ratios entries must be in [0, 1], got {value!r}"
                )


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
    #: PPO optimisation knobs.  ``learning_rate``/``batch_size``/``n_epochs``
    #: are the PLAN.md §7 anchor values; ``ent_coef`` defaults to SB3 2.9's PPO
    #: default of ``0.0`` (the runner historically never passed the kwarg), so
    #: together they preserve the historical training behaviour.  The primary
    #: 0.05 % pool overrides them (``configs/sim_eth_usdc_500.toml``) to
    #: stabilize noisy gradient updates -- a 64-sample mini-batch over ten
    #: epochs moved the policy too far from the behaviour policy, so the hold
    #: fraction thrashed update-to-update even though the captured advantages
    #: favoured holding (ADR-036).  The primary recipe is ``learning_rate=1e-4``,
    #: ``batch_size=512``, ``n_epochs=4``, ``ent_coef=0.001``, ``target_kl=0.01``.
    learning_rate: float = 3e-4
    #: PPO mini-batch size, snapped to a divisor of the rollout by
    #: ``ppo_runner._derive_rollout`` so SB3's ``rollout % batch_size == 0``
    #: invariant always holds.
    batch_size: int = 64
    #: Number of PPO epochs over each rollout.
    n_epochs: int = 10
    #: Entropy bonus coefficient.  Defaults to SB3 2.9's PPO default ``0.0``;
    #: the primary 0.05 % pool sets ``0.001`` (ADR-036).
    ent_coef: float = 0.0
    #: SB3 KL early-stopping threshold; ``None`` disables it.  The primary pool
    #: sets a small finite value so an update cannot drift far from the
    #: behaviour policy that collected the rollout.
    target_kl: float | None = None
    parallel_envs: int = 4
    total_timesteps: int = 1_000_000
    checkpoint_freq_steps: int = 100_000
    log_freq_steps: int = 10_000
    #: Vector-env backend.  ``"subproc"`` runs each ``parallel_envs`` slot in
    #: its own process (true multicore replay); ``"dummy"`` steps them
    #: sequentially in-process and is what synthetic/test views use.
    vec_env: Literal["dummy", "subproc"] = "dummy"
    #: Append ``telemetry.jsonl`` + ``status.json`` sidecars during training
    #: (U00, CONTRACTS.md §1). Additive: defaults keep the existing artifacts
    #: and only add the sidecars.
    telemetry: bool = True
    #: Record deterministic evaluation rollouts to parquet (U06). ON by default:
    #: the Policy Replay view and the action heatmap need them, and skipping the
    #: post-training pass is not worth the loss of observability. Set to false
    #: only when the extra ~5 evaluation episodes must be avoided.
    record_rollouts: bool = True
    #: Maximum evaluation episodes recorded when ``record_rollouts`` is true (U06).
    rollout_max_episodes: int = 5
    #: Positive offset added to the ``hold`` action's logit when the PPO model is
    #: constructed. ``hold`` is index 0 of the discrete action grid (ADR-011) and
    #: the flat softmax otherwise samples it only ~1/N of the time -- too rarely
    #: for PPO to discover that avoiding the per-rebalance gas dominates. ``0.0``
    #: disables the prior and preserves today's behaviour exactly.
    hold_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.vec_env not in ("dummy", "subproc"):
            raise SimConfigError(
                f"training.vec_env must be 'dummy' or 'subproc', got {self.vec_env!r}"
            )
        learning_rate = float(self.learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise SimConfigError(
                f"training.learning_rate must be finite and > 0, got {self.learning_rate!r}"
            )
        batch_size = float(self.batch_size)
        if not math.isfinite(batch_size) or not batch_size.is_integer():
            raise SimConfigError(
                f"training.batch_size must be a finite integer, got {self.batch_size!r}"
            )
        if batch_size < 2:
            raise SimConfigError(
                f"training.batch_size must be >= 2 (SB3 requires > 1), got {self.batch_size!r}"
            )
        n_epochs = float(self.n_epochs)
        if not math.isfinite(n_epochs) or not n_epochs.is_integer():
            raise SimConfigError(
                f"training.n_epochs must be a finite integer, got {self.n_epochs!r}"
            )
        if n_epochs < 1:
            raise SimConfigError(f"training.n_epochs must be >= 1, got {self.n_epochs!r}")
        ent_coef = float(self.ent_coef)
        if not math.isfinite(ent_coef) or ent_coef < 0.0:
            raise SimConfigError(
                f"training.ent_coef must be finite and >= 0, got {self.ent_coef!r}"
            )
        if self.target_kl is not None:
            target_kl = float(self.target_kl)
            if not math.isfinite(target_kl) or target_kl <= 0.0:
                raise SimConfigError(
                    f"training.target_kl must be None or finite and > 0, got {self.target_kl!r}"
                )
        hold_bias = float(self.hold_bias)
        if not math.isfinite(hold_bias) or hold_bias < 0.0:
            raise SimConfigError(
                f"training.hold_bias must be finite and >= 0, got {self.hold_bias!r}"
            )


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
    #: S20 short-perp hedge overlay (additive, defaulted: disabled).
    hedge: HedgeConfig = field(default_factory=HedgeConfig)

    def rng(self) -> np.random.Generator:
        """Construct a seeded ``numpy.random.Generator`` from ``self.seed``."""
        return np.random.default_rng(self.seed)


_T = typing.TypeVar("_T")


def load_sim_config(path: str | Path) -> SimConfig:
    """Load a ``SimConfig`` from a TOML file.

    The file follows the structure of ``configs/sim_default.toml`` — one
    ``[section]`` per config subtree, with keys matching the dataclass fields.
    Omitted sections/keys fall back to defaults.

    Raises ``SimConfigError`` if the file contains unknown top-level sections.
    """
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
        "hedge",
    }
    unknown = set(raw.keys()) - _known_sections
    if unknown:
        raise SimConfigError(
            f"Unknown config section(s): {', '.join(sorted(unknown))}"
        )

    def _populate(cls: type[_T], data: dict | None) -> _T:
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
        hedge=_populate(HedgeConfig, raw.get("hedge")),
    )
