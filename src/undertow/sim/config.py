"""Simulation configuration for ``undertow.sim``.

All user-facing knobs live here. The ``SimConfig`` tree is a frozen dataclass
hierarchy — mutable state belongs in the environment, not the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Fee tier → tick spacing mapping (Uniswap V3 protocol)
# ---------------------------------------------------------------------------
FEE_TIER_TO_SPACING: dict[int, int] = {
    1: 1,       # 0.01%
    5: 10,      # 0.05%
    30: 60,     # 0.30%
    100: 200,   # 1.00%
}


# ---------------------------------------------------------------------------
# Gas-unit constants for typical Uniswap V3 actions (approximate).
# ---------------------------------------------------------------------------
GAS_MINT: int = 200_000
GAS_BURN: int = 150_000
GAS_COLLECT: int = 100_000
GAS_REBALANCE_SWAP: int = 150_000


# ---------------------------------------------------------------------------
# Config tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Episode-level parameters."""

    duration_days: int = 30
    step_minutes: int = 60
    agent_capital_usdc: float = 100_000.0
    marginal_agent: bool = True
    fee_tier_bps: int = 30
    tick_spacing: int = 60

    def __post_init__(self) -> None:
        if self.fee_tier_bps not in FEE_TIER_TO_SPACING:
            raise ValueError(f"Unsupported fee tier: {self.fee_tier_bps} bps")
        expected = FEE_TIER_TO_SPACING[self.fee_tier_bps]
        if self.tick_spacing != expected:
            raise ValueError(
                f"tick_spacing {self.tick_spacing} != expected "
                f"{expected} for fee tier {self.fee_tier_bps} bps"
            )

    @property
    def steps_per_episode(self) -> int:
        return self.duration_days * 24 * 60 // self.step_minutes


@dataclass(frozen=True, slots=True)
class ActionGrid:
    """Discrete action grid for the RL agent."""

    center_offsets: tuple[float, ...] = (
        -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20,
    )
    widths: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    include_hold: bool = True

    @property
    def n_actions(self) -> int:
        return len(self.center_offsets) * len(self.widths) + (
            1 if self.include_hold else 0
        )

    def decode(self, action: int) -> tuple[float, float] | None:
        """Decode an action index into (center_offset, width) or None for hold."""
        if self.include_hold:
            if action == self.n_actions - 1:
                return None  # hold
            n_offsets = len(self.center_offsets)
            off_idx = action // len(self.widths)
            w_idx = action % len(self.widths)
            return self.center_offsets[off_idx], self.widths[w_idx]
        off_idx = action // len(self.widths)
        w_idx = action % len(self.widths)
        return self.center_offsets[off_idx], self.widths[w_idx]


@dataclass(frozen=True, slots=True)
class GasConfig:
    """Gas cost parameters."""

    mint_units: int = GAS_MINT
    burn_units: int = GAS_BURN
    collect_units: int = GAS_COLLECT
    rebalance_swap_units: int = GAS_REBALANCE_SWAP
    base_fee_per_gas_gwei: float = 20.0
    priority_fee_p50_gwei: float = 1.0

    @property
    def total_gas_price_gwei(self) -> float:
        return self.base_fee_per_gas_gwei + self.priority_fee_p50_gwei


@dataclass(frozen=True, slots=True)
class SlippageConfig:
    """Slippage parameters (simple model)."""

    fixed_impact_bps: float = 5.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward function parameters and ablation flags."""

    risk_penalty_lambda: float = 0.0
    risk_rolling_window_steps: int = 24
    normalize_by_capital: bool = True
    # Ablation flags
    fees_enabled: bool = True
    gas_enabled: bool = True
    slippage_enabled: bool = True
    il_enabled: bool = True
    risk_penalty_enabled: bool = False


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """RL training parameters (mapped to stable-baselines3 PPO kwargs)."""

    algorithm: str = "ppo"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    hidden_size: int = 128
    n_hidden: int = 2
    parallel_envs: int = 4
    total_timesteps: int = 1_000_000
    checkpoint_freq_steps: int = 50_000
    log_freq_steps: int = 10_000


@dataclass(frozen=True, slots=True)
class PriceConfig:
    """Price process parameters."""

    mode: str = "calibrated"  # "replay" | "calibrated" | "gbm"
    mu: float = 0.0           # annual drift
    sigma: float = 0.80       # annual vol
    jump_intensity: float = 0.0  # jumps per year
    jump_mean: float = 0.0
    jump_std: float = 0.05
    # Regime-switching
    regime_transition_scale: float = 30.0  # days per regime
    high_vol_sigma: float = 1.20
    sideways_sigma: float = 0.40


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Root simulation configuration."""

    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    action_grid: ActionGrid = field(default_factory=ActionGrid)
    gas: GasConfig = field(default_factory=GasConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    price: PriceConfig = field(default_factory=PriceConfig)


def default_sim_config() -> SimConfig:
    """Return the default SimConfig (USDC/WETH 0.30% pool)."""
    return SimConfig()


def load_sim_config(path: str) -> SimConfig:
    """Load a ``SimConfig`` from a TOML file.

    The file follows the structure of ``configs/sim_default.toml`` — one
    ``[section]`` per config subtree, with keys matching the dataclass fields.
    Omitted sections/keys fall back to defaults.
    """
    import tomllib
    import typing
    from dataclasses import fields as dc_fields

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    def _populate(cls, data: dict | None):
        """Build a frozen dataclass instance from a dict, falling back to defaults."""
        data = data or {}
        type_hints = typing.get_type_hints(cls)
        kwargs: dict[str, object] = {}
        for f in dc_fields(cls):
            if f.name in data:
                val = data[f.name]
                # Convert TOML arrays to tuples for tuple-typed fields
                f_type = type_hints.get(f.name)
                if isinstance(val, list) and f_type is not None:
                    origin = typing.get_origin(f_type)
                    if origin is tuple:
                        val = tuple(val)
                kwargs[f.name] = val
        return cls(**kwargs)

    return SimConfig(
        episode=_populate(EpisodeConfig, raw.get("episode")),
        action_grid=_populate(ActionGrid, raw.get("action_grid")),
        gas=_populate(GasConfig, raw.get("gas")),
        slippage=_populate(SlippageConfig, raw.get("slippage")),
        reward=_populate(RewardConfig, raw.get("reward")),
        training=_populate(TrainingConfig, raw.get("training")),
        price=_populate(PriceConfig, raw.get("price")),
    )