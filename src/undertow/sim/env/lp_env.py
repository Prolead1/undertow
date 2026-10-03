"""Gymnasium environment wiring prices + pool + frictions + reward (S12).

Implements ``docs/plans/sim/CONTRACTS.md`` §12. :class:`LpEnvironment` is the
training loop's world: it composes a :class:`~undertow.sim.marketview.MarketView`
(history + regime + gas), a :class:`~undertow.sim.core.pool.PoolEngine`, a
:class:`~undertow.sim.prices.base.PriceProcess`, a
:class:`~undertow.sim.frictions.gas.GasModel`, a
:class:`~undertow.sim.frictions.slippage.SlippageModel` and a
:class:`~undertow.sim.config.SimConfig`.

MDP shape
---------
* **Observation**: ``Box(shape=(18,), dtype=float64)`` — the stable-order
  :meth:`~undertow.sim.env.observations.Observation.to_vector`.
* **Action**: ``Discrete(1 + |center_offsets| * |widths|)``; index ``0`` is
  ``hold`` and every subsequent index maps to a ``rebalance`` action on the
  ``ActionGridConfig`` grid (ADR-011: only the agent head uses this grid).

Replay vs calibrated (ADR-009/ADR-010)
--------------------------------------
In **replay** mode decisions are made at **reference-bar boundaries** and the
pool engine is stepped at **tape-event granularity**: every tape ``swap`` event
between two decision boundaries is fed to :meth:`PoolEngine.step` individually,
so all volume is accounted for and the fees are per-event. A final zero-volume
step aligns the pool's raw ``sqrt_price``/``tick`` with the decision bar's
reference price. Swap volume uses the tape's raw ``amount0``/``amount1``; the
engine works in raw units and the fees are humanised once
(``10**dec0`` / ``10**dec1``, ADR-009).

In **calibrated** mode the process is only reset and advanced one step per
decision; swap volume comes from the synthetic constants supplied at
construction. The observation timestamp in both modes is the decision time and
the builder only ever reads reference bars at or before it.

Determinism
-----------
:meth:`reset` seeds a private :class:`numpy.random.Generator` from ``seed`` (or
``SimConfig.seed`` when ``seed`` is ``None``); the same seed reproduces the
same episode window, price stream and observation sequence.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np

from undertow.data import Q96, sqrt_price_x96_to_tick
from undertow.sim.config import SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1, Position, initial_deposit
from undertow.sim.env.observations import (
    OBSERVATION_VECTOR_LENGTH,
    Observation,
    ObservationBuilder,
)
from undertow.sim.env.reward import RewardBreakdown, compute_reward, normalize_reward
from undertow.sim.marketview import MarketView
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.prices.replay import ReplayPriceProcess
from undertow.sim.types import (
    Action,
    EnvError,
    SqrtPrice,
    Tick,
)

__all__ = ["LpEnvironment"]

#: Default half-width (in tick-spacings) of the position the env deploys at reset.
DEFAULT_INITIAL_WIDTH: int = 10

#: Default synthetic swap volume (raw token units) in calibrated mode.
DEFAULT_CALIBRATED_SWAP_VOLUME_0: float = 1_000_000.0
DEFAULT_CALIBRATED_SWAP_VOLUME_1: float = 1_000_000.0


class LpEnvironment(gym.Env):
    """Gymnasium environment for concentrated-liquidity provision (CONTRACTS §12).

    The passed ``pool_engine`` must describe the *external* pool only (base
    active liquidity and any background lattice), with no agent positions; the
    environment captures that base at construction and restores it on every
    :meth:`reset`, then deploys its own initial position. This keeps episodes
    independent and reproducible.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        market_view: MarketView,
        pool_engine: PoolEngine,
        price_process: object,
        gas_model: object,
        slippage_model: object,
        config: SimConfig,
        *,
        initial_width: int = DEFAULT_INITIAL_WIDTH,
        initial_center_offset: int = 0,
        calibrated_swap_volume_0: float = DEFAULT_CALIBRATED_SWAP_VOLUME_0,
        calibrated_swap_volume_1: float = DEFAULT_CALIBRATED_SWAP_VOLUME_1,
    ) -> None:
        super().__init__()
        if initial_width <= 0:
            raise ValueError(f"initial_width must be > 0, got {initial_width!r}")
        self.market_view = market_view
        self.pool_engine = pool_engine
        self.price_process = price_process
        self.gas_model = gas_model
        self.slippage_model = slippage_model
        self.config = config
        self._initial_width = int(initial_width)
        self._initial_center_offset = int(initial_center_offset)
        self._calibrated_swap_volume_0 = float(calibrated_swap_volume_0)
        self._calibrated_swap_volume_1 = float(calibrated_swap_volume_1)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_VECTOR_LENGTH,),
            dtype=np.float64,
        )

        # Index 0 = hold; 1+ enumerate the (center_offset, width) grid row-major.
        offsets = tuple(int(c) for c in config.action_grid.center_offsets)
        widths = tuple(int(w) for w in config.action_grid.widths)
        self._action_lookup: list[Action] = [Action(action_type="hold")]
        for center_offset in offsets:
            for width in widths:
                self._action_lookup.append(
                    Action(
                        action_type="rebalance",
                        center_offset=center_offset,
                        width=width,
                    )
                )
        self.action_space = gym.spaces.Discrete(len(self._action_lookup))

        self._builder = ObservationBuilder(market_view)
        self._capture_pool_base()

        # State populated by reset().
        self._seed: int = int(config.seed)
        self._rng: np.random.Generator = np.random.default_rng(self._seed)
        self._start_seq: int = 0
        self._end_seq: int = 0
        self._episode_length: int = 0
        self._step_count: int = 0
        self._bar_k: int = -1
        self._done: bool = True
        self._position_id: int | None = None
        self._position: Position | None = None
        self._entry_price: float = 0.0
        self._entry_sqrt: SqrtPrice = 0.0
        self._capital: float = float(config.episode.agent_capital_usdc)
        self._equity: float = self._capital
        self._last_fees_usdc: float = 0.0
        self._collected_fees_usdc: float = 0.0
        self._il_offset: float = 0.0
        self._last_il_total: float = 0.0
        self._pnl_history: list[float] = [self._capital]
        self._last_pool_result: dict[str, Any] = {}
        self._last_tape_row: dict[str, object] = {}
        self._last_at_time: datetime | None = None

        # Replay event tape (populated on reset for the replay mode).
        self._bar_close_times: list[datetime] = []
        self._ev_times: list[datetime] = []
        self._ev_types: list[object] = []
        self._ev_sqrt_x96: list[object] = []
        self._ev_ticks: list[object] = []
        self._ev_amount0: list[object] = []
        self._ev_amount1: list[object] = []
        self._ev_rows: list[dict[str, object]] = []
        self._event_idx: int = 0

    # ------------------------------------------------------------------
    # Construction-time snapshot of the external pool
    # ------------------------------------------------------------------
    def _capture_pool_base(self) -> None:
        if self.pool_engine.positions:
            raise EnvError(
                "LpEnvironment requires a pool engine with no agent positions at "
                "construction; it deploys its own on reset()"
            )
        self._base_state: PoolState = replace(self.pool_engine.state)
        self._base_ticks: dict[Tick, TickState] = {
            tick: replace(tick_state)
            for tick, tick_state in self.pool_engine.ticks.items()
        }

    def _restore_pool_base(self) -> None:
        state = self.pool_engine.state
        state.sqrt_price = self._base_state.sqrt_price
        state.tick = self._base_state.tick
        state.liquidity = self._base_state.liquidity
        state.fee_growth_global_0 = self._base_state.fee_growth_global_0
        state.fee_growth_global_1 = self._base_state.fee_growth_global_1
        state.fee_tier_bps = self._base_state.fee_tier_bps
        state.tick_spacing = self._base_state.tick_spacing
        self.pool_engine.ticks.clear()
        self.pool_engine.ticks.update(
            {tick: replace(ts) for tick, ts in self._base_ticks.items()}
        )
        self.pool_engine.positions.clear()
        self.pool_engine.position_fee_snapshots.clear()
        # Position ids restart per episode so ``info`` is reproducible for a
        # fixed seed. ``PoolEngine`` has no public reset hook and adding one is
        # outside S12's file ownership, so the counter is reset directly here.
        self.pool_engine._next_position_id = 0

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        """Size of the discrete action space."""
        return len(self._action_lookup)

    @property
    def episode_length(self) -> int:
        """Number of decision steps in the current episode (valid after reset)."""
        return self._episode_length

    @property
    def step_count(self) -> int:
        """Decision steps taken since the last reset."""
        return self._step_count

    @property
    def position_id(self) -> int | None:
        """Id of the agent's current position in the pool engine."""
        return self._position_id

    @property
    def is_replay(self) -> bool:
        """True when the active price process replays a fixed bar window."""
        return isinstance(self.price_process, ReplayPriceProcess)

    def action_for_index(self, action_idx: int) -> Action:
        """Map a discrete action index to its :class:`Action` (ADR-011)."""
        idx = int(action_idx)
        if idx < 0 or idx >= len(self._action_lookup):
            raise EnvError(
                f"action index {action_idx} is outside the action space "
                f"[0, {len(self._action_lookup)})"
            )
        return self._action_lookup[idx]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh episode and return ``(observation_vector, info)``."""
        del options  # no reset options are defined
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

        episode = self.config.episode
        config_steps = int(
            (episode.duration_days * 24 * 60) // episode.step_minutes
        )
        if config_steps <= 0:
            raise EnvError(
                "episode configuration yields a non-positive step budget: "
                f"{episode.duration_days}d / {episode.step_minutes}min"
            )

        start_seq, end_seq = self.market_view.build_episode(
            self._seed, max_steps=config_steps + 1
        )
        self._start_seq = int(start_seq)
        self._end_seq = int(end_seq)

        if self.is_replay:
            # The replay source's window is fixed at construction; re-create it
            # for the freshly sampled episode window so resets are independent.
            self.price_process = ReplayPriceProcess(
                self.market_view, self._start_seq, self._end_seq
            )
        self.price_process.reset(self._rng)

        if self.is_replay:
            self._prepare_replay_events()
            self._episode_length = max(1, min(config_steps, len(self._bar_close_times)))
        else:
            self._episode_length = config_steps

        self._step_count = 0
        self._bar_k = -1
        self._event_idx = 0
        self._done = False

        self._restore_pool_base()
        self._last_pool_result = {}
        self._last_tape_row = {}
        self._last_at_time = None

        entry_row = self.market_view.tape_at(self._start_seq)
        entry_time = entry_row["block_timestamp"]
        reference_price = self.market_view.reference_close(entry_time)
        if reference_price is None:
            reference_price = self._price_from_sqrt(self.pool_engine.state.sqrt_price)
        entry_price = float(reference_price)
        _, entry_sqrt, entry_tick = human_price_to_triplet(entry_price)
        self.pool_engine.state.sqrt_price = entry_sqrt
        self.pool_engine.state.tick = entry_tick
        self._entry_price = entry_price
        self._entry_sqrt = entry_sqrt
        self._last_tape_row = entry_row
        self._last_at_time = entry_time

        spacing = int(episode.tick_spacing)
        deploy = Action(
            action_type="rebalance",
            center_offset=self._initial_center_offset,
            width=self._initial_width,
        )
        lower = deploy.lower_tick(entry_tick, spacing)
        upper = deploy.upper_tick(entry_tick, spacing)
        position = initial_deposit(
            entry_sqrt,
            entry_price,
            lower,
            upper,
            self._capital,
            spacing,
        )
        self._position_id = self.pool_engine.open_position(
            lower, upper, position.liquidity
        )
        self._position = self.pool_engine.positions[self._position_id]

        self._equity = self._capital
        self._last_fees_usdc = 0.0
        self._collected_fees_usdc = 0.0
        self._il_offset = 0.0
        self._last_il_total = self._current_il_total(entry_sqrt, entry_price)
        self._pnl_history = [self._capital]

        self._builder.configure_episode(self._start_seq, self._episode_length)
        observation = self._observe(
            at_time=entry_time,
            tape_row=entry_row,
            step_index=0,
        )
        info = self._make_info(
            breakdown=RewardBreakdown(equity=self._equity),
            raw=RewardBreakdown(equity=self._equity),
            action=self._action_lookup[0],
        )
        return observation, info

    def step(
        self, action_idx: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision step.

        Returns ``(observation, reward, terminated, truncated, info)`` where
        ``reward`` is capital-normalized and ``info`` carries the full
        :class:`RewardBreakdown` under ``"reward_breakdown"`` (normalized) and
        ``"reward_breakdown_raw"`` (USDC).
        """
        if self._done:
            raise EnvError("step() called after the episode ended; call reset()")
        action = self.action_for_index(action_idx)

        if self.is_replay:
            price, sqrt_price, at_time, tape_row, price_exhausted = (
                self._advance_replay()
            )
        else:
            price, sqrt_price, at_time, tape_row, price_exhausted = (
                self._advance_calibrated()
            )

        # Fees earned this step = growth in the position's uncollected fees.
        fees_usdc_before = self._current_fees_usdc(price)
        fees_earned = fees_usdc_before - self._last_fees_usdc

        gas_cost = 0.0
        slippage_cost = 0.0
        if action.action_type == "rebalance" and not price_exhausted:
            gas_cost, slippage_cost = self._rebalance(
                action, sqrt_price, price, tape_row
            )
            # New position starts with zero uncollected fees.
            fees_usdc_before = self._current_fees_usdc(price)
        self._last_fees_usdc = fees_usdc_before

        il_total = self._current_il_total(sqrt_price, price)
        il_change = il_total - self._last_il_total
        self._last_il_total = il_total

        position_value = self._position.value(sqrt_price, price)
        equity = position_value + self._current_fees_usdc(price)
        self._equity = equity
        self._pnl_history.append(equity)

        raw_breakdown = compute_reward(
            fees_earned=fees_earned,
            gas_cost=gas_cost,
            slippage_cost=slippage_cost,
            il_change=il_change,
            pnl_volatility=self._pnl_volatility(),
            config=self.config.reward,
        )
        raw_breakdown = replace(
            raw_breakdown,
            equity=equity,
            pnl_net=fees_earned - gas_cost - slippage_cost + il_change,
        )
        breakdown = normalize_reward(
            raw_breakdown, self._capital, self.config.reward
        )

        self._step_count += 1
        truncated = self._step_count >= self._episode_length
        terminated = price_exhausted
        self._done = truncated or terminated

        observation = self._observe(
            at_time=at_time,
            tape_row=tape_row,
            step_index=self._step_count,
        )
        info = self._make_info(breakdown=breakdown, raw=raw_breakdown, action=action)
        return observation, float(breakdown.reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Episode advance (mode-specific)
    # ------------------------------------------------------------------
    def _prepare_replay_events(self) -> None:
        """Cache the episode window's reference bars and tape swap events."""
        assert isinstance(self.price_process, ReplayPriceProcess)
        self._bar_close_times = list(self.price_process.close_times)
        window = self.market_view.slice(
            self._start_seq, self._end_seq
        ).active_split_data()
        self._ev_times = window["block_timestamp"].to_list()
        self._ev_types = window["event_type"].to_list()
        self._ev_sqrt_x96 = window["sqrt_price_x96"].to_list()
        self._ev_ticks = window["tick"].to_list()
        self._ev_amount0 = window["amount0"].to_list()
        self._ev_amount1 = window["amount1"].to_list()
        self._ev_rows = window.to_dicts()

    def _advance_replay(
        self,
    ) -> tuple[float, SqrtPrice, datetime, dict[str, object], bool]:
        """Advance one reference bar, stepping the pool over the interval's swaps."""
        assert isinstance(self.price_process, ReplayPriceProcess)
        try:
            price, sqrt_price, tick = self.price_process.step(self._rng)
        except StopIteration:
            at_time = self._last_at_time or self.market_view.tape_at(
                self._end_seq - 1
            )["block_timestamp"]
            return (
                self._entry_price,
                self.pool_engine.state.sqrt_price,
                at_time,
                self._last_tape_row,
                True,
            )

        self._bar_k += 1
        at_time = self._bar_close_times[self._bar_k]
        interval_fee_0 = 0.0
        interval_fee_1 = 0.0
        while (
            self._event_idx < len(self._ev_times)
            and self._ev_times[self._event_idx] <= at_time
        ):
            index = self._event_idx
            if self._ev_types[index] == "swap":
                event_sqrt, event_tick = self._event_triplet(index)
                volume_0, volume_1 = self._event_volumes(index)
                result = self.pool_engine.step(
                    event_sqrt, event_tick, volume_0, volume_1
                )
                interval_fee_0 += result["pool_fee_earned_0"]
                interval_fee_1 += result["pool_fee_earned_1"]
            self._last_tape_row = self._ev_rows[index]
            self._event_idx += 1

        # Align the pool's raw price with the decision bar (zero volume).
        self.pool_engine.step(sqrt_price, tick, 0.0, 0.0)
        self._last_pool_result = {
            "fees_accrued": {},
            "ticks_crossed": [],
            "active_liquidity": self.pool_engine.state.liquidity,
            "pool_fee_earned_0": interval_fee_0,
            "pool_fee_earned_1": interval_fee_1,
        }
        self._last_at_time = at_time
        return price, sqrt_price, at_time, self._last_tape_row, False

    def _advance_calibrated(
        self,
    ) -> tuple[float, SqrtPrice, datetime, dict[str, object], bool]:
        """Advance one calibrated price step with synthetic swap volume."""
        try:
            price, sqrt_price, tick = self.price_process.step(self._rng)
        except StopIteration:
            at_time = self._last_at_time or self.market_view.tape_at(
                self._end_seq - 1
            )["block_timestamp"]
            return (
                self._entry_price,
                self.pool_engine.state.sqrt_price,
                at_time,
                self._last_tape_row,
                True,
            )
        cursor = min(
            self._start_seq + self._step_count + 1, self._end_seq - 1
        )
        tape_row = dict(self.market_view.tape_at(cursor))
        self._last_pool_result = self.pool_engine.step(
            sqrt_price,
            tick,
            self._calibrated_swap_volume_0,
            self._calibrated_swap_volume_1,
        )
        at_time = tape_row["block_timestamp"]
        self._last_tape_row = tape_row
        self._last_at_time = at_time
        return price, sqrt_price, at_time, tape_row, False

    def _event_triplet(self, index: int) -> tuple[SqrtPrice, Tick]:
        """Raw ``(sqrt_price, tick)`` for a swap tape event (ADR-009)."""
        x96 = self._ev_sqrt_x96[index]
        if x96 not in (None, ""):
            int_x96 = int(x96)
            sqrt_price = int_x96 / float(Q96)
            tick_value = self._ev_ticks[index]
            tick = (
                Tick(int(tick_value))
                if tick_value is not None
                else Tick(int(sqrt_price_x96_to_tick(int_x96)))
            )
            return float(sqrt_price), tick
        return self.pool_engine.state.sqrt_price, self.pool_engine.state.tick

    def _event_volumes(self, index: int) -> tuple[float, float]:
        """Raw swap volume for a tape event (the engine's units, ADR-012)."""
        amount0 = self._ev_amount0[index]
        amount1 = self._ev_amount1[index]
        volume_0 = abs(float(amount0)) if amount0 not in (None, "") else 0.0
        volume_1 = abs(float(amount1)) if amount1 not in (None, "") else 0.0
        return volume_0, volume_1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _price_from_sqrt(
        sqrt_price: SqrtPrice,
        dec0: int = DEFAULT_DEC0,
        dec1: int = DEFAULT_DEC1,
    ) -> float:
        if sqrt_price <= 0.0:
            return 0.0
        return 10.0 ** (dec1 - dec0) / (sqrt_price * sqrt_price)

    def _current_fees_usdc(self, price: float) -> float:
        """The current position's uncollected fees, valued in USDC.

        ``position_fees`` returns **raw** ``L * d fee_growth``; the human token
        split follows ADR-009 (``10**dec0`` / ``10**dec1``), and token1 fees are
        marked at the human ``price``.
        """
        if self._position_id is None or self._position is None:
            return 0.0
        raw_0, raw_1 = self.pool_engine.position_fees(self._position_id)
        fees_0 = raw_0 / 10.0**self._position.dec0
        fees_1 = raw_1 / 10.0**self._position.dec1
        return fees_0 + fees_1 * price

    def _current_il_total(self, sqrt_price: SqrtPrice, price: float) -> float:
        """Total IL level (current position + realised offset), in USDC."""
        if self._position is None:
            return self._il_offset
        return self._il_offset + self._position.il_vs_hodl(
            self._entry_sqrt, self._entry_price, sqrt_price, price
        )

    def _pnl_volatility(self) -> float:
        """Trailing std of per-step equity changes over the risk window."""
        window = int(self.config.reward.risk_rolling_window_steps)
        if window < 2:
            return 0.0
        history = np.asarray(self._pnl_history[-(window + 1) :], dtype=float)
        if history.size < 2:
            return 0.0
        deltas = np.diff(history)
        if deltas.size < 2:
            return 0.0
        return float(np.std(deltas, ddof=1))

    def _gas_row(self, tape_row: dict[str, object]) -> dict[str, object]:
        """Gas context for a step, preferring the gas feed (as the builder does)."""
        block_number = int(tape_row.get("block_number", 0) or 0)
        row = self.market_view.gas_at_block(block_number)
        return row if row is not None else tape_row

    def _rebalance(
        self,
        action: Action,
        sqrt_price: SqrtPrice,
        price: float,
        tape_row: dict[str, object],
    ) -> tuple[float, float]:
        """Close the current position, open the action's range, return costs.

        Returns ``(gas_cost, slippage_cost)`` as positive magnitudes. The new
        position is funded with the old position's mark-to-market value **plus
        its collected fees**, net of the rebalance costs, so equity is
        continuous and cost-honest. The old position's IL is folded into
        ``_il_offset`` so closing does not create a spurious IL jump.
        """
        if self._position is None or self._position_id is None:
            raise EnvError("rebalance requested with no open position")
        spacing = int(self.config.episode.tick_spacing)
        lower = action.lower_tick(self.pool_engine.state.tick, spacing)
        upper = action.upper_tick(self.pool_engine.state.tick, spacing)

        notional = self._position.value(sqrt_price, price)
        collected = self._current_fees_usdc(price)

        gas_row = self._gas_row(tape_row)
        base_fee = int(gas_row.get("base_fee_per_gas") or 0)
        priority_fee = int(gas_row.get("priority_fee_p50_wei") or 0)
        gas_cost = float(
            self.gas_model.gas_cost_usdc(
                "rebalance",
                int(tape_row.get("block_number", 0) or 0),
                base_fee,
                priority_fee,
                price,
            )
        )
        slippage_cost = float(
            self.slippage_model.slippage_cost_usdc(
                notional,
                int(self.config.episode.fee_tier_bps),
                float(self.config.slippage.fixed_impact_bps),
            )
        )

        # Realise the closing IL into the running offset (no IL jump on close).
        self._il_offset += self._position.il_vs_hodl(
            self._entry_sqrt, self._entry_price, sqrt_price, price
        )
        self._collected_fees_usdc += collected
        self.pool_engine.close_position(self._position_id)

        # Re-deploy the position value plus collected fees, net of costs, so
        # equity (and every reported PnL number) is continuous and cost-honest.
        new_capital = max(notional + collected - gas_cost - slippage_cost, 1.0)
        new_position = initial_deposit(
            sqrt_price, price, lower, upper, new_capital, spacing
        )
        self._position_id = self.pool_engine.open_position(
            lower, upper, new_position.liquidity
        )
        self._position = self.pool_engine.positions[self._position_id]
        self._entry_price = price
        self._entry_sqrt = sqrt_price
        return gas_cost, slippage_cost

    def _observe(
        self,
        *,
        at_time: datetime,
        tape_row: dict[str, object],
        step_index: int,
    ) -> np.ndarray:
        if self._position is None:
            raise EnvError("cannot observe without an open position")
        observation: Observation = self._builder.build_at(
            at_time=at_time,
            tape_row=tape_row,
            step_in_episode=step_index,
            steps_remaining=max(0, self._episode_length - 1 - step_index),
            position=self._position,
            pool_engine=self.pool_engine,
            gas_model=self.gas_model,
            entry_price=self._entry_price,
            entry_sqrt_price=self._entry_sqrt,
            initial_capital=self._capital,
        )
        return observation.to_vector()

    def _make_info(
        self,
        breakdown: RewardBreakdown,
        raw: RewardBreakdown,
        action: Action,
    ) -> dict[str, Any]:
        return {
            "reward_breakdown": asdict(breakdown),
            "reward_breakdown_raw": asdict(raw),
            "action": action,
            "action_type": action.action_type,
            "step_in_episode": self._step_count,
            "equity": self._equity,
            "position_id": self._position_id,
            "tick_lower": int(self._position.tick_lower)
            if self._position is not None
            else 0,
            "tick_upper": int(self._position.tick_upper)
            if self._position is not None
            else 0,
            "position_liquidity": self._position.liquidity
            if self._position is not None
            else 0.0,
            "active_liquidity": self.pool_engine.state.liquidity,
            "entry_price": self._entry_price,
            "collected_fees_usdc": self._collected_fees_usdc,
            "pool_fee_earned_0": self._last_pool_result.get("pool_fee_earned_0", 0.0),
            "pool_fee_earned_1": self._last_pool_result.get("pool_fee_earned_1", 0.0),
        }
