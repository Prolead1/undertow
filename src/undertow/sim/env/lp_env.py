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
* **Observation**: ``Box(shape=(OBSERVATION_VECTOR_LENGTH,), dtype=float64)`` —
  the stable-order reference-bar, position and band-geometry features
  (CONTRACTS §11; ADR-029) flattened by
  :meth:`~undertow.sim.env.observations.Observation.to_vector`.
* **Action**: ``Discrete(1 + |center_offsets| * |widths|)``; index ``0`` is
  ``hold`` and every subsequent index maps to a ``rebalance`` action on the
  ``ActionGridConfig`` grid (ADR-011: only the agent head uses this grid).

Replay vs calibrated (ADR-009/ADR-010)
--------------------------------------
In **replay** mode decisions are made at **reference-bar boundaries** and the
pool engine is stepped at **tape-event granularity**: every tape ``swap`` event
between two decision boundaries is fed to :meth:`PoolEngine.step` individually,
so all volume is accounted for and the fees are per-event.``mint``/``burn``
events in the same interval are applied to the external lattice
(:meth:`PoolEngine.apply_liquidity_event`), mirroring the backtester's exact fee
engine. For hand-built views without a boundary snapshot a final zero-volume
step aligns the pool's raw ``sqrt_price``/``tick`` with the decision bar's
reference price; for tape-driven (snapshot-seeded) views the pool state follows
the tape alone and the reference bar is used for valuation only (ADR-019). Swap
volume uses the tape's raw ``amount0``/``amount1`` and is charged on the
**input** leg only; the engine works in raw units and the fees are humanised
once (``10**dec0`` / ``10**dec1``, ADR-009).

In **calibrated** mode the process is only reset and advanced one step per
decision; swap volume comes from the synthetic constants supplied at
construction. The observation timestamp in both modes is the decision time and
the builder only ever reads reference bars at or before it.

Determinism
-----------
A seeded :meth:`reset` is reproducible: ``reset(seed=k)`` always rebuilds window
``k`` — the same episode window, price stream and observation sequence — no
matter how many resets preceded it.  A no-seed reset (exactly what SB3's
``VecEnv`` auto-reset calls) instead advances to the next window, walking
``SimConfig.seed``, ``SimConfig.seed + 1``, ... so a training loop samples a
fresh market window every episode rather than replaying one.  The private
:class:`numpy.random.Generator` that drives the price process is seeded from the
same window seed.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any, cast

import gymnasium as gym
import numpy as np
import polars as pl

from undertow.data import Q96, sqrt_price_x96_to_tick
from undertow.sim.coerce import as_float, as_int
from undertow.sim.config import SimConfig
from undertow.sim.core.pool import PoolEngine, PoolState, TickState
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1, Position, initial_deposit
from undertow.sim.env.observations import (
    OBSERVATION_VECTOR_LENGTH,
    Observation,
    ObservationBuilder,
)
from undertow.sim.env.reward import RewardBreakdown, compute_reward, normalize_reward
from undertow.sim.frictions.gas import GasModel
from undertow.sim.frictions.slippage import SlippageModel
from undertow.sim.marketview import MarketView
from undertow.sim.prices.base import PriceProcess, human_price_to_triplet
from undertow.sim.prices.replay import ReplayPriceProcess
from undertow.sim.types import (
    Action,
    EnvError,
    SqrtPrice,
    Tick,
    TickSpacing,
)

__all__ = ["LpEnvironment"]

#: Default half-width (in tick-spacings) of the position the env deploys at reset.
DEFAULT_INITIAL_WIDTH: int = 10

#: Default synthetic swap volume (raw token units) in calibrated mode.
DEFAULT_CALIBRATED_SWAP_VOLUME_0: float = 1_000_000.0
DEFAULT_CALIBRATED_SWAP_VOLUME_1: float = 1_000_000.0

#: External-lattice warmup checkpoint stride (events).  The first reset replays
#: the backed lane once from its ADR-018 boundary snapshot while recording a
#: lattice snapshot every ``WARMUP_CHECKPOINT_STRIDE`` rows; later resets then
#: restore the nearest checkpoint and replay at most this many rows to reach
#: their episode window instead of re-replaying the whole lane (ADR-020).
#: Tests lower it to exercise the checkpoint path on short fixtures.
WARMUP_CHECKPOINT_STRIDE: int = 10_000

#: A warmup checkpoint: the lattice state *after* the tape row at ``seq``.
WarmupCheckpoint = tuple[int, PoolState, dict[Tick, TickState]]


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
        price_process: PriceProcess,
        gas_model: GasModel,
        slippage_model: SlippageModel,
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

        # State populated by reset().  ``_base_seed`` anchors the run and
        # ``_episode_index`` advances on every no-seed reset (SB3's auto-reset),
        # so successive windows are seeded ``base, base + 1, ...``.
        self._base_seed: int = int(config.seed)
        self._episode_index: int = -1
        self._rng: np.random.Generator = np.random.default_rng(self._base_seed)
        self._start_seq: int = 0
        self._end_seq: int = 0
        self._external_seeded: bool = False
        self._warmup_checkpoints: list[WarmupCheckpoint] | None = None
        self._warmup_checkpoint_seqs: list[int] = []
        self._warmup_source: tuple[pl.DataFrame, ...] | None = None
        self._episode_length: int = 0
        self._step_count: int = 0
        self._bar_k: int = -1
        self._done: bool = True
        self._position_id: int | None = None
        self._position: Position | None = None
        #: The deposit deployed at ``reset()``: the fixed HODL basket every
        #: excess-vs-HODL measure is benchmarked against (ADR-029).  It is a
        #: value object, so it stays valid after the engine closes it.
        self._initial_position: Position | None = None
        self._entry_price: float = 0.0
        self._entry_sqrt: SqrtPrice = 0.0
        self._capital: float = float(config.episode.agent_capital_usdc)
        self._equity: float = self._capital
        self._last_fees_usdc: float = 0.0
        self._collected_fees_usdc: float = 0.0
        self._last_il_total: float = 0.0
        self._pnl_history: list[float] = [self._capital]
        self._last_pool_result: dict[str, Any] = {}
        self._last_tape_row: dict[str, object] = {}
        self._last_at_time: datetime | None = None
        # Human price used to mark the last advanced step.  Kept so the
        # price-exhaustion fallback reuses the *last mark* rather than the
        # episode entry price, which would pair a stale human price with the
        # engine's current raw sqrt and misvalue the terminal step.
        self._last_mark_price: float = 0.0

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
    def _reset_external_pool(self) -> None:
        """Rebuild the external lattice for the sampled episode window (ADR-019/020).

        When the view carries a boundary pool-state snapshot (ADR-018), restore
        the lattice from the nearest warmup checkpoint and replay the remaining
        pre-window rows — the exact same seed+warmup the backtester runs
        (``_seed_fee_growth_replay``), but bounded by
        :data:`WARMUP_CHECKPOINT_STRIDE` events per reset rather than by the
        distance to the lane start.  Hand-built/fixture views without a snapshot
        keep the construction-time empty-lattice base.
        """
        self._external_seeded = bool(
            self.market_view.pool_state is not None
            and self.market_view.pool_state.height > 0
        )
        if self._external_seeded:
            self._restore_from_warmup_checkpoint()
        self._restore_pool_base()

    def _restore_from_warmup_checkpoint(self) -> None:
        """Seed the live engine from the nearest checkpoint and replay the gap."""
        checkpoints = self._ensure_warmup_checkpoints()
        if not checkpoints:
            raise EnvError(
                "the market view carries a pool_state snapshot but no tape rows "
                "to warm the external lattice"
            )
        # ``bisect_left`` (not ``_right``): a checkpoint holds the state *after*
        # its row, so a start at the checkpoint's own seq must replay from the
        # previous checkpoint, not skip the row.
        index = bisect_left(self._warmup_checkpoint_seqs, self._start_seq) - 1
        checkpoint_seq, checkpoint_state, checkpoint_ticks = checkpoints[index]
        # Warm a fresh external-only engine: the live engine still holds the
        # previous episode's agent positions, and replaying swaps through them
        # would look up their synthetic boundaries in a lattice they are not on.
        seeded = PoolEngine(
            state=replace(checkpoint_state),
            ticks=self._copy_ticks(checkpoint_ticks),
        )
        for row in self._warmup_rows_between(checkpoint_seq, self._start_seq):
            self._apply_warmup_row(seeded, row)
        self._base_state = replace(seeded.state)
        self._base_ticks = self._copy_ticks(seeded.ticks)

    def _ensure_warmup_checkpoints(self) -> list[WarmupCheckpoint]:
        """Replay the backed lane once, memoising lattice snapshots (ADR-020).

        Checkpoints are spaced by :data:`WARMUP_CHECKPOINT_STRIDE` *events*, not
        blocks, so each one is the state after the row at its ``seq`` and a reset
        at ``seq`` needs rows with ``checkpoint_seq < seq < start_seq``.  The
        initial checkpoint is ``first_seq - 1`` (the raw snapshot), which keeps
        an episode starting at the lane's first event warmup-free.
        """
        if self._warmup_checkpoints is not None:
            return self._warmup_checkpoints
        view = self.market_view
        if view.pool_state is None or view.pool_state.height == 0:
            self._warmup_checkpoints = []
            return self._warmup_checkpoints

        engine = PoolEngine.from_snapshot(
            self.pool_engine.state, view.pool_state.to_dicts()
        )
        frames = self._warmup_frames()
        first_seq = int(frames[0]["seq"][0])
        checkpoints: list[WarmupCheckpoint] = [
            (first_seq - 1, replace(engine.state), self._copy_ticks(engine.ticks))
        ]
        seqs = [first_seq - 1]
        seen = 0
        for frame in frames:
            for row in frame.iter_rows(named=True):
                self._apply_warmup_row(engine, row)
                seen += 1
                if seen % WARMUP_CHECKPOINT_STRIDE == 0:
                    seq = int(row["seq"])
                    seqs.append(seq)
                    checkpoints.append(
                        (seq, replace(engine.state), self._copy_ticks(engine.ticks))
                    )
        self._warmup_checkpoint_seqs = seqs
        self._warmup_checkpoints = checkpoints
        return checkpoints

    def _warmup_frames(self) -> tuple[pl.DataFrame, ...]:
        """Seq-ordered frames the external lattice replays (warmup + active).

        ``MarketView.slice`` keeps the pre-window rows in ``warmup`` and every
        one of them precedes the active window, so iterating the two frames in
        order replays the lane in ``seq`` order without a concat copy.
        """
        if self._warmup_source is None:
            view = self.market_view
            frames: list[pl.DataFrame] = []
            if view.warmup is not None and view.warmup.height:
                frames.append(view.warmup.sort("seq"))
            frames.append(view.active_split_data().sort("seq"))
            self._warmup_source = tuple(frames)
        return self._warmup_source

    def _warmup_rows_between(
        self, after_seq: int, before_seq: int
    ) -> Iterator[dict[str, object]]:
        """Yield pre-window rows with ``after_seq < seq < before_seq``."""
        for frame in self._warmup_frames():
            mask = (frame["seq"] > after_seq) & (frame["seq"] < before_seq)
            sub = frame.filter(mask)
            if sub.height:
                yield from sub.iter_rows(named=True)

    @staticmethod
    def _copy_ticks(ticks: dict[Tick, TickState]) -> dict[Tick, TickState]:
        """Deep-copy a tick lattice (``TickState`` is mutable)."""
        return {tick: replace(tick_state) for tick, tick_state in ticks.items()}

    @staticmethod
    def _entry_triplet(
        entry_row: dict[str, object], entry_price: float
    ) -> tuple[SqrtPrice, Tick, int | None]:
        """Raw ``(sqrt_price, tick, sqrt_price_x96)`` for the window's first row.

        Mirrors the backtester's ``_resolve_market(0)``: prefer the tape row's
        exact ``sqrt_price_x96``/``tick``; fall back to the as-of reference
        close when the row is a liquidity event (or a fixture without an exact
        price).  The returned ``sqrt_price_x96`` is ``None`` on the fallback.
        """
        x96 = entry_row.get("sqrt_price_x96")
        tick_value = entry_row.get("tick")
        if x96 not in (None, ""):
            int_x96 = as_int(x96)
            sqrt_price = int_x96 / float(Q96)
            tick = (
                Tick(as_int(tick_value))
                if tick_value is not None
                else Tick(int(sqrt_price_x96_to_tick(int_x96)))
            )
            return float(sqrt_price), tick, int_x96
        _, sqrt_price, tick = human_price_to_triplet(entry_price)
        return sqrt_price, tick, None

    @staticmethod
    def _positive_volume(value: object) -> float:
        """The input-side volume of one signed raw amount cell (0 if absent/out)."""
        if value in (None, ""):
            return 0.0
        raw = as_float(value)
        return raw if raw > 0.0 else 0.0

    @staticmethod
    def _apply_warmup_row(engine: PoolEngine, row: dict[str, object]) -> None:
        """Apply one pre-window tape row to a freshly seeded engine."""
        event_type = str(row.get("event_type", "")).lower()
        if event_type == "swap":
            x96 = row.get("sqrt_price_x96")
            tick_value = row.get("tick")
            exact_x96: int | None = None
            if x96 not in (None, ""):
                int_x96 = as_int(x96)
                exact_x96 = int_x96
                sqrt_price = int_x96 / float(Q96)
                tick = (
                    Tick(as_int(tick_value))
                    if tick_value is not None
                    else Tick(int(sqrt_price_x96_to_tick(int_x96)))
                )
            else:
                sqrt_price = engine.state.sqrt_price
                tick = engine.state.tick
            engine.step(
                sqrt_price,
                tick,
                LpEnvironment._positive_volume(row.get("amount0")),
                LpEnvironment._positive_volume(row.get("amount1")),
                new_sqrt_price_x96=exact_x96,
            )
        elif event_type in ("mint", "burn"):
            engine.apply_liquidity_event(row)

    def _capture_pool_base(self) -> None:
        if self.pool_engine.positions:
            raise EnvError(
                "LpEnvironment requires a pool engine with no agent positions at "
                "construction; it deploys its own on reset()"
            )
        self._base_state = replace(self.pool_engine.state)
        self._base_ticks = {
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
        state.sqrt_price_x96 = self._base_state.sqrt_price_x96
        state.fee_protocol = self._base_state.fee_protocol
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
            # A seeded reset is reproducible: start again at that exact window,
            # however many resets preceded it.
            self._base_seed = int(seed)
            self._episode_index = 0
        else:
            # SB3's auto-reset passes no seed; advance to the next window so a
            # training loop samples a new episode instead of replaying one.
            self._episode_index += 1
        window_seed = self._base_seed + self._episode_index
        self._rng = np.random.default_rng(window_seed)

        episode = self.config.episode
        config_steps = int(
            (episode.duration_days * 24 * 60) // episode.step_minutes
        )
        if config_steps <= 0:
            raise EnvError(
                "episode configuration yields a non-positive step budget: "
                f"{episode.duration_days}d / {episode.step_minutes}min"
            )

        # Pin the window to the env's own cadence: the view may carry a different
        # ``episode_config`` (or none), and the re-created price process below uses
        # ``episode.step_minutes``. The +1 step is a buffer for the last decision.
        start_seq, end_seq = self.market_view.build_episode(
            window_seed, max_steps=config_steps + 1, episode_config=episode
        )
        self._start_seq = int(start_seq)
        self._end_seq = int(end_seq)

        if self.is_replay:
            # The replay source's window is fixed at construction; re-create it
            # for the freshly sampled episode window so resets are independent.
            self.price_process = ReplayPriceProcess(
                self.market_view,
                self._start_seq,
                self._end_seq,
                step_minutes=int(episode.step_minutes),
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

        self._reset_external_pool()
        self._last_pool_result = {}
        self._last_tape_row = {}
        self._last_at_time = None
        self._last_mark_price = 0.0

        entry_row = self.market_view.tape_at(self._start_seq)
        entry_time = cast(datetime, entry_row["block_timestamp"])
        reference_price = self.market_view.reference_close(entry_time)
        if reference_price is None:
            reference_price = self._price_from_sqrt(self.pool_engine.state.sqrt_price)
        entry_price = float(reference_price)
        # Replicate the backtester's ``_resolve_market(0)``: the as-of reference
        # close is the valuation price, while the pool's raw sqrt/tick come from
        # the tape row when present.  When the view is backed by a boundary
        # snapshot the engine is already at the warmed window-start state and is
        # left untouched; hand-built/fixture views keep the old anchor-on-entry
        # behaviour (ADR-019).
        entry_sqrt, entry_tick, entry_x96 = self._entry_triplet(
            entry_row, entry_price
        )
        if not self._external_seeded:
            self.pool_engine.state.sqrt_price = entry_sqrt
            self.pool_engine.state.tick = entry_tick
            self.pool_engine.state.sqrt_price_x96 = entry_x96
        self._entry_price = entry_price
        self._entry_sqrt = entry_sqrt
        self._last_tape_row = entry_row
        self._last_at_time = entry_time
        self._last_mark_price = entry_price

        spacing = TickSpacing(int(episode.tick_spacing))
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
        # The deposit deployed here is the episode's fixed HODL basket.
        self._initial_position = self._position

        self._equity = self._capital
        self._last_fees_usdc = 0.0
        self._collected_fees_usdc = 0.0
        self._last_il_total = self._current_il_total(entry_sqrt, entry_price)
        self._pnl_history = [self._capital]

        self._builder.configure_episode(self._start_seq, self._episode_length)
        observation = self._observe(
            at_time=entry_time,
            tape_row=entry_row,
            step_index=0,
        )
        info = self._make_info(
            at_time=entry_time,
            observation=observation,
            breakdown=RewardBreakdown(equity=self._equity),
            raw=RewardBreakdown(equity=self._equity),
            action=self._action_lookup[0],
            action_index=0,
        )
        return observation.to_vector(), info

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
        # The excess-vs-HODL step change is measured on the *pre-action*
        # position, exactly as the backtester's ledger does (ADR-029), so the
        # reward decomposition reconciles with ``excess_vs_hodl``.
        il_change = self._current_il_total(sqrt_price, price) - self._last_il_total

        gas_cost = 0.0
        slippage_cost = 0.0
        # A requested rebalance that cannot execute (the market is exhausted, or
        # the trade cannot pay for itself) is reported as rejected so turnover
        # counts match the backtester's decision log.
        rebalance_rejected = action.action_type == "rebalance" and price_exhausted
        if action.action_type == "rebalance" and not price_exhausted:
            gas_cost, slippage_cost, executed = self._rebalance(
                action, sqrt_price, price, tape_row
            )
            rebalance_rejected = not executed
            # New position starts with zero uncollected fees.
            fees_usdc_before = self._current_fees_usdc(price)
        self._last_fees_usdc = fees_usdc_before
        self._last_il_total = self._current_il_total(sqrt_price, price)

        position = self._position
        assert position is not None, "reset deploys a position before any step"
        position_value = position.value(sqrt_price, price)
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
        info = self._make_info(
            at_time=at_time,
            observation=observation,
            breakdown=breakdown,
            raw=raw_breakdown,
            action=action,
            action_index=int(action_idx),
        )
        if rebalance_rejected:
            # The requested rebalance did not execute (it could not pay for the
            # trade, or the market was exhausted); the position is unchanged.
            # ``action_type`` records that nothing executed while ``action``
            # above keeps the policy's intent.
            info["action_type"] = "rebalance_rejected"
            info["rebalance_rejected"] = True
        return observation.to_vector(), float(breakdown.reward), terminated, truncated, info

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
        except StopIteration:  # pragma: no cover - unreachable in normal replay
            # ``reset`` caps ``_episode_length`` at ``len(_bar_close_times)``,
            # so the episode truncates before a step can ask the exhausted
            # process for another bar.  Defensive fallback only.
            at_time = self._last_at_time or cast(
                datetime,
                self.market_view.tape_at(self._end_seq - 1)["block_timestamp"],
            )
            return (
                self._last_mark_price,
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
            event_type = str(self._ev_types[index]).lower()
            if event_type == "swap":
                event_sqrt, event_tick = self._event_triplet(index)
                volume_0, volume_1 = self._event_volumes(index)
                raw_x96 = self._ev_sqrt_x96[index]
                event_x96 = (
                    as_int(raw_x96) if raw_x96 not in (None, "") else None
                )
                result = self.pool_engine.step(
                    event_sqrt,
                    event_tick,
                    volume_0,
                    volume_1,
                    new_sqrt_price_x96=event_x96,
                )
                interval_fee_0 += result["pool_fee_earned_0"]
                interval_fee_1 += result["pool_fee_earned_1"]
            elif event_type in ("mint", "burn"):
                # External liquidity moves the fee-growth denominator exactly as
                # the backtester's tracker does (ADR-019). ``collect``/``flash``
                # move no fee state and are skipped, as in ``apply_event``.
                self.pool_engine.apply_liquidity_event(self._ev_rows[index])
            self._last_tape_row = self._ev_rows[index]
            self._event_idx += 1

        # Fixture-only zero-volume alignment (ADR-019).  Hand-built views (the
        # shared ``tiny_market_view``) hold a constant tape price while their
        # reference feed moves, so the pool is aligned to the decision bar to
        # keep the agent in range.  When the view is backed by a boundary
        # snapshot the pool state is driven by the tape alone, exactly like the
        # backtester's tracker: aligning onto a reference tick can land exactly
        # on an agent boundary and flip ``fee_growth_outside`` in a way the
        # tracker never does, producing spurious multi-hundred-thousand-USDC fee
        # jumps.  The reference ``sqrt_price``/``tick`` returned above are still
        # used for valuation.
        if not self._external_seeded:
            self.pool_engine.step(
                sqrt_price,
                tick,
                0.0,
                0.0,
                new_sqrt_price_x96=None,
            )
        self._last_pool_result = {
            "fees_accrued": {},
            "ticks_crossed": [],
            "active_liquidity": self.pool_engine.state.liquidity,
            "pool_fee_earned_0": interval_fee_0,
            "pool_fee_earned_1": interval_fee_1,
        }
        self._last_at_time = at_time
        # Mark the position at the **last tape event's own reference price** — the
        # same ``(raw sqrt, human price)`` pair the backtester's
        # ``_resolve_market`` returns for that event.  ``price_process.step()``
        # gives the decision bar's close, which can be a full reference bar later
        # than the last event's as-of close; pairing a bar-close human price with
        # the last swap's raw ``sqrt_price`` values the position with two
        # different timestamps and injects a one-bar mark-to-market error into
        # IL/PnL parity (ADR-019 parity follow-up).  The event's raw sqrt is
        # already on the engine, so only the human leg needs correcting.
        mark_price = price
        if self._external_seeded:
            for key in ("price_reference", "price_pool"):
                raw = self._last_tape_row.get(key)
                if raw not in (None, ""):
                    mark_price = as_float(raw)
                    break
        self._last_mark_price = mark_price
        return (
            mark_price,
            self.pool_engine.state.sqrt_price,
            at_time,
            self._last_tape_row,
            False,
        )

    def _advance_calibrated(
        self,
    ) -> tuple[float, SqrtPrice, datetime, dict[str, object], bool]:
        """Advance one calibrated price step with synthetic swap volume."""
        try:
            price, sqrt_price, tick = self.price_process.step(self._rng)
        except StopIteration:
            at_time = self._last_at_time or cast(
                datetime,
                self.market_view.tape_at(self._end_seq - 1)["block_timestamp"],
            )
            return (
                self._last_mark_price,
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
        at_time = cast(datetime, tape_row["block_timestamp"])
        self._last_tape_row = tape_row
        self._last_at_time = at_time
        self._last_mark_price = price
        return price, sqrt_price, at_time, tape_row, False

    def _event_triplet(self, index: int) -> tuple[SqrtPrice, Tick]:
        """Raw ``(sqrt_price, tick)`` for a swap tape event (ADR-009)."""
        x96 = self._ev_sqrt_x96[index]
        if x96 not in (None, ""):
            int_x96 = as_int(x96)
            sqrt_price = int_x96 / float(Q96)
            tick_value = self._ev_ticks[index]
            tick = (
                Tick(as_int(tick_value))
                if tick_value is not None
                else Tick(int(sqrt_price_x96_to_tick(int_x96)))
            )
            return float(sqrt_price), tick
        return self.pool_engine.state.sqrt_price, self.pool_engine.state.tick

    def _event_volumes(self, index: int) -> tuple[float, float]:
        """Raw swap volume for a tape event (the engine's units, ADR-012).

        Only the **input** leg is returned: ``amount0``/``amount1`` are signed
        (input positive, output negative) and Uniswap — like the backtester's
        exact ``FeeGrowthTracker`` (CONTRACTS.md §4.1) — charges the fee on the
        input side alone.  The output leg is reported as ``0.0`` so the engine's
        piecewise step never double-charges the same swap.
        """
        return (
            self._positive_volume(self._ev_amount0[index]),
            self._positive_volume(self._ev_amount1[index]),
        )

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
        """Strategy value minus the fixed initial HODL basket, in USDC.

        This is the position's mark-to-market value (excluding uncollected
        fees, which the reward scores separately) measured against the episode's
        initial deposit held outside the pool.  The benchmark is fixed for the
        whole episode (ADR-029), so the level accumulates the rebalancing PnL
        instead of resetting at every rebalance: its per-step change is the
        backtester's ``il`` column and its episode sum is ``excess_vs_hodl``.
        """
        if self._position is None or self._initial_position is None:
            return 0.0
        return self._position.value(sqrt_price, price) - (
            self._initial_position.hodl_value(
                self._entry_sqrt, self._entry_price, price
            )
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
        block_number = as_int(tape_row.get("block_number", 0) or 0)
        row = self.market_view.gas_at_block(block_number)
        return row if row is not None else tape_row

    def _rebalance(
        self,
        action: Action,
        sqrt_price: SqrtPrice,
        price: float,
        tape_row: dict[str, object],
    ) -> tuple[float, float, bool]:
        """Close the current position, open the action's range, return costs.

        Returns ``(gas_cost, slippage_cost, executed)`` as positive magnitudes.
        When the position cannot pay the gas/slippage out of its own equity the
        trade reverts (``executed=False``): nothing is charged and the position,
        fee balance and HODL benchmark are left untouched. The new position is
        funded with the old position's mark-to-market value **plus its collected
        fees**, net of the rebalance costs, so equity is continuous and
        cost-honest.  The fixed initial HODL benchmark is never rebased: the
        rebalancing PnL stays in the ``_current_il_total`` level (ADR-029).
        """
        if self._position is None or self._position_id is None:
            raise EnvError("rebalance requested with no open position")
        spacing = TickSpacing(int(self.config.episode.tick_spacing))
        lower = action.lower_tick(self.pool_engine.state.tick, spacing)
        upper = action.upper_tick(self.pool_engine.state.tick, spacing)

        notional = self._position.value(sqrt_price, price)
        collected = self._current_fees_usdc(price)

        gas_row = self._gas_row(tape_row)
        base_fee = as_int(gas_row.get("base_fee_per_gas") or 0)
        priority_fee = as_int(gas_row.get("priority_fee_p50_wei") or 0)
        gas_cost = float(
            self.gas_model.gas_cost_usdc(
                "rebalance",
                as_int(tape_row.get("block_number", 0) or 0),
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

        # The trade must pay for itself out of the position's own equity; when
        # it cannot, it reverts on-chain. Charge nothing and leave the book
        # untouched (the old ``max(..., 1.0)`` floor silently recapitalised the
        # position, diverging from the ground-truth backtester).
        new_capital = notional + collected - gas_cost - slippage_cost
        if new_capital <= 0.0:
            return 0.0, 0.0, False

        # Realise the closing fees into the redeployed capital; the fixed HODL
        # benchmark is not rebased, so the position-value change flows through
        # ``_current_il_total`` on the next observation (ADR-029).
        self._collected_fees_usdc += collected
        self.pool_engine.close_position(self._position_id)

        # Re-deploy the position value plus collected fees, net of costs, so
        # equity (and every reported PnL number) is continuous and cost-honest.
        new_position = initial_deposit(
            sqrt_price, price, lower, upper, new_capital, spacing
        )
        self._position_id = self.pool_engine.open_position(
            lower, upper, new_position.liquidity
        )
        self._position = self.pool_engine.positions[self._position_id]
        return gas_cost, slippage_cost, True

    def _observe(
        self,
        *,
        at_time: datetime,
        tape_row: dict[str, object],
        step_index: int,
    ) -> Observation:
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
            hodl_position=self._initial_position,
        )
        return observation

    def _make_info(
        self,
        *,
        at_time: datetime,
        observation: Observation,
        breakdown: RewardBreakdown,
        raw: RewardBreakdown,
        action: Action,
        action_index: int,
    ) -> dict[str, Any]:
        """Assemble the step ``info`` dict.

        ``at_time`` and ``action_index`` are not derivable from the action object
        alone, so they are passed explicitly.  The market/position fields mirror
        the observation vector: U06's rollout recorder persists them without
        peeking at ``LpEnvironment`` internals (CONTRACTS.md §2.1).  The
        ``entry_price`` field is the **episode** entry (the fixed HODL anchor,
        ADR-029), not a per-rebalance entry.
        """
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
            # -- U06 rollout fields (CONTRACTS.md §2.1) --
            "action_index": int(action_index),
            "timestamp": at_time,
            "price": float(observation.price),
            "tick": int(observation.tick),
            "in_range": bool(observation.position_in_range),
            "position_value": float(observation.position_value),
            "regime": observation.regime,
            "gas_price_wei": float(observation.gas_price_recent_wei),
        }
