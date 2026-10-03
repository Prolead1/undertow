"""Frozen-policy event replay backtester (S11, CONTRACTS.md §13).

``run_backtest(policy, market_view, config)`` replays a **frozen** policy over the
real block-ordered event tape of a :class:`~undertow.sim.marketview.MarketView`.
The policy only ever receives :meth:`Policy.act` — the :class:`Policy` protocol has
no learning method, so "the policy updated during the backtest" is unrepresentable
(PLAN.md §4, "Frozen means frozen").

Exactness discipline
--------------------
The fee path stays in the data module's currency: Q128 integer accumulators.  The
position's fees are accrued by ``undertow.data``'s exact fee-growth engine, reached
through the top-level ``FeeGrowthReplay`` facade (ADR-013).  Each tape event is
replayed in order (swaps through the tracker's tick lattice, mints/burns through
its liquidity bookkeeping) and the position's ``ΔfeeGrowthInside`` is evaluated on
Python integers; the result is divided by ``10**dec`` only when it is marked to
USDC.  No accumulator is ever round-tripped through ``float`` (PLAN.md §4, "Two
numeric worlds").

This handles boundary-crossing swaps and pre-existing ``feeGrowthOutside`` exactly
(the exact segment split the data module already reconciles), superseding the
earlier tape-columns apportionment, which was exact only for an in-range position
whose boundaries were not crossed.  The tape's ``fee_growth_global_*_x128``
columns are retained as a cross-check in the tests, not as the accrual source.
Residuals that remain are the tracker's own, documented in ADR-013: a swap that
crosses a tick before its price is known is flagged ``exact=False`` and is recorded
on the ledger (``fee_growth_exact``); flash fees are out of scope; and the tick
lattice is built from the events present in the tape window.

Observation
-----------
S12 owns the canonical :class:`~undertow.sim.env.observations.Observation`
(CONTRACTS §11) and lands in the same wave as this task.  To keep this module
independently importable and mergeable it passes a structurally identical,
read-only :class:`BacktestObservation` snapshot with exactly the §11 field set;
baseline policies read it by attribute and are agnostic to the class.  S12 can
swap in its builder at the composition root without touching this module.

Units (ADR-009/ADR-012)
-----------------------
``sqrt_price``/``tick``/``liquidity`` are **raw** protocol quantities; ``price`` is
the human USDC-per-WETH reference close.  Pool fees are Uniswap pips
(``fee_tier_bps / 1_000_000``); slippage adds true basis points
(``fixed_impact_bps / 10_000``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from undertow.data import (
    Q96,
    FeeGrowthReplay,
    PoolConfig,
    default_pools,
    price_to_tick,
)
from undertow.sim.backtest.ledger import (
    BacktestLedger,
    config_hash,
    decompose_equity_curve,
    git_commit,
)
from undertow.sim.config import SimConfig
from undertow.sim.core.position import (
    DEFAULT_DEC0,
    DEFAULT_DEC1,
    Position,
    initial_deposit,
)
from undertow.sim.frictions import GasCostCalculator, ProportionalSlippageModel
from undertow.sim.marketview import MarketView
from undertow.sim.types import Action, BacktestError, Tick, TickSpacing

if TYPE_CHECKING:
    from undertow.sim.policies import Policy

LOGGER = logging.getLogger("undertow.sim.backtest.engine")

__all__ = ["DEFAULT_INITIAL_HALF_WIDTH_TICKS", "BacktestObservation", "run_backtest"]

#: Annualization basis when the tape has fewer than two events (matches S05).
_DEFAULT_PERIODS_PER_YEAR: int = 365 * 24 * 6

#: Half-width, in ticks, of the initial position deployed before the first
#: decision.  Mirrors the S04 ``entry_position`` fixture (±120 ticks) so the
#: buy-and-hold benchmark's initial token mix has a definite, testable shape.
#: Every policy starts from this initial position; HODL simply never rebalances.
DEFAULT_INITIAL_HALF_WIDTH_TICKS: int = 120

#: Tape columns the replay reads.  All are part of the frozen
#: ``EVENT_TAPE_SCHEMA``; validating them up front turns a missing column into a
#: clear ``BacktestError`` instead of a ``KeyError`` deep in the buffer.
_REQUIRED_TAPE_COLUMNS: tuple[str, ...] = (
    "seq",
    "block_number",
    "block_timestamp",
    "price_reference",
    "price_pool",
    "sqrt_price_x96",
    "tick",
    "liquidity",
    "base_fee_per_gas",
    "priority_fee_p50_wei",
    "fee_growth_global_0_x128",
    "fee_growth_global_1_x128",
    "regime",
    # ADR-013: the exact replay reads the raw event payload, not the derived
    # global columns (which stay validated as a cross-check anchor).
    "event_type",
    "amount0",
    "amount1",
    "tick_lower",
    "tick_upper",
    "liquidity_amount",
)


@dataclass(frozen=True, slots=True)
class BacktestObservation:
    """Read-only snapshot with the CONTRACTS §11 field set.

    A structural stand-in for S12's canonical ``Observation`` (see the module
    docstring).  All fields are backward-looking and closed on the right.  Once
    S12 lands, the composition root can pass its builder/type here without
    changing the frozen :func:`run_backtest` signature.
    """

    price: float
    sqrt_price: float
    tick: int
    price_returns_1h: float
    price_returns_24h: float
    realized_vol_24h: float
    position_in_range: bool
    position_value: float
    uncollected_fees_token0: float
    uncollected_fees_token1: float
    il_vs_hodl: float
    pool_active_liquidity: float
    pool_fee_tier_bps: int
    gas_price_recent_wei: float
    eth_usd_price_recent: float
    regime: str
    step_in_episode: int
    steps_remaining: int

    def to_vector(self) -> np.ndarray:
        """Flatten to a stable float64 vector (the §11 ordinal regime encoding)."""
        ordinal = {"bull": 0, "bear": 1, "sideways": 2, "high_vol": 3, "unknown": 4}
        return np.asarray(
            [
                self.price,
                self.sqrt_price,
                float(self.tick),
                self.price_returns_1h,
                self.price_returns_24h,
                self.realized_vol_24h,
                1.0 if self.position_in_range else 0.0,
                self.position_value,
                self.uncollected_fees_token0,
                self.uncollected_fees_token1,
                self.il_vs_hodl,
                self.pool_active_liquidity,
                float(self.pool_fee_tier_bps),
                self.gas_price_recent_wei,
                self.eth_usd_price_recent,
                float(ordinal.get(self.regime, 4)),
                float(self.step_in_episode),
                float(self.steps_remaining),
            ],
            dtype=np.float64,
        )


class _TapeBuffer:
    """Column-oriented view of the active split's event tape.

    Pulling the columns into Python lists once keeps the replay loop free of
    per-row polars overhead and of repeated look-ahead queries.
    """

    def __init__(self, tape: pl.DataFrame) -> None:
        self.seq: list[int] = [int(v) for v in tape["seq"].to_list()]
        self.time: list[datetime] = list(tape["block_timestamp"].to_list())
        self.block: list[int] = [int(v) for v in tape["block_number"].to_list()]
        self.price_reference: list[float | None] = list(
            tape["price_reference"].to_list()
        )
        self.price_pool: list[float | None] = list(tape["price_pool"].to_list())
        self.sqrt_price_x96: list[object] = list(tape["sqrt_price_x96"].to_list())
        self.tick: list[object] = list(tape["tick"].to_list())
        self.liquidity: list[object] = list(tape["liquidity"].to_list())
        self.base_fee: list[object] = list(tape["base_fee_per_gas"].to_list())
        self.priority_fee: list[object] = list(
            tape["priority_fee_p50_wei"].to_list()
        )
        self.fee_growth_0: list[object] = list(
            tape["fee_growth_global_0_x128"].to_list()
        )
        self.fee_growth_1: list[object] = list(
            tape["fee_growth_global_1_x128"].to_list()
        )
        self.regime = list(tape["regime"].to_list())
        # Raw event payload consumed by the ADR-013 exact fee-growth replay.
        self.event_type: list[object] = list(tape["event_type"].to_list())
        self.amount0: list[object] = list(tape["amount0"].to_list())
        self.amount1: list[object] = list(tape["amount1"].to_list())
        self.tick_lower: list[object] = list(tape["tick_lower"].to_list())
        self.tick_upper: list[object] = list(tape["tick_upper"].to_list())
        self.liquidity_amount: list[object] = list(tape["liquidity_amount"].to_list())

    def __len__(self) -> int:
        return len(self.seq)

    def row(self, i: int) -> dict[str, object]:
        """The event payload for row ``i`` in the shape ``FeeGrowthReplay`` expects."""
        return {
            "event_type": self.event_type[i],
            "block_number": self.block[i],
            "amount0": self.amount0[i],
            "amount1": self.amount1[i],
            "sqrt_price_x96": self.sqrt_price_x96[i],
            "tick": self.tick[i],
            "tick_lower": self.tick_lower[i],
            "tick_upper": self.tick_upper[i],
            "liquidity_amount": self.liquidity_amount[i],
        }


def _as_int(value: object, default: int = 0) -> int:
    """Coerce a tape value (decimal string, int or ``None``) to ``int``."""
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(str(value))


def _sqrt_from_x96(x96: object) -> float:
    """Raw sqrt price from an exact Q64.96 integer (ADR-009)."""
    return _as_int(x96) / float(Q96)


def _sqrt_from_price(price: float, dec0: int = DEFAULT_DEC0, dec1: int = DEFAULT_DEC1) -> float:
    """Raw sqrt price paired with a human price (ADR-009).

    ``sqrt_price_raw = 10**((dec1 - dec0) / 2) / sqrt(price)``.
    """
    return (10.0 ** ((dec1 - dec0) / 2.0)) / math.sqrt(price)


def _tick_from_price(price: float, dec0: int = DEFAULT_DEC0, dec1: int = DEFAULT_DEC1) -> int:
    """Human price → raw tick with data's floor semantics (ADR-009)."""
    return int(price_to_tick(Decimal(str(price)), dec0, dec1))


def _snap(tick: int, spacing: int) -> int:
    """Floor ``tick`` to the spacing grid."""
    return (int(tick) // spacing) * spacing


def _default_initial_bounds(
    entry_tick: int,
    spacing: int,
    half_width_ticks: int = DEFAULT_INITIAL_HALF_WIDTH_TICKS,
) -> tuple[int, int]:
    """Snapped symmetric bounds around ``entry_tick`` for the initial position."""
    lower = _snap(entry_tick - half_width_ticks, spacing)
    upper = -((-(entry_tick + half_width_ticks)) // spacing) * spacing
    if lower >= upper:  # pragma: no cover - defensive; half width is positive
        raise BacktestError("degenerate initial range after snapping")
    return lower, upper


def _resolve_pool(market_view: MarketView, config: SimConfig) -> PoolConfig:
    """The pool the exact fee-growth replay runs against (ADR-013).

    Prefers the ``PoolConfig`` the tape was pulled from (``MarketView.pool``,
    populated by :func:`~undertow.sim.marketview.build_market_view` from its
    ``DataConfig``).  Hand-built views leave it ``None``; the fallback matches
    the sim's configured fee tier / tick spacing against the pinned pool
    registry, and otherwise uses the 0.30% USDC/WETH pool as the canonical
    identity (the engine always passes the sim's own ``fee_tier_bps`` to
    ``FeeGrowthReplay`` as ``fee_pips``, so an ablation fee tier of 0 is still
    honoured exactly).
    """
    if market_view.pool is not None:
        return market_view.pool
    pools = default_pools()
    for pool in pools.values():
        if (
            pool.fee_tier.value == config.episode.fee_tier_bps
            and pool.tick_spacing == config.episode.tick_spacing
        ):
            return pool
    return pools["USDC_WETH_3000"]


def _periods_per_year(times: list[datetime]) -> int:
    """Annualization basis from the median event spacing (365-day year)."""
    if len(times) < 2:
        return _DEFAULT_PERIODS_PER_YEAR
    deltas = [
        (times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)
    ]
    positive = [d for d in deltas if d > 0]
    if not positive:
        return _DEFAULT_PERIODS_PER_YEAR
    step_seconds = median(positive)
    if step_seconds <= 0:  # pragma: no cover - filtered above
        return _DEFAULT_PERIODS_PER_YEAR
    return max(1, int(round((365 * 24 * 60 * 60) / step_seconds)))


class _Replay:
    """Mutable per-run state for the replay loop."""

    def __init__(
        self,
        policy: Policy,
        market_view: MarketView,
        config: SimConfig,
        tape: _TapeBuffer,
    ) -> None:
        self.policy = policy
        self.market_view = market_view
        self.config = config
        self.tape = tape
        self.capital = float(config.episode.agent_capital_usdc)
        self.tick_spacing = int(config.episode.tick_spacing)
        self.dec0 = DEFAULT_DEC0
        self.dec1 = DEFAULT_DEC1

        self.gas_calculator = GasCostCalculator(config.gas)
        self.slippage_model = ProportionalSlippageModel()
        self.periods_per_year = _periods_per_year(tape.time)

        # Price/tick carry-forward state (mint/burn rows carry no pool price).
        self.last_price: float | None = None
        self.last_sqrt: float | None = None
        self.last_tick: int | None = None
        self.pool_liquidity = 0.0
        self.price_history: list[float] = []

        # Position + exact fee-growth state.  ``replay`` (ADR-013) is set by
        # ``run_backtest`` once the pool and the tape's opening tick are known.
        # ``fee_growth_last_*`` is ``None`` until the first event has been
        # replayed: the position is seeded there without crediting the
        # pre-deployment gap, so it earns fees only from the next event onward.
        self.position: Position | None = None
        self.replay: FeeGrowthReplay | None = None
        self.fee_growth_last_0: int | None = None
        self.fee_growth_last_1: int | None = None
        self.accrued_raw_0: int = 0
        self.accrued_raw_1: int = 0
        # Mark-to-market value of accrued fees at the end of the previous step;
        # the per-step `fees` term is its change (including price revaluation of
        # the WETH leg), which is what makes the equity change reconcile exactly.
        self.fees_value_prev: float = 0.0
        self.il_prev: float = 0.0
        self.equity: float = self.capital

        # The initial buy-and-hold benchmark is fixed by the first deployment.
        self.initial_position: Position | None = None
        self.initial_entry_sqrt: float = 0.0
        self.initial_entry_price: float = 0.0

        # Ledger rows (parallel lists; assembled into polars frames at the end).
        self.rows: dict[str, list[object]] = {
            "step": [],
            "equity": [],
            "pnl": [],
            "fees": [],
            "gas": [],
            "slippage": [],
            "il": [],
        }
        self.decisions: dict[str, list[object]] = {
            "step": [],
            "action_type": [],
            "center_offset": [],
            "width": [],
            "tick_lower": [],
            "tick_upper": [],
            "gas_paid": [],
            "slippage_paid": [],
            "fees_collected": [],
        }
        self.costs: dict[str, list[object]] = {
            "step": [],
            "cost_type": [],
            "amount_usdc": [],
        }

    # -- price / tick resolution ------------------------------------------
    def _resolve_market(self, i: int) -> tuple[float, float, int]:
        """Resolve ``(price, raw_sqrt, tick)`` for tape row ``i``.

        Uses the tape's as-of reference close and the pool's own raw sqrt/tick,
        carrying the last known pool state forward across liquidity events.
        """
        time = self.tape.time[i]
        raw_price = self.tape.price_reference[i]
        if raw_price is None:
            # The first event can precede the reference feed's first close; the
            # same-block pool price is the look-ahead-safe fallback.
            raw_price = self.tape.price_pool[i]
        if raw_price is None:
            raw_price = self.market_view.reference_close(time)
        price = float(raw_price) if raw_price is not None else self.last_price
        if price is None or price <= 0.0:
            raise BacktestError(
                f"no usable reference price at tape step {self.tape.seq[i]}"
            )

        x96 = self.tape.sqrt_price_x96[i]
        sqrt = _sqrt_from_x96(x96) if x96 is not None else self.last_sqrt
        if sqrt is None or sqrt <= 0.0:
            sqrt = _sqrt_from_price(price, self.dec0, self.dec1)

        tick_value = self.tape.tick[i]
        if tick_value is not None:
            tick = int(tick_value)
        elif self.last_tick is not None:
            tick = self.last_tick
        else:
            tick = _tick_from_price(price, self.dec0, self.dec1)

        liq = self.tape.liquidity[i]
        if liq is not None:
            self.pool_liquidity = float(int(str(liq)))

        self.last_price = price
        self.last_sqrt = sqrt
        self.last_tick = tick
        self.price_history.append(price)
        return price, sqrt, tick

    # -- fees --------------------------------------------------------------
    def _accrue_fees(self, price: float) -> tuple[float, float]:
        """Accrue this step's fees through the exact replay (ADR-013).

        The step's event has already been fed to ``self.replay``.  The position's
        ``ΔfeeGrowthInside`` is evaluated by the data module's tracker on Q128
        integers; the human division happens once, here.

        The first accrual after deployment (or after a rebalance) *seeds* the
        position's inside snapshots and credits nothing, so the position never
        earns fee growth from before it existed.
        """
        replay = self.replay
        position = self.position
        assert replay is not None and position is not None
        liquidity = int(round(position.liquidity))
        if self.fee_growth_last_0 is None or self.fee_growth_last_1 is None:
            _f0, _f1, g0, g1 = replay.accrue(
                tick_lower=position.tick_lower,
                tick_upper=position.tick_upper,
                liquidity=liquidity,
                g_inside_last_0=0,
                g_inside_last_1=0,
            )
        else:
            fees0, fees1, g0, g1 = replay.accrue(
                tick_lower=position.tick_lower,
                tick_upper=position.tick_upper,
                liquidity=liquidity,
                g_inside_last_0=self.fee_growth_last_0,
                g_inside_last_1=self.fee_growth_last_1,
            )
            self.accrued_raw_0 += fees0
            self.accrued_raw_1 += fees1
        self.fee_growth_last_0 = g0
        self.fee_growth_last_1 = g1

        fees_value = (
            self.accrued_raw_0 / 10.0**self.dec0
            + self.accrued_raw_1 / 10.0**self.dec1 * price
        )
        # The `fees` term is the mark-to-market change of the accrued fee balance:
        # the new accrual plus the revaluation of the previously accrued WETH leg.
        # Without the revaluation term the equity change would not reconcile with
        # the decomposition after any hold-then-price-move.
        step_fees_value = fees_value - self.fees_value_prev
        return fees_value, step_fees_value

    # -- observation -------------------------------------------------------
    def _build_observation(
        self,
        price: float,
        sqrt: float,
        tick: int,
        position_value: float,
        fees_value: float,
        il_now: float,
        gas_wei: float,
        step_in_episode: int,
        total_decisions_hint: int,
        regime: str,
    ) -> BacktestObservation:
        history = self.price_history
        r1h = 0.0
        r24h = 0.0
        vol24h = 0.0
        if len(history) >= 2:
            n1 = min(6, len(history) - 1)
            r1h = math.log(history[-1] / history[-1 - n1])
            n24 = min(144, len(history) - 1)
            r24h = math.log(history[-1] / history[-1 - n24])
            window = history[-min(144, len(history)) :]
            if len(window) >= 3:
                logs = [
                    math.log(window[k + 1] / window[k]) for k in range(len(window) - 1)
                ]
                mean = sum(logs) / len(logs)
                var = sum((x - mean) ** 2 for x in logs) / max(1, len(logs) - 1)
                vol24h = math.sqrt(var) * math.sqrt(self.periods_per_year)

        position = self.position
        in_range = bool(position is not None and position.tick_lower <= tick < position.tick_upper)
        fees0 = self.accrued_raw_0 / 10.0**self.dec0
        fees1 = self.accrued_raw_1 / 10.0**self.dec1
        return BacktestObservation(
            price=price,
            sqrt_price=sqrt,
            tick=tick,
            price_returns_1h=r1h,
            price_returns_24h=r24h,
            realized_vol_24h=vol24h,
            position_in_range=in_range,
            position_value=position_value,
            uncollected_fees_token0=fees0,
            uncollected_fees_token1=fees1,
            il_vs_hodl=(il_now / self.capital) if self.capital else 0.0,
            pool_active_liquidity=self.pool_liquidity,
            pool_fee_tier_bps=int(self.config.episode.fee_tier_bps),
            gas_price_recent_wei=gas_wei,
            eth_usd_price_recent=price,
            regime=regime,
            step_in_episode=step_in_episode,
            steps_remaining=max(0, total_decisions_hint - step_in_episode - 1),
        )

    # -- decisions ---------------------------------------------------------
    def _decision_due(self, i: int, last_decision_idx: int | None) -> bool:
        if last_decision_idx is None:
            return True
        step_minutes = max(1, int(self.config.episode.step_minutes))
        start = self.tape.time[last_decision_idx]
        elapsed = (self.tape.time[i] - start).total_seconds() / 60.0
        return elapsed >= step_minutes

    def _apply_rebalance(
        self,
        action: Action,
        tick: int,
        sqrt: float,
        price: float,
        position_value: float,
        fees_value: float,
        i: int,
    ) -> tuple[float, float, float, int, int]:
        """Close + reopen at the action's range; return the new equity and costs.

        Returns ``(equity, gas_cost_magnitude, slippage_cost_magnitude,
        new_lower, new_upper)``.
        """
        block = self.tape.block[i]
        base_fee = _as_int(self.tape.base_fee[i])
        priority = _as_int(self.tape.priority_fee[i])
        gas_cost = self.gas_calculator.gas_cost_usdc(
            "rebalance", block, base_fee, priority, price
        )
        notional = max(0.0, position_value)
        slippage_cost = self.slippage_model.slippage_cost_usdc(
            notional,
            int(self.config.episode.fee_tier_bps),
            float(self.config.slippage.fixed_impact_bps),
        )

        new_capital = position_value + fees_value - gas_cost - slippage_cost
        if new_capital <= 0.0:
            raise BacktestError(
                f"rebalance at step {self.tape.seq[i]} would wipe out the position "
                f"(equity {new_capital:.6f})"
            )

        lower = int(action.lower_tick(Tick(tick), TickSpacing(self.tick_spacing)))
        upper = int(action.upper_tick(Tick(tick), TickSpacing(self.tick_spacing)))
        new_position = initial_deposit(
            sqrt,
            price,
            Tick(lower),
            Tick(upper),
            new_capital,
            TickSpacing(self.tick_spacing),
        )
        if not new_position.in_range(sqrt) and not (
            new_position.below_range(sqrt) or new_position.above_range(sqrt)
        ):  # pragma: no cover - defensive
            raise BacktestError("new position is not valvable at the rebalance price")

        # Collect: realise fee income into the redeployed capital and rebase the
        # exact inside snapshots for the new range so the new position only earns
        # future growth (the fees returned by this accrue call are discarded; it
        # exists only to read the new range's current inside values).
        self.position = new_position
        self.accrued_raw_0 = 0
        self.accrued_raw_1 = 0
        assert self.replay is not None
        self.replay.register_position(
            tick_lower=new_position.tick_lower,
            tick_upper=new_position.tick_upper,
        )
        _f0, _f1, g0, g1 = self.replay.accrue(
            tick_lower=new_position.tick_lower,
            tick_upper=new_position.tick_upper,
            liquidity=int(round(new_position.liquidity)),
            g_inside_last_0=0,
            g_inside_last_1=0,
        )
        self.fee_growth_last_0 = g0
        self.fee_growth_last_1 = g1
        self.equity = new_capital
        return new_capital, gas_cost, slippage_cost, lower, upper


def run_backtest(
    policy: Policy,
    market_view: MarketView,
    config: SimConfig,
) -> BacktestLedger:
    """Replay a frozen policy over the real event tape (CONTRACTS §13).

    The policy is reset once and then only ever asked for ``act()``; there is no
    learning call anywhere on the path.  The tape is replayed at its own event
    granularity — the equity curve records every tape event, while the policy
    decides at the ``config.episode.step_minutes`` cadence.

    Raises
    ------
    BacktestError
        If the tape is empty, carries no usable reference price, or a rebalance
        would wipe out the position.
    """
    tape_frame = market_view.active_split_data()
    if tape_frame.height == 0:
        raise BacktestError("run_backtest: the active split has no tape rows")
    for required in _REQUIRED_TAPE_COLUMNS:
        if required not in tape_frame.columns:
            raise BacktestError(f"run_backtest: tape is missing '{required}'")

    tape = _TapeBuffer(tape_frame)
    run = _Replay(policy, market_view, config, tape)

    # Construct the exact fee-growth replay (ADR-013) once.  The opening tick is
    # the first tick the tape observes, used only to place the tracker on the
    # real grid; the first swap's pre-price is legitimately unknown (the
    # tracker flags an approximation if that swap crosses).  An empty tick
    # lattice is built by the tape's own mint/burn events.
    seed_tick = 0
    for tick_value in tape.tick:
        if tick_value is not None:
            seed_tick = _as_int(tick_value)
            break
    run.replay = FeeGrowthReplay(
        _resolve_pool(market_view, config),
        fee_pips=int(config.episode.fee_tier_bps),
        block_number=tape.block[0] if tape.block else 0,
        current_tick=seed_tick,
    )

    total_events = len(tape)
    periods_per_year = run.periods_per_year

    policy.reset()
    rng = config.rng()

    last_decision_idx: int | None = None
    decision_index = 0
    # A backward-looking hint for the observation's `steps_remaining`; the exact
    # decision count is cadence-dependent.
    total_decisions_hint = max(
        1,
        int(
            (tape.time[-1] - tape.time[0]).total_seconds()
            / 60.0
            / max(1, int(config.episode.step_minutes))
        )
        + 1,
    )

    for i in range(total_events):
        price, sqrt, tick = run._resolve_market(i)
        if i == 0:
            # Deploy the initial position before the first `act()` call for every
            # policy (S10 brief): HODL holds this position, the others rebalance
            # away from it at their first decision.  The fee-growth baseline is
            # seeded by `_accrue_fees` from the first non-null snapshot below.
            lower0, upper0 = _default_initial_bounds(tick, run.tick_spacing)
            initial_position = initial_deposit(
                sqrt,
                price,
                Tick(lower0),
                Tick(upper0),
                run.capital,
                TickSpacing(run.tick_spacing),
            )
            run.position = initial_position
            run.initial_position = initial_position
            run.initial_entry_sqrt = sqrt
            run.initial_entry_price = price
            run.equity = run.capital
            # Register the synthetic position's boundaries so a later real mint
            # at the same tick cannot re-seed feeGrowthOutside (ADR-013).
            run.replay.register_position(
                tick_lower=initial_position.tick_lower,
                tick_upper=initial_position.tick_upper,
            )

        # Replay this event through the exact tracker, then accrue the position's
        # inside growth from the updated state (ADR-013).
        run.replay.apply_event(tape.row(i))
        fees_value, step_fees_value = run._accrue_fees(price)

        position = run.position
        assert position is not None  # always deployed before the loop
        position_value = position.value(sqrt, price)
        # The HODL benchmark is the *initial* deposit's token mix held for the
        # whole window (PLAN §7 "ΔIL definition"), so it does not move when the
        # policy rebalances.  This makes the per-step `il` column and the
        # decomposition reconcile exactly with `excess_vs_hodl`.
        hodl_now = run.initial_position.hodl_value(
            run.initial_entry_sqrt, run.initial_entry_price, price
        )
        il_now = position_value - hodl_now
        step_il = il_now - run.il_prev

        # Equity is the mark-to-market LP value plus uncollected fees; a
        # rebalance pays its costs out of that equity before redeploying.
        run.equity = position_value + fees_value

        step_gas = 0.0
        step_slippage = 0.0
        step_fees_collected = 0.0
        rebalanced = False

        is_decision = run._decision_due(i, last_decision_idx)
        if is_decision:
            gas_wei = float(_as_int(tape.base_fee[i]) + _as_int(tape.priority_fee[i]))
            observation = run._build_observation(
                price,
                sqrt,
                tick,
                position_value,
                fees_value,
                il_now,
                gas_wei,
                decision_index,
                total_decisions_hint,
                str(tape.regime[i] or "unknown"),
            )
            action = policy.act(observation, rng)
            if not action.is_hold():
                (
                    new_equity,
                    gas_cost,
                    slippage_cost,
                    lower,
                    upper,
                ) = run._apply_rebalance(
                    action,
                    tick,
                    sqrt,
                    price,
                    position_value,
                    fees_value,
                    i,
                )
                run.equity = new_equity
                step_gas = -gas_cost
                step_slippage = -slippage_cost
                step_fees_collected = fees_value
                rebalanced = True
                run.costs["step"].append(int(tape.seq[i]))
                run.costs["cost_type"].append("gas")
                run.costs["amount_usdc"].append(-gas_cost)
                run.costs["step"].append(int(tape.seq[i]))
                run.costs["cost_type"].append("slippage")
                run.costs["amount_usdc"].append(-slippage_cost)
                run.decisions["step"].append(int(tape.seq[i]))
                run.decisions["action_type"].append("rebalance")
                run.decisions["center_offset"].append(int(action.center_offset))
                run.decisions["width"].append(int(action.width))
                run.decisions["tick_lower"].append(int(lower))
                run.decisions["tick_upper"].append(int(upper))
                run.decisions["gas_paid"].append(-gas_cost)
                run.decisions["slippage_paid"].append(-slippage_cost)
                run.decisions["fees_collected"].append(step_fees_collected)
                position = run.position
                assert position is not None
                # Mark the redeployed value against the fixed initial basket; the
                # gas/slippage/fee jumps are separate decomposition terms.
                il_now = run.equity - hodl_now
            else:
                run.decisions["step"].append(int(tape.seq[i]))
                run.decisions["action_type"].append("hold")
                run.decisions["center_offset"].append(int(action.center_offset))
                run.decisions["width"].append(int(action.width))
                run.decisions["tick_lower"].append(None)
                run.decisions["tick_upper"].append(None)
                run.decisions["gas_paid"].append(0.0)
                run.decisions["slippage_paid"].append(0.0)
                run.decisions["fees_collected"].append(0.0)
            last_decision_idx = i
            decision_index += 1

        run.il_prev = il_now
        # After a collect the new position starts with zero accrued fees; on a
        # hold the marked fee balance carries into the next step's delta.
        run.fees_value_prev = 0.0 if rebalanced else fees_value
        run.rows["step"].append(int(tape.seq[i]))
        run.rows["equity"].append(run.equity)
        run.rows["pnl"].append(run.equity - run.capital)
        run.rows["fees"].append(step_fees_value)
        run.rows["gas"].append(step_gas)
        run.rows["slippage"].append(step_slippage)
        run.rows["il"].append(step_il)

    # Buy-and-hold benchmark: the initial deposit's token mix held for the window.
    assert run.initial_position is not None
    final_price = run.last_price if run.last_price is not None else run.initial_entry_price
    hodl_final = run.initial_position.hodl_value(
        run.initial_entry_sqrt, run.initial_entry_price, final_price
    )
    hodl_return = hodl_final - run.capital

    equity_curve = pl.DataFrame(run.rows)
    decision_log = pl.DataFrame(run.decisions)
    cost_ledger = pl.DataFrame(run.costs)
    # Guarantee the frozen column shapes even if no action was ever taken.
    decision_log = decision_log.cast(
        {
            "step": pl.Int64,
            "center_offset": pl.Int64,
            "width": pl.Int64,
            "tick_lower": pl.Int64,
            "tick_upper": pl.Int64,
            "gas_paid": pl.Float64,
            "slippage_paid": pl.Float64,
            "fees_collected": pl.Float64,
        }
    )
    cost_ledger = cost_ledger.cast(
        {"step": pl.Int64, "cost_type": pl.String, "amount_usdc": pl.Float64}
    )

    LOGGER.debug(
        "backtest complete: policy=%s split=%s steps=%d decisions=%d net=%s",
        policy.name,
        "train" if market_view.is_train else "eval",
        total_events,
        decision_log.height,
        run.equity - run.capital,
    )

    return BacktestLedger(
        equity_curve=equity_curve,
        decision_log=decision_log,
        cost_ledger=cost_ledger,
        pnl_decomposition=decompose_equity_curve(equity_curve),
        config_hash=config_hash(config),
        git_commit=git_commit(),
        policy_name=policy.name,
        split="train" if market_view.is_train else "eval",
        hodl_return=hodl_return,
        initial_capital=run.capital,
        periods_per_year=periods_per_year,
        n_steps=total_events,
        fee_growth_exact=run.replay.exact,
    )
