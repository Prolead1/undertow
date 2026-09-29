"""Shared test fixtures for undertow.sim."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from undertow.data import (
    EVENT_TAPE_SCHEMA,
    GAS_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    DataConfig,
    EndpointConfig,
    RegimeConfig,
    WindowConfig,
    default_pools,
    tick_to_sqrt_price_x96,
)
from undertow.sim.config import EpisodeConfig
from undertow.sim.core.pool import PoolEngine
from undertow.sim.core.position import Position
from undertow.sim.marketview import MarketView


def pytest_configure(config: pytest.Config) -> None:
    """Register sim-specific markers."""
    config.addinivalue_line("markers", "slow: > 5s (default-on, budgeted)")


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded Generator for deterministic tests."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# S03 append — shared MarketView fixtures (CONTRACTS.md §18).
# S00 owns the seeds/markers above; these are added at the end for S04-S15.
# ---------------------------------------------------------------------------

_TINY_TRAIN_START = datetime(2022, 1, 1, tzinfo=UTC)
_TINY_TRAIN_END = datetime(2022, 2, 1, tzinfo=UTC)
# Tape ends just inside the (half-open) train wall so `build_market_view`
# includes every row; the reference feed still spans up to `_TINY_TRAIN_END`.
_TINY_TAPE_END = _TINY_TRAIN_END - timedelta(seconds=1)
_TINY_EVAL_START = datetime(2024, 1, 1, tzinfo=UTC)
_TINY_EVAL_END = datetime(2024, 12, 31, tzinfo=UTC)
_TINY_SYMBOL = "ETHUSDC"
_TINY_TICK = 196242
_TINY_FEE_GROWTH = str(12345 * 2**64)


def _tiny_linspace(start: datetime, end: datetime, n: int) -> list[datetime]:
    """``n`` evenly spaced UTC datetimes from ``start`` to ``end`` inclusive."""
    span = end - start
    return [start + span * (i / (n - 1)) for i in range(n)]


@pytest.fixture
def tiny_data_config() -> DataConfig:
    """A ``DataConfig`` pointing at a tiny synthetic cache (no real data read)."""
    return DataConfig(
        pool=default_pools()["USDC_WETH_3000"],
        window=WindowConfig(
            start_utc=_TINY_TRAIN_START,
            end_utc=_TINY_TRAIN_END,
        ),
        regime=RegimeConfig(),
        endpoints=EndpointConfig(
            graph_url="https://graph.test",
            rpc_url="https://rpc.test",
            reference_base_url="https://reference.test",
        ),
        cache_dir=Path("data/cache/tiny"),
        output_dir=Path("data/datasets/tiny"),
    )


@pytest.fixture
def tiny_market_view(rng: np.random.Generator) -> MarketView:
    """Shared ``MarketView`` with the train split active (S03, CONTRACTS §18).

    Synthesizes a schema-valid event tape (~1000 rows), reference feed (~100
    bars), gas feed (~100 rows) and regime labels (~100 rows) spanning
    2022-01-01 → 2022-02-01 UTC. It is the shared test data for S04-S15, so its
    values are internally consistent: tape ``price_reference`` is the as-of
    reference close, ``seq`` is dense, and the feeds span the split.
    """
    n_tape = 1000
    n_ref = 100
    n_gas = 100
    n_regime = 100

    pool_address = str(default_pools()["USDC_WETH_3000"].address)
    tape_times = _tiny_linspace(_TINY_TRAIN_START, _TINY_TAPE_END, n_tape)
    ref_open = _tiny_linspace(_TINY_TRAIN_START, _TINY_TRAIN_END, n_ref)
    ref_close = [t + timedelta(minutes=1) for t in ref_open]
    regime_times = _tiny_linspace(_TINY_TRAIN_START, _TINY_TRAIN_END, n_regime)

    # Random-walk pool price; a smoother reference path. Both start near $3000.
    steps = rng.normal(0.0, 0.0005, size=n_tape)
    pool_prices = 3000.0 * np.exp(np.cumsum(steps))
    ref_prices = 3000.0 * np.exp(
        np.linspace(-0.05, 0.05, n_ref) + rng.normal(0.0, 0.002, size=n_ref)
    )
    sqrt_price_x96 = str(tick_to_sqrt_price_x96(_TINY_TICK))

    # As-of reference close for each tape row: last ref close <= block time.
    close_times = np.array([t.timestamp() for t in ref_close])
    tape_close = [t.timestamp() for t in tape_times]
    ref_asof = []
    for ts in tape_close:
        idx = int(np.searchsorted(close_times, ts, side="right")) - 1
        ref_asof.append(float(ref_prices[idx]) if idx >= 0 else None)

    tape_table = pa.table(
        {
            "block_number": list(range(n_tape)),
            "log_index": [0] * n_tape,
            "block_timestamp": tape_times,
            "tx_hash": [f"0x{i:064x}" for i in range(n_tape)],
            "pool_address": [pool_address] * n_tape,
            "event_type": ["swap" if i % 2 == 0 else "mint" for i in range(n_tape)],
            "amount0": ["1000000" if i % 2 == 0 else "2000000" for i in range(n_tape)],
            "amount1": ["-1000000" if i % 2 == 0 else "2000000" for i in range(n_tape)],
            "sqrt_price_x96": [sqrt_price_x96 if i % 2 == 0 else None for i in range(n_tape)],
            "liquidity": [str(2**80) if i % 2 == 0 else None for i in range(n_tape)],
            "tick": [_TINY_TICK if i % 2 == 0 else None for i in range(n_tape)],
            "sender": ["0x" + "aa" * 20] * n_tape,
            "recipient": ["0x" + "bb" * 20 if i % 2 == 0 else None for i in range(n_tape)],
            "owner": [None if i % 2 == 0 else "0x" + "cc" * 20 for i in range(n_tape)],
            "tick_lower": [None if i % 2 == 0 else _TINY_TICK - 120 for i in range(n_tape)],
            "tick_upper": [None if i % 2 == 0 else _TINY_TICK + 120 for i in range(n_tape)],
            "liquidity_amount": [None if i % 2 == 0 else str(10**18) for i in range(n_tape)],
            "seq": list(range(n_tape)),
            "price_pool": [float(p) for p in pool_prices],
            "price_reference": ref_asof,
            "regime": [
                ["bull", "bear", "sideways", "high_vol"][(i // 25) % 4]
                for i in range(n_tape)
            ],
            "base_fee_per_gas": [str(20 * 10**9) for _ in range(n_tape)],
            "priority_fee_p50_wei": [str(10**9) for _ in range(n_tape)],
            "fee_growth_global_0_x128": [_TINY_FEE_GROWTH] * n_tape,
            "fee_growth_global_1_x128": [_TINY_FEE_GROWTH] * n_tape,
            "fee_growth_source": ["exact"] * n_tape,
        },
        schema=EVENT_TAPE_SCHEMA,
    )

    reference_table = pa.table(
        {
            "open_time": ref_open,
            "close_time": ref_close,
            "open": [float(p) for p in ref_prices],
            "high": [float(p) * 1.001 for p in ref_prices],
            "low": [float(p) * 0.999 for p in ref_prices],
            "close": [float(p) for p in ref_prices],
            "volume_base": [100.0] * n_ref,
            "quote_volume": [300_000.0] * n_ref,
            "trades": [1000] * n_ref,
            "symbol": [_TINY_SYMBOL] * n_ref,
            "is_gap_filled": [False] * n_ref,
        },
        schema=REFERENCE_SCHEMA,
    )

    gas_table = pa.table(
        {
            "block_number": list(range(n_gas)),
            "base_fee_per_gas": [str(20 * 10**9)] * n_gas,
            "gas_used": [15_000_000] * n_gas,
            "gas_limit": [30_000_000] * n_gas,
            "priority_fee_p50_wei": [str(10**9)] * n_gas,
            "priority_fee_p90_wei": [str(3 * 10**9)] * n_gas,
        },
        schema=GAS_SCHEMA,
    )

    regime_table = pa.table(
        {
            "timestamp": regime_times,
            "sigma_rv": [0.5 + 0.1 * (i % 4) for i in range(n_regime)],
            "mu": [0.01 * ((i % 4) - 2) for i in range(n_regime)],
            "regime": [
                ["bull", "bear", "sideways", "high_vol"][(i // 25) % 4]
                for i in range(n_regime)
            ],
            "window_complete": [True] * n_regime,
            "symbol": [_TINY_SYMBOL] * n_regime,
        },
        schema=REGIME_SCHEMA,
    )

    return MarketView(
        train=pl.from_arrow(tape_table),
        eval=None,
        reference=pl.from_arrow(reference_table),
        regimes=pl.from_arrow(regime_table),
        gas=pl.from_arrow(gas_table),
        train_start_utc=_TINY_TRAIN_START,
        train_end_utc=_TINY_TRAIN_END,
        eval_start_utc=_TINY_EVAL_START,
        eval_end_utc=_TINY_EVAL_END,
        is_train=True,
        episode_config=EpisodeConfig(),
    )


@pytest.fixture
def entry_position() -> Position:
    """S04: a ±120-tick position built via ``initial_deposit`` at tick 196242.

    ADR-009 orientation: ``entry_sqrt = calc_sqrt_price_a(196242)`` is the raw
    sqrt price (~18244.326) and ``entry_price = tick_to_price(196242, 6, 18)``
    is the human USDC-per-WETH price (~3004.307).  The position's ``value`` at
    that point consumes exactly the 100,000 capital budget, i.e.
    ``entry_position.value(entry_sqrt, entry_price) == 100_000`` to float64
    precision.  Fee-growth snapshots start at zero.
    """
    from undertow.data import tick_to_price
    from undertow.sim.core.position import calc_sqrt_price_a, initial_deposit
    from undertow.sim.types import Tick, TickSpacing

    entry_tick = 196242
    entry_sqrt = calc_sqrt_price_a(Tick(entry_tick))
    entry_price = float(tick_to_price(entry_tick, 6, 18))
    return initial_deposit(
        entry_sqrt,
        entry_price,
        Tick(entry_tick - 120),
        Tick(entry_tick + 120),
        100_000.0,
        TickSpacing(60),
    )


# ---------------------------------------------------------------------------
# S07 append — tiny_pool_engine fixture (CONTRACTS.md §18).
# Kept at the very end so parallel tasks can append after S04/S07 without
# touching each other's fixtures.
# ---------------------------------------------------------------------------

#: Reference tick for ~$3004 USDC/WETH (CONTRACTS §19), and the fee tier /
#: spacing of the WETH/USDC 0.30% pool.  The current tick need not be a
#: multiple of the spacing; position bounds are snapped by the engine.
_TINY_POOL_ENTRY_TICK = 196242
_TINY_POOL_FEE_TIER_BPS = 3000
_TINY_POOL_TICK_SPACING = 60
#: Nearest spacing-grid anchor to the reference tick, used to lay out the
#: background lattice so that entry bounds land on initialized ticks.
_TINY_POOL_GRID_ANCHOR = 196200
#: External active liquidity the agent's positions are marginal against.
_TINY_POOL_BASE_LIQUIDITY = 1.0e15


@pytest.fixture
def tiny_pool_engine(entry_position: Position) -> PoolEngine:
    """S07: a `PoolEngine` at raw tick 196242 with a populated lattice.

    ADR-009 raw orientation: ``sqrt_price = calc_sqrt_price_a(196242)``
    (~18244.3), ``tick = 196242``, raw ``liquidity``.  The lattice has 21
    initialized background ticks from tick 195600 to 196800 (the reference
    tick ±600 on the 60-spacing grid), and one open position whose bounds come
    from ``entry_position`` (snapped to the spacing) carrying its raw ``L``.
    """
    from undertow.sim.core.pool import PoolState, TickState
    from undertow.sim.core.position import calc_sqrt_price_a
    from undertow.sim.types import Tick as _Tick

    entry_tick = _Tick(_TINY_POOL_ENTRY_TICK)
    state = PoolState(
        sqrt_price=calc_sqrt_price_a(entry_tick),
        tick=entry_tick,
        liquidity=_TINY_POOL_BASE_LIQUIDITY,
        fee_growth_global_0=0.0,
        fee_growth_global_1=0.0,
        fee_tier_bps=_TINY_POOL_FEE_TIER_BPS,
        tick_spacing=_TINY_POOL_TICK_SPACING,
    )

    ticks: dict[_Tick, TickState] = {}
    for step in range(-10, 11):
        tick = _Tick(_TINY_POOL_GRID_ANCHOR + step * _TINY_POOL_TICK_SPACING)
        ticks[tick] = TickState(initialized=True)

    engine = PoolEngine(state=state, ticks=ticks)
    engine.open_position(
        entry_position.tick_lower,
        entry_position.tick_upper,
        entry_position.liquidity,
    )
    return engine


# ---------------------------------------------------------------------------
# S08 append — friction fixtures (CONTRACTS.md §8).
# Appended at the very end so parallel tasks can edit this file without
# conflicting with S00/S03/S04 fixtures above.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from undertow.sim.frictions import FlatGasModel, ProportionalSlippageModel


@pytest.fixture
def mock_gas_model() -> FlatGasModel:
    """A ``FlatGasModel`` at $3000 ETH, 30 gwei base fee + 3% tip surcharge.

    Explicit expected USDC costs (hand-computed, ADR-005 base-fee-only
    arithmetic) are asserted in ``tests/sim/test_frictions.py``:

    * ``mint``      -> 460_000 * 30e9 * 1.03 * 3000 / 1e18 == 42.642
    * ``rebalance`` -> 825_000 * 30e9 * 1.03 * 3000 / 1e18 == 76.4775
    * ``hold``      -> 0.0

    The ``priority_fee_gwei=1.0`` argument is retained for the S08 brief's
    constructor but is ignored under ADR-005.
    """
    from undertow.sim.config import GasConfig
    from undertow.sim.frictions import FlatGasModel

    return FlatGasModel(
        GasConfig(),
        eth_usd_price=3000.0,
        base_fee_gwei=30.0,
        priority_fee_gwei=1.0,
    )


@pytest.fixture
def mock_slippage_model() -> ProportionalSlippageModel:
    """A ``ProportionalSlippageModel`` with the pinned default impact.

    Applies ``notional * (fee_pips/1_000_000 + impact_bps/10_000)`` (ADR-012 /
    CONTRACTS §8): with the pinned ``EpisodeConfig.fee_tier_bps=3000`` (Uniswap
    pips = 0.30%) and ``fixed_impact_bps=5`` that is
    ``notional * 0.0035``.
    """
    from undertow.sim.frictions import ProportionalSlippageModel

    return ProportionalSlippageModel()


# ---------------------------------------------------------------------------
# S10 append — shared baseline-policy fixture (CONTRACTS.md §18).
# S00 owns the seeds/markers above; S03 the MarketView/position fixtures; this
# is added at the very end so the other tasks' append points stay intact.
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_policy() -> object:
    """S10: an always-hold :class:`~undertow.sim.policies.HODLPolicy`.

    The fixture is reused by S11-S15 as a deterministic stand-in policy.  It is
    returned as ``object`` only because the policy import is done lazily here to
    keep the shared conftest's import block untouched.
    """
    from undertow.sim.policies import HODLPolicy

    return HODLPolicy()


# ---------------------------------------------------------------------------
# S11 append — shared BacktestLedger fixture (CONTRACTS.md §18).
# Appended at the very end so S12 can append its own `tiny_env` fixture
# without touching this block.  Imports stay lazy, matching the S10 style.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from undertow.sim.backtest import BacktestLedger


@pytest.fixture
def tiny_ledger(tiny_market_view: MarketView) -> BacktestLedger:
    """S11: a ``BacktestLedger`` from a 100-step replay of ``dummy_policy``.

    Deterministically built from the shared ``tiny_market_view`` (sliced to the
    first 100 tape events) and a default ``SimConfig``; reused by S14-S15.
    """
    from undertow.sim.backtest import run_backtest
    from undertow.sim.config import SimConfig
    from undertow.sim.policies import HODLPolicy

    window = tiny_market_view.slice(0, 100)
    return run_backtest(HODLPolicy(), window, SimConfig())
