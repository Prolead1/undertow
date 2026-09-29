"""Shared test fixtures for undertow.sim."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
