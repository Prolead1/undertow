"""Shared synthetic end-to-end dataset factory for the ``undertow.data`` tests (T11).

``tiny_dataset()`` (and its backing ``build_tiny_dataset()``) hand-build a small,
fully-synthetic dataset spanning all seven streams plus regime labels, shaped to
exercise the awkward alignment cases the event tape must handle. It is a **shared
deliverable** (``CONTRACTS.md`` §10, ``tests/data/conftest.py`` — owned by T11, used
by T13/T14/T16): other tasks build their backtester/validator tests on it, so it is
designed as an API, not a test detail.

What it covers (the awkward cases — keep this list in sync with the data):

- **Pre-history window** — the first five blocks precede the first reference bar's
  ``close_time`` and the first regime label, so tape rows there must carry null
  ``price_reference`` and ``regime = "unknown"`` and still be *retained* (test 11).
- **A swap in each direction** — buys (``amount0 > 0``, pool tick falls) and sells
  (``amount0 < 0``, pool tick rises) (test 13).
- **A tick crossing** — the pool tick moves up *and* down across the 60-tick spacing
  grid (196140 → … → 196440 → 196140), so grid boundaries are crossed both ways.
- **An out-of-range position** — position P3 is minted at ``[196300, 196360]`` but the
  pool tick later rises to 196380/196440 (``>= tick_upper``), i.e. price moved below
  the range; T13's per-position logic must handle the "no interior growth" consequence.
- **A mint and its matching burn+collect** — P1 (owner A) and P4 (owner D) are minted
  then burned and cleaned up in the same block, so reconciliation has a real Burn to
  separate principal from fee.
- **A subgraph-style null ``recipient``** — the collect for P2 has ``recipient = None``
  (the hosted subgraph's ``Collect`` entity has no recipient; ``RPC-only``).
- **A block with multiple events** — block 30 (swap + mint), block 34 (burn + collect),
  block 36 (burn + collect) exercise ``(block_number, log_index)`` ordering.
- **A gas spike** — block 37 has a 20x ``base_fee_per_gas`` / priority-fee spike, the
  "get included during a spike" cost a backtester must not average away.
- **A forward-filled reference gap** — bars marked ``is_gap_filled = True`` (forward
  filled over an exchange outage, per CONTRACTS §4.6); their (still-valid) close is
  what the as-of join must land on.
- **An incomplete regime window** — the first regime rows have ``regime = "unknown"``
  and ``window_complete = False``; a complete-but-not-full window must never be
  silently dropped or back-filled.
- **Exact vs stale fee-growth snapshots** — snapshots exist at blocks 5, 12, 35, 60;
  a tape row at a snapshot block is ``"exact"``, between snapshots ``"stale_prior"``,
  and before block 5 there is no snapshot at all.

All values are synthetic (no real on-chain payloads exist in the shared fixture), but
they are **internally consistent**: swap ``sqrt_price_x96`` values are derived from the
real ``fixedpoint.tick_to_sqrt_price_x96`` and reference closes from ``tick_to_price``,
so ``price_pool`` agrees with ``price_reference`` to well within the 5% tolerance T11
test 9 pins. Ticks use the 0.30%-pool spacing grid (60). Blocks are 12 s apart
(Ethereum ballpark), reference bars 60 s, starting 2023-05-31 23:58:00 UTC.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from undertow.data.config import GLOBAL_TICK_SENTINEL, PoolConfig, default_pools
from undertow.data.fixedpoint import tick_to_price, tick_to_sqrt_price_x96
from undertow.data.schemas import (
    BURN_SCHEMA,
    COLLECT_SCHEMA,
    FEE_GROWTH_SCHEMA,
    GAS_SCHEMA,
    MINT_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    SWAP_SCHEMA,
)
from undertow.data.types import Regime

POOL: PoolConfig = default_pools()["USDC_WETH_3000"]

# Convenient, valid (lowercase 0x-prefixed, 42-char) addresses for owners/senders.
_A = "0x" + "a" * 40
_B = "0x" + "b" * 40
_C = "0x" + "c" * 40
_D = "0x" + "d" * 40
_E = "0x" + "e" * 40
_F = "0x" + "f" * 40
_G = "0x" + "g" * 40
_H = "0x" + "h" * 40

_BASE_UTC = datetime(2023, 5, 31, 23, 58, 0, tzinfo=UTC)
_BLOCK_SECONDS = 12  # s/block, ~Ethereum
_BAR_SECONDS = 60  # reference/regime bars are minute closes
_FIRST_BAR_CLOSE = _BASE_UTC + timedelta(seconds=_BAR_SECONDS)  # 23:59:00 UTC
_BASE_BLOCK = 17_000_000


@dataclass(frozen=True, slots=True)
class TinyDataset:
    """Everything ``build_event_tape`` needs, plus the pool, in one object.

    ``streams`` is the ``Mapping[str, pa.Table]`` keyed by the four log-stream names;
    ``gas`` / ``reference`` / ``regime`` / ``fee_growth`` are the non-log streams. The
    ``physical_constants`` mapping records the block-timespan and the Q128 globals so a
    test can make numeric assertions without re-deriving them (test 13).
    """

    pool: PoolConfig
    streams: Mapping[str, pa.Table]
    gas: pa.Table
    reference: pa.Table
    regime: pa.Table
    fee_growth: pa.Table
    # Derived facts a test may want to assert against (row counts, the spike block,
    # the snapshot blocks, the pre-history count). Kept out of the tables themselves.
    physical_constants: Mapping[str, object] = field(default_factory=dict)


def _s(value: object) -> str:
    """Big-integer fields are decimal strings in parquet; stringify without coercion."""
    return str(value)


def _bn(index: int) -> int:
    """Block number for a block index (12 s spacing from the base block)."""
    return _BASE_BLOCK + index


def _ts(index: int) -> datetime:
    """Block timestamp for a block index."""
    return _BASE_UTC + timedelta(seconds=_BLOCK_SECONDS * index)


def _tx(index: int, log: int, kind: str) -> str:
    """Deterministic pseudo-tx hash for a (block, log)."""
    return f"0x{kind}{index:04d}{log:02d}" + "0" * (40 - len(f"{kind}{index:04d}{log:02d}"))


def _table(schema: pa.Schema, rows: list[dict[str, object]]) -> pa.Table:
    """Build a schema-valid table from a list of partial row dicts (missing keys -> null)."""
    arrays = {
        f.name: pa.array([r.get(f.name) for r in rows], type=f.type) for f in schema
    }
    return pa.table(arrays, schema=schema)


def _swap_row(
    index: int,
    log: int,
    tick: int,
    amount0: int,
    amount1: int,
    *,
    sender: str = _E,
    recipient: str = _F,
) -> dict[str, object]:
    return {
        "block_number": _bn(index),
        "log_index": log,
        "block_timestamp": _ts(index),
        "tx_hash": _tx(index, log, "s"),
        "pool_address": POOL.address,
        "event_type": "swap",
        "amount0": _s(amount0),
        "amount1": _s(amount1),
        "sqrt_price_x96": _s(tick_to_sqrt_price_x96(tick)),
        "liquidity": _s(2_000_000_000_000),
        "tick": tick,
        "sender": sender,
        "recipient": recipient,
    }


def _mint_row(
    index: int,
    log: int,
    owner: str,
    tick_lower: int,
    tick_upper: int,
    amount0: int,
    amount1: int,
    liquidity: int,
) -> dict[str, object]:
    return {
        "block_number": _bn(index),
        "log_index": log,
        "block_timestamp": _ts(index),
        "tx_hash": _tx(index, log, "m"),
        "pool_address": POOL.address,
        "event_type": "mint",
        "owner": owner,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity_amount": _s(liquidity),
        "amount0": _s(amount0),
        "amount1": _s(amount1),
        "sender": _B,
    }


def _burn_row(
    index: int,
    log: int,
    owner: str,
    tick_lower: int,
    tick_upper: int,
    amount0: int,
    amount1: int,
    liquidity: int,
) -> dict[str, object]:
    return {
        "block_number": _bn(index),
        "log_index": log,
        "block_timestamp": _ts(index),
        "tx_hash": _tx(index, log, "b"),
        "pool_address": POOL.address,
        "event_type": "burn",
        "owner": owner,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity_amount": _s(liquidity),
        "amount0": _s(amount0),
        "amount1": _s(amount1),
        "sender": None,  # Burn has no sender (CONTRACTS §4.2)
    }


def _collect_row(
    index: int,
    log: int,
    owner: str,
    tick_lower: int,
    tick_upper: int,
    amount0: int,
    amount1: int,
    *,
    recipient: str | None = _C,
) -> dict[str, object]:
    return {
        "block_number": _bn(index),
        "log_index": log,
        "block_timestamp": _ts(index),
        "tx_hash": _tx(index, log, "c"),
        "pool_address": POOL.address,
        "event_type": "collect",
        "owner": owner,
        "recipient": recipient,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "amount0": _s(amount0),
        "amount1": _s(amount1),
    }


# The pool-tick path through the window: a swap at `index` sets the active tick for
# that and the following blocks (raw USDC/WETH 0.30% pool spacing 60).
_SWAP_PATH: list[tuple[int, int]] = [
    (1, 196140),  # buy, tick down across the grid
    (6, 196200),
    (10, 196260),
    (14, 196320),
    (17, 196260),  # sell, tick up
    (21, 196200),
    (25, 196140),  # tick down again
    (29, 196200),
    (30, 196260),
    (33, 196320),
    (35, 196380),  # crosses P3's upper bound (196360): P3 out of range
    (38, 196440),  # deepest upward leg
    (42, 196200),  # buy pressure, tick falls
    (46, 196320),
    (50, 196260),
    (54, 196140),
    (58, 196200),
    (62, 196140),
]

def build_tiny_dataset() -> TinyDataset:
    """Build the shared synthetic dataset (see the module docstring for coverage)."""
    n_blocks = 63  # blocks 0..62 -> ~12.6 minutes; same span as _SWAP_PATH

    # --- active tick per block, from the swap path ---
    active_tick = [196200] * n_blocks
    for index, tick in _SWAP_PATH:
        active_tick[index] = tick

    # --- swaps ---
    # Buy: token0 (USDC) in, token1 (WETH) out, tick falls. Sell: the reverse.
    swap_rows = [
        _swap_row(1, 0, 196140, +3_050_000_000, -700_000_000_000_000_000),
        _swap_row(6, 0, 196200, -2_900_000_000, +600_000_000_000_000_000),
        _swap_row(10, 0, 196260, -2_800_000_000, +580_000_000_000_000_000),
        _swap_row(14, 0, 196320, -2_700_000_000, +560_000_000_000_000_000),
        _swap_row(17, 0, 196260, +3_100_000_000, -640_000_000_000_000_000),
        _swap_row(21, 0, 196200, +3_000_000_000, -620_000_000_000_000_000),
        _swap_row(25, 0, 196140, +3_200_000_000, -660_000_000_000_000_000),
        _swap_row(29, 0, 196200, +3_000_000_000, -620_000_000_000_000_000),
        _swap_row(30, 0, 196260, -2_900_000_000, +600_000_000_000_000_000),
        _swap_row(33, 0, 196320, -2_800_000_000, +570_000_000_000_000_000),
        _swap_row(35, 2, 196380, -2_600_000_000, +540_000_000_000_000_000),
        _swap_row(38, 0, 196440, -2_500_000_000, +520_000_000_000_000_000),
        _swap_row(42, 0, 196200, +3_400_000_000, -700_000_000_000_000_000),
        _swap_row(46, 0, 196320, -2_700_000_000, +560_000_000_000_000_000),
        _swap_row(50, 0, 196260, +3_100_000_000, -640_000_000_000_000_000),
        _swap_row(54, 0, 196140, +3_300_000_000, -670_000_000_000_000_000),
        _swap_row(58, 0, 196200, +3_000_000_000, -620_000_000_000_000_000),
        _swap_row(62, 0, 196140, +3_200_000_000, -650_000_000_000_000_000),
    ]

    # --- mints / burns / collects (with matching burn+collect lifecycles) ---
    mint_rows = [
        # P1 owner A [196200, 196260] — in range at mint (tick 196200 == lower).
        _mint_row(
            3,
            0,
            _A,
            196200,
            196260,
            2_000_000_000_000,
            500_000_000_000_000_000,
            1_000_000_000_000),
        _mint_row(
            8,
            0,
            _B,
            196140,
            196260,
            1_500_000_000_000,
            400_000_000_000_000_000,
            800_000_000_000),
        _mint_row(
            12,
            0,
            _D,
            196320,
            196380,
            1_800_000_000_000,
            420_000_000_000_000_000,
            900_000_000_000),
        # P3 owner C [196300, 196360] — THE out-of-range position (tick later tops 196360).
        _mint_row(
            20,
            0,
            _C,
            196300,
            196360,
            2_200_000_000_000,
            520_000_000_000_000_000,
            1_100_000_000_000),
        # Same block as the swap at 30: exercises (block_number, log_index) ordering.
        _mint_row(
            30,
            1,
            _G,
            196200,
            196260,
            1_900_000_000_000,
            460_000_000_000_000_000,
            950_000_000_000),
        _mint_row(
            40,
            0,
            _H,
            196200,
            196260,
            2_100_000_000_000,
            500_000_000_000_000_000,
            1_050_000_000_000),
        _mint_row(
            48,
            0,
            _F,
            196260,
            196320,
            1_700_000_000_000,
            400_000_000_000_000_000,
            850_000_000_000),
    ]
    burn_rows = [
        # P1: burn (principal) followed by collect (fees) in the SAME block (34).
        _burn_row(
            34,
            1,
            _A,
            196200,
            196260,
            2_000_000_000_000,
            500_000_000_000_000_000,
            1_000_000_000_000),
        _burn_row(
            36,
            0,
            _D,
            196320,
            196380,
            1_800_000_000_000,
            420_000_000_000_000_000,
            900_000_000_000),
    ]
    collect_rows = [
        _collect_row(34, 2, _A, 196200, 196260, 12_000_000_000, 3_000_000_000_000_000),
        _collect_row(36, 1, _D, 196320, 196380, 15_000_000_000, 4_000_000_000_000_000),
        # Subgraph-style: recipient is NULL on the RPC-fetched... rather here the
        # subgraph-shaped row (Collect has no recipient on that route).
        _collect_row(
            44, 0, _B, 196140, 196260, 70_000_000_000,
            18_000_000_000_000_000,
            recipient=None,
        ),
    ]

    swap = _table(SWAP_SCHEMA, swap_rows)
    mint = _table(MINT_SCHEMA, mint_rows)
    burn = _table(BURN_SCHEMA, burn_rows)
    collect = _table(COLLECT_SCHEMA, collect_rows)

    # --- reference bars: one per 60 s close, close = pool price at that minute ---
    n_bars = ((n_blocks - 1) * _BLOCK_SECONDS) // _BAR_SECONDS + 2  # covers the span
    reference_rows: list[dict[str, object]] = []
    previous_close: float | None = None
    for m in range(n_bars):
        close_time = _FIRST_BAR_CLOSE + timedelta(seconds=_BAR_SECONDS * m)
        # block index nearest the start of this bar (for the pool tick -> price)
        probe = int(max(0, (m * _BAR_SECONDS) // _BLOCK_SECONDS))
        probe = min(probe, n_blocks - 1)
        close = float(tick_to_price(active_tick[probe], POOL.token0_decimals, POOL.token1_decimals))
        effective_close = previous_close if m >= 2 and m % 5 == 0 else close  # gap-filled bars
        reference_rows.append(
            {
                "open_time": close_time - timedelta(seconds=_BAR_SECONDS),
                "close_time": close_time,
                "open": previous_close if previous_close is not None else close,
                "high": max(
                    effective_close if previous_close is not None else close, effective_close
                ),
                "low": min(
                    effective_close if previous_close is not None else close, effective_close
                ),
                "close": effective_close,
                "volume_base": 100.0 if effective_close == close else 0.0,  # filled bars, no volume
                "quote_volume": effective_close * 100.0,
                "trades": 10,
                "symbol": "ETHUSDT",
                "is_gap_filled": m >= 2 and m % 5 == 0,  # forward-filled outage gap
            }
        )
        previous_close = effective_close
    reference = _table(REFERENCE_SCHEMA, reference_rows)

    # --- regime labels: an incomplete warmup, then real labels ---
    regime_rows: list[dict[str, object]] = []
    for m in range(min(n_bars, 14)):
        ts = _FIRST_BAR_CLOSE + timedelta(seconds=_BAR_SECONDS * m)
        if m < 3:  # window not yet full
            regime_rows.append(
                {
                    "timestamp": ts,
                    "sigma_rv": 0.0,  # non-nullable column; window not full -> 0/unused
                    "mu": 0.0,
                    "regime": Regime.UNKNOWN.value,
                    "window_complete": False,
                    "symbol": "ETHUSDT",
                }
            )
            continue
        regime = ("bull", "sideways", "bear", "high_vol")[m % 4]
        regime_rows.append(
            {
                "timestamp": ts,
                "sigma_rv": 0.5 if regime != "high_vol" else 1.2,
                "mu": 0.08 if regime == "bull" else (-0.09 if regime == "bear" else 0.01),
                "regime": regime,
                "window_complete": True,
                "symbol": "ETHUSDT",
            }
        )
    regime = _table(REGIME_SCHEMA, regime_rows)

    # --- gas: one row per block; block 37 is a 20x base-fee / priority-fee spike ---
    gas_rows = []
    for index in range(n_blocks):
        spike = index == 37
        gas_rows.append(
            {
                "block_number": _bn(index),
                "block_timestamp": _ts(index),
                "base_fee_per_gas": _s(600_000_000_000 if spike else 30_000_000_000),
                "gas_used": 12_000_000 if not spike else 29_000_000,
                "gas_limit": 30_000_000,
                "priority_fee_p50_wei": _s(40_000_000_000 if spike else 2_000_000_000),
                "priority_fee_p90_wei": _s(120_000_000_000 if spike else 14_000_000_000),
                "eth_usd_price": None,
            }
        )
    gas = _table(GAS_SCHEMA, gas_rows)

    # --- fee-growth snapshots at blocks 5, 12, 35, 60 (global rows) ---
    snapshot_blocks = [5, 12, 35, 60]
    fg_rows = []
    for j, block_index in enumerate(snapshot_blocks):
        fg_rows.append(
            {
                "block_number": _bn(block_index),
                "tick": GLOBAL_TICK_SENTINEL,
                "fee_growth_outside_0_x128": None,
                "fee_growth_outside_1_x128": None,
                "liquidity_gross": _s(50_000_000_000_000),
                "liquidity_net": _s(-5_000_000_000_000),
                "initialized": True,
                "fee_growth_global_0_x128": _s(2 ** (120 + j)),
                "fee_growth_global_1_x128": _s(2 ** (121 + j)),
                "current_tick": active_tick[block_index],
                "current_liquidity": _s(1_000_000_000_000),
                "source": "rpc_call",
                "pool_address": POOL.address,
            }
        )
    fee_growth = _table(FEE_GROWTH_SCHEMA, fg_rows)

    return TinyDataset(
        pool=POOL,
        streams={"swap": swap, "mint": mint, "burn": burn, "collect": collect},
        gas=gas,
        reference=reference,
        regime=regime,
        fee_growth=fee_growth,
        physical_constants={
            "n_blocks": n_blocks,
            "n_swaps": len(swap_rows),
            "n_mints": len(mint_rows),
            "n_burns": len(burn_rows),
            "n_collects": len(collect_rows),
            "n_events": len(swap_rows) + len(mint_rows) + len(burn_rows) + len(collect_rows),
            "spike_block": 37,
            "spike_block_number": _bn(37),
            "snapshot_blocks": [_bn(b) for b in snapshot_blocks],
            "prehistory_blocks": 5,  # blocks 0..4 have no reference bar / regime label yet
        },
    )


@pytest.fixture(scope="session")
def tiny_dataset() -> TinyDataset:
    """Session-scoped pytest wrapper around :func:`build_tiny_dataset` (immutable)."""
    return build_tiny_dataset()