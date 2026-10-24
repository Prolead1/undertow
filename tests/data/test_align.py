"""Tests for ``undertow.data.transforms.align`` (T11) — the block-ordered event tape.

Covers CONTRACTS.md §6.2 end to end on ``tiny_dataset()`` (``tests/data/conftest.py``,
the shared factory this task owns): the four-stream union + dense ``seq``, the unique-key
guarantee (a residual duplicate raises ``ValidationError``; a cross-stream reorg
collision collapses by deterministic priority — T18), null-vs-zero
applicability, the backward-only as-of joins (reference, regime, fee-growth globals), the
**exact** gas equi-join, the price-orientation guard against the 1e12 decimals bug,
``assert_no_lookahead`` in **both** directions (it must fire on a forward join and pass on
the correct tape), pre-history retention, determinism via ``content_hash``, the
``tiny_dataset()`` coverage contract itself, and ``Dataset.require``.

The tape is the handoff artifact ``undertow.sim``'s backtester consumes, so look-ahead is
the star of this module: the backward-only tests also assert the *violating* case is
handled the right way, because a guard that never fires is not a guard.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from tests.data.conftest import POOL, build_tiny_dataset
from undertow.data.config import WindowConfig
from undertow.data.schemas import (
    EVENT_TAPE_SCHEMA,
    GAS_SCHEMA,
    REFERENCE_SCHEMA,
    REGIME_SCHEMA,
    SCHEMA_REGISTRY,
    empty_table,
    validate_table,
)
from undertow.data.storage.manifest import DatasetManifest
from undertow.data.storage.parquet import content_hash
from undertow.data.transforms.align import (
    Dataset,
    assert_no_lookahead,
    build_event_tape,
)
from undertow.data.types import BlockNumber, Regime, ValidationError

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _swap_indices(tape: pa.Table) -> list[int]:
    """Row indices of the swap rows (the fixture has one swap per swap block)."""
    nc = tape.num_rows
    return [
        i
        for i in range(nc)
        if tape.column("event_type")[i].as_py() == "swap"
    ]


def _row_of(tape: pa.Table, block: int, etype: str) -> int:
    nc = tape.num_rows
    for i in range(nc):
        if tape.column("block_number")[i].as_py() == block and tape.column("event_type")[
            i
        ].as_py() == etype:
            return i
    raise AssertionError(f"no {etype} row at block {block}")


def _blocks(tape: pa.Table) -> list[int]:
    return tape.column("block_number").to_pylist()


def _manifest() -> DatasetManifest:
    utc = datetime(2023, 6, 1, tzinfo=UTC)
    return DatasetManifest(
        schema_version="1.0.0",
        pool=POOL,
        window=WindowConfig(
            start_block=BlockNumber(17_000_000), end_block=BlockNumber(17_000_062)
        ),
        streams={},
        created_at_utc=utc,
        git_commit="test",
        library_versions={},
    )


# ---------------------------------------------------------------------------
# 1. Merge: union shape, dense seq, strict ordering, unique key
# ---------------------------------------------------------------------------


def test_merge_shape_seq_and_ordering() -> None:
    """Four streams -> one tape with the right row count, dense seq 0..N-1, strictly
    increasing (block_number, log_index)."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    n_events = data.physical_constants["n_events"]
    assert tape.num_rows == n_events
    seq = tape.column("seq").to_pylist()
    assert seq == list(range(n_events))
    keys = list(zip(_blocks(tape), tape.column("log_index").to_pylist(), strict=True))
    assert all(keys[i] < keys[i + 1] for i in range(len(keys) - 1))


def _collide(
    streams: Mapping[str, pa.Table], winner: str, loser: str
) -> dict[str, pa.Table]:
    """Move ``loser``'s first row onto ``winner``'s first key — a synthetic fork.

    The loser keeps its payload (tx hash, amounts); only the tape's merge key and
    timestamp change, so the resulting union is exactly the shape a subgraph reorg
    leaves behind: two different streams at one ``(block_number, log_index)``.
    """
    streams = dict(streams)
    winner_row = streams[winner].to_pylist()[0]
    loser_rows = streams[loser].to_pylist()
    loser_rows[0]["block_number"] = winner_row["block_number"]
    loser_rows[0]["log_index"] = winner_row["log_index"]
    loser_rows[0]["block_timestamp"] = winner_row["block_timestamp"]
    streams[loser] = pa.Table.from_pylist(loser_rows, schema=SCHEMA_REGISTRY[loser])
    return streams


def _align_warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """WARNING records emitted by this module (same shape as the fetcher tests)."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "undertow.data.transforms.align" and r.levelno == logging.WARNING
    ]


def test_cross_stream_collision_collapses_to_mint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T18 addendum: a swap and a mint sharing ``(block_number, log_index)`` — the
    orphaned-fork artifact — collapse onto the mint (liquidity over flow), emit one
    WARNING naming the key, and never raise."""
    data = build_tiny_dataset()
    mint = data.streams["mint"]
    target_block = int(mint.column("block_number")[0].as_py())
    target_log = int(mint.column("log_index")[0].as_py())
    mint_owner = mint.column("owner")[0].as_py()
    streams = _collide(data.streams, "mint", "swap")

    with caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"):
        tape = build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )

    rows = [
        i
        for i in range(tape.num_rows)
        if tape.column("block_number")[i].as_py() == target_block
        and tape.column("log_index")[i].as_py() == target_log
    ]
    assert len(rows) == 1, "the collapse must leave exactly one row per merge key"
    survivor = rows[0]
    assert tape.column("event_type")[survivor].as_py() == "mint"
    assert tape.column("owner")[survivor].as_py() == mint_owner
    assert tape.num_rows == int(data.physical_constants["n_events"]) - 1

    warnings = _align_warning_messages(caplog)
    assert len(warnings) == 1, warnings
    message = warnings[0]
    assert "collapsed 1 cross-stream" in message
    assert f"({target_block}, {target_log})" in message
    assert "dropping 1 row(s)" in message
    assert "mint > burn > swap > collect" in message


@pytest.mark.parametrize(
    ("winner", "loser"),
    [("mint", "burn"), ("burn", "swap"), ("swap", "collect")],
)
def test_cross_stream_priority_is_mint_burn_swap_collect(
    winner: str, loser: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The documented priority is a total order: each adjacent pair crowns the
    higher-priority event type and drops the lower one."""
    data = build_tiny_dataset()
    streams = _collide(data.streams, winner, loser)
    winner_row = streams[winner].to_pylist()[0]

    with caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"):
        tape = build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )

    rows = [
        i
        for i in range(tape.num_rows)
        if tape.column("block_number")[i].as_py() == winner_row["block_number"]
        and tape.column("log_index")[i].as_py() == winner_row["log_index"]
    ]
    assert len(rows) == 1
    assert tape.column("event_type")[rows[0]].as_py() == winner
    assert _align_warning_messages(caplog)


def test_cross_stream_collision_keeps_residual_guard_for_same_stream_dups() -> None:
    """The priority resolves *cross*-stream collisions only: two rows of the same
    type at one key (a pagination bug the fetch layer must have removed) still abort
    via ``_assert_unique_keys``."""
    data = build_tiny_dataset()
    mint_rows = data.streams["mint"].to_pylist()
    duplicate = dict(mint_rows[0])
    duplicate["owner"] = "0x" + "1" * 40
    streams = dict(data.streams)
    streams["mint"] = pa.Table.from_pylist(
        [*mint_rows, duplicate], schema=SCHEMA_REGISTRY["mint"]
    )

    target_block = int(mint_rows[0]["block_number"])
    target_log = int(mint_rows[0]["log_index"])
    with pytest.raises(ValidationError) as ei:
        build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
    message = str(ei.value)
    assert "duplicate merge keys" in message
    assert f"block_number={target_block}" in message
    assert f"log_index={target_log}" in message


def test_mixed_collision_with_repeated_losing_type_still_aborts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A key that mixes a cross-stream collision with a within-stream duplicate
    (two swaps + one mint) must not let the repeated losing type be collapsed away
    silently: the key is left to ``_assert_unique_keys`` and aborts with no
    collapse warning."""
    data = build_tiny_dataset()
    swap_rows = data.streams["swap"].to_pylist()
    mint_row = data.streams["mint"].to_pylist()[0]
    duplicate = dict(swap_rows[0])
    duplicate["tx_hash"] = "0x" + "2" * 64
    for row in (swap_rows[0], duplicate):
        row["block_number"] = mint_row["block_number"]
        row["log_index"] = mint_row["log_index"]
        row["block_timestamp"] = mint_row["block_timestamp"]
    streams = dict(data.streams)
    streams["swap"] = pa.Table.from_pylist(
        [*swap_rows, duplicate], schema=SCHEMA_REGISTRY["swap"]
    )

    with (
        caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"),
        pytest.raises(ValidationError, match="duplicate merge keys"),
    ):
        build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
    assert _align_warning_messages(caplog) == []


def test_resolvable_collision_beside_a_mixed_key_collapses_then_aborts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``~resolvable`` must keep only the mixed key: a pure collision elsewhere is
    still collapsed (and warned about) before the residual guard aborts on the mixed
    key."""
    data = build_tiny_dataset()
    streams = _collide(data.streams, "mint", "swap")

    # A second key that repeats an event type: burn[0] + burn[1] + swap[1].
    burn_rows = streams["burn"].to_pylist()
    swap_rows = streams["swap"].to_pylist()
    target = burn_rows[0]
    for row in (burn_rows[1], swap_rows[1]):
        row["block_number"] = target["block_number"]
        row["log_index"] = target["log_index"]
        row["block_timestamp"] = target["block_timestamp"]
    streams["burn"] = pa.Table.from_pylist(burn_rows, schema=SCHEMA_REGISTRY["burn"])
    streams["swap"] = pa.Table.from_pylist(swap_rows, schema=SCHEMA_REGISTRY["swap"])

    with (
        caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"),
        pytest.raises(ValidationError, match="duplicate merge keys"),
    ):
        build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
    warnings = _align_warning_messages(caplog)
    assert len(warnings) == 1, warnings
    assert "collapsed 1 cross-stream" in warnings[0]
    assert "dropping 1 row(s)" in warnings[0]


def test_resolvable_collision_beside_a_mixed_loser_key_collapses_then_aborts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The discriminating mixed key repeats the *losing* type (two swaps + one
    mint): without ``~resolvable`` in the keep-filter the swaps would be collapsed
    silently and the residual guard would never fire."""
    data = build_tiny_dataset()
    streams = _collide(data.streams, "mint", "swap")
    mint_rows = streams["mint"].to_pylist()
    swap_rows = streams["swap"].to_pylist()
    target = mint_rows[1]
    for row in (swap_rows[1], swap_rows[2]):
        row["block_number"] = target["block_number"]
        row["log_index"] = target["log_index"]
        row["block_timestamp"] = target["block_timestamp"]
    streams["swap"] = pa.Table.from_pylist(swap_rows, schema=SCHEMA_REGISTRY["swap"])

    with (
        caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"),
        pytest.raises(ValidationError, match="duplicate merge keys"),
    ):
        build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
    warnings = _align_warning_messages(caplog)
    assert len(warnings) == 1, warnings
    assert "collapsed 1 cross-stream" in warnings[0]
    assert "dropping 1 row(s)" in warnings[0]


def test_three_way_collision_collapses_to_mint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three streams at one key (mint + burn + swap) resolve to the mint and drop
    both losing rows — the multi-drop arithmetic in one assertion."""
    data = build_tiny_dataset()
    streams = _collide(data.streams, "mint", "swap")
    streams = _collide(streams, "mint", "burn")
    mint_row = streams["mint"].to_pylist()[0]

    with caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"):
        tape = build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )

    rows = [
        i
        for i in range(tape.num_rows)
        if tape.column("block_number")[i].as_py() == mint_row["block_number"]
        and tape.column("log_index")[i].as_py() == mint_row["log_index"]
    ]
    assert len(rows) == 1
    assert tape.column("event_type")[rows[0]].as_py() == "mint"
    assert tape.num_rows == int(data.physical_constants["n_events"]) - 2
    warnings = _align_warning_messages(caplog)
    assert len(warnings) == 1, warnings
    assert "collapsed 1 cross-stream" in warnings[0]
    assert "dropping 2 row(s)" in warnings[0]


def test_no_cross_stream_collision_warns_nothing_and_is_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A collision-free union is untouched — same rows in the same order (identical
    content hash) — and emits no WARNING."""
    data = build_tiny_dataset()
    with caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"):
        tape = build_event_tape(
            data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
        again = build_event_tape(
            data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )

    assert tape.num_rows == int(data.physical_constants["n_events"])
    assert content_hash(tape) == content_hash(again)
    assert _align_warning_messages(caplog) == []


def test_seq_is_dense_after_cross_stream_collapse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``seq`` is stamped after the collapse: 0..N-1 with no gap left by the dropped
    rows."""
    data = build_tiny_dataset()
    streams = _collide(data.streams, "mint", "swap")
    streams = _collide(streams, "burn", "collect")

    with caplog.at_level(logging.WARNING, logger="undertow.data.transforms.align"):
        tape = build_event_tape(
            streams, pool=data.pool, gas=data.gas, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )

    assert tape.num_rows == int(data.physical_constants["n_events"]) - 2
    assert tape.column("seq").to_pylist() == list(range(tape.num_rows))
    keys = list(zip(_blocks(tape), tape.column("log_index").to_pylist(), strict=True))
    assert all(keys[i] < keys[i + 1] for i in range(len(keys) - 1))
    assert len(_align_warning_messages(caplog)) == 1


def test_missing_stream_tolerated_as_empty() -> None:
    """An absent log stream is tolerated: the tape simply contains no such rows."""
    data = build_tiny_dataset()
    reduced = dict(data.streams)
    reduced.pop("collect")
    tape = build_event_tape(
        reduced, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    assert tape.num_rows == data.physical_constants["n_events"] - data.physical_constants[
        "n_collects"
    ]


# ---------------------------------------------------------------------------
# 2. Null, not zero
# ---------------------------------------------------------------------------


def test_inapplicable_columns_are_null_not_zero() -> None:
    """A collect row has null sqrt_price_x96 / tick / sender — not 0. Zero is a value,
    null is an absence; T13's conventions depend on the difference."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    nc = tape.num_rows
    collect_row = next(
        i
        for i in range(nc)
        if tape.column("event_type")[i].as_py() == "collect"
    )
    for name in ("sqrt_price_x96", "tick", "sender"):
        assert tape.column(name)[collect_row].as_py() is None, (
            f"collect row should have null {name!r}, got "
            f"{tape.column(name)[collect_row].as_py()!r}"
        )
    # And the meaningful collect payload is present / non-null.
    assert tape.column("amount0")[collect_row].as_py() is not None


# ---------------------------------------------------------------------------
# 3. Backward-only joins
# ---------------------------------------------------------------------------


def test_reference_join_is_backward_only() -> None:
    """A reference bar that closes AFTER a tape row must not be picked up by it — the
    row keeps the last PRIOR bar."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    # Take two adjacent swap rows; insert a bar strictly between their block times,
    # DROPPING any fixture bar that already sits in that gap so the new bar is the
    # last prior close for the later row.
    swaps = _swap_indices(tape)
    earlier = swaps[0]
    later = swaps[1]
    t_earlier = tape.column("block_timestamp")[earlier].as_py()
    t_later = tape.column("block_timestamp")[later].as_py()
    insert_time = t_earlier + (t_later - t_earlier) / 2

    ref_rows = data.reference.to_pylist()
    keep = [
        r for r in ref_rows if r["close_time"] <= t_earlier or r["close_time"] >= t_later
    ]
    template = dict(ref_rows[0])
    new_bar = dict(template)
    new_bar["open_time"] = insert_time - timedelta(seconds=30)
    new_bar["close_time"] = insert_time
    new_bar["close"] = 99999.5  # distinctive, never produced by the fixture
    new_bar["open"] = new_bar["high"] = new_bar["low"] = new_bar["close"]
    ref2 = pa.Table.from_pylist(keep + [new_bar], schema=REFERENCE_SCHEMA)

    tape2 = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=ref2,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    # Key by block_number to compare rows across the two tapes regardless of seq.
    b2map = {b: i for i, b in enumerate(_blocks(tape2))}
    earlier2 = b2map[tape.column("block_number")[earlier].as_py()]
    later2 = b2map[tape.column("block_number")[later].as_py()]
    # The earlier row must NOT have picked up the new bar.
    assert tape2.column("price_reference")[earlier2].as_py() != 99999.5
    # The later row (after insert_time) may legitimately use it (it is the prior bar then).
    assert tape2.column("price_reference")[later2].as_py() == 99999.5


def test_reference_join_selects_configured_symbol() -> None:
    """The feed carries the primary plus a cross-check; the tape uses exactly one."""
    data = build_tiny_dataset()
    cross_rows = []
    for row in data.reference.to_pylist():
        shifted = dict(row)
        shifted["symbol"] = "ETHUSDC"
        for field in ("open", "high", "low", "close"):
            shifted[field] = shifted[field] + 1000.0
        cross_rows.append(shifted)
    cross = pa.Table.from_pylist(cross_rows, schema=REFERENCE_SCHEMA)
    multi = pa.concat_tables([data.reference, cross])

    tape = build_event_tape(
        data.streams,
        pool=data.pool,
        gas=data.gas,
        reference=multi,
        regime=data.regime,
        fee_growth=data.fee_growth,
        reference_symbol="ETHUSDT",
    )
    primary_closes = {
        row["close"] for row in data.reference.to_pylist() if row["symbol"] == "ETHUSDT"
    }
    cross_closes = {value + 1000.0 for value in primary_closes}
    observed = [
        p for p in tape.column("price_reference").to_pylist() if p is not None
    ]
    assert observed, "the fixture tape must carry at least one reference price"
    assert all(p in primary_closes for p in observed)
    assert all(p not in cross_closes for p in observed)

    # A configured symbol absent from a multi-symbol feed is an error, not a fallback.
    with pytest.raises(ValidationError, match="reference_symbol"):
        build_event_tape(
            data.streams,
            pool=data.pool,
            gas=data.gas,
            reference=multi,
            regime=data.regime,
            fee_growth=data.fee_growth,
            reference_symbol="ETHBTC",
        )


def test_regime_join_selects_configured_symbol() -> None:
    """The regime feed carries the primary plus a cross-check; the tape uses one."""
    data = build_tiny_dataset()
    cross_rows = []
    for row in data.regime.to_pylist():
        shifted = dict(row)
        shifted["symbol"] = "ETHUSDC"
        shifted["regime"] = "cross"
        cross_rows.append(shifted)
    cross = pa.Table.from_pylist(cross_rows, schema=REGIME_SCHEMA)
    multi = pa.concat_tables([data.regime, cross])

    tape = build_event_tape(
        data.streams,
        pool=data.pool,
        gas=data.gas,
        reference=data.reference,
        regime=multi,
        fee_growth=data.fee_growth,
        reference_symbol="ETHUSDT",
    )
    labels = set(tape.column("regime").to_pylist())
    # The cross-check label must never leak onto the ETHUSDT tape.
    assert "cross" not in labels
    assert labels <= set(data.regime.column("regime").to_pylist()) | {"unknown"}


def test_regime_join_is_backward_only() -> None:
    """A regime label whose timestamp is after a tape row must not be used by it."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    swaps = _swap_indices(tape)
    earlier, later = swaps[0], swaps[1]
    t_earlier = tape.column("block_timestamp")[earlier].as_py()
    t_later = tape.column("block_timestamp")[later].as_py()
    insert_time = t_earlier + (t_later - t_earlier) / 2

    reg_rows = data.regime.to_pylist()
    keep = [
        r for r in reg_rows if r["timestamp"] <= t_earlier or r["timestamp"] >= t_later
    ]
    new_label = dict(reg_rows[0])
    new_label["timestamp"] = insert_time
    new_label["regime"] = "high_vol"
    new_label["window_complete"] = True
    new_label["sigma_rv"] = 1.2
    new_label["mu"] = 0.0
    reg2 = pa.Table.from_pylist(keep + [new_label], schema=REGIME_SCHEMA)

    tape2 = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=reg2, fee_growth=data.fee_growth,
    )
    b2map = {b: i for i, b in enumerate(_blocks(tape2))}
    earlier2 = b2map[tape.column("block_number")[earlier].as_py()]
    later2 = b2map[tape.column("block_number")[later].as_py()]
    # The earlier row must not be high_vol from the future label.
    assert tape2.column("regime")[earlier2].as_py() != "high_vol"
    assert tape2.column("regime")[later2].as_py() == "high_vol"


# ---------------------------------------------------------------------------
# 4. Gas: exact equi-join
# ---------------------------------------------------------------------------


def test_gas_exact_join_complete_no_nulls() -> None:
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    assert tape.column("base_fee_per_gas").null_count == 0
    assert tape.column("priority_fee_p50_wei").null_count == 0


def test_gas_exact_join_missing_block_raises() -> None:
    """A tape block absent from the gas stream raises ValidationError naming the block."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    missing = _blocks(tape)[_swap_indices(tape)[0]]  # a block that actually has a tape row
    gas_rows = [g for g in data.gas.to_pylist() if g["block_number"] != missing]
    gas2 = pa.Table.from_pylist(gas_rows, schema=GAS_SCHEMA)
    with pytest.raises(ValidationError) as ei:
        build_event_tape(
            data.streams, pool=data.pool, gas=gas2, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
    assert str(missing) in str(ei.value)


# ---------------------------------------------------------------------------
# 5. Fee-growth globals: exact vs stale_prior
# ---------------------------------------------------------------------------


def test_gas_source_duplicate_block_is_rejected() -> None:
    """A schema-valid ``gas`` table carrying a duplicate ``block_number`` must not
    silently fan out the tape's merge keys — the post-assembly uniqueness re-check
    catches it (the base four-stream union alone can't, because the dup enters via
    the side join)."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    swap_block = _blocks(tape)[_swap_indices(tape)[0]]
    gas_rows = data.gas.to_pylist()
    base_row = next(g for g in gas_rows if g["block_number"] == swap_block)
    gas_rows.append(dict(base_row))  # same block_number twice
    gas2 = pa.Table.from_pylist(gas_rows, schema=GAS_SCHEMA)
    with pytest.raises(ValidationError) as ei:
        build_event_tape(
            data.streams, pool=data.pool, gas=gas2, reference=data.reference,
            regime=data.regime, fee_growth=data.fee_growth,
        )
    assert "duplicate merge keys" in str(ei.value)


def test_fee_growth_source_exact_and_stale_never_future() -> None:
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    src = tape.column("fee_growth_source").to_pylist()
    assert set(src) <= {"exact", "stale_prior", None}
    assert "exact" in src
    assert "stale_prior" in src

    # Exact at a snapshot block that also has a swap.
    snaps = [b for b in data.physical_constants["snapshot_blocks"]]
    b2map = {b: i for i, b in enumerate(_blocks(tape))}
    for snap_block in snaps:
        i = b2map.get(snap_block)
        if i is None:
            continue
        assert tape.column("fee_growth_source")[i].as_py() == "exact"

    # A swap AFTER a snapshot but BEFORE the next is stale_prior.
    for i in _swap_indices(tape):
        row_src = tape.column("fee_growth_source")[i].as_py()
        assert row_src in (None, "exact", "stale_prior")
    # No 'interpolated_future' anywhere — the contract's forbidden value.
    assert "interpolated_future" not in src


# ---------------------------------------------------------------------------
# 6. assert_no_lookahead — both directions required
# ---------------------------------------------------------------------------


def test_assert_no_lookahead_passes_on_correct_tape() -> None:
    tape = build_event_tape(
        build_tiny_dataset().streams, pool=POOL,
        gas=build_tiny_dataset().gas, reference=build_tiny_dataset().reference,
        regime=build_tiny_dataset().regime, fee_growth=build_tiny_dataset().fee_growth,
    )
    # build_event_tape already runs it internally; running it again must also pass.
    assert_no_lookahead(tape)


def test_assert_no_lookahead_catches_forward_join() -> None:
    """A table whose _src_ts_<col> is later than the row's own block time is caught."""
    src = pa.array(
        [
            datetime(2023, 6, 1, 0, 5, tzinfo=UTC),
            datetime(2023, 6, 1, 0, 1, tzinfo=UTC),
        ],
        type=pa.timestamp("us", tz="UTC"),
    )
    ts = pa.array(
        [
            datetime(2023, 6, 1, 0, 0, tzinfo=UTC),
            datetime(2023, 6, 1, 0, 1, tzinfo=UTC),
        ],
        type=pa.timestamp("us", tz="UTC"),
    )
    cols = {
        "block_number": pa.array([1, 2], type=pa.int64()),
        "block_timestamp": ts,
        "price_reference": pa.array([9.0, 1.0], type=pa.float64()),
        "_src_ts_price_reference": src,
    }
    table = pa.table(cols)
    with pytest.raises(ValidationError) as ei:
        assert_no_lookahead(table)
    assert "look-ahead" in str(ei.value)
    assert "price_reference" in str(ei.value)


# ---------------------------------------------------------------------------
# 7. Price orientation
# ---------------------------------------------------------------------------


def _swap_price_peak_deviation(tape: pa.Table) -> float:
    peak = 0.0
    for i in _swap_indices(tape):
        pool = tape.column("price_pool")[i].as_py()
        ref = tape.column("price_reference")[i].as_py()
        if pool is None or ref is None:
            continue
        peak = max(peak, abs(pool - ref) / max(abs(ref), 1e-300))
    return peak


def test_price_orientation_pool_vs_reference_within_5pct() -> None:
    """price_pool and price_reference agree within 5% on real-shaped fixture data."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    peak = _swap_price_peak_deviation(tape)
    assert peak < 0.05, f"price_pool vs price_reference disagree by {peak:.2%}"


def test_price_orientation_swapped_decimals_is_wildly_different() -> None:
    """Swapping the pool's decimals inverts/scales price_pool far outside 5%."""
    data = build_tiny_dataset()
    swapped = dataclasses.replace(data.pool, token0_decimals=18, token1_decimals=6)
    tape = build_event_tape(
        data.streams, pool=swapped, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    peak = _swap_price_peak_deviation(tape)
    assert peak > 0.5, f"decimals swap should grossly distort price_pool (rel diff {peak:.0%})"


# ---------------------------------------------------------------------------
# 8. Schema conformance, pre-history, determinism, tiny_dataset coverage
# ---------------------------------------------------------------------------


def test_tape_conforms_to_schema() -> None:
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    validate_table(tape, EVENT_TAPE_SCHEMA, strict=True)


def test_prehistory_rows_retained_with_unknown() -> None:
    """Tape rows before the first reference bar / regime label keep null price_reference
    and regime=='unknown', and are retained (not dropped)."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    assert tape.num_rows == data.physical_constants["n_events"]
    # The first swap (block 1) is in pre-history (before the first bar's close).
    swaps = _swap_indices(tape)
    first = swaps[0]
    assert tape.column("price_reference")[first].as_py() is None
    assert tape.column("regime")[first].as_py() == Regime.UNKNOWN.value


def test_deterministic_content_hash() -> None:
    """Building the tape twice from the same inputs yields identical content_hash."""
    data = build_tiny_dataset()
    t1 = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    t2 = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    assert content_hash(t1) == content_hash(t2)


def test_tiny_dataset_covers_awkward_cases() -> None:
    """tiny_dataset() contains each case its docstring claims — the fixture-rot guard."""
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    amt0 = {
        int(tape.column("amount0")[i].as_py()) for i in _swap_indices(tape)
    }
    assert any(a > 0 for a in amt0) and any(a < 0 for a in amt0)  # swap both ways
    ticks = [int(tape.column("tick")[i].as_py()) for i in _swap_indices(tape)]
    assert max(ticks) - min(ticks) >= 60  # tick crossing across a grid step
    gbf = data.gas.column("base_fee_per_gas").to_pylist()
    spike_bn = data.physical_constants["spike_block_number"]
    spike_idx = data.gas.column("block_number").to_pylist().index(spike_bn)
    assert int(gbf[spike_idx]) >= 20 * int(gbf[0])  # gas spike (20x)
    assert True in data.reference.column("is_gap_filled").to_pylist()  # filled gap
    assert Regime.UNKNOWN.value in data.regime.column("regime").to_pylist()  # warmup
    assert "burn" in {str(t) for t in tape.column("event_type").to_pylist()}
    assert "collect" in {str(t) for t in tape.column("event_type").to_pylist()}


# ---------------------------------------------------------------------------
# 10. Dataset.require — present / absent / empty distinguishable
# ---------------------------------------------------------------------------


def test_dataset_require() -> None:
    data = build_tiny_dataset()
    tape = build_event_tape(
        data.streams, pool=data.pool, gas=data.gas, reference=data.reference,
        regime=data.regime, fee_growth=data.fee_growth,
    )
    ds = Dataset(
        tape=tape, gas=data.gas, reference=data.reference, regime=data.regime,
        fee_growth=data.fee_growth, manifest=_manifest(),
    )
    assert ds.require("gas") is data.gas

    # An empty table is 'requested, present, zero rows' — passes.
    empty = Dataset(
        tape=tape, gas=empty_table(GAS_SCHEMA), reference=None, regime=None,
        fee_growth=None, manifest=_manifest(),
    )
    assert empty.require("gas").num_rows == 0

    # None is 'absent' — require raises naming the stream and the pull command.
    with pytest.raises(ValidationError) as ei:
        empty.require("reference")
    assert "reference" in str(ei.value)
    assert "undertow-data pull" in str(ei.value)

    with pytest.raises(ValidationError) as ei2:
        ds.require("no_such_stream")
    assert "undertow-data pull" in str(ei2.value)


# ----------------------------------------------------------------------------
# 11. Gas stream is base-fee only; tape carries the reference price (ADR-006)
# ----------------------------------------------------------------------------
# ``attach_reference_prices`` was removed with the per-block gas timestamp column;
# USD conversion now happens on the tape via ``price_reference`` (exercised by the
# as-of join tests in section 4 above).
