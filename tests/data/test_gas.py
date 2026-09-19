"""Tests for ``undertow.data.fetchers.gas`` (T07) — gas / block fetcher.

Everything runs offline against a ``responses``-mocked archive node and the committed
``tests/data/fixtures/blocks_gas.json`` fixture (SYNTHETIC — see its header note; no live RPC
token exists in this sandbox, so the payloads are fabricated but wire-faithful). The fetch-path
mock serves a single ``eth_feeHistory`` (empty percentiles, ADR-005) per chunk; per-block
timestamps are no longer fetched (ADR-006). The block index still uses single-block
``eth_getBlockByNumber`` + ``eth_blockNumber``.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import responses

from undertow.data.config import EndpointConfig
from undertow.data.fetchers.base import FetchRequest, FetchResult
from undertow.data.fetchers.gas import (
    EIP1559_LONDON_BLOCK,
    GasFetcher,
    gas_cost_wei,
)
from undertow.data.schemas import GAS_SCHEMA, decode_uint, validate_table
from undertow.data.types import BlockNumber, ConfigError, PermanentFetchError, ValidationError

RPC_URL = "https://rpc.test"
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "blocks_gas.json").read_text(encoding="utf-8")
)
BLOCKS = [int(hex_b, 16) for hex_b in FIXTURE["_blocks"]]
START, END = BLOCKS[0], BLOCKS[-1]

# Deterministic mock timestamps — a simple 12 s/slot model is fine for synthetic
# test data; only the block index consumes timestamps via single eth_getBlockByNumber
# lookups.
_MOCK_MERGE_BLOCK = 15_537_394
_MOCK_MERGE_TS = datetime(2022, 9, 15, 6, 42, 59, tzinfo=UTC)


def _mock_timestamp(block_number: int) -> int:
    """Deterministic timestamp for a block number (synthetic test data only)."""
    if block_number == 0:
        return 1_438_269_988  # Genesis
    if block_number >= _MOCK_MERGE_BLOCK:
        offset = (block_number - _MOCK_MERGE_BLOCK) * 12
        return int((_MOCK_MERGE_TS + timedelta(seconds=offset)).timestamp())
    offset = (_MOCK_MERGE_BLOCK - block_number) * 14
    return int((_MOCK_MERGE_TS - timedelta(seconds=offset)).timestamp())


# ---------------------------------------------------------------------------
# Mock plumbing
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_http():
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


def _make_fetcher(tmp_path: Path, *, rpc_url: str = RPC_URL, retries: int = 2) -> GasFetcher:
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url=rpc_url,
        reference_base_url="https://b.test",
        max_concurrency=1,
        request_timeout_s=5.0,
        max_retries=retries,
    )
    return GasFetcher(endpoints, tmp_path, sleep=lambda _: None, rng=random.Random(1))


def _mock_fetch(rsp, url: str = RPC_URL, *, drop: int | None = None):
    """Register the fetch-path mock: one ``eth_feeHistory`` call per chunk (ADR-005).

    ADR-006 removed the per-block ``eth_getBlockByNumber`` timestamp batch — gas rows are
    base-fee only.
    """
    fee_history = dict(FIXTURE["fee_history"])
    if drop is not None:
        # Remove the base fee for the dropped block to simulate a short array. The slice
        # over range(len(BLOCKS)) deliberately ignores the trailing blockCount+1 projection.
        fee_history["baseFeePerGas"] = [
            fee_history["baseFeePerGas"][i] for i in range(len(BLOCKS)) if BLOCKS[i] != drop
        ]

    def cb(request):
        body = json.loads(request.body or "{}")
        method = body.get("method")
        if method == "eth_feeHistory":
            return (
                200,
                {},
                json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": fee_history}),
            )
        raise AssertionError(f"unexpected method {method!r} in fetch mock")

    rsp.add_callback(responses.POST, url, callback=cb)


def _mock_index(rsp, *, ts_fn, latest: int, url: str = RPC_URL):
    """Register the block-index mock: single eth_getBlockByNumber + eth_blockNumber."""

    def cb(request):
        body = json.loads(request.body or "{}")
        if isinstance(body, list):
            # Index code never sends batch calls — but be tolerant just in case.
            items = [
                {
                    "jsonrpc": "2.0",
                    "id": item.get("id", i),
                    "result": {
                        "number": item["params"][0],
                        "timestamp": hex(ts_fn(int(item["params"][0], 16))),
                    },
                }
                for i, item in enumerate(body)
            ]
            return 200, {}, json.dumps(items)
        method = body.get("method")
        if method == "eth_blockNumber":
            return 200, {}, json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": hex(latest)})
        if method == "eth_getBlockByNumber":
            bn = int(body["params"][0], 16)
            return (
                200,
                {},
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {"number": hex(bn), "timestamp": hex(ts_fn(bn))},
                    }
                ),
            )
        raise AssertionError(f"unexpected method {method!r} in index mock")

    rsp.add_callback(responses.POST, url, callback=cb)


def _gas_request(start: int = START, end: int = END) -> FetchRequest:
    return FetchRequest(
        stream="gas", pool=None, start_block=BlockNumber(start), end_block=BlockNumber(end)
    )


# ---------------------------------------------------------------------------
# 1. GAS_SCHEMA conformance; the no-log/chain-wide column set
# ---------------------------------------------------------------------------


def test_gas_schema_conformance_no_forbidden_columns(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    result = f.fetch(_gas_request())
    table = result.table
    validate_table(table, GAS_SCHEMA, strict=True)  # must not raise
    names = set(table.schema.names)
    for forbidden in ("log_index", "tx_hash", "pool_address", "event_type"):
        assert forbidden not in names, f"gas is chain-wide; must not carry {forbidden!r}"
    # full contiguous run, ascending, sort key (block_number,)
    assert list(table["block_number"].to_pylist()) == list(range(START, END + 1))
    assert isinstance(result, FetchResult)
    assert result.route.value == "rpc"


# ---------------------------------------------------------------------------
# 2. Field decode: real big-int base fee, zeroed gas_used/gas_limit (ADR-005/006)
# ---------------------------------------------------------------------------


def test_field_decode_exact_integers(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    table = f.fetch(_gas_request()).table

    base0 = decode_uint(table["base_fee_per_gas"][0].as_py())
    assert base0 == 12_000_000_000
    assert 1e9 < base0 < 1e11  # ~1e10 wei, realistic EIP-1559 base fee

    # gas_used and gas_limit are 0 (ADR-005 §Decision.5)
    assert int(table["gas_used"][0].as_py()) == 0
    assert int(table["gas_limit"][0].as_py()) == 0
    assert table["gas_used"].type == table["gas_limit"].type

    # ADR-006: gas is base-fee only — no per-block timestamp or price column.
    assert "block_timestamp" not in table.column_names
    assert "eth_usd_price" not in table.column_names


# ---------------------------------------------------------------------------
# 3. Priority fee columns are zero (ADR-005 — flat tip surcharge replaces them)
# ---------------------------------------------------------------------------


def test_priority_fees_are_zero_adr005(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    table = f.fetch(_gas_request()).table

    for col in ("priority_fee_p50_wei", "priority_fee_p90_wei"):
        values = [decode_uint(v.as_py()) for v in table[col]]
        assert all(v == 0 for v in values), f"{col}: all values must be 0 per ADR-005"


# ---------------------------------------------------------------------------
# 4. Completeness: a short baseFeePerGas array → PermanentFetchError
# ---------------------------------------------------------------------------


def test_gap_detection_short_fee_history(mocked_http, tmp_path):
    """A short baseFeePerGas array is rejected (PermanentFetchError)."""
    _mock_fetch(mocked_http, drop=16200004)
    f = _make_fetcher(tmp_path)
    with pytest.raises(PermanentFetchError, match="baseFeePerGas entries"):
        f.fetch(_gas_request())


def test_contiguous_range_ok_no_raise(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    table = f.fetch(_gas_request()).table
    assert table.num_rows == len(BLOCKS)


def test_fee_history_trailing_projection_ignored(mocked_http, tmp_path) -> None:
    """EIP-1559 ``eth_feeHistory`` returns ``blockCount + 1`` base fees.

    The trailing entry is the *projected* fee for ``newestBlock + 1``: it must not be
    emitted as a row, and it must not shift the ``oldestBlock``-anchored mapping.
    """
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    table = f.fetch(_gas_request()).table

    fees = FIXTURE["fee_history"]["baseFeePerGas"]
    assert len(fees) == len(BLOCKS) + 1  # fixture encodes the spec shape
    expected = [int(h, 16) for h in fees[: len(BLOCKS)]]
    got = [decode_uint(v.as_py()) for v in table["base_fee_per_gas"]]
    assert got == expected
    assert table.num_rows == len(BLOCKS)


# ---------------------------------------------------------------------------
# 5. Duplicate block in the response → ValidationError
# ---------------------------------------------------------------------------


def test_duplicate_block_validation_error(mocked_http, tmp_path):
    """A duplicate block_number in rows triggers a validation error."""
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    # The fetcher produces one row per block in the range — duplicates can only
    # happen if the underlying data is corrupt. For this test we test the
    # _check_contiguous guard directly: if the fetcher somehow produced two
    # rows for the same block (which can't happen with the new approach but
    # the guard still exists). We can't easily trigger it with the mock, so
    # test it via the internal method.
    rows = [
        {"block_number": 16200000, "base_fee_per_gas": 0,
         "gas_used": 0, "gas_limit": 0, "priority_fee_p50_wei": 0, "priority_fee_p90_wei": 0},
        {"block_number": 16200000, "base_fee_per_gas": 0,
         "gas_used": 0, "gas_limit": 0, "priority_fee_p50_wei": 0, "priority_fee_p90_wei": 0},
    ]
    with pytest.raises(ValidationError) as ei:
        f._check_contiguous(rows, _gas_request())
    assert "duplicate" in str(ei.value).lower()
    assert "16200000" in str(ei.value)


# ---------------------------------------------------------------------------
# 6. block_for_timestamp: exact hit / between / out-of-range / naive
# ---------------------------------------------------------------------------


def _index_ts_fn(b: int) -> int:
    return 1_600_000_000 + b


def test_block_for_timestamp_boundaries(mocked_http, tmp_path):
    latest = 5
    _mock_index(mocked_http, ts_fn=_index_ts_fn, latest=latest)
    f = _make_fetcher(tmp_path)

    # exact hit returns that block
    assert f.block_for_timestamp(datetime.fromtimestamp(1_600_000_001, tz=UTC)) == 1
    # between two blocks returns the earlier one
    assert f.block_for_timestamp(datetime.fromtimestamp(1_600_000_000, tz=UTC)) == 0
    assert f.block_for_timestamp(datetime.fromtimestamp(1_600_000_000, tz=UTC)) == 0
    # a timestamp exactly at the latest block is allowed (inclusive)
    assert f.block_for_timestamp(datetime.fromtimestamp(1_600_000_005, tz=UTC)) == latest

    # before genesis -> ConfigError
    with pytest.raises(ConfigError) as ei:
        f.block_for_timestamp(datetime.fromtimestamp(1_599_999_999, tz=UTC))
    assert "genesis" in str(ei.value).lower()

    # after latest -> ConfigError
    with pytest.raises(ConfigError):
        f.block_for_timestamp(datetime.fromtimestamp(1_600_000_006, tz=UTC))

    # naive datetime -> ConfigError
    with pytest.raises(ConfigError) as ei2:
        f.block_for_timestamp(datetime(2022, 1, 1))
    assert "naive" in str(ei2.value).lower()


def test_block_between_two_blocks_returns_earlier(mocked_http, tmp_path):
    latest = 5
    _mock_index(mocked_http, ts_fn=_index_ts_fn, latest=latest)
    f = _make_fetcher(tmp_path / "c")
    # halfway between ts(0)=...000 and ts(1)=...001 -> the earlier block 0
    assert f.block_for_timestamp(datetime.fromtimestamp(1_600_000_000.5, tz=UTC)) == 0


# ---------------------------------------------------------------------------
# 7. timestamp_for_block round-trips with block_for_timestamp on the fixture
# ---------------------------------------------------------------------------


def test_timestamp_for_block_and_block_roundtrip(mocked_http, tmp_path):
    def ts_fn(b: int) -> int:
        return _mock_timestamp(b)

    _mock_index(mocked_http, ts_fn=ts_fn, latest=max(BLOCKS))
    f = _make_fetcher(tmp_path)

    for block in BLOCKS:
        ts = f.timestamp_for_block(BlockNumber(block))
        assert isinstance(ts, datetime) and ts.tzinfo is not None
        assert f.block_for_timestamp(ts) == block


# ---------------------------------------------------------------------------
# 8. Pre-London request -> ConfigError (EIP-1559)
# ---------------------------------------------------------------------------


def test_pre_london_block_request_rejected(mocked_http, tmp_path):
    f = _make_fetcher(tmp_path)
    req = _gas_request(start=12_000_000, end=12_900_000)
    assert req.end_block < EIP1559_LONDON_BLOCK
    with pytest.raises(ConfigError) as ei:
        f.fetch(req)
    assert "EIP-1559" in str(ei.value)
    assert "baseFeePerGas" in str(ei.value) or "base fee" in str(ei.value).lower()


# ---------------------------------------------------------------------------
# 9. gas_cost_wei: pure integer math with tip_surcharge_pct (ADR-005)
# ---------------------------------------------------------------------------


def test_gas_cost_wei_integer_math():
    row = {
        "base_fee_per_gas": "12000000000",  # 12 Gwei
    }
    gas_units = 210_000

    # default 3% surcharge
    cost = gas_cost_wei(row, gas_units)
    assert isinstance(cost, int) and not isinstance(cost, bool)
    # base * 1.03 * gas_units = 12e9 * 103/100 * 210000
    expected = 12_000_000_000 * 103 // 100 * 210_000
    assert cost == expected

    # tip_surcharge_pct=0 yields base only
    cost0 = gas_cost_wei(row, gas_units, tip_surcharge_pct=0)
    assert cost0 == 12_000_000_000 * 210_000

    # tip_surcharge_pct=10 yields 10% uplift
    cost10 = gas_cost_wei(row, gas_units, tip_surcharge_pct=10)
    assert cost10 == 12_000_000_000 * 110 // 100 * 210_000
    assert cost10 > cost

    # accepts int-typed values too
    int_row = {"base_fee_per_gas": 12_000_000_000}
    assert gas_cost_wei(int_row, gas_units) == cost

    # negative tip_surcharge_pct is rejected
    with pytest.raises(ValueError, match="non-negative"):
        gas_cost_wei(row, gas_units, tip_surcharge_pct=-1)


def test_gas_cost_wei_uses_gas_units_never_hardcoded():
    row = {"base_fee_per_gas": "1000000000"}
    assert gas_cost_wei(row, 400_000) == 1_000_000_000 * 103 // 100 * 400_000
    assert gas_cost_wei(row, 150_000) == 1_000_000_000 * 103 // 100 * 150_000


# ---------------------------------------------------------------------------
# 10. Spike preservation — no averaging/interpolation anywhere
# ---------------------------------------------------------------------------


def test_20x_spike_preserved_exactly(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    table = f.fetch(_gas_request()).table
    numbers = table["block_number"].to_pylist()
    spike_idx = numbers.index(16200004)
    spike = decode_uint(table["base_fee_per_gas"][spike_idx].as_py())
    neighbours = [
        decode_uint(table["base_fee_per_gas"][i].as_py())
        for i in (spike_idx - 1, spike_idx + 1)
    ]
    assert spike == 240_000_000_000  # the exact 20x value, bit-for-bit
    assert spike >= 20 * max(neighbours)  # 20x above EITHER neighbour, no smoothing
    assert all(n != spike for n in neighbours)  # nothing averaged toward the spike
    # the whole column round-trips exactly: no value changed from the fixture. The fixture
    # carries the EIP-1559 trailing projection (blockCount+1); only the first len(BLOCKS)
    # entries map to the requested range.
    raw = [int(h, 16) for h in FIXTURE["fee_history"]["baseFeePerGas"]][: len(BLOCKS)]
    assert [decode_uint(v) for v in table["base_fee_per_gas"].to_pylist()] == raw


# ---------------------------------------------------------------------------
# 11. Warm cache: zero HTTP calls; RPC key scrubbed from logs
# ---------------------------------------------------------------------------


def test_cache_hit_zero_network_and_secret_scrubbed(mocked_http, tmp_path, caplog):
    secret = "SECRETKEY_a1b2c3"
    url = f"https://rpc.test/{secret}"
    _mock_fetch(mocked_http, url=url)
    f = _make_fetcher(tmp_path, rpc_url=url)

    with caplog.at_level("WARNING"):
        cold = f.fetch(_gas_request())
        assert cold.from_cache is False
        assert cold.n_requests == 1  # one chunk fetched (1 HTTP call — eth_feeHistory only)
    calls_after_cold = len(mocked_http.calls)
    assert calls_after_cold > 0

    warm = f.fetch(_gas_request())
    assert warm.from_cache is True
    assert warm.n_requests == 0
    assert len(mocked_http.calls) == calls_after_cold  # no new network

    assert secret not in caplog.text
    assert "SECRETKEY" not in caplog.text


# ---------------------------------------------------------------------------
# 12. Date-mode fetch auto-resolves through block_for_timestamp (ADR-003)
# ---------------------------------------------------------------------------


def _mock_node(rsp, *, url: str = RPC_URL):
    """A single combined mock serving the index AND the fetch path on one URL.

    ``responses`` dispatches to the first matching callback only, so index and fetch must share
    one handler. Handles both single RPC calls (index) and batch calls (timestamp fetch via
    batched ``eth_getBlockByNumber``).
    """
    end = BLOCKS[-1]

    def _batch_block_by_number_response(body: list) -> tuple[int, dict, str]:
        items = [
            {
                "jsonrpc": "2.0",
                "id": item.get("id", i),
                "result": {
                    "number": item["params"][0],
                    "timestamp": hex(_mock_timestamp(int(item["params"][0], 16))),
                    "baseFeePerGas": "0x1",
                },
            }
            for i, item in enumerate(body)
        ]
        return 200, {}, json.dumps(items)

    def cb(request):
        body = json.loads(request.body or "{}")
        if isinstance(body, list):
            return _batch_block_by_number_response(body)
        method = body.get("method")
        if method == "eth_blockNumber":
            return 200, {}, json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": hex(end)})
        if method == "eth_getBlockByNumber":
            bn = int(body["params"][0], 16)
            return (
                200,
                {},
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "number": hex(bn),
                            "timestamp": hex(_mock_timestamp(bn)),
                        },
                    }
                ),
            )
        if method == "eth_feeHistory":
            return (
                200,
                {},
                json.dumps(
                    {"jsonrpc": "2.0", "id": body["id"], "result": FIXTURE["fee_history"]}
                ),
            )
        raise AssertionError(f"unexpected RPC method {method!r} in combined mock")

    rsp.add_callback(responses.POST, url, callback=cb)


def test_date_mode_fetch_resolves_bounds_via_block_index(mocked_http, tmp_path):
    _mock_node(mocked_http)
    f = _make_fetcher(tmp_path)
    request = FetchRequest(
        stream="gas",
        pool=None,
        start_block=None,
        end_block=None,
        start_utc=datetime.fromtimestamp(_mock_timestamp(START), tz=UTC),
        end_utc=datetime.fromtimestamp(_mock_timestamp(END), tz=UTC),
    )
    result = f.fetch(request)
    table = result.table
    # both dates land exactly on fixture blocks (last block with ts <= ts is the block itself)
    assert table["block_number"].to_pylist() == list(range(START, END + 1))
    validate_table(table, GAS_SCHEMA, strict=True)


def test_fetch_with_neither_bounds_returns_empty_table(mocked_http, tmp_path):
    f = _make_fetcher(tmp_path)
    result = f.fetch(FetchRequest(stream="gas", pool=None, start_block=None, end_block=None))
    assert result.table.num_rows == 0
    assert result.n_requests == 0
    assert any("nothing to fetch" in w for w in result.warnings)
    validate_table(result.table, GAS_SCHEMA, strict=True)  # conforms even when empty
