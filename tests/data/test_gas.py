"""Tests for ``undertow.data.fetchers.gas`` (T07) — gas / block fetcher.

Everything runs offline against a ``responses``-mocked archive node and the committed
``tests/data/fixtures/blocks_gas.json`` fixture (SYNTHETIC — see its header note; no live RPC
token exists in this sandbox, so the payloads are fabricated but wire-faithful). The mock
dispatches on JSON-RPC method, so the same callback serves ``eth_feeHistory``, batched and
single ``eth_getBlockByNumber``, and ``eth_blockNumber``.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
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
FIXTURE_TS = {int(k, 16): int(v, 16) for k, v in FIXTURE["block_timestamps"].items()}


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


def _mock_fetch(rsp, url: str = RPC_URL, *, drop: int | None = None, dup: int | None = None):
    """Register the fetch-path mock: batched eth_getBlockByNumber (authoritative header
    fields; optionally a missing/duplicated block) + one eth_feeHistory (percentiles)."""
    headers = {k: dict(v) for k, v in FIXTURE["block_headers"].items()}
    if drop is not None:
        headers.pop(hex(drop), None)
    dup_hex = hex(dup) if dup is not None else None

    def cb(request):
        body = json.loads(request.body or "{}")
        if isinstance(body, list):  # batched eth_getBlockByNumber
            results = []
            for call in body:
                hdr = headers.get(call["params"][0])
                if hdr is None:
                    continue  # dropped block -> no header -> a gap the fetcher must catch
                results.append({"jsonrpc": "2.0", "id": call["id"], "result": hdr})
                if dup_hex == call["params"][0]:
                    results.append({"jsonrpc": "2.0", "id": -1, "result": hdr})
            return 200, {}, json.dumps(results)
        method = body.get("method")
        if method == "eth_feeHistory":
            return (
                200,
                {},
                json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": FIXTURE["fee_history"]}),
            )
        raise AssertionError(f"unexpected single RPC method {method!r} in fetch mock")

    rsp.add_callback(responses.POST, url, callback=cb)


def _mock_index(rsp, *, ts_fn, latest: int, url: str = RPC_URL):
    """Register the block-index mock: single eth_getBlockByNumber + eth_blockNumber."""

    def cb(request):
        body = json.loads(request.body or "{}")
        assert isinstance(body, dict), "index mock sees only single (non-batch) calls"
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
# 2. Field decode: real big-int base fee, gas used/limit, UTC timestamp
# ---------------------------------------------------------------------------


def test_field_decode_exact_integers(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    table = f.fetch(_gas_request()).table

    base0 = decode_uint(table["base_fee_per_gas"][0].as_py())
    assert base0 == 12_000_000_000
    assert 1e9 < base0 < 1e11  # ~1e10 wei, realistic EIP-1559 base fee

    assert int(table["gas_used"][1].as_py()) == 12_500_000
    assert int(table["gas_limit"][0].as_py()) == 30_000_000
    assert table["gas_used"].type == table["gas_limit"].type

    ts = table["block_timestamp"][0].as_py()
    assert ts.tzinfo is not None and ts.utcoffset() is not None
    assert ts.tzname() == "UTC"

    # eth_usd_price is null as written by T07 (T11 joins it later)
    assert table["eth_usd_price"].null_count == table.num_rows


# ---------------------------------------------------------------------------
# 3. Percentiles from eth_feeHistory reward; reward=null -> 0 + warning
# ---------------------------------------------------------------------------


def test_percentiles_and_null_reward(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path)
    result = f.fetch(_gas_request())
    table = result.table
    numbers = table["block_number"].to_pylist()

    # block 16200000 p50 / p90 from the fixture reward array
    assert decode_uint(table["priority_fee_p50_wei"][0].as_py()) == 2_500_000_000
    assert decode_uint(table["priority_fee_p90_wei"][0].as_py()) == 20_000_000_000
    # spike block 16200004: p90 is the "get included during a spike" cost
    assert decode_uint(table["priority_fee_p90_wei"][4].as_py()) == 120_000_000_000

    # block 16200002 has reward=null (no transactions) -> both percentiles 0, flagged
    idx = numbers.index(16200002)
    assert decode_uint(table["priority_fee_p50_wei"][idx].as_py()) == 0
    assert decode_uint(table["priority_fee_p90_wei"][idx].as_py()) == 0
    assert any("reward=null" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 4. Completeness: a missing interior block raises ValidationError naming it;
#    a contiguous range does not
# ---------------------------------------------------------------------------


def test_gap_detection_names_first_missing_block(mocked_http, tmp_path):
    _mock_fetch(mocked_http, drop=16200004)
    f = _make_fetcher(tmp_path / "a")
    with pytest.raises(ValidationError) as ei:
        f.fetch(_gas_request())
    assert "16200004" in str(ei.value)


def test_contiguous_range_ok_no_raise(mocked_http, tmp_path):
    _mock_fetch(mocked_http)
    f = _make_fetcher(tmp_path / "b")
    table = f.fetch(_gas_request()).table
    assert table.num_rows == len(BLOCKS)


# ---------------------------------------------------------------------------
# 5. Duplicate block in the response -> ValidationError
# ---------------------------------------------------------------------------


def test_duplicate_block_validation_error(mocked_http, tmp_path):
    _mock_fetch(mocked_http, dup=16200002)
    f = _make_fetcher(tmp_path)
    with pytest.raises(ValidationError) as ei:
        f.fetch(_gas_request())
    assert "duplicate" in str(ei.value).lower()
    assert "16200002" in str(ei.value)


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
    anchor, anchor_ts = min(FIXTURE_TS), FIXTURE_TS[min(FIXTURE_TS)]

    def ts_fn(b: int) -> int:
        if b in FIXTURE_TS:
            return FIXTURE_TS[b]
        return anchor_ts + (b - anchor) * 12  # monotone fallback for intermediate blocks

    _mock_index(mocked_http, ts_fn=ts_fn, latest=max(FIXTURE_TS))
    f = _make_fetcher(tmp_path)

    for block in FIXTURE_TS:
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
# 9. gas_cost_wei: pure integer math, p90 > p50 when the percentiles differ
# ---------------------------------------------------------------------------


def test_gas_cost_wei_integer_math():
    row = {
        "base_fee_per_gas": "12000000000",  # 12 Gwei
        "priority_fee_p50_wei": "2500000000",  # 2.5 Gwei
        "priority_fee_p90_wei": "20000000000",  # 20 Gwei
    }
    gas_units = 210_000
    cost = gas_cost_wei(row, gas_units, percentile=50)
    assert isinstance(cost, int) and not isinstance(cost, bool)
    assert cost == (12_000_000_000 + 2_500_000_000) * 210_000
    # p90 selection yields a strictly larger cost when percentiles differ
    cost90 = gas_cost_wei(row, gas_units, percentile=90)
    assert cost90 > cost
    assert isinstance(cost90, int)
    # accepts int-typed values too
    int_row = {k: int(v) for k, v in row.items()}
    assert gas_cost_wei(int_row, gas_units) == cost
    # unsupported percentile is rejected
    with pytest.raises(ValueError):
        gas_cost_wei(row, gas_units, percentile=75)


def test_gas_cost_wei_uses_gas_units_never_hardcoded():
    row = {"base_fee_per_gas": "1000000000", "priority_fee_p50_wei": "0"}
    assert gas_cost_wei(row, 400_000) == 1_000_000_000 * 400_000
    assert gas_cost_wei(row, 150_000) == 1_000_000_000 * 150_000


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
    assert spike >= 20 * max(neighbours)  # 20x above EITHER neighbour, no smoothng
    assert all(n != spike for n in neighbours)  # nothing averaged toward the spike
    # the whole column round-trips exactly: no value changed from the fixture
    raw = [int(h, 16) for h in FIXTURE["fee_history"]["baseFeePerGas"]]
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
        assert cold.n_requests == 1
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
    one handler. Blocks outside the fixture get a monotone virtual header (the constant-12s
    chain both tests 6/7 rely on), so the binary search stays exact at the fixture's own blocks.
    """
    start, end = BLOCKS[0], BLOCKS[-1]
    base_ts = min(FIXTURE_TS.values())

    def virtual_header(bn: int) -> dict:
        hx = hex(bn)
        if hx in FIXTURE["block_headers"]:
            return FIXTURE["block_headers"][hx]
        return {
            "number": hx,
            "timestamp": hex(base_ts + (bn - start) * 12),
            "baseFeePerGas": hex(10_000_000_000 + (bn % 1000) * 1_000_000),
            "gasUsed": hex(8_000_000),
            "gasLimit": hex(30_000_000),
        }

    def cb(request):
        body = json.loads(request.body or "{}")
        if isinstance(body, list):  # batched eth_getBlockByNumber
            results = []
            for call in body:
                results.append(
                    {
                        "jsonrpc": "2.0",
                        "id": call["id"],
                        "result": virtual_header(int(call["params"][0], 16)),
                    }
                )
            return 200, {}, json.dumps(results)
        method = body.get("method")
        if method == "eth_blockNumber":
            return 200, {}, json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": hex(end)})
        if method == "eth_getBlockByNumber":
            bn = int(body["params"][0], 16)
            return (
                200,
                {},
                json.dumps(
                    {"jsonrpc": "2.0", "id": body["id"], "result": virtual_header(bn)}
                ),
            )
        if method == "eth_feeHistory":
            return (
                200,
                {},
                json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": FIXTURE["fee_history"]}),
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
        start_utc=datetime.fromtimestamp(FIXTURE_TS[START], tz=UTC),
        end_utc=datetime.fromtimestamp(FIXTURE_TS[END], tz=UTC),
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