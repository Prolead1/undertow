"""Tests for ``undertow.data.fetchers.rpc`` (T06) — Route C RPC fetcher.

Fully offline: every RPC endpoint is a ``responses`` callback speaking
JSON-RPC 2.0 against a tiny in-test dispatcher (which may answer batch calls in
shuffled ``id`` order and inject per-item errors). Fixture logs come from
``tests/data/fixtures/rpc_logs_*.json`` (synthetic but protocol-shaped; see the
T06 brief). Covers the 13 required scenarios: field-identical fetch of every
log stream, empty/zero-row ranges, warm-cache zero-calls, ``fee_growth_at``
syntax+exactness, archive-node absence, out-of-order batch reassembly, per-item
errors naming the failing params, the batch-per-item fallback, the finality
guard, the reorg guard, and secret hygiene in ``caplog``.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pytest
import responses

from undertow.data.config import EndpointConfig, GLOBAL_TICK_SENTINEL, default_pools
from undertow.data.fetchers.abi import (
    SELECTOR_FEE_GROWTH_GLOBAL_0_X128,
    SELECTOR_FEE_GROWTH_GLOBAL_1_X128,
    SELECTOR_LIQUIDITY,
    SELECTOR_SLOT0,
    SELECTOR_TICKS,
    TOPIC_SWAP,
    encode_ticks_calldata,
)
from undertow.data.fetchers.base import FetchRequest
from undertow.data.fetchers.rpc import RpcFetcher
from undertow.data.schemas import SCHEMA_REGISTRY, decode_uint, validate_table
from undertow.data.types import (
    BlockNumber,
    FetchError,
    PermanentFetchError,
    ReorgDetectedError,
    Route,
)

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
POOL = default_pools()["USDC_WETH_3000"]
RPC_URL = "https://rpc.example.test/"


@pytest.fixture
def mocked_http():
    """Activated ``responses`` mock; unused registered responses never fail the test.

    Defined locally (pytest fixtures do not cross test modules) — the identical
    fixture in ``test_fetcher_base.py`` belongs to T04's module.
    """
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


class _RpcError(Exception):
    def __init__(self, message: str, code: int = -32000) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _base_ts() -> int:
    return int(datetime(2022, 2, 8, 12, 0, 0, tzinfo=UTC).timestamp())


class _Responder:
    """One JSON-RPC callback over the mocked URL.

    ``handlers`` maps method name -> ``fn(params) -> result`` (or raises
    ``_RpcError``). Batch responses echo every request id in *reversed* order
    unless ``shuffle=False``, so out-of-order reassembly is exercised on every
    multi-call test, not just the dedicated one.
    """

    def __init__(
        self, mocked: "responses.RequestsMock", *, shuffle: bool = True, url: str = RPC_URL
    ) -> None:
        self.calls: list[dict] = []
        self.handlers: dict[str, Callable[[list[object]], object]] = {}
        self.shuffle = shuffle
        # Defaults shared by every test on this pool.
        self.latest_block = 13_400_000

        def block_number(_: list[object]) -> object:
            return hex(self.latest_block)

        def block_by_number(params: list[object]) -> object:
            block = int(str(params[0]), 16)
            return {
                "number": hex(block),
                "hash": "0x" + f"h{block:063x}",
                "timestamp": hex(_base_ts() + (block - 13_385_000) * 12),
            }

        self.handlers["eth_blockNumber"] = block_number
        self.handlers["eth_getBlockByNumber"] = block_by_number
        self.handlers["eth_getLogs"] = lambda _params: []
        self.handlers["eth_call"] = lambda _params: "0x"

        mocked.add_callback(
            responses.POST, url, callback=self._callback, content_type="application/json"
        )

    def _callback(self, request):
        body = json.loads(request.body or "{}")
        self.calls.append(body)
        as_batch = isinstance(body, list)
        items = body if as_batch else [body]
        out: list[dict] = []
        for item in items:
            method = item.get("method")
            params = item.get("params", [])
            handler = self.handlers.get(method)
            if handler is None:
                reply: dict = {"jsonrpc": "2.0", "id": item.get("id"),
                               "error": {"code": -32601, "message": f"method {method} not found"}}
            else:
                try:
                    result = handler(params)
                except _RpcError as exc:
                    reply = {"jsonrpc": "2.0", "id": item.get("id"),
                             "error": {"code": exc.code, "message": exc.message}}
                else:
                    reply = {"jsonrpc": "2.0", "id": item.get("id"), "result": result}
            out.append(reply)
        if as_batch and self.shuffle:
            out.reverse()
        payload = json.dumps(out if as_batch else out[0])
        return 200, {"Content-Type": "application/json"}, payload

    def n_posts(self) -> int:
        return len(self.calls)


def _noop_sleep(_: float) -> None:
    pass


def _maker(
    tmp_path: Path, *, confirmations: int = 64, verify_chain: bool = False, seed: int = 1
) -> RpcFetcher:
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url=RPC_URL,
        reference_base_url="https://b.test",
        max_concurrency=1,
        request_timeout_s=5.0,
        max_retries=2,
    )
    return RpcFetcher(
        endpoints,
        tmp_path,
        confirmations=confirmations,
        verify_chain=verify_chain,
        sleep=_noop_sleep,
        rng=random.Random(seed),
    )


def _req(stream: str, start: int, end: int) -> FetchRequest:
    return FetchRequest(
        stream=stream,
        pool=POOL,
        start_block=BlockNumber(start),
        end_block=BlockNumber(end),
    )


def _fixture(name: str) -> list[dict]:
    """An ``rpc_logs_*`` fixture as a plain log list (the ``_comment`` envelope is
    metadata only — the transport never sees it)."""
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "logs" in data:
        return data["logs"]
    return data


# ---------------------------------------------------------------------------
# 1. Log streams: field-identical tables, timestamps resolved, no 0x in rows.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stream", "fixture", "blocks", "n_rows"),
    [
        ("swap", "rpc_logs_swap.json", (13385010, 13385012), 3),
        ("mint", "rpc_logs_mint.json", (13385020, 13385020), 1),
        ("burn", "rpc_logs_burn.json", (13385021, 13385021), 1),
        ("collect", "rpc_logs_collect.json", (13385022, 13385022), 1),
    ],
)
def test_fetch_log_streams(
    mocked_http: "responses.RequestsMock",
    tmp_path: Path,
    stream: str,
    fixture: str,
    blocks: tuple[int, int],
    n_rows: int,
) -> None:
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture(fixture)
    f = _maker(tmp_path)

    start, end = blocks
    result = f.fetch(_req(stream, start, end))
    assert result.route is Route.RPC
    assert result.from_cache is False
    assert result.n_requests == 1
    assert result.warnings == ()
    table = result.table
    validate_table(table, SCHEMA_REGISTRY[stream])  # already done inside, idempotent
    assert table.num_rows == n_rows
    assert table.column("pool_address").to_pylist() == [str(POOL.address)] * n_rows
    # Every row carries a resolved UTC timestamp and a plain-int block/log index.
    first_ts = table.column("block_timestamp").to_pylist()[0]
    assert first_ts.tzinfo is not None
    assert first_ts.utcoffset().total_seconds() == 0
    assert all(isinstance(b, int) for b in table.column("block_number").to_pylist())
    assert all(c.startswith("0x") for c in table.column("tx_hash").to_pylist())
    # tx hashes are lowercase hex
    assert all(h == h.lower() for h in table.column("tx_hash").to_pylist())


def test_fetch_stream_field_values_hand_checked(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture("rpc_logs_swap.json")
    table = _maker(tmp_path).fetch(_req("swap", 13385010, 13385012)).table

    swap0 = table.slice(0, 1).to_pylist()[0]
    assert swap0["amount0"] == "-548466086452"  # negative: signed int256 in a string col
    assert swap0["amount1"] == "182700063432212445192"
    assert swap0["sqrt_price_x96"] == "1446501726624926496477173928747177"
    assert swap0["tick"] == 196242
    assert swap0["event_type"] == "swap"
    # The fixture's third log has a NEGATIVE tick (int24 sign extension); the
    # synthetic row exists solely to prove the decoder's sign handling.
    assert table.column("tick").to_pylist()[-1] == -202010


def test_fetch_block_with_no_swaps_produces_no_rows(mocked_http, tmp_path: Path) -> None:
    """13385012 is inside the requested range but the fixture holds no log for it."""
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture("rpc_logs_swap.json")
    table = _maker(tmp_path).fetch(_req("swap", 13385010, 13385012)).table
    assert set(table.column("block_number").to_pylist()) == {13385010, 13385011}


def test_fetch_empty_page_returns_empty_table(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: []
    table = _maker(tmp_path).fetch(_req("swap", 13385040, 13385040)).table
    assert table.num_rows == 0
    validate_table(table, SCHEMA_REGISTRY["swap"])


def test_fetch_flash_inline_log(mocked_http, tmp_path: Path) -> None:
    """Flash decode through the transport (no flash fixture exists: the pinned
    pools have negligible flash volume; T10 needs to know it is decoded)."""
    from undertow.data.fetchers.abi import TOPIC_FLASH, decode_flash

    log = {
        "address": str(POOL.address),
        "topics": [TOPIC_FLASH,
                   "0x" + "aa" * 32,  # sender word (indexed)
                   "0x" + "bb" * 32],  # recipient word (indexed)
        "data": "0x" + "".join(
            (v & ((1 << 256) - 1)).to_bytes(32, "big").hex() for v in (1000, 2000, 1300, 2600)
        ),
        "blockNumber": hex(13385030),
        "blockHash": "0x" + "ab" * 32,
        "transactionHash": "0x" + "cd" * 32,
        "transactionIndex": "0x0",
        "logIndex": "0x0",
        "removed": False,
    }
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: [log]
    table = _maker(tmp_path).fetch(_req("flash", 13385030, 13385030)).table
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["event_type"] == "flash"
    assert row["paid0"] == "1300"
    assert row["paid1"] == "2600"
    assert row["sender"] == "0x" + "aa" * 20


def test_unsupported_stream_and_missing_pool(mocked_http, tmp_path: Path) -> None:
    f = _maker(tmp_path)
    with pytest.raises(FetchError, match="does not support stream 'gas'"):
        f.fetch(FetchRequest(stream="gas", pool=POOL, start_block=BlockNumber(1),
                             end_block=BlockNumber(2)))
    with pytest.raises(FetchError, match="fee_growth_at"):
        f.fetch(FetchRequest(stream="fee_growth", pool=POOL, start_block=BlockNumber(1),
                             end_block=BlockNumber(2)))
    no_pool = FetchRequest(stream="swap", pool=None, start_block=BlockNumber(13385010),
                           end_block=BlockNumber(13385010))
    _Responder(mocked_http)
    with pytest.raises(FetchError, match="pool is required"):
        f.fetch(no_pool)


def test_supported_streams_contract() -> None:
    assert RpcFetcher.supported_streams(RpcFetcher) == {
        "swap", "mint", "burn", "collect", "flash", "fee_growth"
    }


# ---------------------------------------------------------------------------
# 2. fee_growth_at against rpc_ticks_call.json.
# ---------------------------------------------------------------------------


def _ticks_fixture() -> dict:
    return json.loads((FIXTURES / "rpc_ticks_call.json").read_text(encoding="utf-8"))


def _install_fee_growth_calls(resp: _Responder, fx: dict) -> None:
    by_selector = {
        SELECTOR_SLOT0: fx["slot0"],
        SELECTOR_LIQUIDITY: fx["liquidity"],
        SELECTOR_FEE_GROWTH_GLOBAL_0_X128: fx["feeGrowthGlobal0X128"],
        SELECTOR_FEE_GROWTH_GLOBAL_1_X128: fx["feeGrowthGlobal1X128"],
    }

    def eth_call(params: list[object]) -> object:
        data = str(params[0]["data"])
        selector = "0x" + data[2:10]
        if selector == SELECTOR_TICKS:
            tick = int(data[10:], 16)
            tick = tick - (1 << 256) if tick >= (1 << 255) else tick
            return fx["ticks"][str(tick)]
        if selector in by_selector:
            return by_selector[selector]
        raise _RpcError(f"unknown selector {selector}")

    resp.handlers["eth_call"] = eth_call


def test_fee_growth_at_rows_exact_and_schema_conformant(
    mocked_http, tmp_path: Path
) -> None:
    resp = _Responder(mocked_http)
    fx = _ticks_fixture()
    _install_fee_growth_calls(resp, fx)
    f = _maker(tmp_path)

    table = f.fee_growth_at(POOL, BlockNumber(fx["block_number"]), [196242, 196238])
    validate_table(table, SCHEMA_REGISTRY["fee_growth"])
    assert table.num_rows == 3  # global + 2 ticks

    tick_col = table.column("tick").to_pylist()
    assert tick_col[0] == GLOBAL_TICK_SENTINEL
    assert tick_col[1:] == [196242, 196238]

    row0 = table.slice(0, 1).to_pylist()[0]
    assert row0["fee_growth_outside_0_x128"] is None  # null ON global rows
    assert row0["fee_growth_outside_1_x128"] is None
    assert row0["liquidity_gross"] == "0"  # non-nullable schema placeholders
    assert row0["liquidity_net"] == "0"
    assert row0["initialized"] is False
    assert row0["current_tick"] == 196242
    assert row0["current_liquidity"] == "15000000000000000000"
    assert row0["source"] == "rpc_call"
    assert row0["pool_address"] == str(POOL.address)

    row_tick = table.slice(1, 1).to_pylist()[0]
    # All Q128 values decode back to EXACT integers.
    assert decode_uint(row_tick["fee_growth_global_0_x128"]) == decode_uint(row0["fee_growth_global_0_x128"])
    assert decode_uint(row_tick["fee_growth_global_0_x128"]) == 245472016230501318647318615910821729000
    assert decode_uint(row_tick["liquidity_net"]) == -2_000_000_000_000_000_000  # signed
    assert decode_uint(row_tick["liquidity_gross"]) == 15_000_000_000_000_000_000
    assert row_tick["initialized"] is True
    assert row_tick["source"] == "rpc_call"
    # The outsize tick: uninitialized, zeroed.
    row_uninit = table.slice(2, 1).to_pylist()[0]
    assert row_uninit["initialized"] is False
    assert decode_uint(row_uninit["liquidity_net"]) == 0


def test_fee_growth_at_warm_cache_zero_calls(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    fx = _ticks_fixture()
    _install_fee_growth_calls(resp, fx)
    f = _maker(tmp_path)
    block = BlockNumber(fx["block_number"])
    t1 = f.fee_growth_at(POOL, block, [196242])
    posts_after_cold = resp.n_posts()  # eth_blockNumber + one fee-growth batch
    t2 = f.fee_growth_at(POOL, block, [196242])
    assert t1 == t2
    assert resp.n_posts() == posts_after_cold  # warm: zero network calls


def test_fee_growth_duplicate_ticks_collapsed(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    fx = _ticks_fixture()
    _install_fee_growth_calls(resp, fx)
    table = _maker(tmp_path).fee_growth_at(POOL, BlockNumber(fx["block_number"]), [196242, 196242])
    assert table.num_rows == 2  # global + one row per DISTINCT tick


# ---------------------------------------------------------------------------
# 3. Archive-node absence.
# ---------------------------------------------------------------------------


def test_missing_trie_node_raises_archive_permanent(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    fx = _ticks_fixture()

    def eth_call(params: list[object]) -> object:
        data = str(params[0]["data"])
        if data.startswith(SELECTOR_SLOT0):
            raise _RpcError("missing trie node", -32000)
        return "0x"

    resp.handlers["eth_call"] = eth_call
    with pytest.raises(PermanentFetchError) as excinfo:
        _maker(tmp_path).fee_growth_at(POOL, BlockNumber(fx["block_number"]), [196242])
    message = str(excinfo.value)
    assert "archive node required" in message


def test_state_not_available_also_classified_archive(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    fx = _ticks_fixture()

    def eth_call(params: list[object]) -> object:
        raise _RpcError("state not available", -32000)

    resp.handlers["eth_call"] = eth_call
    with pytest.raises(PermanentFetchError, match="archive node required"):
        _maker(tmp_path).fee_growth_at(POOL, BlockNumber(fx["block_number"]), [196242])


# ---------------------------------------------------------------------------
# 4. Batching: out-of-order ids, per-item errors, batch-per-item fallback.
# ---------------------------------------------------------------------------


def test_batch_out_of_order_reassembly(mocked_http, tmp_path: Path) -> None:
    """A batch response whose items come back in reversed id order is
    reassembled by id — the table rows must come out in input order."""
    resp = _Responder(mocked_http, shuffle=True)  # responder reverses by default
    fx = _ticks_fixture()
    _install_fee_growth_calls(resp, fx)
    f = _maker(tmp_path)
    table = f.fee_growth_at(POOL, BlockNumber(fx["block_number"]), [196242, 196238])
    # If reassembly were positional, the tick rows would be swapped.
    assert table.column("tick").to_pylist() == [
        GLOBAL_TICK_SENTINEL, 196242, 196238
    ]
    assert decode_uint(table.slice(1, 1).to_pylist()[0]["liquidity_net"]) == -2_000_000_000_000_000_000
    # The last request recorded was a single JSON-RPC batch (one HTTP POST):
    # slot0 + liquidity + 2 globals + 2 ticks, all in ONE batch.
    assert any(isinstance(call, list) and len(call) == 6 for call in resp.calls)


def test_batch_per_item_error_raises_for_that_item_only(
    mocked_http, tmp_path: Path
) -> None:
    resp = _Responder(mocked_http)
    fx = _ticks_fixture()
    fail_on = "196238"

    def eth_call(params: list[object]) -> object:
        data = str(params[0]["data"])
        if data.startswith(SELECTOR_TICKS):
            tick = int(data[10:], 16)
            if str(tick) == fail_on:
                raise _RpcError("execution reverted: VM Exception while processing", -32000)
        return "0x"

    resp.handlers["eth_call"] = eth_call
    with pytest.raises(PermanentFetchError) as excinfo:
        _maker(tmp_path).fee_growth_at(POOL, BlockNumber(fx["block_number"]), [196242, 196238])
    # The failing item is named with its params (the brief's requirement).
    assert "ticks(196238)" in str(excinfo.value)


def test_batch_falls_back_to_per_item_calls(mocked_http, tmp_path: Path) -> None:
    """Some providers reject batch POSTs outright; we fall back to individual
    calls and still succeed. Two blocks => one (rejected) batch + two singles."""
    seen_batch = False
    seen_single: list[int] = []

    def callback(request):
        nonlocal seen_batch
        body = json.loads(request.body or "{}")
        if isinstance(body, list):
            seen_batch = True
            return 400, {}, json.dumps(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32000, "message": "batch requests are not supported"}}
            )
        block = int(str(body["params"][0]), 16)
        seen_single.append(block)
        return 200, {"Content-Type": "application/json"}, json.dumps(
            {"jsonrpc": "2.0", "id": body["id"],
             "result": {"number": hex(block), "hash": f"0x{block:064x}",
                        "timestamp": hex(1_600_000_000)}}
        )

    mocked_http.add_callback(
        responses.POST, RPC_URL, callback=callback, content_type="application/json"
    )
    f = _maker(tmp_path)
    got = f._resolve_block_timestamps([101, 202])
    assert seen_batch is True
    assert sorted(seen_single) == [101, 202]
    assert list(got) == [101, 202]


# ---------------------------------------------------------------------------
# 5. Finality guard + warm-cache zero calls.
# ---------------------------------------------------------------------------


def test_finality_guard_refuses_near_tip(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: []
    f = _maker(tmp_path, confirmations=64)
    resp.latest_block = 13386020  # latest-confirmed = 13386020 - 64 = 13385956
    with pytest.raises(PermanentFetchError, match="confirmations"):
        f.fetch(_req("swap", 13385957, 13386000))
    # a window ending exactly at latest-confirmed is fine
    result = f.fetch(_req("swap", 13385950, 13385956))
    assert result.table.num_rows == 0


def test_cache_hit_makes_zero_network_calls(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture("rpc_logs_swap.json")
    f = _maker(tmp_path)
    req = _req("swap", 13385010, 13385012)
    cold = f.fetch(req)
    assert cold.n_requests == 1 and cold.from_cache is False
    posts_after_cold = resp.n_posts()  # eth_blockNumber + logs + headers
    warm = f.fetch(req)
    assert warm.table == cold.table  # byte-identical table
    assert warm.n_requests == 0
    assert warm.from_cache is True
    assert resp.n_posts() == posts_after_cold  # zero calls on the warm path


def test_empty_range_empty_table_and_zero_requests(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    f = _maker(tmp_path)
    result = f.fetch(_req("swap", 100, 50))  # start > end
    assert result.table.num_rows == 0
    assert result.n_requests == 0
    assert resp.n_posts() == 0  # no honest network call at all


# ---------------------------------------------------------------------------
# 6. Reorg guard (verify_chain=True on a cache hit whose block hash changed).
# ---------------------------------------------------------------------------


def test_reorg_detected_on_cached_hash_change(mocked_http, tmp_path: Path) -> None:
    # Cold fetch records hashes `h<block>...` in the sidecar.
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture("rpc_logs_swap.json")
    cold = _maker(tmp_path).fetch(_req("swap", 13385010, 13385011))
    assert cold.table.num_rows == 3

    # The chain reorgs under us: the same blocks now report a different hash,
    # and the pool's logs moved too. A FRESH fetcher (no in-memory caches) on
    # the same cache dir now verifies on the cache hit.
    resp = _Responder(mocked_http)  # new responder, same URL

    def block_by_number(params: list[object]) -> object:
        block = int(str(params[0]), 16)
        return {
            "number": hex(block),
            "hash": "0x00000000000000000000000000000000000000000000000000000000000" + f"{block:03x}",
            "timestamp": hex(1),
        }

    resp.handlers["eth_getBlockByNumber"] = block_by_number
    f = _maker(tmp_path, verify_chain=True)
    with pytest.raises(ReorgDetectedError, match="reorgani"):
        f.fetch(_req("swap", 13385010, 13385011))


def test_verify_chain_off_keeps_cache_hit_zero_call(mocked_http, tmp_path: Path) -> None:
    resp = _Responder(mocked_http)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture("rpc_logs_swap.json")
    f = _maker(tmp_path, verify_chain=False)
    req = _req("swap", 13385010, 13385011)
    f.fetch(req)
    posts = resp.n_posts()
    f.fetch(req)
    assert resp.n_posts() == posts  # verify_chain off: no verification calls


# ---------------------------------------------------------------------------
# 7. Secret hygiene: the RPC URL/key never reaches logs or error messages.
# ---------------------------------------------------------------------------


def test_no_rpc_url_or_key_in_caplog_or_errors(mocked_http, tmp_path: Path, caplog) -> None:
    secret = "SUPERSECRET_TOKEN_42"
    secret_url = f"https://rpc.example.test/rpc/v1/{secret}"

    class _SecretResponder(_Responder):
        pass

    resp = _SecretResponder(mocked_http, url=secret_url)
    resp.handlers["eth_getLogs"] = lambda _params: _fixture("rpc_logs_swap.json")

    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url=secret_url,
        reference_base_url="https://b.test",
        max_concurrency=1,
        request_timeout_s=5.0,
        max_retries=1,
    )
    f = RpcFetcher(endpoints, tmp_path, sleep=_noop_sleep)
    with caplog.at_level("DEBUG"):
        f.fetch(_req("swap", 13385010, 13385011))
        # Force an error path too (a bad JSON-RPC result type) on a block that
        # was NOT already resolved into the instance cache.
        resp.handlers["eth_getBlockByNumber"] = lambda _params: 42
        with pytest.raises(PermanentFetchError) as excinfo:
            f._block_headers([13385012])
    assert secret not in caplog.text
    assert secret_url not in caplog.text
    assert secret not in str(excinfo.value)
    assert secret_url not in str(excinfo.value)