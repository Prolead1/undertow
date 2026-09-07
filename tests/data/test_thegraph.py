"""Tests for ``undertow.data.fetchers.thegraph`` (T05, Route A).

Everything runs offline against ``responses``-mocked gateway POSTs plus the
committed fixtures ``tests/data/fixtures/graph_{swaps,mints,burns,collects}_page.json``.
Those fixtures are synthetic (live capture is impossible in this sandbox — no
``GRAPH_API_KEY``) but internally consistent: sqrt-price/tick values sit on the
CONTRACTS.md §3.0 anchors of the pinned pool (tick ~196256, positive ~+196k),
amounts keep the pool sign convention, and BigDecimal amounts carry token0=6dp /
token1=18dp decimal adjustment. Provenance is recorded in each fixture header.

Covers the T05 brief's twelve required tests plus the extras the acceptance
criteria imply: query-files-as-files, unsupported-stream gating, burn ``sender``
nullability, and a fixedpoint-independent cross-check of the mint rows.

Pagination behaviour under the cursor: a chunk whose first page is ``page_size``
rows keeps paging; a short page or an all-duplicate page terminates; the boundary
overlap is de-duplicated on ``(block_number, log_index)``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import responses

from undertow.data.config import EndpointConfig, default_pools
from undertow.data.fetchers import thegraph
from undertow.data.fetchers.base import FetchRequest
from undertow.data.fetchers.thegraph import SUBGRAPH_ID, TheGraphFetcher
from undertow.data.fixedpoint import amounts_for_liquidity, tick_to_sqrt_price_x96
from undertow.data.schemas import (
    BURN_SCHEMA,
    COLLECT_SCHEMA,
    MINT_SCHEMA,
    SCHEMA_REGISTRY,
    SWAP_SCHEMA,
    validate_table,
)
from undertow.data.types import (
    BlockNumber,
    FetchError,
    PermanentFetchError,
    SchemaViolationError,
    TransientNetworkError,
)

POOL3000 = default_pools()["USDC_WETH_3000"]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GRAPH_SECRET = "topsecret-api-key-9f8e7d"
GRAPH_URL = f"https://gateway.test/api/{GRAPH_SECRET}"
LIQ = "123456789012345678901234567890"  # pool active liquidity carried by every synthetic swap row


def _noop_sleep(_: float) -> None:
    """Injected sleep — offline tests never actually sleep."""


BASE_BLOCK = 12370624  # the fixture block anchor (first swap row)
_BLOCK_TS_STEP = 13  # seconds per block, mainnet cadence (see the fixture headers)


def _block_ts(block: int) -> str:
    """The fixture timestamp convention recorded in the committed fixture
    headers: ts(block) = ts(12370624) + 13 * (block - 12370624), with ts(12370624)
    anchored to 2021-05-08T00:00:00Z."""
    base = int(datetime(2021, 5, 8, tzinfo=UTC).timestamp())
    return str(base + _BLOCK_TS_STEP * (block - BASE_BLOCK))


_ENTITY_FIXTURE = {"swap": "swaps", "mint": "mints", "burn": "burns", "collect": "collects"}


def _load_pages(name: str) -> list[dict]:
    """Load the committed fixture pages for a stream.

    ``name`` may be the singular stream ("swap") or the plural entity ("swaps");
    both resolve to the same ``graph_<entity>_page.json`` file.
    """
    entity = _ENTITY_FIXTURE.get(name, name)
    fixture = json.loads((FIXTURES / f"graph_{entity}_page.json").read_text(encoding="utf-8"))
    return fixture["pages"]


def _make(
    tmp_path: Path,
    *,
    max_concurrency: int = 1,
    max_retries: int = 3,
    graph_url: str = GRAPH_URL,
    seed: int = 7,
) -> TheGraphFetcher:
    endpoints = EndpointConfig(
        graph_url=graph_url,
        rpc_url="https://rpc.test",
        reference_base_url="https://ref.test",
        max_concurrency=max_concurrency,
        request_timeout_s=5.0,
        max_retries=max_retries,
    )
    return TheGraphFetcher(endpoints, tmp_path, sleep=_noop_sleep, rng=random.Random(seed))


def _req(
    stream: str = "swap",
    start: int = 12370624,
    end: int = 12370633,
    pool=POOL3000,
) -> FetchRequest:
    return FetchRequest(
        stream=stream,
        pool=pool,
        start_block=BlockNumber(start),
        end_block=BlockNumber(end),
    )


@pytest.fixture
def mocked_http():
    """Activated ``responses`` mock; unused registered responses never fail the test."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


def _register(fetcher: TheGraphFetcher, rsps, pages: list[dict]) -> None:
    """Queue ``pages`` as consecutive POST responses to the fetcher's endpoint.

    Stale unconsumed registrations for the same URL are dropped first, so a
    fixture's trailing pages (e.g. the committed empty-page case) never leak
    into the next stream's fetch on the shared mock.
    """
    rsps.remove(responses.POST, fetcher._endpoint())
    for page in pages:
        rsps.add(responses.POST, fetcher._endpoint(), json=page, status=200)


def _swap_row(block: int, log_index: int, *, amount0: str = "-1234.567891",
              amount1: str = "+0.411522630333333333") -> dict:
    """One synthetic Swap entity (used only for synthetic-envelope pagination tests)."""
    tx = "0x" + hashlib.sha256(f"{block}:{log_index}".encode()).hexdigest()
    return {
        "id": f"{tx}#{log_index}",
        "transaction": {"id": tx, "blockNumber": str(block)},
        "timestamp": _block_ts(block),
        "logIndex": log_index,
        "pool": {"id": POOL3000.address},
        "amount0": amount0,
        "amount1": amount1,
        "sqrtPriceX96": "1446501726624926496477173928747177",
        "liquidity": LIQ,
        "tick": 196256,
        "sender": "0xe592427a0aece92de3edea1f18e0157c05861564",
        "recipient": "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",
    }


def _page(entities: list[dict], meta: int = 16_210_000) -> dict:
    return {"data": {"swaps": entities, "_meta": {"block": {"number": str(meta)}}}}


# ---------------------------------------------------------------------------
# 1. Fixture -> schema-conforming tables for every supported stream
# ---------------------------------------------------------------------------


def test_fixture_tables_validate_and_streams_supported(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    assert f.supported_streams() == frozenset({"swap", "mint", "burn", "collect"})
    schemas = {
        "swap": SWAP_SCHEMA,
        "mint": MINT_SCHEMA,
        "burn": BURN_SCHEMA,
        "collect": COLLECT_SCHEMA,
    }
    for stream, _schema in schemas.items():
        _register(f, mocked_http, _load_pages(stream))
        result = f.fetch(_req(stream=stream))
        validate_table(result.table, SCHEMA_REGISTRY[stream], strict=True)
        assert result.table.num_rows >= 1, f"fixture for {stream} must be non-empty"
        assert result.route.value == "thegraph"
        assert result.n_requests == 1  # a single 10-block chunk covers the fixture range
        assert result.from_cache is False
        assert result.warnings == ()
        # every row is pool-scoped to the requested pool
        assert set(result.table.column("pool_address").to_pylist()) == {POOL3000.address}
        assert set(result.table.column("event_type").to_pylist()) == {stream}


# ---------------------------------------------------------------------------
# 2. Field-by-field decode of one real (fixture) swap row
# ---------------------------------------------------------------------------


def test_decode_swap_row_field_by_field(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    fixture_rows = _load_pages("swaps")[0]["data"]["swaps"]
    _register(f, mocked_http, _load_pages("swaps"))
    table = f.fetch(_req()).table
    rows = table.to_pylist()
    assert len(rows) == 7
    assert rows[0]["block_number"] == 12370624
    assert rows[0]["log_index"] == 0

    r0 = fixture_rows[0]
    row = rows[0]
    assert row["amount0"] == str(int(Decimal("-1234.567891") * 10**6))
    assert row["amount1"] == str(int(Decimal("+0.411522630333333333") * 10**18))
    assert row["sqrt_price_x96"] == str(int(r0["sqrtPriceX96"]))
    assert row["liquidity"] == str(int(r0["liquidity"]))
    assert row["tick"] == int(r0["tick"])
    assert row["sender"] == r0["sender"].lower()
    assert row["recipient"] == r0["recipient"].lower()
    tx = r0["transaction"]["id"]
    assert row["tx_hash"] == tx
    assert row["tx_hash"] == row["tx_hash"].lower()
    assert len(row["tx_hash"]) == 66 and row["tx_hash"].startswith("0x")
    assert row["pool_address"] == POOL3000.address
    assert row["event_type"] == "swap"
    expected_ts = datetime.fromtimestamp(int(r0["timestamp"]), tz=UTC)
    assert row["block_timestamp"] == expected_ts
    assert row["block_timestamp"].tzinfo is not None
    assert row["block_timestamp"].utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# 3. Raw-amount reconstruction: exact at 6dp and full 18-decimal precision,
#    non-integral BigDecimal values raise SchemaViolationError
# ---------------------------------------------------------------------------


def test_raw_amount_reconstruction_exact_and_non_integral(mocked_http, tmp_path: Path) -> None:
    raw = thegraph._amount_to_raw_units
    # 6dp USDC amount -> exact integer
    assert raw("-1234.567891", 6, field="amount0") == int(Decimal("-1234.567891") * 10**6)
    # full 18-decimal-precision WETH amount -> exact integer (brief test 3)
    assert raw("+0.411522630333333333", 18, field="amount1") == int(
        Decimal("+0.411522630333333333") * 10**18
    )
    assert raw("1200.000000000000000000", 18, field="amount1") == 1200 * 10**18
    assert raw("0.000000000000000001", 18, field="amount1") == 1
    # a value that would require rounding must raise, never silently round
    with pytest.raises(SchemaViolationError):
        raw("-1234.5678915", 6, field="amount0")  # 7 fraction digits on a 6dp token
    with pytest.raises(SchemaViolationError) as excinfo:
        raw("0.0833210802684362519", 18, field="amount1")  # 19 fraction digits
    assert "0.0833210802684362519" in str(excinfo.value)  # offending value carried

    # the full-18-decimal-precision fixture row decodes to the exact integer through
    # the whole pipeline (not just the helper)
    f = _make(tmp_path)
    _register(f, mocked_http, _load_pages("swaps"))
    table = f.fetch(_req()).table
    full = next(
        r for r in table.to_pylist()
        if r["amount1"] == str(int(Decimal("+0.083321080268436251") * 10**18))
    )
    assert full["amount0"] == str(int(Decimal("-250.000000") * 10**6))


def test_non_integral_amount_fails_through_fetch(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    bad = _swap_row(12370624, 0, amount0="-1234.5678915")  # 7 fraction digits on 6dp token
    _register(f, mocked_http, [_page([bad])])
    with pytest.raises(SchemaViolationError) as excinfo:
        f.fetch(_req())
    assert "amount0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Sign convention on a real row: WETH-sell => amount1 > 0, amount0 < 0
#    (T02 orientation sentence: raw pool price is token1-per-token0, so selling
#    WETH/token1 into the pool is positive amount1, negative amount0.)
# ---------------------------------------------------------------------------


def test_swap_sign_convention(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    _register(f, mocked_http, _load_pages("swaps"))
    rows = f.fetch(_req()).table.to_pylist()
    # at least one WETH-sell row with the documented signs
    sells = [r for r in rows if int(r["amount1"]) > 0]
    assert sells, "fixture must contain a WETH-sell row"
    for r in sells:
        assert int(r["amount1"]) > 0 and int(r["amount0"]) < 0
    # and the mirror buys (amount0 > 0, amount1 < 0)
    buys = [r for r in rows if int(r["amount1"]) < 0]
    assert buys, "fixture must contain a WETH-buy row"
    for r in buys:
        assert int(r["amount0"]) > 0 and int(r["amount1"]) < 0
    # contract: amount0/amount1 always have opposite signs on real swaps
    for r in rows:
        a0, a1 = int(r["amount0"]), int(r["amount1"])
        assert (a0 > 0) != (a1 > 0), f"expected strictly opposite signs, got {a0} / {a1}"


# ---------------------------------------------------------------------------
# 5. Cursor pagination: three mocked pages spanning >1000 rows return every row
#    exactly once, keys strictly increasing
# ---------------------------------------------------------------------------


def test_cursor_pagination_spans_over_1000_rows(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    rows_per_block, n_blocks = 100, 26
    corpus: dict[tuple[int, int], dict] = {}
    for b in range(n_blocks):
        for li in range(rows_per_block):
            corpus[(100_000 + b, li)] = _swap_row(
                100_000 + b, li, amount0="-1234.567891", amount1="+0.411522630333333333"
            ) if (b + li) % 2 == 0 else _swap_row(
                100_000 + b, li, amount0="+876.543210", amount1="-0.292011167432130721"
            )

    def cb(request):
        body = json.loads(request.body)
        variables = body["variables"]
        lo, hi, first = (
            int(variables["cursorBlock"]),
            int(variables["endBlock"]),
            int(variables["first"]),
        )
        selected = [
            row
            for (blk, li), row in sorted(corpus.items())
            if lo <= blk <= hi
        ]
        page = selected[:first]
        payload = _page(page, meta=hi)
        return 200, {"Content-Type": "application/json"}, json.dumps(payload)

    mocked_http.add_callback(responses.POST, f._endpoint(), callback=cb)
    result = f.fetch(_req(start=100_000, end=100_025))  # one 4096-block chunk
    assert result.table.num_rows == rows_per_block * n_blocks
    assert result.table.num_rows > 1000
    assert result.n_requests == 1  # the whole range is one base chunk
    keys = list(
        zip(
            result.table.column("block_number").to_pylist(),
            result.table.column("log_index").to_pylist(),
            strict=True,
        )
    )
    assert keys == sorted(keys)  # block-ordered
    assert len(set(keys)) == len(keys) == result.table.num_rows  # every row exactly once
    assert len(mocked_http.calls) == 3  # a full page, a boundary-overlap page, a short page


# ---------------------------------------------------------------------------
# 6. Boundary de-dup: two pages whose cursor overlaps by one row -> once
# ---------------------------------------------------------------------------


def test_boundary_overlap_dedup(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    f.page_size = 3  # small pages so the two-page overlap is cheap to mock
    shared = dict(_swap_row(101, 1))  # the row both pages carry
    page1 = [_swap_row(100, 0), _swap_row(101, 0), shared]
    page2 = [shared, _swap_row(102, 0), _swap_row(102, 1)]
    page3 = [_swap_row(102, 0), _swap_row(102, 1)]  # trailing short page -> end
    _register(f, mocked_http, [_page(page1), _page(page2), _page(page3)])

    rows = f._fetch_chunk(_req(start=100, end=102), 100, 102)
    keys = [
        (int(r["transaction"]["blockNumber"]), int(r["logIndex"])) for r in rows
    ]
    assert keys == [(100, 0), (101, 0), (101, 1), (102, 0), (102, 1)]
    assert len(keys) == len(set(keys)) == 5
    assert len(mocked_http.calls) == 3


def test_mid_block_boundary_recovers_tail_in_any_order(mocked_http, tmp_path: Path) -> None:
    """A page boundary that falls MID-block must not lose the block's tail rows,
    and the de-dup must not assume the re-served block arrives in logIndex order
    (Graph Node's intra-block order is by entity id, i.e. arbitrary logIndex).

    page_size = 3; block 101's subgraph order is [1, 0]:
      page 1 (cursor 100, first=3): (100,0), (100,1), (101,1)  -- ends mid-101
      page 2 (cursor 101, first=3): (101,1), (101,0), (102,0)  -- tail (101,0) + next
      page 3 (cursor 102, first=3): (102,0)                    -- short page -> end
    """
    f = _make(tmp_path)
    f.page_size = 3
    page1 = [_swap_row(100, 0), _swap_row(100, 1), _swap_row(101, 1)]
    page2 = [_swap_row(101, 1), _swap_row(101, 0), _swap_row(102, 0)]
    page3 = [_swap_row(102, 0)]
    _register(f, mocked_http, [_page(page1), _page(page2), _page(page3)])

    rows = f._fetch_chunk(_req(start=100, end=102), 100, 102)
    keys = [(int(r["transaction"]["blockNumber"]), int(r["logIndex"])) for r in rows]
    # arrival order is page order; the tail (101,0) arrives on page 2
    assert keys == [(100, 0), (100, 1), (101, 1), (101, 0), (102, 0)]
    assert len(keys) == len(set(keys)) == 5
    assert sorted(keys) == [(100, 0), (100, 1), (101, 0), (101, 1), (102, 0)]
    assert len(mocked_http.calls) == 3


def test_giant_block_refusal_is_loud_and_duplicate_free(mocked_http, tmp_path: Path) -> None:
    """A block holding >= page_size rows can never be fully served (every page
    re-serves the same leading rows), so the fetcher must refuse LOUDLY — and the
    overlap de-dup must accumulate across the pages that re-serve the block so no
    duplicate rows are ever emitted on the way to the refusal.

    page_size = 3; block 101 holds 4 rows, served in id order [3, 0, 1], so the
    tail (101,2) is unreachable whatever we do.
    """
    f = _make(tmp_path)
    f.page_size = 3
    page_101 = [_swap_row(101, 3), _swap_row(101, 0), _swap_row(101, 1)]
    _register(f, mocked_http, [
        _page([_swap_row(100, 0), *page_101]),
        _page(page_101),
        _page(page_101),
    ])

    with pytest.raises(FetchError) as excinfo:
        f._fetch_chunk(_req(start=100, end=101), 100, 101)
    msg = str(excinfo.value)
    assert "cannot advance" in msg
    assert "101" in msg  # names the unpageable block


def test_pagination_refuses_duplicate_keys(mocked_http, tmp_path: Path) -> None:
    """A page stream that genuinely duplicates a key raises ValidationError."""
    f = _make(tmp_path)
    pages = [_load_pages("swaps")[0]]
    dup = [dict(pages[0]["data"]["swaps"][0]), dict(pages[0]["data"]["swaps"][0])]
    _register(f, mocked_http, [{"data": {"swaps": dup}, "_meta": pages[0]["data"]["_meta"]}])
    with pytest.raises(Exception) as excinfo:
        f.fetch(_req())
    assert "duplicate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 7. GraphQL `errors` in a 200: permanent -> PermanentFetchError (no retry);
#    transient phrase -> retried by the fetcher; exhaustion -> TransientNetworkError
# ---------------------------------------------------------------------------


def test_graphql_errors_permanent_and_transient(mocked_http, tmp_path: Path) -> None:
    # permanent GraphQL error inside a 200 body
    f = _make(tmp_path / "perm")
    _register(f, mocked_http, [{"errors": [{"message": "unknown entity type 'swaps'"}]}])
    with pytest.raises(PermanentFetchError) as excinfo:
        f.fetch(_req())
    assert "unknown entity type" in str(excinfo.value)
    assert len(mocked_http.calls) == 1  # permanent errors are never retried

    # transient-phrased error (rate limit) is retried and then succeeds
    f2 = _make(tmp_path / "retry-ok")
    before = len(mocked_http.calls)
    _register(f2, mocked_http, [
        {"errors": [{"message": "rate limit exceeded, retry later"}]},
        _load_pages("swaps")[0],
    ])
    result = f2.fetch(_req())
    assert result.table.num_rows == 7
    assert len(mocked_http.calls) - before == 2  # error attempt + success

    # transient errors exhausting retries terminate as TransientNetworkError.
    # f3 gets its own cache dir: f2's successful retry must not warm f3's cache.
    f3 = _make(tmp_path / "retry-exhaust", max_retries=1)
    before3 = len(mocked_http.calls)
    _register(f3, mocked_http, [
        {"errors": [{"message": "rate limit exceeded, retry later"}]},
        {"errors": [{"message": "rate limit exceeded, retry later"}]},
    ])
    with pytest.raises(TransientNetworkError):
        f3.fetch(_req())
    assert len(mocked_http.calls) - before3 == 2


# ---------------------------------------------------------------------------
# 8. Indexer lag: _meta.block.number below end_block -> FetchError naming both
# ---------------------------------------------------------------------------


def test_indexer_lag_raises_fetch_error(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    lagged = {"data": {"swaps": [], "_meta": {"block": {"number": "12370000"}}}}
    _register(f, mocked_http, [lagged])
    with pytest.raises(FetchError) as excinfo:
        f.fetch(_req())  # end_block = 12370633
    msg = str(excinfo.value)
    assert "12370000" in msg  # the indexed head
    assert "12370633" in msg  # the requested end
    assert "indexer lag" in msg


# ---------------------------------------------------------------------------
# 9/9b. Empty range and empty data array -> empty_table(SWAP_SCHEMA), no errors
# ---------------------------------------------------------------------------


def test_empty_range_returns_empty_table(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    result = f.fetch(_req(start=500, end=400))
    assert result.table.num_rows == 0
    validate_table(result.table, SWAP_SCHEMA, strict=True)
    assert result.table.schema == SWAP_SCHEMA
    assert len(mocked_http.calls) == 0  # an inverted range never touches the network


def test_empty_data_page_returns_empty_table(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    empty_page = _load_pages("swaps")[1]  # the committed empty data-array page
    assert empty_page["data"]["swaps"] == []
    _register(f, mocked_http, [empty_page])
    result = f.fetch(_req())
    assert result.table.num_rows == 0
    validate_table(result.table, SWAP_SCHEMA, strict=True)


# ---------------------------------------------------------------------------
# 10. Ordering: table is sorted by (block_number, log_index) even though the
#     base class executes chunks concurrently (arrival order is scrambled)
# ---------------------------------------------------------------------------


def test_output_sorted_when_chunks_complete_out_of_order(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path, max_concurrency=4)
    # three chunks: [0,4095], [4096,8191], [8192,9000]
    corpus = {
        500: 0, 2500: 2, 4090: 1,  # chunk 1
        5000: 0, 7000: 3,          # chunk 2
        8500: 1, 8999: 0,          # chunk 3
    }

    def cb(request):
        body = json.loads(request.body)
        v = body["variables"]
        lo, hi, first = int(v["cursorBlock"]), int(v["endBlock"]), int(v["first"])
        selected = [_swap_row(blk, li) for blk, li in sorted(corpus.items()) if lo <= blk <= hi]
        payload = _page(selected[:first], meta=hi)
        return 200, {"Content-Type": "application/json"}, json.dumps(payload)

    mocked_http.add_callback(responses.POST, f._endpoint(), callback=cb)
    result = f.fetch(_req(start=0, end=9000))
    keys = list(
        zip(
            result.table.column("block_number").to_pylist(),
            result.table.column("log_index").to_pylist(),
            strict=True,
        )
    )
    assert keys == sorted(keys), "rows must be sorted by (block_number, log_index)"
    assert len(keys) == len(corpus) == len(set(keys))
    assert result.n_requests == 3  # three disjoint chunks, no loss, no duplication


# ---------------------------------------------------------------------------
# 11. Cache: an identical second fetch makes zero HTTP calls
# ---------------------------------------------------------------------------


def test_second_fetch_hits_cache_zero_requests(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    _register(f, mocked_http, _load_pages("swaps"))
    first = f.fetch(_req())
    assert first.n_requests == 1
    assert first.from_cache is False
    calls_after_first = len(mocked_http.calls)
    second = f.fetch(_req())
    assert second.n_requests == 0
    assert second.from_cache is True
    assert len(mocked_http.calls) == calls_after_first  # zero HTTP calls on the second fetch
    assert second.table.equals(first.table)


# ---------------------------------------------------------------------------
# 12. No API key ever reaches logs or cache filenames
# ---------------------------------------------------------------------------


def test_no_api_key_in_logs_or_cache_filename(mocked_http, tmp_path: Path, caplog) -> None:
    f = _make(tmp_path, graph_url=GRAPH_URL)  # GRAPH_SECRET lives in the URL path
    _register(f, mocked_http, _load_pages("swaps"))
    with caplog.at_level(logging.DEBUG, logger="undertow.data.fetchers.thegraph"):
        result = f.fetch(_req())
    assert result.table.num_rows == 7
    assert GRAPH_SECRET not in caplog.text
    assert "gateway.test" in caplog.text  # only the host is ever logged
    for record in caplog.records:
        assert SUBGRAPH_ID not in record.getMessage()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert GRAPH_SECRET not in path.name, f"secret in cache filename {path.name}"
            assert GRAPH_SECRET not in path.read_bytes().decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Extras implied by the acceptance criteria
# ---------------------------------------------------------------------------


def test_queries_are_files_with_cursor_variables() -> None:
    query_dir = Path(thegraph.__file__).resolve().parent / "queries"
    for entity in ("swaps", "mints", "burns", "collects"):
        text = (query_dir / f"{entity}.graphql").read_text(encoding="utf-8")
        for needle in (
            "$pool: String!",
            "$cursorBlock: BigInt!",
            "$endBlock: BigInt!",
            "$first: Int!",
            "transaction__blockNumber",
            "transaction_: { blockNumber_gte: $cursorBlock",
            "_meta {",
        ):
            assert needle in text, f"{entity}.graphql missing {needle!r}"


def test_fetch_rejects_unsupported_stream_and_missing_pool(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    with pytest.raises(PermanentFetchError) as excinfo:
        f.fetch(_req(stream="flash"))
    assert "flash" in str(excinfo.value)
    with pytest.raises(PermanentFetchError):
        f.fetch(
            FetchRequest(
                stream="swap",
                pool=None,
                start_block=BlockNumber(1),
                end_block=BlockNumber(2),
            )
        )


def test_burn_sender_null_and_mint_rows_independent_crosscheck(
    mocked_http, tmp_path: Path
) -> None:
    # Burn rows carry null sender (the Burn event has no sender field)
    f = _make(tmp_path)
    _register(f, mocked_http, _load_pages("burn"))
    burns = f.fetch(_req(stream="burn")).table.to_pylist()
    assert burns and all(r["sender"] is None for r in burns)
    validate_table(f.fetch(_req(stream="burn")).table, BURN_SCHEMA, strict=True)

    # Mint rows: unsigned amounts on the 60-tick grid; the first fixture row is
    # cross-checked against the T02 fixedpoint ports (independent of the decode path)
    f2 = _make(tmp_path)
    _register(f2, mocked_http, _load_pages("mint"))
    mints = f2.fetch(_req(stream="mint")).table.to_pylist()
    validate_table(f2.fetch(_req(stream="mint")).table, MINT_SCHEMA, strict=True)
    for r in mints:
        assert int(r["amount0"]) >= 0 and int(r["amount1"]) >= 0
        assert r["tick_lower"] % 60 == 0 and r["tick_upper"] % 60 == 0
        assert r["tick_upper"] > r["tick_lower"]
    first = next(r for r in mints if r["tick_lower"] == 195000 and r["tick_upper"] == 198960)
    liquidity = 512345678901234567890123
    a0, a1 = amounts_for_liquidity(
        tick_to_sqrt_price_x96(197100),
        tick_to_sqrt_price_x96(195000),
        tick_to_sqrt_price_x96(198960),
        liquidity,
    )
    assert int(first["liquidity_amount"]) == liquidity
    assert int(first["amount0"]) == a0
    assert int(first["amount1"]) == a1
    assert first["owner"] == "0xc36442b4a4522e871399cd717abdd847ab11fe88"


def test_collect_rows_decode(mocked_http, tmp_path: Path) -> None:
    f = _make(tmp_path)
    _register(f, mocked_http, _load_pages("collect"))
    result = f.fetch(_req(stream="collect"))
    validate_table(result.table, COLLECT_SCHEMA, strict=True)
    rows = result.table.to_pylist()
    assert len(rows) == 3
    for r in rows:
        assert int(r["amount0"]) >= 0 and int(r["amount1"]) >= 0
        assert r["tick_lower"] % 60 == 0 and r["tick_upper"] % 60 == 0
    assert int(rows[0]["amount1"]) == int(Decimal("0.000012345678901234") * 10**18)