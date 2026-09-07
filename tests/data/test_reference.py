"""Tests for ``undertow.data.fetchers.reference`` (T08) — Binance kline feed.

Covers, offline: both fixture shapes -> ``REFERENCE_SCHEMA`` conformance (no block
columns, global not pool-partitioned), integer-exact ms -> UTC-µs conversion and
the ``close_time = open_time + 59_999_000 µs`` interval convention, float64 price
round-tripping, forward-fill-and-flag gap handling (3-minute hole, grid
completeness, leading-edge nulls with NO back-fill), gap statistics in
``FetchResult.warnings``, bulk-archive vs REST API agreement on an overlapping
day, REST pagination, ``compare_symbols`` (returns ``types.CheckResult``),
range clamping with warnings, warm-cache zero-call behaviour, and error paths.

Fixture provenance: **fully synthetic** — live capture is impossible in this
sandbox (no network to Binance). ``binance_klines_api.json`` holds 33 ETHUSDT
1m klines for 2022-08-01T00:00..00:35Z **with minutes 00:20..00:22Z missing**
(the 3-minute hole), in the REST reply shape; ``binance_klines_bulk.zip`` holds
the same first hour (minutes 0..59, same hole) in the ``data.binance.vision``
monthly-CSV shape. Both fixtures are explicitly marked synthetic in their
headers/notes and must never be presented as real captures. All prices are
float64-exact (.25 grid: ``x.0``/``x.25``/``x.5``/``x.75``) and within the
realistic $1600-1760 range.

Fetch windows are **half-open** ``[start_utc, end_utc)`` (matching the REST
API's exclusive ``endTime``): the grid covers every minute whose open time is
``< end``, so a calendar-day request returns exactly 1440 bars and the bar
whose open equals the end belongs to the next day.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pyarrow as pa
import pytest
import requests
import responses

from undertow.data.config import EndpointConfig
from undertow.data.fetchers.base import FetchRequest
from undertow.data.fetchers.reference import (
    BINANCE_VISION_BASE_URL,
    INTERVAL_CLOSE_DELTA_US,
    ReferenceFetcher,
    compare_symbols,
    forward_fill_klines,
    ms_epoch_to_utc,
)
from undertow.data.schemas import REFERENCE_SCHEMA, validate_table
from undertow.data.types import BlockNumber, CheckResult, ConfigError, Route

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
API_FIXTURE = FIXTURES / "binance_klines_api.json"
BULK_FIXTURE = FIXTURES / "binance_klines_bulk.zip"

REST_BASE = "https://binance.test"
T0 = datetime(2022, 8, 1, 0, 0, tzinfo=UTC)  # 2022-08-01T00:00:00Z = 1659312000000 ms
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _epoch_ms(dt: datetime) -> int:
    """Integer-exact epoch-ms for a UTC-aware datetime (test-side mirror)."""
    return ((dt - _EPOCH) // timedelta(microseconds=1)) // 1000


def _noop_sleep(_seconds: float) -> None:
    """Injected sleep so tests never actually block on backoff."""


@pytest.fixture
def mocked_http() -> responses.RequestsMock:
    """Activated ``responses`` mock; unmatched registrations never fail the test."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


def _fetcher(
    tmp_path: Path, *, symbols: tuple[str, ...] = ("ETHUSDT",), max_retries: int = 2
) -> ReferenceFetcher:
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url="https://r.test",
        reference_base_url=REST_BASE,
        max_concurrency=1,
        request_timeout_s=10.0,
        max_retries=max_retries,
    )
    return ReferenceFetcher(
        endpoints,
        tmp_path,
        symbols=symbols,
        sleep=_noop_sleep,
        rng=random.Random(1),
    )


def _req(start: datetime, end: datetime, *, stream: str = "reference") -> FetchRequest:
    return FetchRequest(
        stream=stream,
        pool=None,
        start_block=None,
        end_block=None,
        start_utc=start,
        end_utc=end,
    )


def _load_api_fixture() -> dict:
    return json.loads(API_FIXTURE.read_text(encoding="utf-8"))


def _register_rest(
    mocked_http: responses.RequestsMock, fixture: dict, symbol: str = "ETHUSDT"
) -> None:
    """Serve ``/api/v3/klines`` from the fixture, honouring startTime/endTime.

    The mock slices the fixture by the query window (like Binance does) and
    returns an empty page for any other symbol.
    """
    klines = fixture["klines"]

    def callback(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        q = parse_qs(urlsplit(request.url).query)
        req_symbol = q.get("symbol", [symbol])[0]
        start = int(q.get("startTime", ["0"])[0])
        end = int(q.get("endTime", [str(2**63)])[0])
        if req_symbol != symbol:
            return 200, {}, json.dumps([])
        return 200, {}, json.dumps(
            [k for k in klines if start <= k[0] < end]  # endTime is exclusive, like Binance
        )

    mocked_http.add_callback(
        responses.GET,
        re.compile(re.escape(REST_BASE) + r"/api/v3/klines.*"),
        callback=callback,
    )


def _register_bulk(
    mocked_http: responses.RequestsMock,
    zip_bytes: bytes,
    symbol: str = "ETHUSDT",
    year: int = 2022,
    month: int = 8,
) -> None:
    url = (
        f"{BINANCE_VISION_BASE_URL}/data/spot/monthly/klines/{symbol}/1m/"
        f"{symbol}-1m-{year:04d}-{month:02d}.zip"
    )
    mocked_http.add(responses.GET, url, body=zip_bytes, status=200)


def _close_table(pairs: list[tuple[int, float]]) -> pa.Table:
    """A minimal reference-shaped table with only ``open_time`` + ``close``."""
    opens = pa.array(
        [T0 + timedelta(minutes=m) for m, _ in pairs],
        type=REFERENCE_SCHEMA.field("open_time").type,
    )
    closes = pa.array([c for _, c in pairs], type=pa.float64())
    return pa.table({"open_time": opens, "close": closes})


# ---------------------------------------------------------------------------
# 1. Both fixtures -> REFERENCE_SCHEMA conformance; global stream, no blocks
# ---------------------------------------------------------------------------


def test_fixtures_conform_to_reference_schema(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    api = _load_api_fixture()
    fetcher = _fetcher(tmp_path)
    _register_rest(mocked_http, api)

    rest_res = fetcher.fetch(_req(T0, T0 + timedelta(minutes=35)))
    assert rest_res.route is Route.REFERENCE
    assert rest_res.table.column_names == REFERENCE_SCHEMA.names
    validate_table(rest_res.table, REFERENCE_SCHEMA)  # fetcher validates internally too
    # no block columns, no pool columns — the stream is global (CONTRACTS §4.0/§4.6)
    for banned in ("block_number", "log_index", "block_timestamp", "tx_hash",
                   "pool_address", "event_type"):
        assert banned not in rest_res.table.column_names

    _register_bulk(mocked_http, BULK_FIXTURE.read_bytes())
    bulk_res = fetcher.fetch(_req(T0, T0 + timedelta(hours=1)), access="bulk")
    assert bulk_res.table.column_names == REFERENCE_SCHEMA.names
    validate_table(bulk_res.table, REFERENCE_SCHEMA)
    assert bulk_res.table.num_rows == 60  # grid hour 0 (half-open [00:00, 01:00)), 3 forward-filled


# ---------------------------------------------------------------------------
# 2. ms -> UTC µs exactly; close_time - open_time == 59_999_000 µs
# ---------------------------------------------------------------------------


def test_ms_to_utc_exact_and_interval_delta(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    # the known bar from the fixture header: 2022-08-01T00:00:00Z
    assert ms_epoch_to_utc(1659312000000) == datetime(2022, 8, 1, 0, 0, tzinfo=UTC)
    # sub-minute ms precision carries exact microseconds (1000 µs = 1 ms)
    assert ms_epoch_to_utc(1659312000000 + 1) == datetime(
        2022, 8, 1, 0, 0, 0, 1000, tzinfo=UTC
    )

    _register_rest(mocked_http, _load_api_fixture())
    res = _fetcher(tmp_path).fetch(_req(T0, T0 + timedelta(minutes=36)))
    opens = res.table.column("open_time").to_pylist()
    closes = res.table.column("close_time").to_pylist()
    assert len(opens) == 36
    for o, c in zip(opens, closes, strict=True):
        assert (c - o) // timedelta(microseconds=1) == INTERVAL_CLOSE_DELTA_US
    # and the first bar is exactly the fixture's first bar, in UTC
    assert opens[0] == datetime(2022, 8, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 3. float64 round-trip for realistic ETH price strings
# ---------------------------------------------------------------------------


def test_price_strings_roundtrip_exactly_as_float64(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    api = _load_api_fixture()
    _register_rest(mocked_http, api)
    res = _fetcher(tmp_path).fetch(_req(T0, T0 + timedelta(minutes=36)))

    closes_by_ms = {
        _epoch_ms(dt): close
        for dt, close in zip(
            res.table.column("open_time").to_pylist(),
            res.table.column("close").to_pylist(),
            strict=True,
        )
    }
    for kline in api["klines"]:
        assert closes_by_ms[kline[0]] == float(kline[4])  # source string, bit-exact
    for price_str, expected in [
        ("1000.0", 1000.0),
        ("2000.25", 2000.25),
        ("3000.125", 3000.125),
        ("4000.5", 4000.5),
        ("1648.25", 1648.25),
    ]:
        parsed = float(price_str)
        assert parsed == expected
        assert float.fromhex(parsed.hex()) == parsed  # stable, no rounding drift


# ---------------------------------------------------------------------------
# 4. Gap fill: the 3-minute hole -> 3 flagged bars, prices = last prior close
# ---------------------------------------------------------------------------


def test_three_minute_gap_forward_filled_and_flagged(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_rest(mocked_http, _load_api_fixture())
    res = _fetcher(tmp_path).fetch(_req(T0, T0 + timedelta(minutes=36)))
    table = res.table
    opens = table.column("open_time").to_pylist()

    # complete 1-minute grid over the half-open window, no duplicates
    assert table.num_rows == 36
    assert opens == [T0 + timedelta(minutes=m) for m in range(36)]
    assert len(set(opens)) == 36

    is_filled = table.column("is_gap_filled").to_pylist()
    last_prior_close = table.column("close").to_pylist()[19]  # the 00:19 bar
    for minute in (20, 21, 22):
        i = minute  # rows are dense on the grid
        assert is_filled[i] is True
        assert table.column("close").to_pylist()[i] == last_prior_close
        assert table.column("open").to_pylist()[i] == last_prior_close
        assert table.column("high").to_pylist()[i] == last_prior_close
        assert table.column("low").to_pylist()[i] == last_prior_close
        assert table.column("volume_base").to_pylist()[i] == 0.0
        assert table.column("quote_volume").to_pylist()[i] == 0.0
        assert table.column("trades").to_pylist()[i] == 0
    # observed bars are not flagged
    for minute in (0, 10, 19, 23, 35):
        assert is_filled[minute] is False


# ---------------------------------------------------------------------------
# 5. Leading-edge gap: nulls + warning, never a back-fill (look-ahead guard)
# ---------------------------------------------------------------------------


def test_leading_edge_gap_is_null_and_never_backfilled(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    # pure gap-fill: 5 leading minutes before the first observed bar get null
    # prices (a "no bar" placeholder) — never a value copied from later
    rows = [
        {
            "symbol": "ETHUSDT",
            "open_time_ms": _epoch_ms(T0),
            "open": 1648.0,
            "high": 1650.0,
            "low": 1647.0,
            "close": 1648.25,
            "volume_base": 10.0,
            "quote_volume": 16482.5,
            "trades": 42,
        }
    ]
    grid, stats = forward_fill_klines(
        rows, start_utc=T0 - timedelta(minutes=5), end_utc=T0 + timedelta(minutes=5)
    )
    leading = [r for r in grid if r["close"] is None]
    assert len(leading) == 5
    for r in leading:
        assert r["open"] is None and r["high"] is None
        assert r["low"] is None and r["close"] is None
        assert r["is_gap_filled"] is False  # not forward-filled — it has no prior bar
    assert stats["ETHUSDT"].leading_missing == 5

    # fetch level: the returned (schema-validated) table starts at the first
    # observed bar — nothing was copied backwards in time, and a warning says so
    _register_rest(mocked_http, _load_api_fixture())
    res = _fetcher(tmp_path).fetch(
        _req(T0 - timedelta(minutes=5), T0 + timedelta(minutes=6))
    )
    opens = res.table.column("open_time").to_pylist()
    assert opens[0] == T0  # clamped to the first observed bar
    assert all(dt >= T0 for dt in opens)  # no pre-coverage rows at all
    assert len(opens) == 6  # bars 00:00..00:05; window is half-open at 00:06
    warnings = "\n".join(res.warnings)
    assert "leading edge" in warnings
    assert "no back-fill" in warnings


# ---------------------------------------------------------------------------
# 6. Gap statistics surface in FetchResult.warnings
# ---------------------------------------------------------------------------


def test_gap_stats_in_warnings(tmp_path: Path, mocked_http: responses.RequestsMock) -> None:
    _register_rest(mocked_http, _load_api_fixture())
    res = _fetcher(tmp_path).fetch(_req(T0, T0 + timedelta(minutes=35)))
    warnings = "\n".join(res.warnings)
    assert "filled 3 minute bar(s)" in warnings
    assert "longest run 3" in warnings


# ---------------------------------------------------------------------------
# 7. Grid completeness: a 24h window yields exactly 1440 consecutive bars
# ---------------------------------------------------------------------------


def test_24h_window_has_exactly_1440_rows(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_rest(mocked_http, _load_api_fixture())
    res = _fetcher(tmp_path).fetch(_req(T0, T0 + timedelta(days=1)))
    table = res.table
    assert table.num_rows == 1440
    opens = table.column("open_time").to_pylist()
    assert len(set(opens)) == 1440  # every minute exactly once
    assert opens == [T0 + timedelta(minutes=m) for m in range(1440)]  # dense grid
    is_filled = table.column("is_gap_filled").to_pylist()
    assert is_filled.count(True) == 1440 - 33  # 33 observed bars in the fixture
    assert is_filled.count(False) == 33


# ---------------------------------------------------------------------------
# 8. Bulk-archive and REST paths agree on an overlapping day (mocked)
# ---------------------------------------------------------------------------


def test_bulk_and_rest_agree_on_overlap(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_rest(mocked_http, _load_api_fixture())
    _register_bulk(mocked_http, BULK_FIXTURE.read_bytes())

    fetcher = _fetcher(tmp_path)
    window = _req(T0, T0 + timedelta(minutes=35))
    rest_res = fetcher.fetch(window)  # auto -> REST (single partial month)
    bulk_res = fetcher.fetch(window, access="bulk")  # whole month zip, trimmed

    assert rest_res.table.equals(bulk_res.table)
    assert rest_res.n_requests == 1
    assert bulk_res.n_requests == 1
    urls = [c.request.url for c in mocked_http.calls]
    assert any("api/v3/klines" in u for u in urls)
    assert any(BINANCE_VISION_BASE_URL in u for u in urls)


# ---------------------------------------------------------------------------
# 9. compare_symbols -> types.CheckResult; fractions and edge cases
# ---------------------------------------------------------------------------


def test_compare_symbols_identical_and_diverged() -> None:
    pairs_a = [(m, 1600.0 + m * 0.25) for m in range(100)]

    identical = compare_symbols(_close_table(pairs_a), _close_table(pairs_a))
    assert isinstance(identical, CheckResult)
    assert identical.passed is True
    assert identical.severity == "warning"
    assert identical.metrics["bars_compared"] == 100
    assert identical.metrics["bars_diverged"] == 0
    assert identical.metrics["frac_diverged"] == 0.0
    assert identical.metrics["max_rel_diff_bps"] == 0.0

    # 100bp divergence on exactly 10 of 100 bars -> frac_diverged == 0.10
    pairs_b = [
        (m, (1600.0 + m * 0.25) * 1.01 if m >= 90 else 1600.0 + m * 0.25)
        for m in range(100)
    ]
    diverged = compare_symbols(_close_table(pairs_a), _close_table(pairs_b))
    assert diverged.metrics["bars_compared"] == 100
    assert diverged.metrics["bars_diverged"] == 10
    assert abs(diverged.metrics["frac_diverged"] - 0.10) < 1e-9
    assert diverged.passed is False
    assert "100.00 bps" in diverged.detail or "100 bps" in diverged.detail

    # zero overlapping bars -> failed with a clear reason, still a CheckResult
    disjoint = compare_symbols(
        _close_table(pairs_a), _close_table([(m, 1.0) for m in range(200, 300)])
    )
    assert isinstance(disjoint, CheckResult)
    assert disjoint.passed is False
    assert disjoint.metrics["bars_compared"] == 0


# ---------------------------------------------------------------------------
# 10. Range clamping: out-of-coverage requests warn instead of silently shrinking
# ---------------------------------------------------------------------------


def test_out_of_coverage_requests_warn_and_clamp(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_rest(mocked_http, _load_api_fixture())
    fetcher = _fetcher(tmp_path)

    # leading edge: window begins 30 minutes before any coverage
    res = fetcher.fetch(
        _req(T0 - timedelta(minutes=30), T0 + timedelta(minutes=6))
    )
    opens = res.table.column("open_time").to_pylist()
    assert opens[0] == T0  # clamped: 6 rows (00:00..00:05), never fabricated earlier
    assert "leading edge" in "\n".join(res.warnings)

    # entirely outside coverage: empty table and a loud warning — not silence
    res2 = fetcher.fetch(_req(T0 - timedelta(days=30), T0 - timedelta(days=29)))
    assert res2.table.num_rows == 0
    assert "no observed bars" in "\n".join(res2.warnings)


# ---------------------------------------------------------------------------
# 11. Warm cache -> zero HTTP calls
# ---------------------------------------------------------------------------


def test_warm_cache_makes_zero_http_calls(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_rest(mocked_http, _load_api_fixture())
    fetcher = _fetcher(tmp_path)
    window = _req(T0, T0 + timedelta(minutes=35))

    cold = fetcher.fetch(window)
    assert cold.n_requests == 1
    assert cold.from_cache is False
    assert len(mocked_http.calls) == 1

    warm = fetcher.fetch(window)
    assert warm.n_requests == 0
    assert warm.from_cache is True
    assert len(mocked_http.calls) == 1  # no new HTTP call
    assert warm.table.equals(cold.table)


# ---------------------------------------------------------------------------
# Extras: pagination, multi-symbol, error paths, fixture provenance
# ---------------------------------------------------------------------------


def test_rest_pagination_crosses_1000_bar_pages(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    # a synthetic 1440-bar stream served with the real 1000-bar page limit
    def callback(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        q = parse_qs(urlsplit(request.url).query)
        start = int(q["startTime"][0])
        end = int(q["endTime"][0])
        rows: list[list[object]] = []
        t = start
        while t <= end and len(rows) < 1000:
            minute = (t - _epoch_ms(T0)) // 60_000
            if 0 <= minute < 1440:
                c = 1600.0 + (minute % 120) * 0.25
                rows.append(
                    [t, f"{c:.2f}", f"{c:.2f}", f"{c:.2f}", f"{c:.2f}",
                     "1.0", t + 59_999, f"{c:.2f}", 1, "0.5", f"{c / 2:.2f}", "0"]
                )
            t += 60_000
        return 200, {}, json.dumps(rows)

    mocked_http.add_callback(
        responses.GET,
        re.compile(re.escape(REST_BASE) + r"/api/v3/klines.*"),
        callback=callback,
    )
    fetcher = _fetcher(tmp_path)
    res = fetcher.fetch(_req(T0, T0 + timedelta(days=1)))
    assert res.n_requests == 2  # 1000-bar page + 440-bar page
    assert res.table.num_rows == 1440
    assert res.table.column("is_gap_filled").to_pylist().count(True) == 0


def test_all_configured_symbols_fetch_missing_symbol_warns(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_rest(mocked_http, _load_api_fixture())  # serves only ETHUSDT
    fetcher = _fetcher(tmp_path, symbols=("ETHUSDT", "ETHUSDC"))
    res = fetcher.fetch(_req(T0, T0 + timedelta(minutes=35)))
    symbols = sorted(set(res.table.column("symbol").to_pylist()))
    assert symbols == ["ETHUSDT"]  # ETHUSDC: no rows, but a warning — never silent
    assert "ETHUSDC: no observed bars" in "\n".join(res.warnings)
    # two symbols requested => two source fetches happen (both cached on rerun)


def test_fetch_requires_date_window_and_reference_stream(tmp_path: Path) -> None:
    fetcher = _fetcher(tmp_path)
    with pytest.raises(ConfigError):
        fetcher.fetch(
            FetchRequest(
                stream="reference",
                pool=None,
                start_block=BlockNumber(1),
                end_block=BlockNumber(2),
            )
        )
    with pytest.raises(ConfigError):
        fetcher.fetch(_req(T0, T0 + timedelta(minutes=1), stream="swap"))
    with pytest.raises(ConfigError):
        fetcher.fetch(_req(T0 + timedelta(minutes=10), T0))  # start > end


def test_fetch_block_machinery_not_usable(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        _fetcher(tmp_path)._fetch_chunk(
            FetchRequest(
                stream="reference",
                pool=None,
                start_block=None,
                end_block=None,
                start_utc=T0,
                end_utc=T0 + timedelta(minutes=1),
            ),
            0,
            1,
        )


def test_fixtures_are_marked_synthetic() -> None:
    api = json.loads(API_FIXTURE.read_text(encoding="utf-8"))
    assert "_synthetic_note" in api
    assert "synthetic" in api["_synthetic_note"].lower()
    assert api["symbol"] == "ETHUSDT"

    with zipfile.ZipFile(BULK_FIXTURE) as zf:
        names = zf.namelist()
        assert names and names[0] == "ETHUSDT-1m-2022-08.csv"
        first_lines = zf.read(names[0]).decode("utf-8-sig").splitlines()
    assert first_lines[0].startswith("# SYNTHETIC")
    assert "live capture unavailable" in first_lines[0]


def test_corrupt_cache_entry_is_a_miss(tmp_path: Path, mocked_http: responses.RequestsMock) -> None:
    fetcher = _fetcher(tmp_path)
    window = _req(T0, T0 + timedelta(minutes=36))
    key = fetcher._ref_cache_key(
        {"symbol": "ETHUSDT", "path": "rest", "start_ms": _epoch_ms(T0),
         "end_ms": _epoch_ms(T0 + timedelta(minutes=36))}
    )
    path = fetcher._ref_cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("not gzip json at all")

    _register_rest(mocked_http, _load_api_fixture())
    res = fetcher.fetch(window)
    assert res.n_requests == 1  # corrupt entry treated as a miss, re-fetched
    assert res.table.num_rows == 36


def test_bulk_404_falls_back_to_rest(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    """A bulk month that has not been published yet (HTTP 404 archive) falls
    back to the REST API for the window's sub-range, with a loud warning."""
    _register_rest(mocked_http, _load_api_fixture())
    bulk_url = (
        f"{BINANCE_VISION_BASE_URL}/data/spot/monthly/klines/ETHUSDT/1m/"
        "ETHUSDT-1m-2022-08.zip"
    )
    mocked_http.add(responses.GET, bulk_url, status=404)

    res = _fetcher(tmp_path).fetch(_req(T0, T0 + timedelta(hours=1)), access="bulk")

    assert res.table.num_rows == 60  # full hour grid, filled from REST bars
    assert any("fell back to REST API" in w for w in res.warnings)
    urls = [c.request.url for c in mocked_http.calls]
    assert any("api/v3/klines" in u for u in urls)
    assert any(bulk_url in u for u in urls)