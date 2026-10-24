"""Tests for ``undertow.data.fetchers.funding`` (S19) — Binance USDT-perp funding.

Covers, offline against synthetic recorded-page payloads: ``FUNDING_SCHEMA``
conformance (dtypes, nullable ``mark_price``, the ``is_gap_filled=False``
default), millisecond-jitter normalization onto the 8 h settlement grid, REST
pagination across two pages with boundary de-duplication, forward-fill-and-flag
gap handling (a complete series is never flagged), inclusive window clamping with
a typed empty frame for an empty window, the ``load_funding`` bridge loader's
sort order and determinism, and the typed error paths (non-2xx page, malformed
``fundingRate``/``fundingTime``, bad requests).

Fixture provenance: **fully synthetic** — no live capture is committed. Every
record is marked by construction; the one live test is ``@pytest.mark.network``
and skipped unless ``--run-network`` is passed.
"""

from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import polars as pl
import pyarrow as pa
import pytest
import requests
import responses

from undertow.data.config import (
    DataConfig,
    EndpointConfig,
    RegimeConfig,
    WindowConfig,
    default_pools,
)
from undertow.data.fetchers.base import FetchRequest
from undertow.data.fetchers.funding import (
    FUNDING_INTERVAL_MS,
    REST_LIMIT,
    FundingFetcher,
    floor_funding_ms,
    forward_fill_funding,
    ms_epoch_to_utc,
)
from undertow.data.loader import load_funding
from undertow.data.schemas import FUNDING_SCHEMA, validate_table
from undertow.data.storage.parquet import write_stream
from undertow.data.types import ConfigError, PermanentFetchError, Route

FAPI_BASE = "https://fapi.test"
T0 = datetime(2022, 1, 1, 0, 0, tzinfo=UTC)  # 2022-01-01T00:00:00Z
T0_MS = 1640995200000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

_NEEDLE = re.escape(FAPI_BASE) + r"/fapi/v1/fundingRate.*"


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


def _fetcher(tmp_path: Path, *, symbol: str = "ETHUSDT", max_retries: int = 1) -> FundingFetcher:
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url="https://r.test",
        reference_base_url="https://api.binance.com",
        futures_base_url=FAPI_BASE,
        max_concurrency=1,
        request_timeout_s=10.0,
        max_retries=max_retries,
    )
    return FundingFetcher(
        endpoints, tmp_path, symbol=symbol, sleep=_noop_sleep, rng=random.Random(1)
    )


def _req(start: datetime, end: datetime, *, stream: str = "funding") -> FetchRequest:
    return FetchRequest(
        stream=stream,
        pool=None,
        start_block=None,
        end_block=None,
        start_utc=start,
        end_utc=end,
    )


def _record(ms: int, rate: str = "0.00010000", mark: str | None = None) -> dict:
    """One synthetic Binance fundingRate record (``mark=None`` -> the live ``""``)."""
    return {
        "symbol": "ETHUSDT",
        "fundingTime": ms,
        "fundingRate": rate,
        "markPrice": "" if mark is None else mark,
        "rateType": "Regular",
    }


def _series(n: int, *, start_ms: int = T0_MS, missing: set[int] | None = None) -> list[dict]:
    """``n`` observed settlements on the 8 h grid; ``missing`` indexes are omitted."""
    missing = missing or set()
    return [
        _record(start_ms + i * FUNDING_INTERVAL_MS, f"{0.0001 + i * 0.000001:.8f}", f"{3000.5 + i}")
        for i in range(n)
        if i not in missing
    ]


def _register_funding(
    mocked_http: responses.RequestsMock, records: list[dict], *, symbol: str = "ETHUSDT"
) -> None:
    """Serve ``/fapi/v1/fundingRate`` from ``records``, honouring startTime/endTime/limit.

    The selection deliberately starts one interval before ``startTime`` so a page
    boundary repeats the previous page's last record — exactly the overlap the
    fetcher's ``(symbol, funding_time)`` de-duplication must absorb.
    """
    limit_records = records

    def callback(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        q = parse_qs(urlsplit(request.url).query)
        if q.get("symbol", [symbol])[0] != symbol:
            return 200, {}, json.dumps([])
        start = int(q.get("startTime", ["0"])[0])
        end = int(q.get("endTime", [str(2**63)])[0])
        limit = int(q.get("limit", [str(REST_LIMIT)])[0])
        window = [
            r for r in limit_records if start - FUNDING_INTERVAL_MS <= r["fundingTime"] <= end
        ]
        return 200, {}, json.dumps(window[:limit])

    mocked_http.add_callback(responses.GET, re.compile(_NEEDLE), callback=callback)


def _register_raw(
    mocked_http: responses.RequestsMock, records: list[dict], *, symbol: str = "ETHUSDT"
) -> None:
    """Serve every record regardless of startTime/endTime — the window-clamp mock."""

    def callback(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        q = parse_qs(urlsplit(request.url).query)
        if q.get("symbol", [symbol])[0] != symbol:
            return 200, {}, json.dumps([])
        return 200, {}, json.dumps(records)

    mocked_http.add_callback(responses.GET, re.compile(_NEEDLE), callback=callback)


# ---------------------------------------------------------------------------
# 1. Schema conformance: dtypes, the is_gap_filled default, jitter removal
# ---------------------------------------------------------------------------


def test_fetched_rows_conform_to_funding_schema(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    records = [
        _record(T0_MS + 6, "0.00027858", "2283.75"),  # 6 ms jitter -> 00:00 grid
        _record(T0_MS + FUNDING_INTERVAL_MS + 12, "-0.00001234", None),  # empty markPrice
    ]
    _register_funding(mocked_http, records)
    fetcher = _fetcher(tmp_path)

    res = fetcher.fetch(_req(T0, T0 + timedelta(hours=8)))

    assert res.route is Route.REFERENCE
    assert res.table.column_names == FUNDING_SCHEMA.names
    validate_table(res.table, FUNDING_SCHEMA)  # the fetcher validates internally too
    assert res.table.schema.field("funding_time").type == pa.timestamp("us", tz="UTC")
    assert res.table.schema.field("symbol").type == pa.large_string()
    assert res.table.schema.field("funding_rate").type == pa.float64()
    assert res.table.schema.field("mark_price").type == pa.float64()
    assert res.table.schema.field("is_gap_filled").type == pa.bool_()
    assert res.table.column("funding_time").to_pylist() == [
        T0,  # jitter removed: the scheduled settlement instant
        T0 + timedelta(hours=8),
    ]
    assert res.table.column("symbol").to_pylist() == ["ETHUSDT", "ETHUSDT"]
    assert res.table.column("funding_rate").to_pylist() == [0.00027858, -0.00001234]
    assert res.table.column("mark_price").to_pylist() == [2283.75, None]
    # Observed rows default to not-filled; gaps are the only source of True.
    assert res.table.column("is_gap_filled").to_pylist() == [False, False]


# ---------------------------------------------------------------------------
# 2. Pagination: >1000 records across two pages, no duplicates, sorted
# ---------------------------------------------------------------------------


def test_pagination_spans_two_pages_without_duplicates(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    records = _series(1201)
    _register_funding(mocked_http, records)
    fetcher = _fetcher(tmp_path)
    end = T0 + timedelta(hours=8 * 1200)

    res = fetcher.fetch(_req(T0, end))

    assert res.n_requests == 2  # 1000 + 201 => two pages, second short
    assert res.from_cache is False
    assert res.table.num_rows == 1201
    times = res.table.column("funding_time").to_pylist()
    assert times == sorted(times)
    assert len(set(times)) == 1201  # boundary overlap deduplicated
    assert not any(res.table.column("is_gap_filled").to_pylist())
    assert res.warnings == ()


# ---------------------------------------------------------------------------
# 3. Gap fill: a missing 8 h settlement is forward-filled and flagged
# ---------------------------------------------------------------------------


def test_gap_fill_forward_fills_and_flags_missing_settlement(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    records = _series(5, missing={3})
    _register_funding(mocked_http, records)
    fetcher = _fetcher(tmp_path)

    res = fetcher.fetch(_req(T0, T0 + timedelta(hours=8 * 4)))

    assert res.table.num_rows == 5  # the hole in the middle is re-inserted
    flags = res.table.column("is_gap_filled").to_pylist()
    assert flags == [False, False, False, True, False]
    times = res.table.column("funding_time").to_pylist()
    assert times == [T0 + timedelta(hours=8 * i) for i in range(5)]
    rates = res.table.column("funding_rate").to_pylist()
    marks = res.table.column("mark_price").to_pylist()
    # The filled row carries the prior observation's rate and mark.
    assert rates[3] == rates[2]
    assert marks[3] == marks[2]
    assert any("gap-fill" in w for w in res.warnings)


def test_complete_series_has_no_gap_fills(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_funding(mocked_http, _series(5))
    fetcher = _fetcher(tmp_path)

    res = fetcher.fetch(_req(T0, T0 + timedelta(hours=8 * 4)))

    assert res.table.num_rows == 5
    assert res.table.column("is_gap_filled").to_pylist() == [False] * 5
    assert not any("gap-fill" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# 4. Window clamp: outside records dropped; empty window -> typed empty frame
# ---------------------------------------------------------------------------


def test_window_clamp_drops_outside_records_and_empty_window_is_typed(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    # The mock ignores startTime/endTime, so the fetcher must clamp client-side.
    outside_and_inside = [
        _record(T0_MS - FUNDING_INTERVAL_MS, "0.00090000", "1111"),
        _record(T0_MS, "0.00010000", "2222"),
        _record(T0_MS + 2 * FUNDING_INTERVAL_MS, "0.00080000", "3333"),
    ]
    _register_raw(mocked_http, outside_and_inside)
    fetcher = _fetcher(tmp_path)

    clamped = fetcher.fetch(_req(T0, T0 + timedelta(hours=8)))
    assert clamped.table.num_rows == 1
    assert clamped.table.column("funding_time").to_pylist() == [T0]
    assert clamped.table.column("funding_rate").to_pylist() == [0.0001]
    assert clamped.table.column("mark_price").to_pylist() == [2222.0]

    empty = fetcher.fetch(_req(T0 + timedelta(hours=20), T0 + timedelta(hours=24)))
    assert empty.table.num_rows == 0
    assert empty.table.schema == FUNDING_SCHEMA
    validate_table(empty.table, FUNDING_SCHEMA)
    assert any("no funding settlements" in w for w in empty.warnings)


# ---------------------------------------------------------------------------
# 5. Loader: documented sort order; deterministic across two calls
# ---------------------------------------------------------------------------


def _funding_table() -> pa.Table:
    """A deliberately out-of-order two-symbol funding table."""
    rows = [
        {
            "funding_time": T0 + timedelta(hours=16),
            "symbol": "ETHUSDT",
            "funding_rate": 0.0003,
            "mark_price": 3100.0,
            "is_gap_filled": False,
        },
        {
            "funding_time": T0,
            "symbol": "ETHUSDT",
            "funding_rate": 0.0001,
            "mark_price": 3000.0,
            "is_gap_filled": False,
        },
        {
            "funding_time": T0 + timedelta(hours=8),
            "symbol": "ETHUSDC",
            "funding_rate": -0.0002,
            "mark_price": None,
            "is_gap_filled": True,
        },
        {
            "funding_time": T0,
            "symbol": "ETHUSDC",
            "funding_rate": -0.0001,
            "mark_price": 2999.0,
            "is_gap_filled": False,
        },
    ]
    return pa.Table.from_pylist(rows, schema=FUNDING_SCHEMA)


def _loader_config(output_dir: Path, cache_dir: Path) -> DataConfig:
    return DataConfig(
        pool=default_pools()["USDC_WETH_3000"],
        window=WindowConfig(
            start_utc=datetime(2022, 1, 1, tzinfo=UTC),
            end_utc=datetime(2022, 1, 2, tzinfo=UTC),
        ),
        regime=RegimeConfig(),
        endpoints=EndpointConfig(
            graph_url="https://g.test",
            rpc_url="https://r.test",
            reference_base_url="https://api.binance.com",
        ),
        cache_dir=cache_dir,
        output_dir=output_dir,
    )


def test_load_funding_is_sorted_and_deterministic(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    write_stream(
        _funding_table(),
        output_dir,
        "funding",
        pool=default_pools()["USDC_WETH_3000"],
        route=Route.REFERENCE,
    )
    config = _loader_config(output_dir, tmp_path / "cache")

    first = load_funding(config)
    second = load_funding(config)

    assert isinstance(first, pl.DataFrame)
    assert first.columns == list(FUNDING_SCHEMA.names)
    assert first.equals(second)  # deterministic across calls on the same cache
    assert first.height == 4
    # Documented sort order: symbol first, then funding_time within each symbol.
    assert first["symbol"].to_list() == ["ETHUSDC", "ETHUSDC", "ETHUSDT", "ETHUSDT"]
    assert first["funding_time"].to_list() == [
        T0,
        T0 + timedelta(hours=8),
        T0,
        T0 + timedelta(hours=16),
    ]
    assert first.sort(["symbol", "funding_time"]).equals(first)


def test_load_funding_absent_stream_is_typed_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    config = _loader_config(output_dir, tmp_path / "cache")

    frame = load_funding(config)

    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 0
    assert frame.columns == list(FUNDING_SCHEMA.names)


# ---------------------------------------------------------------------------
# 6. Error paths: typed HTTP error; malformed record reported with the record
# ---------------------------------------------------------------------------


def test_non_2xx_page_raises_typed_error(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    mocked_http.add_callback(
        responses.GET,
        re.compile(_NEEDLE),
        callback=lambda request: (404, {}, "not found"),
    )
    fetcher = _fetcher(tmp_path)

    with pytest.raises(PermanentFetchError):
        fetcher.fetch(_req(T0, T0 + timedelta(hours=8)))


def test_malformed_funding_rate_reports_offending_record(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    bad = _record(T0_MS, "not-a-number", "3000.0")
    _register_raw(mocked_http, [bad])
    fetcher = _fetcher(tmp_path)

    with pytest.raises(PermanentFetchError) as exc:
        fetcher.fetch(_req(T0, T0 + timedelta(hours=8)))

    message = str(exc.value)
    assert "fundingRate" in message
    assert "not-a-number" in message
    assert str(T0_MS) in message  # the offending record is quoted, not just the field


@pytest.mark.parametrize(
    ("record", "needle"),
    [
        (_record(T0_MS, "0.0001", "3000.0") | {"fundingTime": "not-a-time"}, "not-a-time"),
        (_record(T0_MS, "0.0001", "3000.0") | {"fundingTime": -1}, "-1"),
        (_record(T0_MS, "nan", "3000.0"), "nan"),
        (_record(T0_MS, "0.0001", "not-a-price"), "not-a-price"),
    ],
    ids=["bad-time", "negative-time", "non-finite-rate", "bad-mark"],
)
def test_malformed_records_raise_with_record(
    tmp_path: Path, mocked_http: responses.RequestsMock, record: dict, needle: str
) -> None:
    _register_raw(mocked_http, [record])
    fetcher = _fetcher(tmp_path)

    with pytest.raises(PermanentFetchError) as exc:
        fetcher.fetch(_req(T0, T0 + timedelta(hours=8)))

    assert needle in str(exc.value)
    assert "ETHUSDT" in str(exc.value)


def test_bad_requests_are_config_errors(tmp_path: Path) -> None:
    fetcher = _fetcher(tmp_path)
    # Wrong stream.
    with pytest.raises(ConfigError, match="unsupported stream"):
        fetcher.fetch(_req(T0, T0 + timedelta(hours=8), stream="reference"))
    # Missing date bounds (block-range request).
    with pytest.raises(ConfigError, match="requires start_utc/end_utc"):
        fetcher.fetch(FetchRequest(stream="funding", pool=None, start_block=1, end_block=2))
    # Inverted window.
    with pytest.raises(ConfigError, match="must be <="):
        fetcher.fetch(_req(T0 + timedelta(hours=8), T0))
    # The block-range machinery is deliberately unreachable.
    with pytest.raises(NotImplementedError):
        fetcher._fetch_chunk(_req(T0, T0), 0, 1)
    # Symbol validation happens at construction.
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url="https://r.test",
        reference_base_url="https://api.binance.com",
    )
    with pytest.raises(ConfigError, match="invalid symbol"):
        FundingFetcher(endpoints, tmp_path, symbol="not a pair!")
    # An empty futures base URL is a config error, not a silent request to nowhere.
    empty_endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url="https://r.test",
        reference_base_url="https://api.binance.com",
        futures_base_url="",
    )
    with pytest.raises(ConfigError, match="futures_base_url is empty"):
        FundingFetcher(empty_endpoints, tmp_path).fetch(_req(T0, T0 + timedelta(hours=8)))


def test_helpers_reject_bool_and_floor_to_grid() -> None:
    with pytest.raises(ConfigError, match="expected an epoch-millisecond int"):
        ms_epoch_to_utc(True)
    assert floor_funding_ms(T0_MS + 31) == T0_MS
    assert floor_funding_ms(T0_MS + FUNDING_INTERVAL_MS) == T0_MS + FUNDING_INTERVAL_MS


def test_warm_cache_makes_zero_requests(
    tmp_path: Path, mocked_http: responses.RequestsMock
) -> None:
    _register_funding(mocked_http, _series(3))
    fetcher = _fetcher(tmp_path)

    cold = fetcher.fetch(_req(T0, T0 + timedelta(hours=16)))
    warm = fetcher.fetch(_req(T0, T0 + timedelta(hours=16)))

    assert cold.n_requests == 1
    assert cold.from_cache is False
    assert warm.n_requests == 0
    assert warm.from_cache is True
    assert warm.table.equals(cold.table)


def test_forward_fill_marks_leading_edge_absent() -> None:
    # A settlement at the first grid instant is missing, but the later one is
    # observed: the leading hole is absent (no back-fill), not invented.
    rows = [
        {
            "symbol": "ETHUSDT",
            "funding_time_ms": T0_MS + FUNDING_INTERVAL_MS,
            "funding_rate": 0.0001,
            "mark_price": 3000.0,
            "is_gap_filled": False,
        }
    ]
    grid, stats = forward_fill_funding(rows, start_ms=T0_MS, end_ms=T0_MS + FUNDING_INTERVAL_MS)
    assert [r["funding_time_ms"] for r in grid] == [T0_MS + FUNDING_INTERVAL_MS]
    assert stats.n_observed == 1
    assert stats.n_filled == 0
    assert stats.leading_missing == 1


# ---------------------------------------------------------------------------
# Live endpoint (network-marked; skipped unless --run-network)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_funding_endpoint_returns_schema_conformant_rows(tmp_path: Path) -> None:
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url="https://r.test",
        reference_base_url="https://api.binance.com",
        futures_base_url="https://fapi.binance.com",
    )
    fetcher = FundingFetcher(
        endpoints, tmp_path, symbol="ETHUSDT", sleep=_noop_sleep, rng=random.Random(1)
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)

    res = fetcher.fetch(_req(start, start + timedelta(hours=23, minutes=59)))

    assert res.table.num_rows == 3  # 00:00, 08:00, 16:00 UTC
    validate_table(res.table, FUNDING_SCHEMA)
    assert res.table.column("symbol").to_pylist() == ["ETHUSDT"] * 3
