"""S19 — Binance USDT-M perpetual funding-rate stream (``FUNDING_SCHEMA``).

The thesis's hedge leg (ADR-030 C3) needs the real cost of carry: Binance's
``ETHUSDT`` perp settles funding every 8 h at 00:00/08:00/16:00 UTC and the
signed rate is what a short hedge receives. This fetcher pins that tape into
``undertow.data`` so the sim (S20) never has to fetch anything itself.

Source
------
``GET {futures_base_url}/fapi/v1/fundingRate?symbol={symbol}&``
``startTime={ms}&endTime={ms}&limit=1000`` (REST only — there is no bulk archive
for funding). The response carries
``fundingRate`` and ``markPrice`` on the same record, so one page supplies both
columns; perp klines are deliberately **not** fetched — ``markPrice`` is the
pinned mark source and basis modelling is S20's problem.

Pagination
----------
Pages hold up to 1000 records. The cursor advances to ``last fundingTime + 1``
(not ``+ interval``) and the loop stops on a short page or once the last record
reaches the window end. Records are de-duplicated on ``(symbol, funding_time)``
because page boundaries can repeat the preceding settlement. The request's
``endTime`` carries a 1 s slack (``_REQUEST_END_SLACK_MS``) so a settlement
jittered a few ms past an exactly grid-aligned window end is still fetched; the
client-side grid clamp is what actually enforces the window.

Settlement grid, window clamp and gap policy (ADR-017 mirror)
-------------------------------------------------------------
Binance stamps the 8 h settlement with a few milliseconds of jitter (observed
0..31 ms over the pinned 2022–2024 window). Every emitted ``funding_time`` is
floored to its 8 h UTC grid instant — the *scheduled* settlement — so the stream
is exactly regular and an as-of join cannot miss a settlement by a few
milliseconds.

The request window is ``[start_utc, end_utc]`` (inclusive, ``WindowConfig``
semantics); records outside it are dropped client-side even if the endpoint
returns them. The output is reindexed onto the grid from the **first observed**
settlement through the **last observed** one:

* observed settlement -> emitted as observed (``is_gap_filled = False``);
* a scheduled settlement missing *between* the first and last observed record ->
  forward-filled (``funding_rate``/``mark_price`` copied from the prior
  observation, ``is_gap_filled = True``);
* missing settlements before the first observation -> absent; they are never
  back-filled, because back-filling would be look-ahead. Silence after the last
  observed record is likewise *not* synthesized — the grid stops there rather
  than inventing settlements the exchange never reported.

``markPrice`` is empty (``""``) on 2005 of the 3288 settlements in the pinned
2022-2024 window (all of 2022 and most of 2023), so ``mark_price`` is
**nullable**: an empty/absent source value becomes null, never ``""`` and never
``0`` (the repo's absent-is-null contract, ``schemas.py``). ``funding_rate`` is
always present and non-nullable.

Caching
-------
Raw observed rows for one ``(symbol, window)`` are cached under the reference
route's on-disk layout (``cache_dir/reference/funding/<sha>.json.gz``), so a warm
re-pull of the same window makes **zero** HTTP calls (``n_requests == 0``,
``from_cache = True``).

Fixtures: fully synthetic — this module is tested offline against recorded-page
payloads in ``tests/data/test_funding.py``; no live capture is committed.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import math
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urlencode

import pyarrow as pa

from undertow.data.config import DEFAULT_REFERENCE_SYMBOL, EndpointConfig
from undertow.data.fetchers.base import BaseHttpFetcher, FetchRequest, FetchResult
from undertow.data.schemas import FUNDING_SCHEMA, empty_table, validate_table
from undertow.data.types import ConfigError, PermanentFetchError, Route

logger = logging.getLogger("undertow.data.fetchers.funding")

# ---------------------------------------------------------------------------
# Pinned source constants.
# ---------------------------------------------------------------------------
FUNDING_INTERVAL_MS: int = 8 * 60 * 60 * 1000  # 28_800_000 ms = one 8 h settlement
REST_LIMIT: int = 1000  # max records per /fapi/v1/fundingRate page

_REQUEST_END_SLACK_MS: int = 1_000  # 1 s
"""Binance filters ``fundingRate`` on the raw ``fundingTime`` (``endTime``
inclusive), so a settlement jittered a few ms past a grid-aligned window end
would be dropped server-side. The request's ``endTime`` is widened by this slack
(observed jitter over 2022–2024 is <= 31 ms) while the client-side grid clamp
still admits only settlements whose scheduled instant is inside ``[start, end]``."""

_EPOCH_UTC: datetime = datetime(1970, 1, 1, tzinfo=UTC)
"""Reference epoch for integer-exact millisecond <-> datetime conversion."""

# ---------------------------------------------------------------------------
# Pure time helpers (integer-exact; no float anywhere).
# ---------------------------------------------------------------------------


def ms_epoch_to_utc(ms: int) -> datetime:
    """Convert an epoch-millisecond value to a UTC-aware ``datetime``, exactly.

    ``ms * 1000`` microseconds are added via ``timedelta`` — integer arithmetic
    only. A bool is rejected: it is an ``int`` subclass and would silently
    produce 1970-01-01 rather than a real timestamp.
    """
    if isinstance(ms, bool) or not isinstance(ms, int):
        raise ConfigError(
            f"ms_epoch_to_utc: expected an epoch-millisecond int, got {type(ms).__name__} {ms!r}"
        )
    return _EPOCH_UTC + timedelta(microseconds=ms * 1000)


def _to_epoch_ms(dt: datetime) -> int:
    """Integer-exact epoch-milliseconds from an aware UTC datetime."""
    return ((dt - _EPOCH_UTC) // timedelta(microseconds=1)) // 1000


def floor_funding_ms(ms: int) -> int:
    """Floor epoch-ms to its 8 h settlement grid instant (00:00/08:00/16:00 UTC).

    Binance's raw ``fundingTime`` carries a few milliseconds of jitter; the
    scheduled instant is the grid it belongs to.
    """
    return ms - (ms % FUNDING_INTERVAL_MS)


# ---------------------------------------------------------------------------
# Record parsing — one JSON object -> one raw row dict.
# ---------------------------------------------------------------------------


def _need_number(value: object, field: str, symbol: str, record: object) -> float:
    """Parse a funding wire number, naming the field and the offending record on failure."""
    if isinstance(value, bool) or value is None:
        raise PermanentFetchError(
            f"{symbol}: malformed funding {field!r} value {value!r} in record {record!r}"
        )
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise PermanentFetchError(
            f"{symbol}: malformed funding {field!r} value {value!r} in record {record!r}"
        ) from exc
    if not math.isfinite(number):
        raise PermanentFetchError(
            f"{symbol}: non-finite funding {field!r} value {value!r} in record {record!r}"
        )
    return number


def _parse_funding_record(record: object, symbol: str) -> dict:
    """Parse one ``/fapi/v1/fundingRate`` JSON object into a raw row dict.

    ``fundingTime`` (ms, int) and ``fundingRate`` are required and must be
    numeric; ``markPrice`` is nullable — the live endpoint sends ``""`` for the
    2022-2023 history, which becomes ``None`` (absent is null, never zero).
    """
    if not isinstance(record, dict):
        raise PermanentFetchError(
            f"{symbol}: malformed funding record: expected an object, got "
            f"{type(record).__name__} {record!r}"
        )
    raw_time = record.get("fundingTime")
    if isinstance(raw_time, bool) or raw_time is None:
        raise PermanentFetchError(
            f"{symbol}: malformed funding 'fundingTime' value {raw_time!r} in record {record!r}"
        )
    try:
        funding_time_ms = int(str(raw_time))
    except (TypeError, ValueError) as exc:
        raise PermanentFetchError(
            f"{symbol}: malformed funding 'fundingTime' value {raw_time!r} in record {record!r}"
        ) from exc
    if funding_time_ms < 0:
        raise PermanentFetchError(
            f"{symbol}: negative funding 'fundingTime' value {raw_time!r} in record {record!r}"
        )
    mark_raw = record.get("markPrice")
    mark_price: float | None
    if mark_raw is None or (isinstance(mark_raw, str) and not mark_raw.strip()):
        mark_price = None
    else:
        mark_price = _need_number(mark_raw, "markPrice", symbol, record)
    return {
        "symbol": symbol,
        "funding_time_ms": funding_time_ms,
        "funding_rate": _need_number(record.get("fundingRate"), "fundingRate", symbol, record),
        "mark_price": mark_price,
        "is_gap_filled": False,
    }


# ---------------------------------------------------------------------------
# Grid re-indexing + the pure forward-fill (public, no I/O — unit-testable).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FundingGapStats:
    """Gap-fill statistics for one symbol over one requested window."""

    n_observed: int  # observed settlements inside the window
    n_filled: int  # settlements forward-filled from the prior observation
    leading_missing: int  # settlements before the first observation (absent, never filled)
    first_observed_ms: int | None = None  # grid ms of the earliest observed settlement


def forward_fill_funding(
    rows: list[dict], *, start_ms: int, end_ms: int
) -> tuple[list[dict], FundingGapStats]:
    """Reindex observed funding rows onto the complete 8 h grid for a window.

    Pure and I/O-free (the fetcher and the tests both call it). Input rows come
    from :func:`_parse_funding_record` (raw, jittered ``funding_time_ms``).
    Times are floored to the settlement grid and clamped to the inclusive
    window ``[start_ms, end_ms]``; per grid instant the first row wins.

    The grid runs from the first observed settlement through the last observed
    settlement. Leading missing settlements (before the first observation) are
    absent — never back-filled (look-ahead); every scheduled settlement missing
    between the first and last observation is forward-filled from the prior one
    with ``is_gap_filled = True`` (ADR-017 semantics). Returns
    ``(grid_rows, FundingGapStats)``.
    """
    observed: dict[int, dict] = {}
    for row in rows:
        grid_ms = floor_funding_ms(int(row["funding_time_ms"]))
        if not start_ms <= grid_ms <= end_ms:
            continue  # window clamp: out-of-window records are dropped
        if grid_ms not in observed:
            observed[grid_ms] = {**row, "funding_time_ms": grid_ms}
    if not observed:
        return [], FundingGapStats(n_observed=0, n_filled=0, leading_missing=0)

    first_grid = floor_funding_ms(start_ms)
    first_observed = min(observed)
    leading_missing = max(0, (first_observed - first_grid) // FUNDING_INTERVAL_MS)
    last_grid = max(observed)
    symbol = str(next(iter(observed.values()))["symbol"])

    grid: list[dict] = []
    filled = 0
    last_rate: float | None = None
    last_mark: float | None = None
    t = first_observed
    while t <= last_grid:
        obs = observed.get(t)
        if obs is not None:
            grid.append(obs)
            last_rate = cast(float, obs["funding_rate"])
            last_mark = cast(float | None, obs["mark_price"])
        else:
            grid.append(
                {
                    "symbol": symbol,
                    "funding_time_ms": t,
                    "funding_rate": last_rate,
                    "mark_price": last_mark,
                    "is_gap_filled": True,
                }
            )
            filled += 1
        t += FUNDING_INTERVAL_MS
    return grid, FundingGapStats(
        n_observed=len(observed),
        n_filled=filled,
        leading_missing=leading_missing,
        first_observed_ms=first_observed,
    )


# ---------------------------------------------------------------------------
# The fetcher.
# ---------------------------------------------------------------------------


class FundingFetcher(BaseHttpFetcher):
    """Binance USDT-M perpetual funding-rate fetcher (S19).

    Date-windowed, not block-windowed: ``fetch()`` requires a ``FetchRequest``
    carrying ``start_utc``/``end_utc`` (funding settlements have no block
    numbers). The base class's block-range machinery (``fetch_rows``) is
    intentionally not used — :meth:`_fetch_chunk` is a loud
    ``NotImplementedError``.

    :param symbol: the USDT-perp symbol, defaulting to the pinned primary
        (:data:`~undertow.data.config.DEFAULT_REFERENCE_SYMBOL`). The pipeline
        passes ``DataConfig.reference_symbol`` — the perp string is identical to
        the spot symbol and no second symbol field exists.
    """

    route = Route.REFERENCE

    def __init__(
        self,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        symbol: str = DEFAULT_REFERENCE_SYMBOL,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(endpoints, cache_dir, sleep=sleep, rng=rng)
        normalized = symbol.upper()
        if not normalized or not normalized.isalnum():
            raise ConfigError(
                f"FundingFetcher: invalid symbol {symbol!r} — expected an uppercase "
                "alphanumeric pair like 'ETHUSDT'"
            )
        self._symbol: str = normalized

    # ------------------------------------------------------------------
    # Fetcher protocol
    # ------------------------------------------------------------------

    def supported_streams(self) -> frozenset[str]:
        return frozenset({"funding"})

    def fetch(self, request: FetchRequest) -> FetchResult:
        """Fetch the funding stream for the request's inclusive UTC window.

        Returns a schema-validated ``FUNDING_SCHEMA`` table with settlements on
        the 8 h grid, gaps forward-filled and flagged; ``FetchResult.warnings``
        carries the leading-edge and gap-fill statistics.
        """
        start_utc, end_utc = self._validate_request(request)
        start_ms = _to_epoch_ms(start_utc)
        end_ms = _to_epoch_ms(end_utc)

        rows, n_requests, from_cache = self._fetch_window(start_ms, end_ms)
        table, stats = self._rows_to_table(rows, start_ms=start_ms, end_ms=end_ms)

        warnings: list[str] = []
        if stats.n_observed == 0:
            warnings.append(
                f"{self._symbol}: no funding settlements in window "
                f"{start_utc.isoformat()}..{end_utc.isoformat()}; table is empty"
            )
        else:
            if stats.leading_missing > 0:
                first = (
                    ms_epoch_to_utc(stats.first_observed_ms).isoformat()
                    if stats.first_observed_ms is not None
                    else "n/a"
                )
                warnings.append(
                    f"{self._symbol}: leading edge: {stats.leading_missing} settlement(s) "
                    f"before the first observed settlement {first} have no prior record to "
                    "fill from and are absent — no back-fill (no look-ahead)"
                )
            if stats.n_filled > 0:
                warnings.append(
                    f"{self._symbol}: gap-fill: forward-filled {stats.n_filled} missing "
                    "8 h settlement(s)"
                )
        return FetchResult(
            table=table,
            route=Route.REFERENCE,
            request=request,
            n_requests=n_requests,
            from_cache=from_cache,
            warnings=tuple(warnings),
        )

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        raise NotImplementedError(
            "FundingFetcher is date-windowed (funding settlements have no block "
            "numbers), so the block-range fetch_rows() machinery does not apply. "
            "Call fetch() with a FetchRequest carrying start_utc/end_utc instead."
        )

    # ------------------------------------------------------------------
    # Request validation / pagination
    # ------------------------------------------------------------------

    def _validate_request(self, request: FetchRequest) -> tuple[datetime, datetime]:
        if request.stream != "funding":
            raise ConfigError(
                f"FundingFetcher: unsupported stream {request.stream!r}; "
                "only 'funding' is supported"
            )
        if request.start_utc is None or request.end_utc is None:
            raise ConfigError(
                "FundingFetcher: funding fetch requires start_utc/end_utc — "
                f"got start_block={request.start_block!r}, end_block={request.end_block!r}. "
                "This stream is date-windowed, not block-windowed."
            )
        start = request.start_utc.astimezone(UTC)
        end = request.end_utc.astimezone(UTC)
        if start > end:
            raise ConfigError(
                f"FundingFetcher: start_utc {start.isoformat()} must be <= "
                f"end_utc {end.isoformat()}"
            )
        return start, end

    def _fetch_window(self, start_ms: int, end_ms: int) -> tuple[list[dict], int, bool]:
        """Fetch the observed raw rows for ``[start_ms, end_ms]`` with pagination.

        Returns ``(rows, n_requests, from_cache)``. The whole window is cached
        under one deterministic key, so a warm re-pull makes zero HTTP calls.
        """
        params: dict[str, object] = {
            "symbol": self._symbol,
            "path": "rest",
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        key = self._funding_cache_key(params)
        cached = self._funding_cache_read(key)
        if cached is not None:
            return cached, 0, True

        configured = self._endpoints.futures_base_url.strip()
        if not configured:
            raise ConfigError(
                "FundingFetcher: endpoints.futures_base_url is empty; a USDT-M futures "
                "REST base URL is required for /fapi/v1/fundingRate"
            )
        endpoint = f"{configured.rstrip('/')}/fapi/v1/fundingRate"
        context = (
            f"FundingFetcher stream=funding symbol={self._symbol} range_ms={start_ms}..{end_ms}"
        )

        rows: list[dict] = []
        cursor = start_ms
        n_pages = 0
        request_end_ms = end_ms + _REQUEST_END_SLACK_MS
        while cursor <= end_ms:
            query = urlencode(
                {
                    "symbol": self._symbol,
                    "startTime": cursor,
                    "endTime": request_end_ms,
                    "limit": REST_LIMIT,
                }
            )
            resp = self._request(f"{endpoint}?{query}", context=context)
            try:
                payload = resp.json()
            except ValueError as exc:
                raise PermanentFetchError(
                    f"FundingFetcher {self._symbol}: fundingRate returned non-JSON content"
                ) from exc
            if not isinstance(payload, list):
                raise PermanentFetchError(
                    f"FundingFetcher {self._symbol}: fundingRate returned "
                    f"{type(payload).__name__}, expected a JSON array of funding records"
                )
            page = [_parse_funding_record(item, self._symbol) for item in payload]
            rows.extend(page)
            n_pages += 1
            if len(page) < REST_LIMIT:
                break  # short page: the window is exhausted
            last_ms = int(page[-1]["funding_time_ms"])
            if last_ms >= end_ms:
                break  # reached endTime
            nxt = last_ms + 1  # spec pagination: last fundingTime + 1
            if nxt <= cursor:  # defensive: no forward progress -> stop
                break
            cursor = nxt

        rows = self._dedupe(rows)
        self._funding_cache_write(key, params, rows)
        return rows, n_pages, False

    @staticmethod
    def _dedupe(rows: list[dict]) -> list[dict]:
        """Drop duplicate ``(symbol, funding_time_ms)`` rows — first wins."""
        seen: set[tuple[str, int]] = set()
        out: list[dict] = []
        for row in rows:
            key = (str(row["symbol"]), int(row["funding_time_ms"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Funding-specific raw cache (base's cache is block-keyed).
    # ------------------------------------------------------------------

    def _funding_cache_key(self, params: Mapping[str, object]) -> str:
        return self.cache_key(self.route, "funding", params)

    def _funding_cache_path(self, key: str) -> Path:
        return self._cache_dir / self.route.value / "funding" / f"{key}.json.gz"

    def _funding_cache_read(self, key: str) -> list[dict] | None:
        """Cached raw rows for a window, or ``None`` on miss / corrupt entry."""
        path = self._funding_cache_path(key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as fh:
                envelope = json.loads(fh.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive (corrupt bytes)
            logger.warning("corrupt funding cache entry at %s (%s); treating as miss", path, exc)
            return None
        if not isinstance(envelope, dict) or not isinstance(envelope.get("body"), list):
            logger.warning("corrupt funding cache envelope at %s; treating as miss", path)
            return None
        return envelope["body"]

    def _funding_cache_write(
        self, key: str, params: Mapping[str, object], rows: list[dict]
    ) -> None:
        """Atomically write rows under the funding cache layout."""
        path = self._funding_cache_path(key)
        envelope: dict[str, object] = {
            "params": {k: v for k, v in params.items()},
            "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "body": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(envelope, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------
    # Table building (the subclass's job per CONTRACTS §5.1).
    # ------------------------------------------------------------------

    def _rows_to_table(
        self, rows: list[dict], *, start_ms: int, end_ms: int
    ) -> tuple[pa.Table, FundingGapStats]:
        """Build and validate a ``FUNDING_SCHEMA`` table from raw funding rows.

        Applies :func:`forward_fill_funding`; the result is sorted by
        ``(symbol, funding_time)`` (the stream's sort key) and validated via
        ``schemas.validate_table``. An empty input yields ``empty_table``.
        """
        grid, stats = forward_fill_funding(rows, start_ms=start_ms, end_ms=end_ms)
        if not grid:
            return empty_table(FUNDING_SCHEMA), stats

        columns: dict[str, list[object]] = {name: [] for name in FUNDING_SCHEMA.names}
        for row in grid:
            columns["funding_time"].append(ms_epoch_to_utc(int(row["funding_time_ms"])))
            columns["symbol"].append(str(row["symbol"]))
            columns["funding_rate"].append(cast(float, row["funding_rate"]))
            mark = cast(float | None, row["mark_price"])
            columns["mark_price"].append(None if mark is None else float(mark))
            columns["is_gap_filled"].append(bool(row["is_gap_filled"]))
        arrays = {
            name: pa.array(values, type=FUNDING_SCHEMA.field(name).type)
            for name, values in columns.items()
        }
        table = pa.table(arrays)
        table = table.sort_by([("symbol", "ascending"), ("funding_time", "ascending")])
        validate_table(table, FUNDING_SCHEMA)
        return table, stats
