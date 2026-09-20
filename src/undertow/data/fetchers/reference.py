"""T08 — external reference price feed fetcher (Binance 1m klines).

Provides the *external* ETH price the pipeline needs for impermanent-loss / LVR
measurement and for T12's regime labels: the pool's own tick is stale and noisy
(it only moves when someone trades against it and is distorted by that trade's
slippage), so "true" price comes from the deepest, longest 1m history — Binance
``ETHUSDT`` primary, ``ETHUSDC`` as the cross-check (pinned in plan §7).

Source and access paths
-----------------------
* **Bulk archive** — ``data.binance.vision`` monthly zipped CSVs, one HTTP GET
  per calendar month (the right tool for three years of minute bars):

  ``https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY}-{MM}.zip``

* **REST API** — ``{reference_base_url}/api/v3/klines`` with ``interval=1m`` and
  1000-bar pages, used for the current/partial month and as a fallback when the
  monthly archive is not yet published (HTTP 404).

Both paths produce byte-identical ``REFERENCE_SCHEMA`` output; ``_plan_sources``
routes full calendar months through the bulk archive and boundary months through
the REST API. Both wire formats carry the same 12 kline fields, in this order:

  ``open_time, open, high, low, close, volume, close_time, quote_volume,
  trades, taker_buy_base, taker_buy_quote, ignore``

Times are **epoch milliseconds** in both formats and are converted to UTC-aware
microsecond timestamps with integer-only arithmetic (``ms * 1000`` then a
``timedelta``); no floats, no naive datetimes.

Interval convention
-------------------
A kline is a **left-closed, right-open one-minute bar**: ``open_time`` is the
bar's open, aligned to the UTC wall-clock minute, and
``close_time = open_time + 59_999_000 µs`` (i.e. ``open_time + 59_999 ms``) —
the last microsecond of the bar. T11's as-of join must use
``close_time <= block_timestamp`` (backward-only); any other comparison shifts
every reference price by exactly one bar, so downstream code must quote this
sentence when writing the join.

A ``fetch`` **window is half-open**: ``[start_utc, end_utc)``. The grid covers
every minute whose ``open_time`` is ``>= floor_minute(start_utc)`` and
``< end_utc``, so a calendar-day request returns exactly 1440 bars (the bar
whose open equals the request end belongs to the next day). This matches the
REST API, whose ``endTime`` parameter is exclusive.

Gap policy
----------
Exchange outages and maintenance windows leave genuine holes in the minute
series. The fetcher forward-fills and flags — it **never interpolates and never
drops**:

* The requested window is reindexed to the complete 1-minute grid.
* A missing bar after at least one observed bar is forward-filled: ``open``,
  ``high``, ``low``, ``close`` = the last observed ``close``, volumes and
  ``trades`` = 0, ``is_gap_filled = True``.
* A missing **leading edge** (no prior observed bar) is left *null* at the grid
  level (see :func:`forward_fill_klines`) and reported via ``FetchResult.warnings``
  — it is never back-filled, because back-filling would be look-ahead. Because
  ``REFERENCE_SCHEMA`` declares the OHLC columns non-nullable, such bars cannot
  be represented in the returned table and are therefore *absent* from it; "null"
  equals "no bar" in the artifact, and the warning carries the count.
* Gap statistics (filled-bar count and longest run) are reported in
  ``FetchResult.warnings`` — T13's ``check_reference_coverage`` consumes them.

USD vs USDT
-----------
``REFERENCE_SCHEMA`` prices are "ETH in USD-stable terms". This fetcher treats
**USDT as a USD proxy (1 USDT ≙ 1 USD)** — an explicit assumption, not a fact:
USDT can deviate from parity under stress (tens of bps within the study window,
2022–2024), which is second-order for the pipeline's uses (IL/LVR measurement,
30-day regime labels). The assumption is configurable: the symbol set is a
constructor argument (``ReferenceFetcher(symbols=...)``), so the feed can be
switched to a fiat-backed pair / basket without code changes. (Flagged for T16 to
record in ``docs/data_dictionary.md``.)

Stream characteristics
----------------------
``reference`` is a **non-log, global** stream (CONTRACTS §4.0): it has **no
block columns** and is **not pool-partitioned**; sort key is ``(symbol,
open_time)``. Output conforms to ``REFERENCE_SCHEMA`` (validated via
``schemas.validate_table``) and is wrapped in a ``FetchResult``.

Caching
-------
Each source fetch (one monthly zip, or one REST window) is cached under a
deterministic key ``(route, stream, symbol, path, range)`` in the on-disk raw
cache, so a warm re-pull of the same window makes **zero** HTTP calls
(``n_requests == 0``, ``from_cache == True``).

Fixtures: this module is tested entirely offline against synthetic fixtures
(``tests/data/fixtures/binance_klines_api.json``,
``binance_klines_bulk.zip``) — live capture is impossible in the sandbox (no
network to Binance); the fixtures are marked synthetic and must never be
presented as real captures.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import io
import json
import logging
import os
import random
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlencode

import pyarrow as pa

from undertow.data.config import EndpointConfig
from undertow.data.fetchers.base import BaseHttpFetcher, FetchRequest, FetchResult
from undertow.data.schemas import REFERENCE_SCHEMA, empty_table, validate_table
from undertow.data.types import (
    CheckResult,
    ConfigError,
    PermanentFetchError,
    Route,
)

logger = logging.getLogger("undertow.data.fetchers.reference")

# ---------------------------------------------------------------------------
# Pinned interval / source constants (PLAN.md §7).
# ---------------------------------------------------------------------------
INTERVAL: str = "1m"  # minute bars — the only supported interval
INTERVAL_MS: int = 60_000  # one minute in milliseconds
INTERVAL_CLOSE_DELTA_US: int = 59_999_000  # close_time = open_time + 59_999_000 µs
INTERVAL_CLOSE_DELTA_MS: int = INTERVAL_CLOSE_DELTA_US // 1000  # = 59_999 ms
REST_LIMIT: int = 1000  # max bars per /api/v3/klines page
BINANCE_VISION_BASE_URL: str = "https://data.binance.vision"

_EPOCH_UTC: datetime = datetime(1970, 1, 1, tzinfo=UTC)
"""Reference epoch for integer-exact millisecond <-> datetime conversion."""

# ---------------------------------------------------------------------------
# Pure time helpers (integer-exact; no float anywhere).
# ---------------------------------------------------------------------------


def ms_epoch_to_utc(ms: int) -> datetime:
    """Convert an epoch-millisecond value to a UTC-aware ``datetime``, exactly.

    ``ms * 1000`` microseconds are added via ``timedelta`` — integer arithmetic
    only, so the conversion is exact for every value (Binance epoch times are
    millisecond integers). A bool is rejected: it is an ``int`` subclass and
    would silently produce 1970-01-01 rather than a real timestamp.
    """
    if isinstance(ms, bool) or not isinstance(ms, int):
        raise ConfigError(
            f"ms_epoch_to_utc: expected an epoch-millisecond int, got {type(ms).__name__} {ms!r}"
        )
    return _EPOCH_UTC + timedelta(microseconds=ms * 1000)


def _to_epoch_ms(dt: datetime) -> int:
    """Integer-exact epoch-milliseconds from an aware UTC datetime."""
    return ((dt - _EPOCH_UTC) // timedelta(microseconds=1)) // 1000


def _floor_minute(ms: int) -> int:
    """Floor epoch-ms to the start of its UTC minute (the kline grid).

    Defined for non-negative epochs (real-world dates), which ``fetch``
    guarantees; kline open times land exactly on this grid.
    """
    return ms - (ms % INTERVAL_MS)


def _month_bounds(year: int, month: int) -> tuple[int, int]:
    """``(first_ms, last_minute_open_ms)`` for a calendar month.

    The last minute's open time is one minute before the next month's first
    instant; the month's final kline covers ``[last_open, last_open + 60 s)``.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    first = _to_epoch_ms(datetime(year, month, 1, tzinfo=UTC))
    next_first = _to_epoch_ms(datetime(next_year, next_month, 1, tzinfo=UTC))
    return first, next_first - INTERVAL_MS


def _overlapping_months(start_ms: int, end_ms: int) -> list[tuple[int, int, int, int]]:
    """Calendar months intersecting ``[start_ms, end_ms]``.

    Returns ``(year, month, month_start_ms, month_last_open_ms)`` tuples in
    chronological order. A month intersects the window when its last minute's
    open time is >= the window start and its first instant is <= the window end.
    """
    start_dt = ms_epoch_to_utc(start_ms)
    end_dt = ms_epoch_to_utc(end_ms)
    year, month = start_dt.year, start_dt.month
    end_key = (end_dt.year, end_dt.month)
    out: list[tuple[int, int, int, int]] = []
    while (year, month) <= end_key:
        mstart, mend = _month_bounds(year, month)
        # Half-open window [start_ms, end_ms): a month whose FIRST instant equals the
        # window end (exact-months boundary) does not intersect it. With `<` an exact
        # calendar-month fetch plans only that month's bulk file, not a zero-width
        # REST spec for the next month's first instant.
        if mend >= start_ms and mstart < end_ms:
            out.append((year, month, mstart, mend))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


# ---------------------------------------------------------------------------
# Kline parsing (positional — the 12-field order is identical in REST and CSV).
# ---------------------------------------------------------------------------


def _need_number(value: object, field: str, symbol: str, row: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise PermanentFetchError(
            f"{symbol}: malformed kline {field!r} value {value!r} in row {row!r}"
        ) from exc


def _need_int(value: object, field: str, symbol: str, row: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise PermanentFetchError(
            f"{symbol}: malformed kline {field!r} value {value!r} in row {row!r}"
        ) from exc


def _parse_kline(seq: object, symbol: str) -> dict:
    """Parse one 12-field kline (REST array element or CSV row) into a row dict.

    Field order (both sources): ``open_time, open, high, low, close, volume,
    close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore``.
    ``taker_*`` and ``ignore`` are accepted but dropped — they are not part of
    ``REFERENCE_SCHEMA``. ``close_time`` is checked for consistency but never
    trusted: the emitted ``close_time`` is re-derived as
    ``open_time + INTERVAL_CLOSE_DELTA_MS`` (see the interval convention in the
    module docstring), so a mismatched source value cannot shift the output.
    """
    if not isinstance(seq, (list, tuple)) or len(seq) < 12:
        raise PermanentFetchError(
            f"{symbol}: malformed kline row: expected 12 fields, got {type(seq).__name__} {seq!r}"
        )
    open_ms = _need_int(seq[0], "open_time", symbol, seq)
    close_ms = _need_int(seq[6], "close_time", symbol, seq)
    if open_ms % INTERVAL_MS != 0:
        raise PermanentFetchError(
            f"{symbol}: kline open_time {open_ms} is not aligned to the {INTERVAL} minute grid"
        )
    if close_ms != open_ms + INTERVAL_CLOSE_DELTA_MS:
        # Source close_time is a consistency check only; the emitted close_time
        # is re-derived from open_time. Binance's own feed carries rare
        # malformed bars (e.g. ETHUSDT at 2023-03-24 12:39Z, in both the monthly
        # archive and the REST API), so warn and keep the bar instead of
        # aborting the whole pull — the re-derived close_time is correct
        # regardless of the source value.
        logger.warning(
            "%s: source kline close_time %d != open_time + %d ms (%d) at open_time %d; "
            "keeping the bar and re-deriving close_time per the pinned interval convention",
            symbol,
            close_ms,
            INTERVAL_CLOSE_DELTA_MS,
            open_ms + INTERVAL_CLOSE_DELTA_MS,
            open_ms,
        )
    return {
        "symbol": symbol,
        "open_time_ms": open_ms,
        "open": _need_number(seq[1], "open", symbol, seq),
        "high": _need_number(seq[2], "high", symbol, seq),
        "low": _need_number(seq[3], "low", symbol, seq),
        "close": _need_number(seq[4], "close", symbol, seq),
        "volume_base": _need_number(seq[5], "volume", symbol, seq),
        "quote_volume": _need_number(seq[7], "quote_volume", symbol, seq),
        "trades": _need_int(seq[8], "trades", symbol, seq),
        "is_gap_filled": False,
    }


def _parse_csv(text: str, symbol: str) -> list[dict]:
    """Parse a bulk-archive CSV (12 fields/row, comma separated).

    Skips blank lines, comment lines (``#``-prefixed — synthetic fixtures carry
    a provenance note there) and the header line (non-numeric first field). The
    header names follow Binance's own CSV (``count`` for the trades field).
    """
    rows: list[dict] = []
    for fields in csv.reader(text.splitlines()):
        if not fields or not fields[0].strip():
            continue
        first = fields[0].strip()
        if first.startswith("#") or not first.lstrip("-").isdigit():
            continue  # comment or column-header line
        rows.append(_parse_kline(fields, symbol))
    return rows


def _parse_bulk_zip(content: bytes, symbol: str) -> list[dict]:
    """Decode a ``data.binance.vision`` monthly zip: first ``*.csv`` member."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not names:
                raise PermanentFetchError(
                    f"{symbol}: bulk archive contains no *.csv member "
                    f"(members: {sorted(zf.namelist())})"
                )
            raw = zf.read(names[0]).decode("utf-8-sig")
    except zipfile.BadZipFile as exc:
        raise PermanentFetchError(f"{symbol}: bulk archive is not a valid zip file") from exc
    except UnicodeDecodeError as exc:
        raise PermanentFetchError(f"{symbol}: bulk archive CSV is not valid UTF-8") from exc
    return _parse_csv(raw, symbol)


# ---------------------------------------------------------------------------
# Gap statistics + the pure forward-fill (public, no I/O — unit-testable).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GapStats:
    """Gap-fill statistics for one symbol over one requested window."""

    n_observed: int  # observed (non-filled) bars in the window
    n_filled: int  # bars forward-filled from a prior close
    longest_run: int  # longest consecutive run of forward-filled bars
    leading_missing: int  # minutes before the first observed bar (null, not filled)
    first_observed_ms: int | None = None  # open_time (ms) of the earliest observed bar


def forward_fill_klines(
    rows: Sequence[Mapping[str, object]],
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[list[dict], dict[str, GapStats]]:
    """Reindex observed kline rows onto the complete 1-minute grid for a window.

    Pure and I/O-free (the fetcher and the tests both call it). Each input row
    must carry ``symbol`` and ``open_time_ms`` plus the price/volume fields
    produced by :func:`_parse_kline`. Per symbol it emits one row per grid
    minute whose ``open_time`` lies in ``[floor_minute(start_utc), end_utc)``
    (right-exclusive — a day-long window yields exactly 1440 bars):

    * observed bar  -> emitted unchanged (``is_gap_filled=False``);
    * missing with a prior close -> **forward-filled** from the last observed
      close (OHLC = close, volumes/trades = 0, ``is_gap_filled=True``);
    * missing **leading edge** (no prior close) -> emitted with ``None`` OHLC
      and ``is_gap_filled=False`` — a placeholder "no bar" that callers must
      not back-fill (that would be look-ahead). The table-building path drops
      these rows because ``REFERENCE_SCHEMA`` prices are non-nullable, and
      warns via the returned :class:`GapStats`.

    Returns ``(grid_rows, {symbol: GapStats})``.
    """
    start_ms = _to_epoch_ms(start_utc)
    end_ms = _to_epoch_ms(end_utc)
    observed_by_symbol: dict[str, dict[int, Mapping[str, object]]] = {}
    symbols: list[str] = []
    for row in rows:
        sym = str(row["symbol"])
        open_ms = cast(int, row["open_time_ms"])
        bucket = observed_by_symbol.get(sym)
        if bucket is None:
            bucket = {}
            observed_by_symbol[sym] = bucket
            symbols.append(sym)
        bucket.setdefault(open_ms, row)  # first occurrence wins (deterministic)

    grid_all: list[dict] = []
    stats: dict[str, GapStats] = {}
    for sym in sorted(symbols):
        observed = observed_by_symbol[sym]
        grid: list[dict] = []
        last_close: float | None = None
        n_filled = 0
        longest_run = 0
        cur_run = 0
        leading = 0
        first_observed_ms: int | None = min(observed) if observed else None
        t = _floor_minute(start_ms)
        while t < end_ms:  # half-open window: the bar at end_ms belongs to the next interval
            obs = observed.get(t)
            if obs is not None:
                close = cast(float, obs["close"])
                grid.append(
                    {
                        "symbol": sym,
                        "open_time_ms": t,
                        "open": cast(float, obs["open"]),
                        "high": cast(float, obs["high"]),
                        "low": cast(float, obs["low"]),
                        "close": close,
                        "volume_base": cast(float, obs["volume_base"]),
                        "quote_volume": cast(float, obs["quote_volume"]),
                        "trades": cast(int, obs["trades"]),
                        "is_gap_filled": False,
                    }
                )
                last_close = close
                cur_run = 0
            elif last_close is not None:
                grid.append(
                    {
                        "symbol": sym,
                        "open_time_ms": t,
                        "open": last_close,
                        "high": last_close,
                        "low": last_close,
                        "close": last_close,
                        "volume_base": 0.0,
                        "quote_volume": 0.0,
                        "trades": 0,
                        "is_gap_filled": True,
                    }
                )
                n_filled += 1
                cur_run += 1
                if cur_run > longest_run:
                    longest_run = cur_run
            else:
                # Leading edge: no prior bar to fill from. Null placeholder —
                # never back-filled (look-ahead). Dropped by the table builder.
                grid.append(
                    {
                        "symbol": sym,
                        "open_time_ms": t,
                        "open": None,
                        "high": None,
                        "low": None,
                        "close": None,
                        "volume_base": 0.0,
                        "quote_volume": 0.0,
                        "trades": 0,
                        "is_gap_filled": False,
                    }
                )
                leading += 1
            t += INTERVAL_MS
        grid_all.extend(grid)
        stats[sym] = GapStats(
            n_observed=len(observed),
            n_filled=n_filled,
            longest_run=longest_run,
            leading_missing=leading,
            first_observed_ms=first_observed_ms,
        )
    return grid_all, stats


# ---------------------------------------------------------------------------
# Cross-symbol sanity check (CONTRACTS §5.3 / brief §3).
# ---------------------------------------------------------------------------


def _closes_by_open_time(table: pa.Table) -> dict[datetime, float]:
    """``{open_time: close}`` for a reference table (last row wins on ties)."""
    times = table.column("open_time").to_pylist()
    closes = table.column("close").to_pylist()
    out: dict[datetime, float] = {}
    for t, c in zip(times, closes, strict=True):
        out[t] = float(c)
    return out


def compare_symbols(a: pa.Table, b: pa.Table, *, tolerance_bps: float = 25.0) -> CheckResult:
    """Compare two symbols' closes aligned on ``open_time``.

    Reports the fraction of aligned bars whose relative difference
    ``|a - b| / |a|`` exceeds ``tolerance_bps`` (basis points). A sustained
    divergence means one series is wrong or stale — the ETUSDT/ETHUSDC sanity
    check the roadmap implies (plan §10.1.7).

    Returns a ``CheckResult`` (``undertow.data.types``) — ``passed`` is True
    only when the overlap is non-empty and zero bars exceed the tolerance;
    severity is ``warning`` (a tracking drift is not a data-corruption error,
    but the human must see it). Metrics: ``bars_compared``, ``bars_diverged``,
    ``frac_diverged``, ``max_rel_diff_bps``, ``tolerance_bps``.
    """
    a_close = _closes_by_open_time(a)
    b_close = _closes_by_open_time(b)
    overlap = sorted(a_close.keys() & b_close.keys())
    n = len(overlap)
    divergence_for_zero: float = 1e300  # 0 vs non-zero price: huge (diverged)

    if n == 0:
        return CheckResult(
            name="compare_symbols",
            passed=False,
            severity="warning",
            detail=(
                f"compare_symbols: no overlapping open_time bars "
                f"(a={len(a_close)}, b={len(b_close)})"
            ),
            metrics={
                "bars_compared": 0,
                "bars_diverged": 0,
                "frac_diverged": 0.0,
                "max_rel_diff_bps": 0.0,
                "tolerance_bps": tolerance_bps,
            },
        )

    diverged = 0
    max_bps = 0.0
    for ts in overlap:
        pa_close = a_close[ts]
        pb_close = b_close[ts]
        if pa_close == 0.0:
            bps = 0.0 if pb_close == 0.0 else divergence_for_zero
        else:
            bps = abs(pa_close - pb_close) / abs(pa_close) * 10_000.0
        if bps > max_bps:
            max_bps = bps
        if bps > tolerance_bps:
            diverged += 1
    frac = diverged / n
    detail = (
        f"compare_symbols: {diverged}/{n} aligned bar(s) ({frac * 100.0:.2f}%) "
        f"exceed {tolerance_bps:g} bps divergence; max relative diff {max_bps:.4g} bps"
    )
    return CheckResult(
        name="compare_symbols",
        passed=diverged == 0,
        severity="warning",
        detail=detail,
        metrics={
            "bars_compared": n,
            "bars_diverged": diverged,
            "frac_diverged": frac,
            "max_rel_diff_bps": max_bps,
            "tolerance_bps": tolerance_bps,
        },
    )


# ---------------------------------------------------------------------------
# The fetcher.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Source:
    """One planned source retrieval: a bulk month or a REST window."""

    kind: Literal["bulk", "rest"]
    symbol: str
    start_ms: int | None = None  # rest window start (clamped to month when partial)
    end_ms: int | None = None  # rest window end
    year: int | None = None  # bulk month
    month: int | None = None  # bulk month
    month_start_ms: int | None = None  # bulk month bounds (for REST fallback)
    month_end_ms: int | None = None


class ReferenceFetcher(BaseHttpFetcher):
    """Binance 1m-kline reference feed fetcher (``CONTRACTS.md`` §5.3, T08).

    Date-windowed, not block-windowed: ``fetch()`` requires a ``FetchRequest``
    carrying ``start_utc``/``end_utc`` (reference bars have no block numbers).
    The base class's block-range machinery (``fetch_rows``) is intentionally not
    used — :meth:`_fetch_chunk` is a loud ``NotImplementedError``.

    :param symbols: the symbols to fetch per call, in table order; default
        ``("ETHUSDT", "ETHUSDC")`` per PLAN.md §7 (config-driven so the
        USDT-as-USD assumption can be revisited without code changes).
    """

    route = Route.REFERENCE

    def __init__(
        self,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        symbols: Sequence[str] = ("ETHUSDT", "ETHUSDC"),
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(endpoints, cache_dir, sleep=sleep, rng=rng)
        normalized = tuple(symbol.upper() for symbol in symbols)
        if not normalized:
            raise ConfigError("ReferenceFetcher: at least one symbol is required")
        for symbol in normalized:
            if not symbol.isalnum() or not symbol.isupper():
                raise ConfigError(
                    f"ReferenceFetcher: invalid symbol {symbol!r} — expected an "
                    "uppercase alphanumeric pair like 'ETHUSDT'"
                )
        self._symbols: tuple[str, ...] = normalized

    # ------------------------------------------------------------------
    # Fetcher protocol
    # ------------------------------------------------------------------

    def supported_streams(self) -> frozenset[str]:
        return frozenset({"reference"})

    def fetch(self, request: FetchRequest, *, access: str = "auto") -> FetchResult:
        """Fetch the configured symbols' 1m klines over the request's UTC window.

        The window is half-open, ``[start_utc, end_utc)``: the returned grid
        contains every minute bar whose open time is ``< end_utc`` (a
        calendar-day request returns exactly 1440 bars; see the module
        docstring's interval convention).

        ``access`` routes sources: ``"auto"`` (bulk archive for full calendar
        months, REST API for the boundary months), ``"bulk"`` (archive only) or
        ``"rest"`` (REST only). Returns a schema-validated ``REFERENCE_SCHEMA``
        table with rows for every configured symbol, gap-filled and flagged per
        the module docstring; ``FetchResult.warnings`` carries gap statistics,
        leading-edge and coverage notes.
        """
        start_utc, end_utc = self._validate_request(request)
        start_ms = _to_epoch_ms(start_utc)
        end_ms = _to_epoch_ms(end_utc)
        if end_ms < 0:
            raise ConfigError(
                "ReferenceFetcher: dates before 1970-01-01 are not supported "
                f"(end_utc={end_utc.isoformat()})"
            )
        if start_ms > end_ms:
            return FetchResult(
                table=empty_table(REFERENCE_SCHEMA),
                route=Route.REFERENCE,
                request=request,
                n_requests=0,
                from_cache=False,
                warnings=(
                    f"reference: empty date range {start_utc.isoformat()}.."
                    f"{end_utc.isoformat()}; nothing to fetch",
                ),
            )

        all_rows: list[dict] = []
        warnings: list[str] = []
        n_requests = 0
        from_cache = True
        for symbol in self._symbols:
            rows, n, cached, sym_warnings = self._fetch_symbol(symbol, start_ms, end_ms, access)
            all_rows.extend(rows)
            n_requests += n
            from_cache = from_cache and cached
            warnings.extend(sym_warnings)

        table, stats = self._rows_to_table(
            all_rows, request.stream, start_ms=start_ms, end_ms=end_ms
        )
        for symbol in self._symbols:
            st = stats.get(symbol)
            if st is None:
                warnings.append(
                    f"{symbol}: no observed bars in window "
                    f"{start_utc.isoformat()}..{end_utc.isoformat()}; "
                    "table has no rows for this symbol"
                )
            else:
                if st.leading_missing > 0:
                    when = (
                        ms_epoch_to_utc(st.first_observed_ms).isoformat()
                        if st.first_observed_ms is not None
                        else "n/a"
                    )
                    warnings.append(
                        f"{symbol}: leading edge: {st.leading_missing} minute(s) "
                        f"before first observed bar {when} have no prior bar to fill "
                        "from and are left null/absent — no back-fill (no look-ahead)"
                    )
                if st.n_filled > 0:
                    warnings.append(
                        f"{symbol}: gap-fill: filled {st.n_filled} minute bar(s), "
                        f"longest run {st.longest_run}"
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
            "ReferenceFetcher is date-windowed (CONTRACTS.md §4.6): klines have no "
            "block numbers, so the block-range fetch_rows() machinery does not apply. "
            "Call fetch() with a FetchRequest carrying start_utc/end_utc instead."
        )

    # ------------------------------------------------------------------
    # Request validation / source planning
    # ------------------------------------------------------------------

    def _validate_request(self, request: FetchRequest) -> tuple[datetime, datetime]:
        if request.stream != "reference":
            raise ConfigError(
                f"ReferenceFetcher: unsupported stream {request.stream!r}; "
                "only 'reference' is supported"
            )
        if request.start_utc is None or request.end_utc is None:
            raise ConfigError(
                "ReferenceFetcher: reference fetch requires start_utc/end_utc — "
                f"got start_block={request.start_block!r}, end_block={request.end_block!r}. "
                "This stream is date-windowed, not block-windowed (CONTRACTS.md §4.6)."
            )
        start = request.start_utc.astimezone(UTC)
        end = request.end_utc.astimezone(UTC)
        if start > end:
            raise ConfigError(
                f"ReferenceFetcher: start_utc {start.isoformat()} must be <= "
                f"end_utc {end.isoformat()}"
            )
        return start, end

    def _plan_sources(self, symbol: str, start_ms: int, end_ms: int, access: str) -> list[_Source]:
        """Decide which HTTP retrievals cover ``[start_ms, end_ms]`` for a symbol.

        ``auto``: full calendar months -> bulk archive (one GET per month);
        boundary months -> REST (only the requested sub-range). ``bulk`` /
        ``rest`` force one path for tests and for the fallback.
        """
        if access == "bulk":
            return [
                _Source("bulk", symbol, year=y, month=m, month_start_ms=ms, month_end_ms=me)
                for y, m, ms, me in _overlapping_months(start_ms, end_ms)
            ]
        if access == "rest":
            return [_Source("rest", symbol, start_ms=start_ms, end_ms=end_ms)]
        if access != "auto":
            raise ConfigError(
                f"ReferenceFetcher: access must be 'auto', 'bulk' or 'rest', got {access!r}"
            )
        specs: list[_Source] = []
        for y, m, mstart, mend in _overlapping_months(start_ms, end_ms):
            if mstart >= start_ms and mend <= end_ms:
                specs.append(
                    _Source(
                        "bulk",
                        symbol,
                        year=y,
                        month=m,
                        month_start_ms=mstart,
                        month_end_ms=mend,
                    )
                )
            else:
                specs.append(
                    _Source(
                        "rest",
                        symbol,
                        start_ms=max(start_ms, mstart),
                        end_ms=min(end_ms, mend),
                    )
                )
        return specs

    def _fetch_symbol(
        self, symbol: str, start_ms: int, end_ms: int, access: str
    ) -> tuple[list[dict], int, bool, list[str]]:
        """Fetch one symbol's observed rows for the window.

        Returns ``(rows, n_requests, from_cache, warnings)``. A bulk month that
        urns up a ``PermanentFetchError`` (e.g. HTTP 404 — archive not yet
        published) falls back to the REST API for that month's window sub-range.
        """
        specs = self._plan_sources(symbol, start_ms, end_ms, access)
        warnings: list[str] = []
        parts: list[list[dict]] = []
        n_requests = 0
        all_cached = True
        for spec in specs:
            try:
                if spec.kind == "bulk":
                    part, n, cached = self._bulk_month(
                        symbol, cast(int, spec.year), cast(int, spec.month)
                    )
                else:
                    part, n, cached = self._rest_klines(
                        symbol, cast(int, spec.start_ms), cast(int, spec.end_ms)
                    )
            except PermanentFetchError as exc:
                if spec.kind != "bulk":
                    raise
                month_label = f"{cast(int, spec.year):04d}-{cast(int, spec.month):02d}"
                warnings.append(
                    f"{symbol}: bulk archive {month_label} unavailable ({exc}); "
                    "fell back to REST API"
                )
                part, n, cached = self._rest_klines(
                    symbol,
                    max(start_ms, cast(int, spec.month_start_ms)),
                    min(end_ms, cast(int, spec.month_end_ms)),
                )
            parts.append(part)
            n_requests += n
            all_cached = all_cached and cached

        rows: list[dict] = []
        for part in parts:
            rows.extend(part)
        observed = [r for r in rows if start_ms <= int(r["open_time_ms"]) < end_ms]
        return self._dedupe(observed), n_requests, all_cached, warnings

    # ------------------------------------------------------------------
    # Source paths
    # ------------------------------------------------------------------

    def _rest_klines(self, symbol: str, start_ms: int, end_ms: int) -> tuple[list[dict], int, bool]:
        """REST ``/api/v3/klines`` with 1000-bar pagination for the window."""
        params: dict[str, object] = {
            "symbol": symbol,
            "path": "rest",
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        key = self._ref_cache_key(params)
        cached = self._ref_cache_read(key)
        if cached is not None:
            return cached, 0, True

        configured = self._endpoints.reference_base_url.strip()
        if not configured:
            raise ConfigError(
                "ReferenceFetcher: endpoints.reference_base_url is empty; a REST "
                "base URL is required for /api/v3/klines"
            )
        endpoint = f"{configured.rstrip('/')}/api/v3/klines"
        context = f"ReferenceFetcher stream=reference symbol={symbol} range_ms={start_ms}..{end_ms}"
        rows: list[dict] = []
        cursor = start_ms
        n_pages = 0
        while cursor < end_ms:  # endTime is exclusive (matches Binance /api/v3/klines)
            query = urlencode(
                {
                    "symbol": symbol,
                    "interval": INTERVAL,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": REST_LIMIT,
                }
            )
            resp = self._request(f"{endpoint}?{query}", context=context)
            try:
                payload = resp.json()
            except ValueError as exc:
                raise PermanentFetchError(
                    f"ReferenceFetcher {symbol}: REST /api/v3/klines returned non-JSON content"
                ) from exc
            if not isinstance(payload, list):
                raise PermanentFetchError(
                    f"ReferenceFetcher {symbol}: REST /api/v3/klines returned "
                    f"{type(payload).__name__}, expected a JSON array of klines"
                )
            page = [_parse_kline(item, symbol) for item in payload]
            rows.extend(page)
            n_pages += 1
            if len(page) < REST_LIMIT:
                break
            last_ms = page[-1]["open_time_ms"]
            nxt = last_ms + INTERVAL_MS
            if nxt <= cursor or nxt > end_ms:
                break
            cursor = nxt
        rows = self._dedupe(rows)
        self._ref_cache_write(key, params, rows)
        return rows, n_pages, False

    def _bulk_month(self, symbol: str, year: int, month: int) -> tuple[list[dict], int, bool]:
        """One ``data.binance.vision`` monthly zip (whole month's rows)."""
        month_label = f"{year:04d}-{month:02d}"
        params: dict[str, object] = {
            "symbol": symbol,
            "path": "bulk",
            "month": month_label,
        }
        key = self._ref_cache_key(params)
        cached = self._ref_cache_read(key)
        if cached is not None:
            return cached, 0, True
        url = (
            f"{BINANCE_VISION_BASE_URL}/data/spot/monthly/klines/{symbol}/{INTERVAL}/"
            f"{symbol}-{INTERVAL}-{month_label}.zip"
        )
        context = f"ReferenceFetcher stream=reference symbol={symbol} bulk archive {month_label}"
        resp = self._request(url, context=context)
        rows = _parse_bulk_zip(resp.content, symbol)
        self._ref_cache_write(key, params, rows)
        return rows, 1, False

    @staticmethod
    def _dedupe(rows: list[dict]) -> list[dict]:
        """Drop duplicate ``(symbol, open_time_ms)`` rows — first wins."""
        seen: set[tuple[str, int]] = set()
        out: list[dict] = []
        for row in rows:
            key = (str(row["symbol"]), int(row["open_time_ms"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Reference-specific raw cache (base's cache is block-keyed).
    # ------------------------------------------------------------------

    def _ref_cache_key(self, params: Mapping[str, object]) -> str:
        return self.cache_key(self.route, "reference", params)

    def _ref_cache_path(self, key: str) -> Path:
        return self._cache_dir / self.route.value / "reference" / f"{key}.json.gz"

    def _ref_cache_read(self, key: str) -> list[dict] | None:
        """Cached raw rows for a source spec, or ``None`` on miss / corrupt entry."""
        path = self._ref_cache_path(key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as fh:
                envelope = json.loads(fh.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive (corrupt bytes)
            logger.warning("corrupt reference cache entry at %s (%s); treating as miss", path, exc)
            return None
        if not isinstance(envelope, dict) or not isinstance(envelope.get("body"), list):
            logger.warning("corrupt reference cache envelope at %s; treating as miss", path)
            return None
        return envelope["body"]

    def _ref_cache_write(self, key: str, params: Mapping[str, object], rows: list[dict]) -> None:
        """Atomically write rows under the reference cache layout."""
        path = self._ref_cache_path(key)
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
        self,
        rows: list[dict],
        stream: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> tuple[pa.Table, dict[str, GapStats]]:
        """Build and validate a ``REFERENCE_SCHEMA`` table from raw kline rows.

        Applies :func:`forward_fill_klines`; leading-edge placeholder rows
        (null prices) are dropped because the frozen schema declares OHLC
        non-nullable — the caller already warns about them. The result is
        sorted by ``(symbol, open_time)`` (the stream's sort key) and validated
        via ``schemas.validate_table``; an empty input yields
        ``schemas.empty_table``.
        """
        if stream != "reference":
            raise ConfigError(
                f"ReferenceFetcher._rows_to_table: unsupported stream {stream!r}; "
                "this fetcher only emits 'reference' tables"
            )
        if not rows:
            return empty_table(REFERENCE_SCHEMA), {}
        grid, stats = forward_fill_klines(
            rows,
            start_utc=ms_epoch_to_utc(start_ms),
            end_utc=ms_epoch_to_utc(end_ms),
        )
        valid = [r for r in grid if r["close"] is not None]
        if not valid:
            return empty_table(REFERENCE_SCHEMA), stats

        columns: dict[str, list[object]] = {name: [] for name in REFERENCE_SCHEMA.names}
        for r in valid:
            open_ms = int(r["open_time_ms"])
            columns["open_time"].append(ms_epoch_to_utc(open_ms))
            columns["close_time"].append(ms_epoch_to_utc(open_ms + INTERVAL_CLOSE_DELTA_MS))
            columns["open"].append(cast(float, r["open"]))
            columns["high"].append(cast(float, r["high"]))
            columns["low"].append(cast(float, r["low"]))
            columns["close"].append(cast(float, r["close"]))
            columns["volume_base"].append(cast(float, r["volume_base"]))
            columns["quote_volume"].append(cast(float, r["quote_volume"]))
            columns["trades"].append(cast(int, r["trades"]))
            columns["symbol"].append(str(r["symbol"]))
            columns["is_gap_filled"].append(bool(r["is_gap_filled"]))
        arrays = {
            name: pa.array(values, type=REFERENCE_SCHEMA.field(name).type)
            for name, values in columns.items()
        }
        table = pa.table(arrays)
        table = table.sort_by([("symbol", "ascending"), ("open_time", "ascending")])
        validate_table(table, REFERENCE_SCHEMA)
        return table, stats
