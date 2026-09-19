"""Per-block gas / block-index fetcher for ``undertow.data`` (T07).

Owns ``GAS_SCHEMA`` production — one row **per block**, no gaps — and the public
``timestamp <-> block`` index that resolves date-mode ``WindowConfig`` windows to block
ranges (``CONTRACTS.md`` §5.3, ADR-003). Roadmap role: ``G_t``, the rebalancing friction
(§10.1.6 / §10.2.3); spikes are real and must survive un-smoothed, which is why there is
**no averaging or interpolation anywhere in this module** — an unavailable value is ``0``
with a warning, or the fetch fails.

Data path (ADR-005 — flat tip surcharge; priority fees are not fetched from the chain):

* ``eth_feeHistory(blockCount, newestBlock, [])`` — one call per ``FEE_HISTORY_BLOCK_COUNT``
  blocks — supplies ``baseFeePerGas[]`` and ``oldestBlock``.
* ``block_timestamp`` is **always fetched** from the chain via batched ``eth_getBlockByNumber``
  — post-merge skipped slots make a linear slot-spacing model inaccurate (cumulative drift of
  many hours over a multi-year window defeats the 30 s join tolerance; ADR-005 §Decision.2
  amended). Per-block gas header fields (``gas_used``, ``gas_limit``) are still written as
  ``0`` — nobody reads them.
* ``priority_fee_p50_wei`` and ``priority_fee_p90_wei`` are always written as ``0`` — they are
  not consumed by any module (ADR-005 §Decision.4).

``eth_usd_price`` is left **null** here; T11 fills it from the reference feed via a backward
as-of join. ``base_fee_per_gas`` is stored as decimal strings per the big-int convention
and handled as ``int`` end-to-end.

The block index (``timestamp_for_block`` / ``block_for_timestamp``) is binary-searched and
memoized in-process; ``block_for_timestamp`` returns the **last** block whose timestamp is
``<= ts`` and is exact at boundaries. Date-mode windows (``start_block is None``) resolve their
bounds through it.

LAYERING RULE (``CONTRACTS.md`` §5.1): the base class knows HTTP, ranges and bytes; this module
owns ``_rows_to_table`` and calls ``validate_table`` itself before wrapping a ``FetchResult``.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from undertow.data.config import EndpointConfig
from undertow.data.fetchers.base import BaseHttpFetcher, FetchRequest, FetchResult
from undertow.data.schemas import (
    GAS_SCHEMA,
    decode_uint,
    empty_table,
    encode_uint,
    validate_table,
)
from undertow.data.types import (
    BlockNumber,
    ConfigError,
    PermanentFetchError,
    RateLimitError,
    Route,
    ValidationError,
)

logger = logging.getLogger("undertow.data.fetchers.gas")

# ---------------------------------------------------------------------------
# Task constants. The London fork flipped Ethereum to EIP-1559 (Aug 2021, block
# ~12,965,000); the pinned window (2022-01-01) is safely past it, so baseFeePerGas is
# always present and we never need a pre-London path. These are module-local because
# config.py is owned by T01 and may not be edited here.
# ---------------------------------------------------------------------------
EIP1559_LONDON_BLOCK: int = 12_965_000
"""First block where ``baseFeePerGas`` is guaranteed (London fork). Any requested range that
starts before this block raises ``ConfigError`` — we never invent a base fee, and stale
headers from pre-London blocks carry no ``baseFeePerGas`` at all."""

FEE_HISTORY_BLOCK_COUNT: int = 1024
"""Max ``blockCount`` per ``eth_feeHistory`` call; also our chunk size (≈ one HTTP call
per 1024 blocks, per the brief)."""

MAX_JSONRPC_BATCH_SIZE: int = 500
"""Max ``eth_getBlockByNumber`` calls in one HTTP JSON-RPC batch.

Archive providers reject oversized batches (Alchemy: "maximum batch request size is 1000",
HTTP 400). A gas chunk spans ``FEE_HISTORY_BLOCK_COUNT`` (1024) blocks, so timestamps for a
chunk are fetched in batches of at most this many calls. Kept below the common provider cap
to leave headroom and to bound the burst size of a single request."""




# ---------------------------------------------------------------------------
# Hex codecs for Ethereum JSON-RPC wire values.
# ---------------------------------------------------------------------------


def _from_hex(value: object) -> int:
    """``0x``-prefixed hex string (or int) -> int. Tolerant of already-int values."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16)
    raise TypeError(f"expected hex string or int, got {type(value).__name__} {value!r}")


def _to_hex(value: int) -> str:
    """int -> lowercase ``0x``-prefixed hex string."""
    return hex(int(value))





def _coerce_int(value: object) -> int:
    """Big-int coercion for ``gas_cost_wei``: accept int or decimal string only."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return decode_uint(value)
    raise TypeError(
        f"expected an int or decimal string, got {type(value).__name__} {value!r}"
    )


def _make_row(block_number: int, timestamp: int, base_fee_per_gas: int) -> dict:
    """Build one raw gas row with the ADR-005 zeroed columns.

    ``gas_used``, ``gas_limit``, ``priority_fee_p50_wei``, and ``priority_fee_p90_wei``
    are not consumed by any module (ADR-005 §Decision.4–5), so they are written as 0.
    """
    return {
        "block_number": block_number,
        "block_timestamp": timestamp,
        "base_fee_per_gas": base_fee_per_gas,
        "gas_used": 0,
        "gas_limit": 0,
        "priority_fee_p50_wei": 0,
        "priority_fee_p90_wei": 0,
    }


def _as_utc(value: object) -> datetime:
    """Unix epoch-seconds -> UTC-aware datetime, or a passthrough datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ConfigError(f"block timestamps must be timezone-aware UTC; got {value!r}")
        return value.astimezone(UTC)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(
            f"expected epoch seconds or a datetime, got {type(value).__name__} {value!r}"
        )
    return datetime.fromtimestamp(int(value), tz=UTC)


# ---------------------------------------------------------------------------
# Public gas-cost helper (CONTRACTS.md §4.5, §5.3) — pure and integer.
# ---------------------------------------------------------------------------


def gas_cost_wei(
    gas_row: Mapping[str, object],
    gas_units: int,
    *,
    tip_surcharge_pct: int = 3,
) -> int:
    """``(base_fee_per_gas * (100 + tip_surcharge_pct) // 100) * gas_units``, exactly, as ``int``.

    ``gas_row`` is a row of the gas stream — ``base_fee_per_gas`` as a decimal string or
    ``int``. ``gas_units`` comes from ``config.GAS_UNITS`` (``DataConfig.gas_units``); it is
    never hardcoded here. ``tip_surcharge_pct`` is a non-negative integer percentage uplift
    on the base fee to approximate priority fees (ADR-005). This is the function
    ``undertow.sim`` will call to price a rebalance, so it stays integer and pure — no
    float, no network, no state.
    """
    if not isinstance(tip_surcharge_pct, int) or tip_surcharge_pct < 0:
        raise ValueError(
            f"gas_cost_wei: tip_surcharge_pct must be a non-negative int, "
            f"got {tip_surcharge_pct!r}"
        )
    base = _coerce_int(gas_row["base_fee_per_gas"])
    total = base * (100 + tip_surcharge_pct) // 100
    units = gas_units
    if isinstance(units, bool) or not isinstance(units, int):
        raise TypeError(f"gas_cost_wei: gas_units must be an int, got {type(units).__name__}")
    cost = total * units
    if isinstance(cost, bool) or not isinstance(cost, int):
        raise TypeError("gas_cost_wei: cost must be an integer; a float here is a bug")
    return cost


# ---------------------------------------------------------------------------
# The fetcher.
# ---------------------------------------------------------------------------


class GasFetcher(BaseHttpFetcher):
    """One finalized ``GAS_SCHEMA`` row per block, no gaps (``CONTRACTS.md`` §5.3).

    Chain-wide and non-log: the table carries **no** ``log_index`` / ``tx_hash`` /
    ``pool_address`` / ``event_type`` columns and is not partitioned per pool.

    :param endpoints: resolved endpoint + policy config; ``rpc_url`` must point at an archive
        node exposing ``eth_feeHistory``.
    """

    route = Route.RPC

    def __init__(
        self,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(endpoints, cache_dir, sleep=sleep, rng=rng)
        self._rpc_url: str = endpoints.rpc_url
        # eth_feeHistory covers ~1024 blocks per call; make that the chunk size so we do not
        # default to the base class's tiny 10-block chunks over a 2.5M-block window.
        self._initial_chunk_size = FEE_HISTORY_BLOCK_COUNT
        # In-process memo for the block index (see block_for_timestamp). This is deliberately
        # in-memory only, NOT the on-disk raw cache: a single-block header row shares the same
        # (route, stream, start=end) cache namespace as a 1-block eth_feeHistory chunk and would
        # silently collide with it.
        self._ts_memo: dict[BlockNumber, datetime] = {}

    def supported_streams(self) -> frozenset[str]:
        return frozenset({"gas"})

    # ------------------------------------------------------------------
    # JSON-RPC plumbing with retry on transient provider errors.
    # ------------------------------------------------------------------

    def _with_jsonrpc_retry(self, fn: Callable[[], object], context: str) -> object:
        """Retry loop for JSON-RPC-body transient errors (rate-limit phrases
        inside a 200 response). HTTP 5xx and native 429 are already retried
        by ``_request`` — this layer only adds retries for per-item errors
        the provider returned in a success response body.
        """
        attempt = 0
        while True:
            try:
                return fn()
            except RateLimitError as exc:
                self._handle_transient(exc, attempt)
                attempt += 1

    def _rpc(self, method: str, params: list[object], context: str) -> object:
        """Single JSON-RPC call; retries transient errors, returns ``result``."""
        def once() -> object:
            body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params,
            }
            response = self._request(self._rpc_url, method="POST", json=body, context=context)
            payload = response.json()
            self._raise_for_jsonrpc(payload, context)
            if not isinstance(payload, dict) or "result" not in payload:
                raise PermanentFetchError(
                    f"{context}: unexpected JSON-RPC response shape for {method!r}"
                )
            return payload.get("result")
        return self._with_jsonrpc_retry(once, context)

    def _fee_history_base_fees(
        self, start: int, end: int, context: str
    ) -> dict[int, int]:
        """``eth_feeHistory`` with empty percentiles -> ``{block: base_fee_per_gas}``.

        No reward percentiles are requested — archive nodes reject them (ADR-005 §Problem.1).
        The response carries ``baseFeePerGas[]`` aligned to blocks via ``oldestBlock``.

        EIP-1559 returns ``block_count + 1`` base-fee entries: entry ``i`` (``i`` in
        ``[0, block_count)``) belongs to block ``oldestBlock + i``, and the trailing entry is
        the *projected* base fee for the block after ``newestBlock``. Only the first
        ``block_count`` are consumed; the trailing projection is never read as a row. An
        array genuinely shorter than ``block_count`` still raises, so a gap can never be
        silently interpreted as a zero fee.
        """
        block_count = end - start + 1
        result = self._rpc(
            "eth_feeHistory",
            [block_count, _to_hex(end), []],
            context,
        )
        if not isinstance(result, dict):
            raise PermanentFetchError(f"{context}: eth_feeHistory returned an unexpected shape")
        base_fees = result.get("baseFeePerGas")
        if not isinstance(base_fees, list):
            raise PermanentFetchError(
                f"{context}: eth_feeHistory returned no baseFeePerGas array for {start}..{end}"
            )
        oldest = result.get("oldestBlock")
        anchor = _from_hex(oldest) if isinstance(oldest, str) else start
        if len(base_fees) < block_count:
            raise PermanentFetchError(
                f"{context}: eth_feeHistory returned {len(base_fees)} baseFeePerGas entries "
                f"for {block_count} blocks ({start}..{end}); refusing to interpret missing "
                "blocks as zero-fee"
            )
        return {anchor + i: _from_hex(base_fees[i]) for i in range(block_count)}

    # ------------------------------------------------------------------
    # The block index (public, owns date->block resolution; ADR-003)
    # ------------------------------------------------------------------

    def timestamp_for_block(self, block: BlockNumber) -> datetime:
        """UTC-aware block time for ``block``, memoized in-process.

        A single ``eth_getBlockByNumber`` header lookup on a miss. Used both directly (T14,
        T11 annotate rows) and as the leaf of the ``block_for_timestamp`` binary search.
        """
        block = BlockNumber(int(block))
        memo = self._ts_memo.get(block)
        if memo is not None:
            return memo
        context = f"gas block index: timestamp_for_block(block={block})"
        result = self._rpc("eth_getBlockByNumber", [_to_hex(block), False], context)
        if not isinstance(result, dict) or "timestamp" not in result:
            raise PermanentFetchError(f"{context}: block header missing 'timestamp'")
        dt = datetime.fromtimestamp(_from_hex(result["timestamp"]), tz=UTC)
        self._ts_memo[block] = dt
        return dt

    def _latest_block(self) -> BlockNumber:
        """Current chain head via ``eth_blockNumber``."""
        result = self._rpc("eth_blockNumber", [], "gas block index: eth_blockNumber")
        if isinstance(result, str):
            return BlockNumber(_from_hex(result))
        if isinstance(result, dict) and "number" in result:
            return BlockNumber(_from_hex(result["number"]))
        raise PermanentFetchError("gas block index: eth_blockNumber returned an unexpected shape")

    def block_for_timestamp(self, ts: datetime) -> BlockNumber:
        """The **last** block with ``timestamp <= ts`` — exact at boundaries (CONTRACTS §5.3).

        Rejects (``ConfigError``) a naive ``ts``, a ``ts`` before the chain genesis, or a
        ``ts`` after the current chain head. This is what T01's date-mode windows and T14 call
        to turn dates into blocks; the memoized ``timestamp_for_block`` makes the ~log(N)
        lookups cheap across calls.
        """
        if not isinstance(ts, datetime) or ts.tzinfo is None or ts.utcoffset() is None:
            raise ConfigError(
                "block_for_timestamp: ts must be a timezone-aware datetime "
                f"(naive datetimes are not allowed); got {ts!r}"
            )
        target = ts.astimezone(UTC)

        genesis = self.timestamp_for_block(BlockNumber(0))
        if target < genesis:
            raise ConfigError(
                "block_for_timestamp: ts "
                f"{target.isoformat()} is before the chain genesis ({genesis.isoformat()})"
            )

        latest = self._latest_block()
        latest_ts = self.timestamp_for_block(latest)
        if target > latest_ts:
            raise ConfigError(
                "block_for_timestamp: ts "
                f"{target.isoformat()} is after the latest block {latest} "
                f"({latest_ts.isoformat()})"
            )

        # Binary search for the last b in [0, latest] with timestamp(b) <= target. Historical
        # block timestamps are monotone increasing, so the predicate is monotone and this is
        # exact at boundaries (a target equal to a block's timestamp returns that block).
        lo, hi = 0, int(latest)
        answer = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.timestamp_for_block(BlockNumber(mid)) <= target:
                answer = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return BlockNumber(answer)

    # ------------------------------------------------------------------
    # Pre-London boundary (EIP-1559 only)
    # ------------------------------------------------------------------

    def _reject_pre_london(self, request: FetchRequest) -> None:
        """Raise ``ConfigError`` if the requested range touches any pre-London block.

        The gate is on ``start_block``: a range that *starts* before London contains blocks
        with no ``baseFeePerGas``, and if those ever reached the batch fetch they would be
        silently read as 0 from a header lacking the field. Rejecting the whole range is the
        only honest response — base fees are never invented or defaulted.
        """
        if request.start_block is not None and request.start_block < EIP1559_LONDON_BLOCK:
            raise ConfigError(
                "gas requires EIP-1559 base fees (the pinned window is post-London). "
                f"Requested range starts at block {request.start_block}, before the London "
                f"fork at block {EIP1559_LONDON_BLOCK}; no baseFeePerGas exists there."
            )

    # ------------------------------------------------------------------
    # fetch() — assemble, validate and wrap
    # ------------------------------------------------------------------

    def fetch(self, request: FetchRequest) -> FetchResult:
        """Fetch ``GAS_SCHEMA`` rows for ``[start_block, end_block]``, one per block, no gaps.

        A date-mode request (block bounds unset, date bounds set) is resolved first through
        :meth:`block_for_timestamp` — ADR-003 semantics: ``block_for_timestamp`` returns the
        **last** block with ``timestamp <= ts``, exact at boundaries. A request with *neither*
        block nor date bounds yields an empty table and a warning; there is nothing to resolve
        and nothing to fetch.
        """
        if request.stream != "gas":
            raise ValueError(
                f"GasFetcher only serves stream 'gas', got {request.stream!r}"
            )
        warnings: list[str] = []
        if request.start_block is None or request.end_block is None:
            if request.start_utc is None or request.end_utc is None:
                logger.warning(
                    "gas fetch received neither block nor date bounds (stream=%r); "
                    "returning an empty table",
                    request.stream,
                )
                return FetchResult(
                    empty_table(GAS_SCHEMA),
                    self.route,
                    request,
                    0,
                    False,
                    ("gas fetch: request has neither block nor date bounds; nothing to fetch",),
                )
            start = self.block_for_timestamp(request.start_utc)
            end = self.block_for_timestamp(request.end_utc)
            logger.info(
                "gas fetch: resolved date window %s..%s to blocks %s..%s",
                request.start_utc.isoformat(),
                request.end_utc.isoformat(),
                start,
                end,
            )
            request = replace(
                request,
                start_block=BlockNumber(start),
                end_block=BlockNumber(end),
            )
        self._reject_pre_london(request)
        rows, n_requests, from_cache, warnings_list = self.fetch_rows(request)
        warnings.extend(warnings_list)

        table = self._rows_to_table(rows, request)
        return FetchResult(table, self.route, request, n_requests, from_cache, tuple(warnings))

    # ------------------------------------------------------------------
    # Table-building (this fetcher OWNS it — the base never touches schemas)
    # ------------------------------------------------------------------

    def _rows_to_table(self, rows: list[dict], request: FetchRequest) -> pa.Table:
        """Build, validate (strict) and return a ``GAS_SCHEMA`` table from raw rows.

        The completeness contract lives here: the block numbers must form the exact contiguous
        ascending run ``start..end`` with no duplicates, or ``ValidationError`` names the first
        missing block. No averaging, no interpolation, ever.
        """
        cleaned = [dict(row) for row in rows]
        cleaned.sort(key=lambda r: int(r["block_number"]))
        self._check_contiguous(cleaned, request)

        n = len(cleaned)
        numbers = [int(r["block_number"]) for r in cleaned]
        timestamps = [_as_utc(r["block_timestamp"]) for r in cleaned]
        base_fees = [encode_uint(_coerce_int(r["base_fee_per_gas"])) for r in cleaned]
        gas_used = [_coerce_int(r["gas_used"]) for r in cleaned]
        gas_limits = [_coerce_int(r["gas_limit"]) for r in cleaned]
        p50 = [encode_uint(_coerce_int(r["priority_fee_p50_wei"])) for r in cleaned]
        p90 = [encode_uint(_coerce_int(r["priority_fee_p90_wei"])) for r in cleaned]

        table = pa.table(
            {
                "block_number": numbers,
                "block_timestamp": timestamps,
                "base_fee_per_gas": base_fees,
                "gas_used": gas_used,
                "gas_limit": gas_limits,
                "priority_fee_p50_wei": p50,
                "priority_fee_p90_wei": p90,
                "eth_usd_price": [None] * n,
            },
            schema=GAS_SCHEMA,
        )
        validate_table(table, GAS_SCHEMA, strict=True)
        return table

    def _check_contiguous(self, rows: list[dict], request: FetchRequest) -> None:
        """Enforce the completeness contract: contiguous ascending run, no duplicates."""
        numbers = [int(r["block_number"]) for r in rows]
        seen: set[int] = set()
        for bn in numbers:
            if bn in seen:
                raise ValidationError(
                    f"gas: duplicate block_number {bn} in the fetched range; "
                    "the stream must have exactly one row per block"
                )
            seen.add(bn)
        start, end = request.start_block, request.end_block
        if start is None or end is None:
            return
        expected = set(range(int(start), int(end) + 1))
        missing = expected - seen
        if missing:
            first = min(missing)
            raise ValidationError(
                f"gas: block range {start}..{end} is not contiguous; first missing block is "
                f"{first} ({len(missing)} missing total)"
            )

    # ------------------------------------------------------------------
    # Raw fetch (schema-free; called by base.fetch_rows per chunk)
    # ------------------------------------------------------------------

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        """Raw ``GAS_SCHEMA``-shaped row dicts for the inclusive range ``[start, end]``.

        Base fees come from ``eth_feeHistory`` with empty percentiles (ADR-005).
        Timestamps are **always** fetched from the chain via batched
        ``eth_getBlockByNumber`` — post-merge skipped slots make a linear slot-spacing
        model inaccurate (cumulative drift of many hours defeats the 30 s join tolerance).
        ``gas_used``, ``gas_limit``, ``priority_fee_p50_wei``, and ``priority_fee_p90_wei``
        are written as ``0`` per ADR-005 §Decision.4–5.

        Rows are emitted for every block in the range: the completeness contract in
        ``_rows_to_table`` turns a missing block into a ``ValidationError`` naming the
        first missing block — never a fabricated row.
        """
        context = self._describe(request, start, end)
        base_fees = self._fee_history_base_fees(start, end, context)
        blocks = list(range(start, end + 1))
        headers = self._batch_headers(blocks, context)
        rows: list[dict] = []
        for bn in blocks:
            if bn not in base_fees:
                raise PermanentFetchError(
                    f"{context}: eth_feeHistory is missing baseFeePerGas for block {bn} "
                    f"in range {start}..{end}"
                )
            if bn not in headers:
                raise PermanentFetchError(
                    f"{context}: batch eth_getBlockByNumber missing block {bn} "
                    f"in range {start}..{end}"
                )
            ts = headers[bn]
            rows.append(_make_row(bn, ts, base_fees[bn]))
        return rows

    def _batch_headers(
        self, blocks: list[int], context: str
    ) -> dict[int, int]:
        """Batched ``eth_getBlockByNumber`` calls for block timestamps.

        Returns ``{block_number: unix_timestamp}``. Base fees come from
        ``_fee_history_base_fees`` — they are not returned here.
        A missing or erroneous response item raises ``PermanentFetchError`` —
        timestamps are never fabricated.

        ``blocks`` is split into batches of at most :data:`MAX_JSONRPC_BATCH_SIZE` because
        archive providers reject a single batch above their cap (HTTP 400). Each sub-batch
        is an independent request with its own retry policy.
        """
        out: dict[int, int] = {}
        for offset in range(0, len(blocks), MAX_JSONRPC_BATCH_SIZE):
            batch = blocks[offset : offset + MAX_JSONRPC_BATCH_SIZE]
            out.update(self._batch_headers_once(batch, context))
        return out

    def _batch_headers_once(
        self, blocks: list[int], context: str
    ) -> dict[int, int]:
        """One batched ``eth_getBlockByNumber`` request for ``blocks``.

        ``blocks`` must not exceed :data:`MAX_JSONRPC_BATCH_SIZE`; the caller owns splitting.
        A violation is an internal programming error, not a provider condition, so it fails
        locally rather than as a provider HTTP 400.
        """
        if len(blocks) > MAX_JSONRPC_BATCH_SIZE:
            raise ValueError(
                f"_batch_headers_once got {len(blocks)} blocks; max is "
                f"{MAX_JSONRPC_BATCH_SIZE} — split before calling"
            )
        calls = [
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "eth_getBlockByNumber",
                "params": [_to_hex(bn), False],
            }
            for i, bn in enumerate(blocks)
        ]
        response = self._request(
            self._rpc_url, method="POST", json=calls, context=context
        )
        payload = response.json()
        if not isinstance(payload, list):
            raise PermanentFetchError(
                f"{context}: batch eth_getBlockByNumber returned a non-batch"
            )
        out: dict[int, int] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "error" in item:
                raise PermanentFetchError(
                    f"{context}: batch eth_getBlockByNumber error: "
                    f"{item.get('error')!r}"
                )
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            number = result.get("number")
            if not isinstance(number, str):
                continue
            ts_raw = result.get("timestamp")
            if not isinstance(ts_raw, str):
                raise PermanentFetchError(
                    f"{context}: block {number} header missing timestamp"
                )
            out[_from_hex(number)] = _from_hex(ts_raw)
        return out