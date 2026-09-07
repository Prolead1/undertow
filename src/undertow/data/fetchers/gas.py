"""Per-block gas / block-index fetcher for ``undertow.data`` (T07).

Owns ``GAS_SCHEMA`` production — one row **per block**, no gaps — and the public
``timestamp <-> block`` index that resolves date-mode ``WindowConfig`` windows to block
ranges (``CONTRACTS.md`` §5.3, ADR-003). Roadmap role: ``G_t``, the rebalancing friction
(§10.1.6 / §10.2.3); spikes are real and must survive un-smoothed, which is why there is
**no averaging or interpolation anywhere in this module** — an unavailable value is ``0``
with a warning, or the fetch fails.

Data path (the brief's preferred route, faithful to the archive-node wire format):

* **Per-block header fields** (``block_number``, ``block_timestamp``, ``base_fee_per_gas``,
  ``gas_used``, ``gas_limit``) come from one **batched** JSON-RPC ``eth_getBlockByNumber`` per
  chunk — their authoritative source (real nodes' ``eth_feeHistory`` returns only
  ``oldestBlock``/``baseFeePerGas``/``gasUsedRatio``/``reward``, never ``gasUsed``/
  ``gasLimit``/per-block ``blockNumber`` — reading them there would silently write ``0``).
* ``eth_feeHistory(blockCount, newestBlock, [50, 90])`` — one call per
  ``FEE_HISTORY_BLOCK_COUNT`` blocks — supplies the per-block ``reward`` array (the two
  effective-priority-fee percentiles in the order requested), aligned to blocks via its
  ``oldestBlock`` field. A ``reward`` entry of ``null`` means the block has no transactions;
  those blocks get ``priority_fee_p*_wei = 0`` and a warning — never ``null``-propagated,
  never smoothed.

``eth_usd_price`` is left **null** here; T11 fills it from the reference feed via a backward
as-of join. ``base_fee_per_gas`` and the priority fees are stored as decimal strings per the
big-int convention and handled as ``int`` end-to-end.

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
    encode_uint,
    empty_table,
    validate_table,
)
from undertow.data.types import (
    BlockNumber,
    ConfigError,
    PermanentFetchError,
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

_PERCENTILE_P50: int = 50
_PERCENTILE_P90: int = 90
_SUPPORTED_PERCENTILES: frozenset[int] = frozenset({_PERCENTILE_P50, _PERCENTILE_P90})

# Marker key carried on a raw row dict when eth_feeHistory reported reward=null (no txs in
# the block). It is popped before table-building and its presence is surfaced as a warning
# by ``fetch``. It round-trips through the raw cache so a cache hit is still flagged.
_NULL_REWARD_MARKER: str = "_priority_reward_null"


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
    gas_row: Mapping[str, object], gas_units: int, *, percentile: int = _PERCENTILE_P50
) -> int:
    """``(base_fee_per_gas + priority_fee_p{percentile}_wei) * gas_units``, exactly, as ``int``.

    ``gas_row`` is a row of the gas stream — big integers as decimal strings or ``int``.
    ``gas_units`` comes from ``config.GAS_UNITS`` (``DataConfig.gas_units``); it is never
    hardcoded here. Only percentiles 50 and 90 exist in the stream; anything else is a
    ``ValueError``. This is the function ``undertow.sim`` will call to price a rebalance, so
    it stays integer and pure — no float, no network, no state.
    """
    if percentile not in _SUPPORTED_PERCENTILES:
        raise ValueError(
            f"gas_cost_wei: percentile must be 50 or 90, got {percentile!r} "
            "(the gas stream only carries those two)"
        )
    base = _coerce_int(gas_row["base_fee_per_gas"])
    priority = _coerce_int(gas_row[f"priority_fee_p{percentile}_wei"])
    units = gas_units
    if isinstance(units, bool) or not isinstance(units, int):
        raise TypeError(f"gas_cost_wei: gas_units must be an int, got {type(units).__name__}")
    cost = (base + priority) * units
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
    # JSON-RPC plumbing (HTTP + transient retry handled by base._request)
    # ------------------------------------------------------------------

    def _rpc(self, method: str, params: list[object], context: str) -> object:
        """Single JSON-RPC call; returns ``result`` or raises a ``FetchError``."""
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

    def _rpc_batch_headers(
        self, blocks: list[int], context: str
    ) -> list[tuple[int, dict[str, object]]]:
        """One batched ``eth_getBlockByNumber``; returns ``(block_number, header)`` pairs.

        Returned in response order — a provider may legitimately reorder a batch, and
        ``_rows_to_table`` sorts before validating. Each header carries ``number``,
        ``timestamp``, ``baseFeePerGas``, ``gasUsed`` and ``gasLimit`` (the authoritative
        per-block fields, per the brief's preferred route). A duplicate block in the response
        is deliberately kept: the completeness contract in ``_rows_to_table`` is what rejects
        it.
        """
        calls = [
            {
                "jsonrpc": "2.0",
                "id": i,
                "method": "eth_getBlockByNumber",
                "params": [_to_hex(block), False],
            }
            for i, block in enumerate(blocks)
        ]
        response = self._request(self._rpc_url, method="POST", json=calls, context=context)
        payload = response.json()
        if not isinstance(payload, list):
            raise PermanentFetchError(f"{context}: batch eth_getBlockByNumber returned a non-batch")
        out: list[tuple[int, dict[str, object]]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "error" in item:
                raise PermanentFetchError(
                    f"{context}: batch eth_getBlockByNumber error: {item.get('error')!r}"
                )
            result = item.get("result")
            if isinstance(result, dict) and "number" in result:
                out.append((_from_hex(result["number"]), dict(result)))
        return out

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

        null_reward = sum(1 for r in rows if r.get(_NULL_REWARD_MARKER, False))
        if null_reward:
            warning = (
                f"{null_reward} block(s) had no transactions (eth_feeHistory reward=null); "
                "their priority_fee_p50_wei and priority_fee_p90_wei are set to 0"
            )
            logger.warning("gas fetch: %s", warning)
            warnings.append(warning)

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
        # Strip the transient null-reward marker; it must not leak into the data.
        cleaned = []
        for row in rows:
            stripped = dict(row)
            stripped.pop(_NULL_REWARD_MARKER, None)
            cleaned.append(stripped)
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

    def _fee_history_rewards(
        self, start: int, end: int, context: str
    ) -> dict[int, object | None]:
        """``eth_feeHistory`` reward (priority-fee percentile) array -> ``{block: reward|null}``.

        Reward entries are aligned to blocks via the response's ``oldestBlock`` (entry ``i``
        corresponds to block ``oldestBlock + i``), with a positional fallback onto ``[start,
        end]`` when a provider omits ``oldestBlock``. An entry ``None`` means the block had no
        transactions; an overall missing reward array is a provider failure and raises —
        priority fees are never invented.
        """
        block_count = end - start + 1
        # percentile ids are requested in the order [50, 90]; reward entries follow suit.
        result = self._rpc(
            "eth_feeHistory",
            [block_count, _to_hex(end), [_PERCENTILE_P50, _PERCENTILE_P90]],
            context,
        )
        if not isinstance(result, dict):
            raise PermanentFetchError(f"{context}: eth_feeHistory returned an unexpected shape")
        rewards = result.get("reward")
        if not isinstance(rewards, list):
            raise PermanentFetchError(
                f"{context}: eth_feeHistory returned no reward array for {start}..{end}; "
                "effective priority fees cannot be computed and are never invented"
            )
        oldest = result.get("oldestBlock")
        anchor = _from_hex(oldest) if isinstance(oldest, str) else start
        out: dict[int, object | None] = {}
        for i, reward in enumerate(rewards):
            out[anchor + i] = reward
        if len(out) != block_count:
            # A short/long reward array from a degraded provider would silently turn trailing
            # blocks into the reward=null (0+warning) case. Real nodes return exactly
            # block_count entries — make any deviation loud instead.
            raise PermanentFetchError(
                f"{context}: eth_feeHistory returned {len(out)} reward entries for "
                f"{block_count} blocks ({start}..{end}); refusing to interpret trailing "
                "blocks as no-transaction"
            )
        return out

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        """Raw ``GAS_SCHEMA``-shaped row dicts for the inclusive range ``[start, end]``.

        One batched ``eth_getBlockByNumber`` (authoritative header fields) plus one
        ``eth_feeHistory`` call (priority-fee percentiles). Rows are emitted only for blocks the
        node actually returned: a provider that omits a block produces a gap that the
        completeness contract in ``_rows_to_table`` turns into a ``ValidationError`` naming the
        first missing block — never a fabricated row.
        """
        context = self._describe(request, start, end)
        headers = self._rpc_batch_headers(list(range(start, end + 1)), context)
        if not headers:
            raise PermanentFetchError(
                f"{context}: batched eth_getBlockByNumber returned no headers for "
                f"{start}..{end}"
            )
        rewards = self._fee_history_rewards(start, end, context)

        rows: list[dict] = []
        for bn, header in headers:
            reward = rewards.get(bn)
            missing = [
                key
                for key in ("timestamp", "baseFeePerGas", "gasUsed", "gasLimit")
                if key not in header
            ]
            if missing:
                # A header present but missing fields must not silently become 0 (or epoch-0).
                # Post-London finalized headers always carry all four — raise instead.
                raise PermanentFetchError(
                    f"{context}: eth_getBlockByNumber header for block {bn} is missing "
                    f"required field(s) {missing}; refusing to write silent defaults"
                )
            row: dict = {
                "block_number": bn,
                "block_timestamp": _from_hex(header["timestamp"]),
                "base_fee_per_gas": _from_hex(header["baseFeePerGas"]),
                "gas_used": _from_hex(header["gasUsed"]),
                "gas_limit": _from_hex(header["gasLimit"]),
                "priority_fee_p50_wei": 0,
                "priority_fee_p90_wei": 0,
            }
            if reward is None:
                # No transactions in the block: priority fee is 0, flagged (never interpolated,
                # never null-propagated). The marker surfaces a warning in fetch().
                row[_NULL_REWARD_MARKER] = True
            else:
                if not isinstance(reward, list) or len(reward) < 2:
                    raise PermanentFetchError(
                        f"{context}: malformed eth_feeHistory reward entry for block {bn}: "
                        f"{reward!r} (expected a [p50, p90] hex pair)"
                    )
                row["priority_fee_p50_wei"] = _from_hex(reward[0])
                row["priority_fee_p90_wei"] = _from_hex(reward[1])
            rows.append(row)
        return rows