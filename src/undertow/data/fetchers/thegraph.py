"""The Graph fetcher (Route A) for ``undertow.data`` — CONTRACTS.md §5.2, T05.

Bulk source for the four log streams ``{"swap", "mint", "burn", "collect"}`` (the
Roadmap §10.1.5 "pragmatic" route; T06's RPC fetcher exists to verify it and T13
asserts the routes agree exactly). Returns tables conforming to
``SCHEMA_REGISTRY[stream]``; all amounts are **raw integer token units** (big
integers stay ``int`` / decimal strings — CONTRACTS.md §0, §4.1), block-ordered by
``(block_number, log_index)`` with UTC-aware timestamps.

Endpoint
--------
The hosted service the roadmap quotes is deprecated. This fetcher targets the
decentralized gateway: ``https://gateway.thegraph.com/api/<key>/subgraphs/id/<id>``
with the ``${GRAPH_API_KEY}`` placeholder expanded by ``config.load_config`` (never
hardcoded; the key is a secret and never logged — only the host is).

Subgraph ID target: ``SUBGRAPH_ID`` below is the published deployment id of the
``uniswap/uniswap-v3`` subgraph (Ethereum, the v3-core data source the roadmap
names). **This must be confirmed at the first live pull** (the sandbox has no
``GRAPH_API_KEY``; every offline test is ``responses``-mocked) — if the confirmed
id differs, only ``config.py``/this constant change; no test needs touching.

``collect`` support: the ``uniswap/uniswap-v3`` subgraph schema indexes
``Collect`` events (the ``collects`` entity), so this fetcher supports all four
streams. If the subgraph whose id is confirmed at first live pull turns out to
lack ``collects``, ``supported_streams()`` must be narrowed to reality and T06
becomes the sole source for ``collect`` (T13/T11 must know — never fabricate an
empty ``collect`` table and call it fetched).

Amount conversion (T05 brief pitfall 3, approach (a))
-----------------------------------------------------
The Uniswap subgraph exposes Swap/Mint/Burn/Collect ``amount0``/``amount1`` as
**decimal-adjusted BigDecimal** strings; our schema requires raw integer units.
We reconstruct: ``raw = Decimal(amount) * 10**decimals`` and **refuse to round**:
a non-integral result raises ``SchemaViolationError`` naming the offending value.
``sqrtPriceX96``, ``liquidity``, tick bounds and liquidity ``amount`` are BigInt
scalars and pass through as exact integers.

Sign convention (T05 brief pitfall 4)
-------------------------------------
The subgraph amounts follow the contract convention — positive = flowed INTO the
pool. For a WETH-sell on the pinned pools, ``amount1 > 0`` and ``amount0 < 0``
(token1 = WETH enters the pool, token0 = USDC leaves). We never flip signs;
``Mint``/``Burn``/``Collect`` amounts are unsigned and validated non-negative.

Pagination (T05 brief pitfall 1)
--------------------------------
``skip`` is a trap on hosted services (capped at 5000, silently empty beyond).
We use cursor pagination on the ordering key ``(transaction.blockNumber,
logIndex)``: order by ``transaction__blockNumber`` asc and page with
``where: { pool, transaction_: { blockNumber_gte: $cursorBlock } }`` (plus the
chunk's ``blockNumber_lte`` ceiling). The subgraph exposes a single ``orderBy``
field, so the intra-block tie-break on ``logIndex`` is done client-side: the
cursor advances by **block** and the boundary overlap — the whole overlap block
re-served by the next page, plus its tail when a page boundary fell mid-block —
is de-duplicated on the ``(block_number, log_index)`` key set (see
``_fetch_chunk``). Rows-per-page limit: 1000 (``PAGE_SIZE``, the Graph Node
``first`` cap). Each 4096-block-wide base chunk paginates internally; a short
page (< ``PAGE_SIZE`` rows) is authoritative end-of-range. The only case that
cannot page under this scheme — a single block holding more than ``PAGE_SIZE``
rows, so the subgraph keeps re-serving the same page — raises ``FetchError``
instead of silently dropping rows (unreachable on the pinned pools).

Indexer lag (T05 brief pitfall 6 companion)
-------------------------------------------
Every query carries ``_meta { block { number } }``. If the indexed head is below
the requested ``end_block``, ``FetchError`` names both numbers — a short page that
looks complete is never returned.

Not used: ``transaction.gasPrice``/``gasUsed`` are transaction-level, not the
per-block base fee T07 needs — the gas stream is T07's, not this fetcher's.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from functools import cache
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import pyarrow as pa

from undertow.data.config import EndpointConfig, PoolConfig
from undertow.data.fetchers.base import (
    TRANSIENT_ERROR_PATTERNS,
    BaseHttpFetcher,
    FetchRequest,
    FetchResult,
)
from undertow.data.schemas import SCHEMA_REGISTRY, empty_table, encode_uint, validate_table
from undertow.data.types import (
    FetchError,
    PermanentFetchError,
    Route,
    SchemaViolationError,
    TransientNetworkError,
    ValidationError,
)

logger = logging.getLogger("undertow.data.fetchers.thegraph")

# ---------------------------------------------------------------------------
# Pinned constants.
# ---------------------------------------------------------------------------

SUBGRAPH_ID: Final[str] = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
"""Published deployment id of the ``uniswap/uniswap-v3`` subgraph (Ethereum).

CONFIRM AT FIRST LIVE PULL: the sandbox has no ``GRAPH_API_KEY`` so this id was
taken from published references, not verified against the gateway. If a different
deployment proves authoritative, update this constant (and the note in the PR).
"""

GATEWAY_PATH: Final[str] = "/subgraphs/id/"
PAGE_SIZE: Final[int] = 1000  # Graph Node's default `first` cap — the rows-per-page limit
MAX_PAGES_PER_CHUNK: Final[int] = 128  # safety valve against a runaway pagination bug
GRAPH_CHUNK_BLOCKS: Final[int] = 4096  # base chunk width; rows are limited by PAGE_SIZE

# entity <- stream mapping, and the query file names (queries/*.graphql).
_STREAM_ENTITY: Final[Mapping[str, str]] = {
    "swap": "swaps",
    "mint": "mints",
    "burn": "burns",
    "collect": "collects",
}

_QUERY_DIR: Final[Path] = Path(__file__).resolve().parent / "queries"
_INT_RE: Final[re.Pattern[str]] = re.compile(r"-?[0-9]+")


@cache
def _query_text(entity: str) -> str:
    """Load one GraphQL query file by entity name — queries are files, never inline strings."""
    return (_QUERY_DIR / f"{entity}.graphql").read_text(encoding="utf-8")


def _host_only(url: str) -> str:
    """``scheme://netloc`` — secrets live in the path and never reach logs."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-url>"
    return f"{parts.scheme}://{parts.netloc}"


# ---------------------------------------------------------------------------
# Scalar decode helpers (all values travel as strings or ints in GraphQL JSON).
# ---------------------------------------------------------------------------


def _as_int(value: object, *, field: str) -> int:
    """A GraphQL integer scalar (``BigInt``/``Int``) to an exact Python ``int``."""
    if isinstance(value, bool) or value is None:
        raise SchemaViolationError(f"{field}: expected an integer scalar, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _INT_RE.fullmatch(text):
            return int(text)
    raise SchemaViolationError(f"{field}: expected a decimal integer string, got {value!r}")


def _as_str(value: object, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    raise SchemaViolationError(f"{field}: expected a string, got {value!r}")


def _address(value: object, *, field: str) -> str:
    """Lowercase, checksum-stripped 42-char ``0x`` address (CONTRACTS.md §1)."""
    text = _as_str(value, field=field).strip().lower()
    if not (text.startswith("0x") and len(text) == 42):
        raise SchemaViolationError(
            f"{field}: expected a lowercase 0x-prefixed 42-char address, got {text!r}"
        )
    return text


def _block_timestamp(value: object, *, field: str = "timestamp") -> datetime:
    """Unix-seconds ``BigInt`` to a UTC-aware datetime (CONTRACTS.md §4.0.1)."""
    return datetime.fromtimestamp(_as_int(value, field=field), tz=UTC)


def _amount_to_raw_units(value: object, decimals: int, *, field: str) -> int:
    """Approach (a): BigDecimal ``amount * 10**decimals`` into raw integer units.

    Integrality is enforced — a silently rounding conversion is a correctness bug
    (T05 brief pitfall 3). A non-integral product raises ``SchemaViolationError``
    carrying the offending value. ``Decimal`` context precision is widened so a
    full-18-decimal amount near ``uint256`` bounds converts exactly.
    """
    if not 0 <= decimals <= 18:
        raise SchemaViolationError(f"{field}: token decimals must be in 0..18, got {decimals}")
    text = str(value).strip()
    try:
        amount = Decimal(text)
    except ArithmeticError as exc:  # InvalidOperation etc. — not a BigDecimal
        raise SchemaViolationError(
            f"{field}: not a decimal BigDecimal value, got {text!r}"
        ) from exc
    with localcontext() as ctx:
        ctx.prec = max(60, len(text) + decimals + 16)
        scaled = amount * (10**decimals)
    if scaled != scaled.to_integral_value():
        raise SchemaViolationError(
            f"{field}: BigDecimal amount {text!r} scaled by 10**{decimals} "
            f"is not an exact integer ({scaled}); refusing to round a raw token amount"
        )
    return int(scaled)


def _amount_raw_unsigned(value: object, decimals: int, *, field: str) -> int:
    """``_amount_to_raw_units`` plus the non-negativity the contract requires for
    mint/burn/collect amounts (CONTRACTS.md §4.2/§4.3)."""
    raw = _amount_to_raw_units(value, decimals, field=field)
    if raw < 0:
        raise SchemaViolationError(
            f"{field}: expected an unsigned amount, got raw {raw} (BigDecimal {value!r})"
        )
    return raw


# ---------------------------------------------------------------------------
# The fetcher.
# ---------------------------------------------------------------------------


class TheGraphFetcher(BaseHttpFetcher):
    """Route A: the Uniswap V3 subgraph through the decentralized gateway.

    Supports ``{"swap", "mint", "burn", "collect"}``. The base class owns retry,
    adaptive shrink, caching and concurrency; this class owns GraphQL transport,
    cursor pagination, raw-amount reconstruction and schema conformance
    (``_rows_to_table`` calls ``validate_table(strict=True)`` — CONTRACTS.md §5.1
    layering rule: the base never imports schemas.py, the subclass does).
    """

    route = Route.THEGRAPH

    def __init__(
        self,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(endpoints, cache_dir, sleep=sleep, rng=rng)
        # The subgraph limits requests by ROWS (PAGE_SIZE), not blocks, so chunks
        # can be far wider than the RPC-oriented 10-block default; each chunk
        # paginates internally with 1000-row pages.
        self._initial_chunk_size = GRAPH_CHUNK_BLOCKS
        # Instance-level so tests can exercise two-page pagination with small pages.
        self.page_size: int = PAGE_SIZE

    # ------------------------------------------------------------------
    # Public API (CONTRACTS.md §5.1/§5.2).
    # ------------------------------------------------------------------

    def supported_streams(self) -> frozenset[str]:
        """What this fetcher actually fetches: all four log streams.

        ``collect`` is included because the ``uniswap/uniswap-v3`` subgraph
        indexes the ``collects`` entity; if the subgraph id confirmed at first
        live pull disagrees, narrow this to reality.
        """
        return frozenset(_STREAM_ENTITY)

    def fetch(self, request: FetchRequest) -> FetchResult:
        if request.stream not in self.supported_streams():
            raise PermanentFetchError(
                f"TheGraphFetcher: stream {request.stream!r} is not supported; "
                f"supported streams: {sorted(self.supported_streams())}"
            )
        if request.pool is None:
            raise PermanentFetchError(
                "TheGraphFetcher: log streams are pool-scoped; request.pool is required"
            )
        rows, n_requests, from_cache, warnings = self.fetch_rows(request)
        table = self._rows_to_table(rows, request.stream, pool=request.pool)
        return FetchResult(
            table=table,
            route=self.route,
            request=request,
            n_requests=n_requests,
            from_cache=from_cache,
            warnings=tuple(warnings),
        )

    # ------------------------------------------------------------------
    # GraphQL transport (base class owns HTTP-level retry/backoff; the
    # GraphQL ``errors`` array in a 200 is classified and retried here).
    # ------------------------------------------------------------------

    def _endpoint(self) -> str:
        """Gateway URL built from the config's (secret-bearing) ``graph_url``."""
        return self._endpoints.graph_url.rstrip("/") + GATEWAY_PATH + SUBGRAPH_ID

    def _graphql(self, query: str, variables: dict[str, object], context: str) -> dict[str, Any]:
        """POST one GraphQL query and return its ``data`` object, or raise.

        A GraphQL ``errors`` array inside a 200 response is a failure — silent
        partial data is the failure mode this exists to prevent (T05 brief §shape).
        Errors matching a transient phrase are retried with the base backoff; any
        other error is ``PermanentFetchError``. HTTP-level transients (429/5xx/
        timeouts) are already retried inside ``self._request``.
        """
        url = self._endpoint()
        logger.debug("GraphQL request to %s for %s", _host_only(url), context)
        body = {"query": query, "variables": variables}
        for attempt in range(self._max_retries + 1):
            response = self._request(url, method="POST", json=body, context=context)
            try:
                payload = response.json()
            except ValueError as exc:
                raise PermanentFetchError(
                    f"{context}: gateway returned a non-JSON body ({type(exc).__name__})"
                ) from exc
            if not isinstance(payload, dict):
                raise PermanentFetchError(
                    f"{context}: gateway returned a non-object JSON body: {str(payload)[:200]!r}"
                )
            if payload.get("errors") is not None:
                self._raise_graphql_errors(payload["errors"], context, attempt)
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                raise PermanentFetchError(
                    f"{context}: GraphQL response carried no data object "
                    "(would be silent partial data)"
                )
            return data
        raise AssertionError("unreachable: retry loop always returns or raises")

    def _raise_graphql_errors(self, errors: object, context: str, attempt: int) -> None:
        """Classify a GraphQL ``errors`` value: transient -> backoff/raise, else permanent."""
        texts = _collect_error_messages(errors)
        blurb = " ; ".join(texts)
        if any(pattern in blurb.lower() for pattern in TRANSIENT_ERROR_PATTERNS):
            self._handle_transient(
                TransientNetworkError(
                    f"{context}: transient GraphQL provider error: {blurb[:200]}"
                ),
                attempt,
            )
            return
        raise PermanentFetchError(f"{context}: GraphQL errors: {blurb[:300]}")

    # ------------------------------------------------------------------
    # Template-method implementations (CONTRACTS.md §5.1).
    # ------------------------------------------------------------------

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        """One inclusive block range as raw subgraph entity dicts.

        Cursor-paginates on ``(block_number, log_index)`` (see the module
        docstring and T05 brief pitfall 1). The Graph subgraph orders by a single
        field (``transaction.blockNumber``), so the intra-block tie-break on
        ``logIndex`` cannot be done server-side and the boundary de-dup must not
        assume the keys inside a page are sorted. The scheme is therefore:

        * each page advances the cursor to the **block** of the last new row;
        * the following page re-delivers that boundary block in full (the
          subgraph has no ``skip``), so the overlap is the whole set of
          ``(block_number, log_index)`` keys already fetched for that block
          (``seen_boundary``) — plus the boundary block's **tail**, rows the
          previous page never reached because the page boundary fell mid-block;
        * rows of the boundary block already in ``seen_boundary`` are dropped;
          anything else is new and kept, so every row is returned exactly once
          and a mid-block page boundary loses nothing. When the next page is
          still full of the same boundary block, ``seen_boundary`` accumulates
          that block's additional keys instead of being reset, so a block the
          subgraph keeps re-serving from the top is never partially re-admitted.

        A page that is full yet yields zero new rows means the overlap block
        holds **at least** ``page_size`` rows and the cursor cannot advance (the
        subgraph re-serves the same leading rows forever); that raises
        ``FetchError`` instead of silently dropping rows — unreachable on the
        pinned pools.
        """
        entity = _STREAM_ENTITY.get(request.stream)
        if entity is None:  # defensive — fetch() gates streams before this
            raise PermanentFetchError(
                f"TheGraphFetcher: no query for stream {request.stream!r}; "
                f"supported: {sorted(_STREAM_ENTITY)}"
            )
        if request.pool is None:
            raise PermanentFetchError(
                f"{self._describe(request, start, end)}: log streams are pool-scoped "
                "(request.pool is required)"
            )
        query = _query_text(entity)
        context = self._describe(request, start, end)
        rows: list[dict] = []
        cursor_block = start
        _prev_cursor_block = start - 1  # forces the first page's else-branch below
        seen_boundary: set[tuple[int, int]] = set()  # keys of the overlap block already fetched
        for _page in range(1, MAX_PAGES_PER_CHUNK + 1):
            variables: dict[str, object] = {
                "pool": request.pool.address,
                "cursorBlock": str(cursor_block),
                "endBlock": str(end),
                "first": self.page_size,
            }
            data = self._graphql(query, variables, context)
            self._check_indexer_lag(data, end, context)
            entities = data.get(entity)
            if not isinstance(entities, list):
                raise PermanentFetchError(
                    f"{context}: GraphQL response is missing the {entity!r} entity array "
                    "(refusing silent partial data)"
                )
            if not entities:
                break  # subgraph returned nothing >= cursor: end of range
            fresh: list[dict] = []
            page_keys: set[tuple[int, int]] = set()
            for item in entities:
                key = self._row_key(item, context)
                if key[0] == cursor_block and key in seen_boundary:
                    continue  # overlap with the previous page's boundary block
                if key in page_keys:
                    # A genuinely duplicated key *within one page* is a data
                    # integrity error, never a benign overlap to de-dup away.
                    raise ValidationError(
                        f"{context}: duplicate (block_number, log_index) key {key} "
                        "within a single GraphQL page; refusing to emit a table "
                        "with duplicate keys"
                    )
                page_keys.add(key)
                fresh.append(item)
            rows.extend(fresh)
            if len(entities) < self.page_size:
                break  # a short page is authoritative: no more matching rows
            if not fresh:
                # Page full yet every row already seen: the overlap block holds
                # more than page_size rows and the cursor cannot advance. Never
                # loop forever, and never drop rows silently.
                raise FetchError(
                    f"{context}: at least {self.page_size} {request.stream} rows in "
                    f"block {cursor_block}; the (block_number, log_index) cursor cannot "
                    "advance — refusing to drop rows"
                )
            cursor_block = max(key[0] for key in page_keys)
            if cursor_block == _prev_cursor_block:
                # The page boundary fell inside the SAME overlap block again: the
                # block's keys already fetched must accumulate, or the overlap
                # de-dup would re-admit rows the subgraph re-serves on every page.
                seen_boundary |= {key for key in page_keys if key[0] == cursor_block}
            else:
                seen_boundary = {key for key in page_keys if key[0] == cursor_block}
            _prev_cursor_block = cursor_block
            if cursor_block > end:  # defensive; server-side blockNumber_lte already bounds it
                break
        else:
            raise FetchError(
                f"{context}: {request.stream} pagination did not converge within "
                f"{MAX_PAGES_PER_CHUNK} pages (last cursor block {cursor_block})"
            )
        return rows

    def _check_indexer_lag(self, data: Mapping[str, Any], end_block: int, context: str) -> None:
        """Refuse a page whose indexed head is below the requested range end."""
        meta = data.get("_meta")
        indexed: object | None = None
        if isinstance(meta, dict):
            block = meta.get("block")
            if isinstance(block, dict):
                indexed = block.get("number")
        if indexed is None:
            logger.warning(
                "%s: response carried no _meta.block.number; indexer-lag detection skipped",
                context,
            )
            return
        indexed_block = _as_int(indexed, field="_meta.block.number")
        if indexed_block < end_block:
            raise FetchError(
                f"{context}: indexer lag — subgraph has indexed to block {indexed_block} "
                f"but blocks through {end_block} were requested; refusing to return a "
                "short page that would masquerade as complete"
            )

    def _row_key(self, row: Mapping[str, object], context: str) -> tuple[int, int]:
        """The ordering key ``(block_number, log_index)`` of a subgraph entity."""
        transaction = row.get("transaction")
        if not isinstance(transaction, Mapping):
            raise PermanentFetchError(f"{context}: entity row has no transaction object: {row!r}")
        return (
            _as_int(transaction.get("blockNumber"), field="transaction.blockNumber"),
            _as_int(row.get("logIndex"), field="logIndex"),
        )

    # ------------------------------------------------------------------
    # Raw rows -> validated table.
    # ------------------------------------------------------------------

    def _rows_to_table(self, rows: list[dict], stream: str, *, pool: PoolConfig) -> pa.Table:
        """Decode, sort by ``(block_number, log_index)``, de-dup, and validate.

        An empty row list yields ``empty_table(schema)`` — schema-valid, zero rows
        (what an empty range must produce; CONTRACTS.md §4.9 test 9). Duplicate
        keys after pagination/dedup raise ``ValidationError`` — never a silent
        ``drop_duplicates``.
        """
        if stream not in SCHEMA_REGISTRY:
            raise PermanentFetchError(f"TheGraphFetcher: unknown stream {stream!r}")
        schema = SCHEMA_REGISTRY[stream]
        if not rows:
            return empty_table(schema)
        decoded = [self._decode_row(row, stream, pool) for row in rows]
        decoded.sort(
            key=lambda row: (
                _as_int(row["block_number"], field="block_number"),
                _as_int(row["log_index"], field="log_index"),
            )
        )
        seen: set[tuple[int, int]] = set()
        for row in decoded:
            key = (
                _as_int(row["block_number"], field="block_number"),
                _as_int(row["log_index"], field="log_index"),
            )
            if key in seen:
                raise ValidationError(
                    f"TheGraphFetcher {stream!r}: duplicate (block_number, log_index) "
                    f"key {key} after pagination; refusing to emit a table with "
                    "duplicate keys"
                )
            seen.add(key)
        columns: dict[str, object] = {}
        for field in schema:
            columns[field.name] = pa.array(
                [row[field.name] for row in decoded], type=field.type
            )
        table = pa.table(columns)
        validate_table(table, schema, strict=True)
        return table

    def _decode_row(
        self, row: Mapping[str, object], stream: str, pool: PoolConfig
    ) -> dict[str, object]:
        """One subgraph entity -> one row dict keyed by canonical column names."""
        transaction = row.get("transaction")
        pool_ref = row.get("pool")
        if not isinstance(transaction, Mapping) or not isinstance(pool_ref, Mapping):
            raise PermanentFetchError(f"TheGraphFetcher {stream!r}: malformed entity row {row!r}")
        decoded: dict[str, object] = {
            "block_number": _as_int(
                transaction.get("blockNumber"), field="transaction.blockNumber"
            ),
            "log_index": _as_int(row.get("logIndex"), field="logIndex"),
            "block_timestamp": _block_timestamp(row.get("timestamp")),
            "tx_hash": _as_str(transaction.get("id"), field="transaction.id").lower(),
            "pool_address": _as_str(pool_ref.get("id"), field="pool.id").lower(),
            "event_type": stream,
        }
        if stream == "swap":
            liquidity = row.get("liquidity")
            decoded.update(
                {
                    "amount0": encode_uint(
                        _amount_to_raw_units(
                            row.get("amount0"), pool.token0_decimals, field="amount0"
                        )
                    ),
                    "amount1": encode_uint(
                        _amount_to_raw_units(
                            row.get("amount1"), pool.token1_decimals, field="amount1"
                        )
                    ),
                    "sqrt_price_x96": encode_uint(
                        _as_int(row.get("sqrtPriceX96"), field="sqrtPriceX96")
                    ),
                    "liquidity": (
                        encode_uint(_as_int(liquidity, field="liquidity"))
                        if liquidity is not None
                        else None
                    ),
                    "tick": _as_int(row.get("tick"), field="tick"),
                    "sender": _address(row.get("sender"), field="sender"),
                    "recipient": _address(row.get("recipient"), field="recipient"),
                }
            )
        elif stream in ("mint", "burn"):
            sender = row.get("sender")  # the Burn event has no sender -> null
            decoded.update(
                {
                    "owner": _address(row.get("owner"), field="owner"),
                    "tick_lower": _as_int(row.get("tickLower"), field="tickLower"),
                    "tick_upper": _as_int(row.get("tickUpper"), field="tickUpper"),
                    "liquidity_amount": encode_uint(
                        _as_int(row.get("amount"), field="amount")
                    ),
                    "amount0": encode_uint(
                        _amount_raw_unsigned(
                            row.get("amount0"), pool.token0_decimals, field="amount0"
                        )
                    ),
                    "amount1": encode_uint(
                        _amount_raw_unsigned(
                            row.get("amount1"), pool.token1_decimals, field="amount1"
                        )
                    ),
                    "sender": _address(sender, field="sender") if sender is not None else None,
                }
            )
        elif stream == "collect":
            recipient = row.get("recipient")
            decoded.update(
                {
                    "owner": _address(row.get("owner"), field="owner"),
                    "recipient": (
                        _address(recipient, field="recipient")
                        if recipient is not None
                        else None
                    ),
                    "tick_lower": _as_int(row.get("tickLower"), field="tickLower"),
                    "tick_upper": _as_int(row.get("tickUpper"), field="tickUpper"),
                    "amount0": encode_uint(
                        _amount_raw_unsigned(
                            row.get("amount0"), pool.token0_decimals, field="amount0"
                        )
                    ),
                    "amount1": encode_uint(
                        _amount_raw_unsigned(
                            row.get("amount1"), pool.token1_decimals, field="amount1"
                        )
                    ),
                }
            )
        else:
            raise PermanentFetchError(
                f"TheGraphFetcher: no row decoder for stream {stream!r}"
            )
        return decoded


def _collect_error_messages(errors: object) -> list[str]:
    """Flatten a GraphQL ``errors`` array/string into message texts."""
    if isinstance(errors, str):
        return [errors]
    if isinstance(errors, list):
        texts: list[str] = []
        for item in errors:
            if isinstance(item, dict):
                message = item.get("message")
                texts.append(message if isinstance(message, str) else str(item))
            else:
                texts.append(str(item))
        return texts
    return [str(errors)]