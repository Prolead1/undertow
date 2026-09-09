"""RPC fetcher — CONTRACTS.md Route C (T06): eth_getLogs + eth_call with ABI decode.

Two jobs (roadmap §10.1.3 / §10.1.5): (1) an independent second opinion on the
four log streams plus ``flash`` (T13 compares against The Graph's tables
field-for-field); (2) the sole source of the fee-growth snapshots T10 reconciles
against.

Architecture (per CONTRACTS.md §5.1's layering rule): :class:`BaseHttpFetcher`
provides retry / block-range chunking / the on-disk raw cache and NEVER imports
``schemas.py``. This module implements the subclass contract — ``_fetch_chunk``
(the eth_getLogs + ABI decode + block-timestamp resolution work) and
``_rows_to_table`` (schema validation, in the subclass, per the rule) — and the
standalone ``fee_growth_at`` for the eth_call snapshots.

Design notes / provider assumptions (handoff items; see the T06 PR body):

* **Provider.** Assumed a Geth-compatible / Erigon-style archive node speaking
  JSON-RPC 2.0 over the ``rpc_url`` endpoint: ``eth_getLogs`` with a
  ``{address, topics, fromBlock, toBlock}`` filter object (Geth's documented
  shape), ``eth_call`` with an explicit historical ``blockNumber`` as the second
  param (requires archive state), ``eth_blockNumber`` and
  ``eth_getBlockByNumber``. Where providers phrase limits differently we extend
  T04's pattern sets *locally* (``_TOO_MANY_PATTERNS`` / ``_TRANSIENT_PATTERNS``)
  and override ``_is_too_many`` so the base's adaptive shrink honours them —
  T04's ``TOO_MANY_RESULTS_PATTERNS`` already carries the common phrasings
  ("Log response size exceeded", JSON-RPC ``-32602``, ...).
* **Archive node required.** ``eth_call`` at a historical block returns
  "missing trie node" / "state not available" on a non-archive node; we classify
  those as :class:`PermanentFetchError` whose message says *archive node
  required* (this error otherwise wastes hours chasing a range bug).
* **Batching.** Batch JSON-RPC calls where we make several at once (block
  timestamps, block hashes, ``fee_growth_at``) but keep a per-item fallback —
  batch support and size limits vary by provider. A batch response may be out
  of `id` order and may carry per-item ``error`` objects: we match on `id`,
  never position, and surface the failing item's method+params.
* **Finality guard.** ``fetch`` / ``fee_growth_at`` refuse blocks newer than
  ``latest - confirmations`` (default 64) and raise a non-retryable error; the
  tip should be pulled by the correct consumer once finalised. The guard costs
  one ``eth_blockNumber`` per fetcher instance (cached) — so a warm-cache
  second fetch makes zero network calls.
* **Reorg guard.** Block hashes of the blocks present in a cached chunk are
  recorded alongside the cache (``<key>.hashes.json``). With ``verify_chain``
  enabled (off by default, so cache hits stay zero-call), a cache hit re-fetches
  the current block hashes and raises :class:`ReorgDetectedError` on the first
  mismatch.
* **Private block-timestamp resolver:** ``_resolve_block_timestamps`` (backed by
  ``_block_headers``, one batched ``eth_getBlockByNumber`` per previously unseen
  block, cached for the instance lifetime). T07 owns the *public*
  timestamp<->block index; this private one only annotates logs. T14 may
  consolidate the two later.
* **Flash is decoded** (``EventType.FLASH``): a flash's ``paid - amount`` is fee
  growth with no liquidity change, and CONTRACTS §4.3.1 wants that fee source in
  the tape so T10's replay does not run negatively biased.

Conventions: big integers stay ``int`` everywhere in Python and become decimal
strings only via ``schemas.encode_uint``; ``(block_number, log_index)`` is the
ordering key; timestamps are tz-aware UTC; no URL / key is ever logged (the
base's ``_describe`` context carries no secrets).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import requests

from undertow.data.config import GLOBAL_TICK_SENTINEL, EndpointConfig, PoolConfig
from undertow.data.fetchers.abi import (
    SELECTOR_FEE_GROWTH_GLOBAL_0_X128,
    SELECTOR_FEE_GROWTH_GLOBAL_1_X128,
    SELECTOR_LIQUIDITY,
    SELECTOR_SLOT0,
    TOPIC_BURN,
    TOPIC_COLLECT,
    TOPIC_FLASH,
    TOPIC_MINT,
    TOPIC_SWAP,
    decode_burn,
    decode_collect,
    decode_flash,
    decode_liquidity_return,
    decode_mint,
    decode_slot0_return,
    decode_swap,
    decode_ticks_return,
    decode_uint256_return,
    encode_ticks_calldata,
)
from undertow.data.fetchers.base import (
    TOO_MANY_RESULTS_PATTERNS,
    TRANSIENT_ERROR_PATTERNS,
    BaseHttpFetcher,
    FetchRequest,
    FetchResult,
)
from undertow.data.schemas import (
    FEE_GROWTH_SCHEMA,
    SCHEMA_REGISTRY,
    empty_table,
    encode_uint,
    validate_table,
)
from undertow.data.types import (
    Address,
    BlockNumber,
    FetchError,
    PermanentFetchError,
    RateLimitError,
    ReorgDetectedError,
    Route,
    SchemaViolationError,
    TransientNetworkError,
)

LOGGER = logging.getLogger("undertow.data.fetchers.rpc")

# ---------------------------------------------------------------------------
# Provider-specific error phrasing — LOCAL extensions of T04's pattern sets
# (T04's base is frozen; subclasses extend by union per its docstring).
# ---------------------------------------------------------------------------

# Phrasings that mark a JSON-RPC error as "range too wide" -> adaptive shrink.
_TOO_MANY_PATTERNS: frozenset[str] = TOO_MANY_RESULTS_PATTERNS | frozenset(
    {
        "block range is too large",
        "range is too wide",
        "exceeds the maximum allowed",
        "max 10000",
        "maximum number of logs",
        "response too large",
    }
)

# Phrasings that mark a JSON-RPC error as retryable.
_TRANSIENT_PATTERNS: frozenset[str] = TRANSIENT_ERROR_PATTERNS | frozenset(
    {
        "upstream request timeout",
        "temporarily unavailable",
        "backoff",
        "try again shortly",
    }
)

# Phrasings that mean "this node has no archive state for eth_call at a
# historical block" -> PermanentFetchError with an *archive node required*
# message. Providers phrase this differently: Erigon "missing trie node",
# Geth "state not available", ...
_ARCHIVE_UNAVAILABLE_PATTERNS: frozenset[str] = frozenset(
    {
        "missing trie node",
        "state not available",
        "historical state is not available",
        "archive state",
        "header to which trie is rooted not found",
    }
)

# Batch-level failure phrasings that trigger the per-item fallback. Deliberately
# narrow: the bare word "batch" also occurs in OUR OWN transport error messages
# ("batch response must be a list, got dict"), so we match only phrasings that
# indicate the PROVIDER does not accept batches, never our own diagnostics.
_BATCH_UNSUPPORTED_PATTERNS: frozenset[str] = frozenset(
    {
        "not support",
        "unsupported",
        "array of requests",
        "only single request",
    }
)

_TOPIC_BY_STREAM: Mapping[str, str] = {
    "swap": TOPIC_SWAP,
    "mint": TOPIC_MINT,
    "burn": TOPIC_BURN,
    "collect": TOPIC_COLLECT,
    "flash": TOPIC_FLASH,
}

_DECODER_BY_STREAM: Mapping[str, Callable[..., dict[str, object]]] = {
    "swap": decode_swap,
    "mint": decode_mint,
    "burn": decode_burn,
    "collect": decode_collect,
    "flash": decode_flash,
}

_JSONRPC_VERSION = "2.0"


def _params_shorthand(params: object) -> str:
    """A short, secret-free description of a JSON-RPC request's params."""
    return json.dumps(params, separators=(",", ":"))[:80]


class RpcFetcher(BaseHttpFetcher):
    """Route C fetcher: eth_getLogs (logs) + eth_call (fee-growth snapshots).

    :param confirmations: finality guard; blocks within ``latest - N`` of the
        tip are refused (default 64).
    :param json_rpc_batch_size: max calls per JSON-RPC batch request.
    :param verify_chain: when True, cache hits verify the recorded block hashes
        against the current chain and raise :class:`ReorgDetectedError` on a
        mismatch. Costs network calls; default False keeps cache hits zero-call.
    """

    route = Route.RPC

    def __init__(
        self,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        confirmations: int = 64,
        json_rpc_batch_size: int = 100,
        verify_chain: bool = False,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(endpoints, cache_dir, sleep=sleep, rng=rng)
        if confirmations < 0:
            raise ValueError(f"confirmations must be >= 0, got {confirmations}")
        if json_rpc_batch_size < 1:
            raise ValueError(f"json_rpc_batch_size must be >= 1, got {json_rpc_batch_size}")
        self._rpc_url: str = endpoints.rpc_url
        self._confirmations: int = confirmations
        self._batch_size: int = json_rpc_batch_size
        self._verify_chain: bool = verify_chain
        self._jsonrpc_id: int = 0
        self._latest_confirmed: int | None = None
        self._block_headers_cache: dict[int, dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Contract surface (CONTRACTS.md §5.2)
    # ------------------------------------------------------------------

    def supported_streams(self) -> frozenset[str]:
        return frozenset({"swap", "mint", "burn", "collect", "flash", "fee_growth"})

    def fetch(self, request: FetchRequest) -> FetchResult:
        if request.stream not in self.supported_streams():
            raise FetchError(
                f"RpcFetcher does not support stream {request.stream!r}; supported streams "
                f"are {sorted(self.supported_streams())}"
            )
        if request.stream == "fee_growth":
            raise FetchError(
                "stream 'fee_growth' has no block-range fetch; use "
                "fee_growth_at(pool, block, ticks) for the eth_call snapshots"
            )
        if (
            request.start_block is not None
            and request.end_block is not None
            and request.start_block > request.end_block
        ):
            # Nothing to fetch: an empty range must not consult the chain (the
            # finality guard costs an eth_blockNumber and would fire on a range
            # that makes no request at all). Mirrors the base's empty-range
            # semantics (n_requests=0, from_cache=False, a warning) and returns
            # a schema-conformant empty table.
            return FetchResult(
                table=empty_table(SCHEMA_REGISTRY[request.stream]),
                route=self.route,
                request=request,
                n_requests=0,
                from_cache=False,
                warnings=(
                    f"fetch: empty block range {int(request.start_block)}.."
                    f"{int(request.end_block)}; nothing to fetch",
                ),
            )
        if request.end_block is not None:
            self._assert_confirmed(request.end_block)
        rows, n_requests, from_cache, warnings = self.fetch_rows(request)
        table = self._rows_to_table(rows, request.stream)
        return FetchResult(
            table=table,
            route=self.route,
            request=request,
            n_requests=n_requests,
            from_cache=from_cache,
            warnings=tuple(warnings),
        )

    def fee_growth_at(
        self, pool: PoolConfig, block: BlockNumber, ticks: Sequence[int]
    ) -> pa.Table:
        """Snapshot the pool's fee-growth state at ``block`` (state AFTER the block).

        One batched set of ``eth_call``s: slot0(), liquidity(),
        feeGrowthGlobal0X128(), feeGrowthGlobal1X128(), then ``ticks(int24)`` for
        each requested tick. Returns a validated ``FEE_GROWTH_SCHEMA`` table with
        one global row (``GLOBAL_TICK_SENTINEL``) plus one row per tick, all with
        ``source="rpc_call"``. The global row carries ``liquidity_gross="0"`` /
        ``liquidity_net="0"`` / ``initialized=False`` placeholders (the schema
        declares those columns non-nullable; the sentinel row's tick fields are
        not meaningful and T10's reconcile ignores them). Requires an **archive
        node**; a non-archive node raises PermanentFetchError naming that
        requirement. Duplicate ticks are collapsed (first occurrence wins).
        """
        if pool is None:
            raise FetchError("fee_growth_at: a pool is required (fee growth is pool-scoped)")
        if int(block) <= 0:
            raise FetchError(f"fee_growth_at: block must be > 0, got {block}")
        for t in ticks:
            if not -(1 << 23) <= t < (1 << 23):
                raise SchemaViolationError(
                    f"fee_growth_at: tick {t} does not fit in the int24 ticks() selector"
                )
        self._assert_confirmed(block)
        deduped = list(dict.fromkeys(int(t) for t in ticks))
        rows = self._fee_growth_rows(pool, block, deduped)
        table = pa.Table.from_pylist(rows, schema=FEE_GROWTH_SCHEMA)
        validate_table(table, FEE_GROWTH_SCHEMA)
        return table

    # ------------------------------------------------------------------
    # Base contract implementation
    # ------------------------------------------------------------------

    def _cache_params(self, request: FetchRequest, start: int, end: int) -> dict[str, object]:
        params = super()._cache_params(request, start, end)
        if request.stream in _TOPIC_BY_STREAM:
            params["topic0"] = _TOPIC_BY_STREAM[request.stream]
        return params

    def _is_too_many(self, error: PermanentFetchError) -> bool:
        """Extended too-many detection so the base's adaptive shrink honours the
        provider phrasings T04 could not have known about."""
        needle = str(error).lower()
        return any(pattern in needle for pattern in _TOO_MANY_PATTERNS)

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        """One block-range chunk of a log stream: eth_getLogs + decode + timestamps.

        Called through the base's cache/retry/adaptive-shrink machinery. The raw
        rows cache therefore stores *decoded schema-shaped* rows (deterministic),
        and the block hashes of every block present in the chunk are recorded
        next to the cache entry for the reorg guard.
        """
        stream = request.stream
        if request.pool is None:
            raise FetchError(
                f"{self._describe(request, start, end)}: a pool is required for RPC log fetches"
            )
        decoder = _DECODER_BY_STREAM.get(stream)
        topic0 = _TOPIC_BY_STREAM.get(stream)
        if decoder is None or topic0 is None:
            raise FetchError(
                f"{self._describe(request, start, end)}: stream {stream!r} has no RPC log "
                "decoder; only " + ", ".join(sorted(_TOPIC_BY_STREAM)) + " are log streams"
            )
        context = self._describe(request, start, end)
        logs = self._eth_get_logs(request.pool.address, topic0, start, end, context)
        if not logs:
            return []
        blocks = sorted({_hex_int(lg["blockNumber"]) for lg in logs})
        headers = self._block_headers(blocks)
        self._record_hashes(
            request, start, end, {b: str(headers[b]["hash"]) for b in blocks}
        )
        rows: list[dict] = []
        for lg in logs:
            block = _hex_int(lg["blockNumber"])
            # Providers return the header timestamp as a hex string: decode it
            # like every other provider integer (never ``int(str)``).
            ts = datetime.fromtimestamp(_hex_int(headers[block]["timestamp"]), tz=UTC)
            # The decoder knows the payload; the transport knows the envelope.
            row = decoder(lg, pool_address=request.pool.address, block_timestamp=ts)
            rows.append(row)
        return rows

    def _rows_to_table(self, rows: list[dict], stream: str) -> pa.Table:
        schema = SCHEMA_REGISTRY[stream]
        if not rows:
            return empty_table(schema)
        table = pa.Table.from_pylist(rows, schema=schema)
        validate_table(table, schema)
        return table

    # ------------------------------------------------------------------
    # Raw-cache (de)serialisation: rows carry tz-aware datetimes in memory,
    # but the on-disk raw cache is JSON and JSON has no datetime. Convert at
    # the boundary so a warm-cache fetch returns the exact same row shape as
    # a cold one (the base's envelope stays untouched).
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_cache_form(rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for row in rows:
            item = dict(row)
            ts = item.get("block_timestamp")
            item["block_timestamp"] = (
                ts.isoformat() if isinstance(ts, datetime) else ts
            )
            out.append(item)
        return out

    @staticmethod
    def _rows_from_cache_form(rows: list[dict]) -> list[dict]:
        out: list[dict] = []
        for row in rows:
            item = dict(row)
            ts = item.get("block_timestamp")
            if isinstance(ts, str):
                item["block_timestamp"] = datetime.fromisoformat(ts)
            out.append(item)
        return out

    def _cache_read(self, request: FetchRequest, start: int, end: int) -> list[dict] | None:
        body = super()._cache_read(request, start, end)
        if body is None:
            return None
        return self._rows_from_cache_form(body)

    def _cache_write(self, request: FetchRequest, start: int, end: int, rows: list[dict]) -> None:
        super()._cache_write(request, start, end, self._rows_to_cache_form(rows))

    def _fetch_one(self, request: FetchRequest, start: int, end: int) -> tuple[list[dict], bool]:
        """Cache-hit path with the optional reorg verification (verify_chain).

        The base template is otherwise untouched: cold fetch -> network -> cache;
        warm fetch -> cached rows. Only when ``verify_chain`` is set do we spend
        network calls on a cache hit to compare the recorded block hashes
        against the current chain.
        """
        rows, used_cache = super()._fetch_one(request, start, end)
        if used_cache and self._verify_chain and self._is_log_stream(request.stream):
            self._verify_cached_hashes(request, start, end)
        return rows, used_cache

    def _is_log_stream(self, stream: str) -> bool:
        return stream in _TOPIC_BY_STREAM

    # ------------------------------------------------------------------
    # Finality guard
    # ------------------------------------------------------------------

    def _assert_confirmed(self, block: BlockNumber) -> None:
        """Refuse blocks within the unconfirmed tip window (per CONTRACTS/brief)."""
        if self._confirmations <= 0:
            return
        latest = self._fetch_latest_block()
        safe = latest - self._confirmations
        if int(block) > safe:
            raise PermanentFetchError(
                f"block {block} is within the last {self._confirmations} confirmations "
                f"(latest block {latest}; latest-confirmed is {safe}); refusing to fetch "
                "unfinalised, reorg-able state — wait for confirmations or narrow the window"
            )

    def _fetch_latest_block(self) -> int:
        if self._latest_confirmed is None:
            result = self._jsonrpc("eth_blockNumber", [], "eth_blockNumber")
            self._latest_confirmed = _hex_int(result)
        return self._latest_confirmed

    # ------------------------------------------------------------------
    # JSON-RPC plumbing (single + batch with id matching and fallback)
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        rid = self._jsonrpc_id
        self._jsonrpc_id += 1
        return rid

    def _post_jsonrpc(self, payload: object, context: str) -> object:
        """POST ``payload`` to the RPC endpoint; return the parsed response body.

        One deliberate difference from the base class' ``_request``: a **4xx body
        often names the failure** ("batch requests are not supported", "missing
        trie node"); the base discards that body and raises a bare "HTTP 4xx".
        That would (a) break the batch->per-item fallback and (b) cost hours on
        a non-archive node. So this layer keeps the base's retry semantics for
        transport / 5xx / 429 but decodes the body *before* raising on a 4xx,
        feeding provider phrasings into :meth:`_classify_jsonrpc_error` and
        naming the (truncated) body when nothing in it classifies.
        """
        attempt = 0
        while True:
            try:
                response = self._session.request(
                    "POST",
                    self._rpc_url,
                    json=payload,  # type: ignore[arg-type]  # JSON-RPC bodies are JSON
                    timeout=self._timeout,
                )
            except requests.exceptions.Timeout:
                self._handle_transient(
                    TransientNetworkError(
                        f"{context}: request timed out after {self._timeout}s"
                    ),
                    attempt,
                )
                attempt += 1
                continue
            except requests.exceptions.ConnectionError:
                self._handle_transient(
                    TransientNetworkError(f"{context}: connection error"), attempt
                )
                attempt += 1
                continue

            if 200 <= response.status_code < 300:
                body = self._try_json(response, context)
                if body is None:
                    raise PermanentFetchError(
                        f"{context}: JSON-RPC response was not JSON: "
                        f"{response.text[:200]!r}"
                    )
                return body
            if response.status_code == 429:
                self._handle_transient(
                    RateLimitError(f"{context}: HTTP 429 rate limited"),
                    attempt,
                    self._parse_retry_after(response),
                )
                attempt += 1
                continue
            if response.status_code >= 500:
                self._handle_transient(
                    TransientNetworkError(f"{context}: HTTP {response.status_code}"),
                    attempt,
                    self._parse_retry_after(response),
                )
                attempt += 1
                continue
            body = self._try_json(response, context)
            if isinstance(body, dict) and body.get("error") is not None:
                self._classify_jsonrpc_error(body["error"], context)
            detail = (
                ": " + json.dumps(body, separators=(",", ":"))[:200]
                if body is not None
                else ""
            )
            raise PermanentFetchError(f"{context}: HTTP {response.status_code}{detail}")

    def _try_json(self, response: requests.Response, context: str) -> object | None:
        """Response body as JSON when it parses; ``None`` otherwise (and never a
        NetworkError — a non-JSON body on a 4xx is just detail for the message)."""
        try:
            body = response.json()
        except ValueError:
            LOGGER.warning("%s: response body was not JSON; body omitted", context)
            return None
        return body

    def _with_jsonrpc_retry(self, fn: Callable[[], object], context: str) -> object:
        """JSON-RPC-level retry: RateLimitError / TransientNetworkError from a
        provider's ``error`` body are retried with the base's backoff; the base's
        ``_request`` already handled HTTP-status transients, this is the layer
        T04's review note asked subclasses to add around classification."""
        attempt = 0
        while True:
            try:
                return fn()
            except (RateLimitError, TransientNetworkError) as exc:
                self._handle_transient(exc, attempt)
                attempt += 1

    def _jsonrpc(self, method: str, params: list[object], context: str = "") -> object:
        """One JSON-RPC call. Raises the classified FetchError on an error item."""
        payload = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": self._next_id(),
            "method": method,
            "params": params,
        }

        def once() -> object:
            raw = self._post_jsonrpc(payload, context)
            if not isinstance(raw, dict):
                raise PermanentFetchError(
                    f"{context}: JSON-RPC response must be an object, got {type(raw).__name__}"
                )
            if raw.get("error") is not None:
                self._classify_jsonrpc_error(raw["error"], context)
            return raw.get("result")

        return self._with_jsonrpc_retry(once, context)

    def _jsonrpc_batch(
        self, entries: Sequence[tuple[str, list[object], str]], *, batch_context: str = ""
    ) -> list[object]:
        """Batch several JSON-RPC calls, results in INPUT order.

        Handles out-of-order responses (match on ``id``, never position) and
        per-item ``error`` objects (raise for that item only, naming its
        method+params). Falls back to individual calls when the provider rejects
        the batch shape (``_BATCH_UNSUPPORTED_PATTERNS``).
        """
        if not entries:
            return []
        if len(entries) == 1:
            method, params, context = entries[0]
            return [self._jsonrpc(method, params, context)]

        calls: list[dict[str, object]] = []
        call_ids: list[int] = []
        by_id: dict[int, tuple[str, list[object], str]] = {}
        for method, params, context in entries:
            rid = self._next_id()
            by_id[rid] = (method, params, context)
            call_ids.append(rid)
            calls.append(
                {"jsonrpc": _JSONRPC_VERSION, "id": rid, "method": method, "params": params}
            )

        try:
            raw = self._with_jsonrpc_retry(
                lambda: self._post_jsonrpc(calls, batch_context), batch_context
            )
        except PermanentFetchError as exc:
            if self._batch_unsupported(str(exc)):
                return [self._jsonrpc(m, p, c) for m, p, c in entries]
            raise

        if not isinstance(raw, list):
            raise PermanentFetchError(
                f"{batch_context}: JSON-RPC batch response must be a list, got "
                f"{type(raw).__name__}"
            )
        results_by_id: dict[int, object] = {}
        for item in raw:
            if not isinstance(item, dict) or "id" not in item:
                raise PermanentFetchError(f"{batch_context}: malformed batch item {item!r}")
            rid = item["id"]
            if not isinstance(rid, int):
                raise PermanentFetchError(
                    f"{batch_context}: batch response id {rid!r} is not an integer "
                    "(the ids this client sends are integers)"
                )
            entry = by_id.get(rid)
            if entry is None:
                raise PermanentFetchError(
                    f"{batch_context}: batch response carried id {rid!r} which this "
                    "request did not send"
                )
            if item.get("error") is not None:
                method, params, context = entry
                # Name the failing call precisely: the per-entry ``context`` is
                # "ticks(196238)" / "slot0()" ... — the params shorthand alone
                # truncates the calldata and would hide which call failed.
                self._classify_jsonrpc_error(
                    item["error"],
                    f"{batch_context}: {context} ({method} {_params_shorthand(params)})",
                )
            results_by_id[rid] = item.get("result")
        missing = [str(call_id) for call_id in call_ids if call_id not in results_by_id]
        if missing:
            raise PermanentFetchError(
                f"{batch_context}: batch response missing result(s) for id(s) "
                + ", ".join(missing)
            )
        return [results_by_id[call_id] for call_id in call_ids]  # input order

    def _batch_unsupported(self, message: str) -> bool:
        needle = message.lower()
        return any(pattern in needle for pattern in _BATCH_UNSUPPORTED_PATTERNS)

    def _classify_jsonrpc_error(self, error: object, context: str) -> None:
        """Raise the right FetchError for one JSON-RPC ``error`` object.

        Never returns. Order matters: archive-state phrasings are permanent and
        its message must say *archive node required*; rate-limit phrasings are
        retryable; anything else is permanent.
        """
        message = ""
        code = ""
        if isinstance(error, dict):
            value = error.get("message")
            if isinstance(value, str):
                message = value
            code = str(error.get("code", ""))
        elif isinstance(error, str):
            message = error
        else:
            message = str(error)
        needle = f"{message} {code}".lower()
        if any(pattern in needle for pattern in _ARCHIVE_UNAVAILABLE_PATTERNS):
            raise PermanentFetchError(
                f"{context}: archive node required — this eth_call/eth_getLogs needs "
                f"historical state and the configured rpc_url is not an archive node "
                f"(provider said {message!r}); configure an archive-capable provider"
            )
        if any(pattern in needle for pattern in _TRANSIENT_PATTERNS):
            raise RateLimitError(f"{context}: transient provider error: {message or code}")
        raise PermanentFetchError(f"{context}: JSON-RPC error: {message or code}")

    # ------------------------------------------------------------------
    # eth_getLogs
    # ------------------------------------------------------------------

    def _eth_get_logs(
        self, address: Address, topic0: str, start: int, end: int, context: str
    ) -> list[dict]:
        params: list[object] = [
            {
                "address": str(address),
                "topics": [topic0],
                "fromBlock": hex(start),
                "toBlock": hex(end),
            }
        ]
        result = self._jsonrpc("eth_getLogs", params, context)
        if not isinstance(result, list):
            raise PermanentFetchError(
                f"{context}: eth_getLogs returned {type(result).__name__}, expected a list"
            )
        return result

    # ------------------------------------------------------------------
    # Block timestamps / hashes — the private resolver (see module docstring)
    # ------------------------------------------------------------------

    def _block_headers(self, blocks: Iterable[int]) -> dict[int, dict[str, object]]:
        """``{block_number: {"timestamp": int, "hash": "0x.."}}``, batched + cached.

        Only previously-unseen blocks hit the network (one batch of
        ``eth_getBlockByNumber`` each), so repeated resolution is cheap across
        chunks within an instance.
        """
        wanted = list(dict.fromkeys(int(b) for b in blocks))
        unseen = [b for b in wanted if b not in self._block_headers_cache]
        if unseen:
            results: list[object] = []
            for i in range(0, len(unseen), self._batch_size):
                chunk = unseen[i : i + self._batch_size]
                entries = [
                    ("eth_getBlockByNumber", [hex(b), False], f"block header {b}")
                    for b in chunk
                ]
                results.extend(
                    self._jsonrpc_batch(entries, batch_context="block header resolution")
                )
            if len(results) != len(unseen):
                raise PermanentFetchError(
                    f"block header resolution: expected {len(unseen)} headers, got "
                    f"{len(results)}"
                )
            for block, result in zip(unseen, results, strict=True):
                if not isinstance(result, dict) or "timestamp" not in result:
                    raise PermanentFetchError(
                        f"block header resolution: malformed header for block {block}: {result!r}"
                    )
                self._block_headers_cache[block] = result
        return {b: self._block_headers_cache[b] for b in wanted}

    def _resolve_block_timestamps(self, blocks: Iterable[int]) -> dict[int, datetime]:
        """PRIVATE block_number -> UTC timestamp resolver (annotates logs only).

        T07 owns the public timestamp<->block index; this one exists so log rows
        can carry ``block_timestamp`` without coordinating across wave 3 (T14
        may consolidate). Cached per instance.
        """
        headers = self._block_headers(blocks)
        return {
            b: datetime.fromtimestamp(_hex_int(headers[b]["timestamp"]), tz=UTC)
            for b in headers
        }

    def _fresh_block_hashes(self, blocks: Iterable[int]) -> dict[int, str]:
        """Current-chain block hashes, ALWAYS fresh (never the instance cache)."""
        wanted = list(dict.fromkeys(int(b) for b in blocks))
        result: dict[int, str] = {}
        for i in range(0, len(wanted), self._batch_size):
            chunk = wanted[i : i + self._batch_size]
            entries = [
                ("eth_getBlockByNumber", [hex(b), False], f"fresh block header {b}")
                for b in chunk
            ]
            for block, header in zip(
                chunk,
                self._jsonrpc_batch(entries, batch_context="reorg verification"),
                strict=True,
            ):
                if not isinstance(header, dict) or "hash" not in header:
                    raise PermanentFetchError(
                        f"reorg verification: malformed header for block {block}: {header!r}"
                    )
                result[block] = str(header["hash"])
        return result

    # ------------------------------------------------------------------
    # Reorg guard — block hashes recorded alongside cached chunks
    # ------------------------------------------------------------------

    def _hashes_path(self, request: FetchRequest, start: int, end: int) -> Path:
        key = self._cache_key(request, start, end)
        return self._cache_path(request, key).with_suffix(".hashes.json")

    def _record_hashes(
        self, request: FetchRequest, start: int, end: int, hashes: Mapping[int, str]
    ) -> None:
        if not hashes:
            return
        path = self._hashes_path(request, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps({str(k): v for k, v in sorted(hashes.items())}, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _read_hashes(self, request: FetchRequest, start: int, end: int) -> dict[int, str] | None:
        path = self._hashes_path(request, start, end)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive, corrupt local file
            LOGGER.warning("corrupt block-hash sidecar at %s (%s); skipping check", path, exc)
            return None
        if not isinstance(data, dict):
            return None
        return {int(k): str(v) for k, v in data.items()}

    def _verify_cached_hashes(self, request: FetchRequest, start: int, end: int) -> None:
        stored = self._read_hashes(request, start, end)
        if not stored:
            return
        current = self._fresh_block_hashes(sorted(stored))
        for block, old in sorted(stored.items()):
            now = current.get(block)
            if now is not None and now.lower() != old.lower():
                raise ReorgDetectedError(
                    f"cached block {block} hash {old} no longer matches the chain hash {now}; "
                    f"a reorganisation was detected for "
                    f"{self._describe(request, start, end)}"
                )

    # ------------------------------------------------------------------
    # fee_growth_at internals
    # ------------------------------------------------------------------

    def _fee_growth_rows(self, pool: PoolConfig, block: int, ticks: list[int]) -> list[dict]:
        cached = self._fg_cache_read(pool, block, ticks)
        if cached is not None:
            return cached
        rows = self._fee_growth_fetch(pool, block, ticks)
        self._fg_cache_write(pool, block, ticks, rows)
        return rows

    def _fg_cache_key(self, pool_address: str, block: int, ticks: list[int]) -> str:
        return self.cache_key(
            self.route,
            "fee_growth",
            {"pool": pool_address, "block": block, "ticks": [int(t) for t in ticks]},
        )

    def _fg_cache_path(self, pool_address: str, block: int, ticks: list[int]) -> Path:
        key = self._fg_cache_key(pool_address, block, ticks)
        return self._cache_dir / "rpc" / "fee_growth" / f"{key}.json.gz"

    def _fg_cache_read(
        self, pool: PoolConfig, block: int, ticks: list[int]
    ) -> list[dict] | None:
        path = self._fg_cache_path(str(pool.address), block, ticks)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as fh:
                envelope = json.loads(fh.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive, corrupt local file
            LOGGER.warning("corrupt fee_growth cache entry %s (%s); treating as miss", path, exc)
            return None
        body = envelope.get("body") if isinstance(envelope, dict) else None
        return body if isinstance(body, list) else None

    def _fg_cache_write(
        self, pool: PoolConfig, block: int, ticks: list[int], rows: list[dict]
    ) -> None:
        path = self._fg_cache_path(str(pool.address), block, ticks)
        envelope = {
            "params": {"pool": str(pool.address), "block": block, "ticks": ticks},
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
            tmp.unlink(missing_ok=True)
            raise

    def _fee_growth_fetch(
        self, pool: PoolConfig, block: int, ticks: list[int]
    ) -> list[dict]:
        address = str(pool.address)
        call_params: list[object] = [{"to": address, "data": SELECTOR_SLOT0}, hex(block)]
        entries: list[tuple[str, list[object], str]] = [
            ("eth_call", call_params, "slot0()"),
            ("eth_call", [{"to": address, "data": SELECTOR_LIQUIDITY}, hex(block)], "liquidity()"),
            (
                "eth_call",
                [{"to": address, "data": SELECTOR_FEE_GROWTH_GLOBAL_0_X128}, hex(block)],
                "feeGrowthGlobal0X128()",
            ),
            (
                "eth_call",
                [{"to": address, "data": SELECTOR_FEE_GROWTH_GLOBAL_1_X128}, hex(block)],
                "feeGrowthGlobal1X128()",
            ),
        ]
        entries += [
            ("eth_call", [{"to": address, "data": encode_ticks_calldata(t)}, hex(block)],
             f"ticks({t})")
            for t in ticks
        ]

        results: list[object] = []
        for i in range(0, len(entries), self._batch_size):
            chunk = entries[i : i + self._batch_size]
            results.extend(
                self._jsonrpc_batch(chunk, batch_context=f"fee_growth_at(block={block})")
            )
        if len(results) != len(entries):
            raise PermanentFetchError(
                f"fee_growth_at(block={block}): expected {len(entries)} results, got "
                f"{len(results)}"
            )

        slot0 = decode_slot0_return(str(results[0]))
        current_liquidity = decode_liquidity_return(str(results[1]))
        g0 = decode_uint256_return(str(results[2]))
        g1 = decode_uint256_return(str(results[3]))
        tick_results = results[4:]

        rows: list[dict] = [
            {
                "block_number": block,
                "tick": GLOBAL_TICK_SENTINEL,
                "fee_growth_outside_0_x128": None,
                "fee_growth_outside_1_x128": None,
                "liquidity_gross": "0",
                "liquidity_net": "0",
                "initialized": False,
                "fee_growth_global_0_x128": encode_uint(g0),
                "fee_growth_global_1_x128": encode_uint(g1),
                "current_tick": slot0.tick,
                "current_liquidity": encode_uint(current_liquidity),
                "source": "rpc_call",
                "pool_address": address,
                "fee_protocol": slot0.fee_protocol,
            }
        ]
        for t, result in zip(ticks, tick_results, strict=True):
            state = decode_ticks_return(str(result))
            rows.append(
                {
                    "block_number": block,
                    "tick": int(t),
                    "fee_growth_outside_0_x128": encode_uint(state.fee_growth_outside_0_x128),
                    "fee_growth_outside_1_x128": encode_uint(state.fee_growth_outside_1_x128),
                    "liquidity_gross": encode_uint(state.liquidity_gross),
                    "liquidity_net": encode_uint(state.liquidity_net),
                    "initialized": bool(state.initialized),
                    "fee_growth_global_0_x128": encode_uint(g0),
                    "fee_growth_global_1_x128": encode_uint(g1),
                    "current_tick": slot0.tick,
                    "current_liquidity": encode_uint(current_liquidity),
                    "source": "rpc_call",
                    "pool_address": address,
                    "fee_protocol": slot0.fee_protocol,
                }
            )
        return rows


def _hex_int(value: object) -> int:
    """Hex string/int -> int; a non-hex value is a provider bug, raise loudly."""
    if isinstance(value, int):
        return value
    try:
        return int(str(value), 16)
    except (TypeError, ValueError):
        raise PermanentFetchError(
            f"provider returned a non-hex block number {value!r}; this is not a "
            "Geth-compatible eth_getLogs/eth_getBlockByNumber response"
        ) from None