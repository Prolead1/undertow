"""Shared HTTP fetcher plumbing for ``undertow.data`` (CONTRACTS.md §5.1, T04).

Provides:
  * retry with exponential backoff + full jitter on transient failures;
  * HTTP-status and JSON-RPC error classification;
  * inclusive block-range chunking with adaptive shrink on "too many results";
  * an atomic, deterministic on-disk raw-response cache.

LAYERING RULE (why T04 builds in parallel with T03): this module imports nothing from
``schemas.py`` and validates no tables. Its template method returns raw row ``dict``s; each
concrete fetcher (T05-T08) owns ``_rows_to_table`` and calls ``validate_table`` itself. This
module knows HTTP, ranges and bytes — not what a swap is.

Concrete subclasses implement exactly::

    _fetch_chunk(request: FetchRequest, start: int, end: int) -> list[dict]
        Raw decoded rows for the inclusive block range ``[start, end]``. It performs the network
        call(s) (usually via ``self._request``) and decodes the response into row dicts. Called
        only on cache miss; the base handles retry, adaptive shrink and caching around it.

and may override::

    _cache_params(request: FetchRequest, start: int, end: int) -> dict
        Extra canonical material included in the cache key (e.g. pool address, RPC topic). Default
        includes the pool address and the requested block range. Never put a secret here.

``fetch_rows(request)`` returns ``(rows, n_requests, from_cache, warnings)`` — the schema-free
core of ``fetch()``. Subclasses combine it with their own ``_rows_to_table`` + ``validate_table``
and wrap the result in ``FetchResult``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import pyarrow as pa
import requests
from requests.adapters import HTTPAdapter

from undertow.data.config import EndpointConfig, PoolConfig
from undertow.data.types import (
    BlockNumber,
    PermanentFetchError,
    RateLimitError,
    Route,
    TransientNetworkError,
)

logger = logging.getLogger("undertow.data.fetchers.base")

# ---------------------------------------------------------------------------
# Retry / backoff policy (CONTRACTS.md §5.1). Full-jitter exponential backoff:
# sleep = random.uniform(0, min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt))
# ---------------------------------------------------------------------------
BACKOFF_BASE_SECONDS: float = 0.5
BACKOFF_CAP_SECONDS: float = 30.0
USER_AGENT: str = "undertow/0.1 (RL-for-AMM-liquidity data pipeline)"
DEFAULT_CHUNK_SIZE: int = 10
CACHE_DIGEST_BYTES: int = 32  # sha256 hex, truncated

# ---------------------------------------------------------------------------
# Provider error-pattern constants.
#
# TRANSIENT_ERROR_PATTERNS — phrases that mark a JSON-RPC ``error`` as retryable (RateLimitError)
# rather than permanent. Extend by OR-ing new phrases in a subclass module, e.g.::
#   from undertow.data.fetchers.base import TRANSIENT_ERROR_PATTERNS
#   _EXTRA = TRANSIENT_ERROR_PATTERNS | frozenset({"backoff", "try again shortly"})
# and use ``_raise_for_jsonrpc`` / your own classification against ``_EXTRA``.
#
# TOO_MANY_RESULTS_PATTERNS — phrases that mark a failure as "range too wide", triggering adaptive
# shrink (§2 of the brief). Extend the same way; providers phrase this differently (the Graph
# "query returned more than 10000 results", RPC code -32602 / "Log response size exceeded", ...).
# ---------------------------------------------------------------------------
TRANSIENT_ERROR_PATTERNS: frozenset[str] = frozenset(
    {
        "rate limit",
        "rate-limit",
        "too many requests",
        "throttl",
        "slow down",
        "please try again",
        "429",
    }
)

TOO_MANY_RESULTS_PATTERNS: frozenset[str] = frozenset(
    {
        "too many results",
        "more than 10000 results",
        "more than 10,000 results",
        "response size exceeded",
        "log response size exceeded",
        "datasize too large",
        "query returned more than",
        "-32602",
    }
)

_SleepFn = Callable[[float], None]

_URL_SCHEMES: tuple[str, ...] = ("http://", "https://")


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """What one fetcher call produces and one stream covers (CONTRACTS.md §5.1).

    Exactly one of the block pair or the date pair is populated, depending on the stream.
    """

    stream: str
    pool: PoolConfig | None
    start_block: BlockNumber | None
    end_block: BlockNumber | None
    start_utc: datetime | None = None
    end_utc: datetime | None = None


@dataclass(frozen=True, slots=True)
class FetchResult:
    """The validated handoff artifact a concrete fetcher returns (CONTRACTS.md §5.1).

    ``table`` conforms to ``SCHEMA_REGISTRY[request.stream]`` — that validation is performed by
    the concrete fetcher ("the subclass's job"), not by :class:`BaseHttpFetcher`.
    """

    table: pa.Table
    route: Route
    request: FetchRequest
    n_requests: int  # network calls actually made (0 => full cache hit)
    from_cache: bool
    warnings: tuple[str, ...]


class Fetcher(Protocol):
    """Structural protocol every concrete fetcher satisfies (CONTRACTS.md §5.1)."""

    route: Route

    def supported_streams(self) -> frozenset[str]: ...

    def fetch(self, request: FetchRequest) -> FetchResult: ...


# ---------------------------------------------------------------------------
# Canonicalization: same logical params => same key, nested dicts/lists included,
# secrets scrubbed (URLs hashed by host only, credentials/path removed).
# ---------------------------------------------------------------------------


def _host_only(value: str) -> str:
    """``scheme://netloc`` for a URL; never a path, query, fragment or userinfo."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value  # not a usable URL; keep as-is (still hashed, not logged)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    return f"{parts.scheme}://{netloc}"


def _canonicalize(value: object) -> object:
    """Recursively build a canonical, JSON-encodable form of ``params``.

    * ints (and bools handled separately) serialise as their decimal string — so a value of
      ``1`` and ``"1"`` are distinct, while insertion order never matters (dicts are sorted).
    * nested dicts/lists recurse.
    * URL strings are reduced to ``scheme://host`` so an API-key path segment never reaches the
      key material or a filename.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        if value.startswith(_URL_SCHEMES):
            return _host_only(value)
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise TypeError(f"unsupported cache param type {type(value).__name__!r}")


class BaseHttpFetcher:
    """Retry / chunking / raw-cache plumbing every HTTP fetcher shares.

    Imports nothing from ``schemas.py`` (see the layering rule). Subclasses implement
    ``_fetch_chunk`` and own ``_rows_to_table`` + ``validate_table`` + the ``FetchResult`` wrap.

    :param endpoints: resolved endpoint + policy config (timeout, concurrency, retries).
    :param cache_dir: root of the on-disk raw cache (``cache_dir/<route>/<stream>/<key>.json.gz``).
    :param sleep: injectable sleep for tests (defaults to ``time.sleep``). Never sleeps in a test
        if you don't want it to.
    :param rng: injectable random source for backoff jitter (defaults to a fresh ``random.Random``).
    """

    route: Route

    def __init__(
        self,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        sleep: _SleepFn | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._endpoints = endpoints
        self._cache_dir = Path(cache_dir)
        self._timeout: float = endpoints.request_timeout_s
        self._max_retries: int = endpoints.max_retries
        self._max_concurrency: int = max(1, endpoints.max_concurrency)
        self._initial_chunk_size: int = DEFAULT_CHUNK_SIZE
        self._sleep: _SleepFn = sleep if sleep is not None else time.sleep
        self._rng: random.Random = rng if rng is not None else random.Random()

        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self._max_concurrency, pool_maxsize=self._max_concurrency
        )
        for prefix in ("https://", "http://"):
            self._session.mount(prefix, adapter)
        self._session.headers["User-Agent"] = USER_AGENT

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunks(self, start: int, end: int, size: int) -> Iterator[tuple[int, int]]:
        """Yield inclusive, gapless, non-overlapping ``[lo, hi]`` sub-ranges of ``[start, end]``.

        The final chunk may be short. ``size <= 0`` raises ``ValueError``; an empty range
        (``start > end``) yields nothing.
        """
        if size <= 0:
            raise ValueError(f"chunk size must be > 0, got {size}")
        if start > end:
            return
        lo = start
        while lo <= end:
            hi = min(lo + size - 1, end)
            yield lo, hi
            lo = hi + 1

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def cache_key(self, route: Route, stream: str, params: Mapping[str, object]) -> str:
        """Deterministic sha256 hex (32 chars) over ``(route, stream, canonical(params))``.

        Canonicalization makes the key independent of dict insertion order and of nested
        structure, renders ints as strings, and strips secrets from URL values (hash the host,
        not the key). Same logic => same key across runs and machines.
        """
        canonical = _canonicalize(dict(params))
        payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(f"{route.value}:{stream}:{payload}".encode("utf-8")).hexdigest()
        return digest[:CACHE_DIGEST_BYTES]

    # ------------------------------------------------------------------
    # Cache read / write (atomic, gzip, envelope)
    # ------------------------------------------------------------------

    def _cache_params(self, request: FetchRequest, start: int, end: int) -> dict[str, object]:
        params: dict[str, object] = {}
        if request.pool is not None:
            params["pool"] = request.pool.address
        params["start_block"] = start
        params["end_block"] = end
        return params

    def _cache_path(self, request: FetchRequest, key: str) -> Path:
        return self._cache_dir / self.route.value / request.stream / f"{key}.json.gz"

    def _cache_key(self, request: FetchRequest, start: int, end: int) -> str:
        params = self._cache_params(request, start, end)
        return self.cache_key(self.route, request.stream, params)

    def _cache_read(self, request: FetchRequest, start: int, end: int) -> list[dict] | None:
        """Return cached raw rows for this chunk, or ``None`` on miss / corrupt entry.

        A corrupt entry (bad gzip, missing envelope fields, non-list body) is logged at WARNING
        and treated as a miss — never crashed on.
        """
        key = self._cache_key(request, start, end)
        path = self._cache_path(request, key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as fh:
                envelope = json.loads(fh.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive; any corrupt bytes land here
            logger.warning("corrupt cache entry at %s (%s); treating as miss", path, exc)
            return None
        if not isinstance(envelope, dict) or not isinstance(envelope.get("body"), list):
            logger.warning("corrupt cache envelope at %s; treating as miss", path)
            return None
        return envelope["body"]

    def _cache_write(self, request: FetchRequest, start: int, end: int, rows: list[dict]) -> None:
        """Atomically write ``rows`` to the cache (``.tmp`` + ``os.replace``).

        The envelope carries the canonical params and a UTC fetch timestamp so a stale entry is
        diagnosable. A crash mid-write leaves no final ``.json.gz`` entry; the temporary file is
        removed on error.
        """
        key = self._cache_key(request, start, end)
        path = self._cache_path(request, key)
        params = _canonicalize(self._cache_params(request, start, end))
        envelope = {
            "params": params,
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
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def clear_cache(self, stream: str | None = None) -> None:
        """Clear the on-disk raw cache. ``--force`` in T14 maps to this.

        ``stream=None`` removes the whole ``cache_dir/<route>/`` partition; otherwise only the
        ``<stream>/`` sub-directory. Idempotent.
        """
        root = self._cache_dir / self.route.value
        target = root / stream if stream is not None else root
        if target.is_dir():
            for child in sorted(target.iterdir(), reverse=True):
                if child.is_dir():
                    for f in child.iterdir():
                        f.unlink(missing_ok=True)
                    child.rmdir()
                else:
                    child.unlink(missing_ok=True)
            target.rmdir()

    # ------------------------------------------------------------------
    # Retry / backoff / error classification
    # ------------------------------------------------------------------

    def _describe(self, request: FetchRequest, start: int | None, end: int | None) -> str:
        """Safe, secret-free description used in error messages. Never logs a URL or key."""
        bits = [f"route={self.route.value!r}", f"stream={request.stream!r}"]
        if request.pool is not None:
            bits.append(f"pool={request.pool.address}")
        if start is not None or end is not None:
            bits.append(f"blocks={start}..{end}")
        return ", ".join(bits)

    def _backoff(self, attempt: int) -> float:
        """Full-jitter backoff for ``attempt`` (0-based): uniform(0, min(cap, base*2**attempt))."""
        bound = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2 ** attempt))
        return self._rng.uniform(0.0, bound)

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float | None:
        """Seconds from a ``Retry-After`` header, or ``None`` if absent/unparseable."""
        header = response.headers.get("Retry-After")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None

    def _handle_transient(
        self,
        exc: RateLimitError | TransientNetworkError,
        attempt: int,
        retry_after: float | None = None,
    ) -> None:
        """Sleep-then-retry gate for a transient error; raises ``exc`` once attempts are exhausted.

        A ``Retry-After`` hint overrides the jittered backoff. When ``attempt >= max_retries`` the
        error is raised unchanged (message already carries the block range via ``context``).
        """
        if attempt >= self._max_retries:
            raise exc
        delay = retry_after if retry_after is not None else self._backoff(attempt)
        self._sleep(delay)

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        context: str = "",
    ) -> requests.Response:
        """Perform one HTTP request to ``url`` with backoff/jitter on transient failures.

        Retries only ``RateLimitError`` (HTTP 429) and ``TransientNetworkError`` (5xx, connection
        reset, read timeout, JSON-RPC transient phrase). A non-retryable 4xx raises
        ``PermanentFetchError`` immediately. Returns the 2xx ``requests.Response``.

        ``context`` is a secret-free string (pass ``self._describe(...)``) folded into every error
        message so a final failure names the block range without naming the URL or any key.
        """
        attempt = 0
        while True:
            try:
                response = self._session.request(
                    method, url, headers=headers, json=json, timeout=self._timeout
                )
            except requests.exceptions.Timeout:
                self._handle_transient(
                    TransientNetworkError(f"{context}: request timed out after {self._timeout}s"),
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
                return response
            retry_after = self._parse_retry_after(response)
            if response.status_code == 429:
                error: RateLimitError | TransientNetworkError = RateLimitError(
                    f"{context}: HTTP 429 rate limited"
                )
            elif response.status_code >= 500:
                error = TransientNetworkError(f"{context}: HTTP {response.status_code}")
            else:
                raise PermanentFetchError(f"{context}: HTTP {response.status_code}")
            self._handle_transient(error, attempt, retry_after)
            attempt += 1

    def _raise_for_jsonrpc(self, payload: object, context: str = "") -> None:
        """Raise the right ``FetchError`` for a JSON-RPC ``error`` object, or no-op on success.

        A body carrying an ``error`` is generally a :class:`PermanentFetchError`; it is a
        retryable :class:`RateLimitError` only when the message matches a phrase in
        ``TRANSIENT_ERROR_PATTERNS``. Subclasses call this after ``_request`` when a response may
        be JSON-RPC-shaped.
        """
        if not isinstance(payload, dict):
            return
        error = payload.get("error")
        if error is None:
            return
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
        if any(pattern in needle for pattern in TRANSIENT_ERROR_PATTERNS):
            raise RateLimitError(f"{context}: transient provider error: {message or code}")
        raise PermanentFetchError(f"{context}: JSON-RPC error: {message or code}")

    # ------------------------------------------------------------------
    # fetch_rows: chunking + cache + retry + adaptive shrink + concurrency
    # ------------------------------------------------------------------

    def _is_too_many(self, error: PermanentFetchError) -> bool:
        needle = str(error).lower()
        return any(pattern in needle for pattern in TOO_MANY_RESULTS_PATTERNS)

    def _fetch_chunk_adaptive(
        self, request: FetchRequest, start: int, end: int
    ) -> list[dict]:
        """Fetch an inclusive range, halving on "too many results" down to a 1-block floor.

        On a single-block failure the :class:`PermanentFetchError` is re-raised with the block
        number in its message.
        """
        try:
            return self._fetch_chunk(request, start, end)
        except PermanentFetchError as error:
            if not self._is_too_many(error):
                raise
            if start >= end:
                raise PermanentFetchError(
                    f"{self._describe(request, start, end)}: {error}"
                ) from error
            mid = start + (end - start) // 2
            left = self._fetch_chunk_adaptive(request, start, mid)
            right = self._fetch_chunk_adaptive(request, mid + 1, end)
            return left + right

    def _fetch_one(self, request: FetchRequest, start: int, end: int) -> tuple[list[dict], bool]:
        """Serve one chunk: cache hit -> rows + ``used_cache=True``; else network -> cache + rows."""
        cached = self._cache_read(request, start, end)
        if cached is not None:
            return cached, True
        rows = self._fetch_chunk_adaptive(request, start, end)
        self._cache_write(request, start, end, rows)
        return rows, False

    def fetch_rows(self, request: FetchRequest) -> tuple[list[dict], int, bool, list[str]]:
        """The schema-free core of every ``fetch``.

        Returns ``(rows, n_requests, from_cache, warnings)``:
          * ``rows`` — raw row ``dict``s across the whole requested range, in determinstic
            (chunk) order regardless of thread completion order;
          * ``n_requests`` — number of chunks fetched from the network (0 == full cache hit);
          * ``from_cache`` — ``True`` only when every chunk was served from cache;
          * ``warnings`` — non-fatal notes (e.g. an empty / missing block range).

        No schema validation here, and no ``schemas.py`` import — T05-T08 own that.
        """
        warnings: list[str] = []
        start = request.start_block
        end = request.end_block
        if start is None or end is None:
            return [], 0, False, [f"fetch_rows: no block range in request ({self._describe(request, None, None)}); nothing to fetch"]
        if start > end:
            return [], 0, False, [f"fetch_rows: empty block range {start}..{end}; nothing to fetch"]

        chunk_edges = list(self.chunks(start, end, self._initial_chunk_size))
        ordered_rows: list[list[dict]] = [[] for _ in chunk_edges]
        all_cached = True
        n_requests = 0

        with ThreadPoolExecutor(max_workers=self._max_concurrency) as executor:
            futures = [
                executor.submit(self._fetch_one, request, lo, hi) for lo, hi in chunk_edges
            ]
            for index, future in enumerate(futures):
                rows, used_cache = future.result()
                ordered_rows[index] = rows
                all_cached = all_cached and used_cache
                if not used_cache:
                    n_requests += 1

        flat: list[dict] = []
        for rows in ordered_rows:
            flat.extend(rows)
        return flat, n_requests, all_cached, warnings

    # ------------------------------------------------------------------
    # Subclass contract (raise if unimplemented)
    # ------------------------------------------------------------------

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        """Raw decoded rows for ``[start, end]``. Subclasses MUST implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _fetch_chunk(request, start, end)"
        )