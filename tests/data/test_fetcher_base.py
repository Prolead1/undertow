"""Tests for ``undertow.data.fetchers.base`` (T04) — retry/backoff, chunking, raw cache.

Everything runs offline against ``responses``-mocked endpoints and an injected no-op sleep so the
suite stays well under 5s. A tiny ``_StubFetcher`` subclass supplies ``_fetch_chunk`` (the only
piece T05-T08 must write) and exercises the template method through the public ``fetch_rows``.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import pytest
import responses

from undertow.data.config import EndpointConfig, default_pools
from undertow.data.fetchers.base import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    BaseHttpFetcher,
    PermanentFetchError,
    RateLimitError,
    TOO_MANY_RESULTS_PATTERNS,
    TRANSIENT_ERROR_PATTERNS,
)
from undertow.data.types import BlockNumber, FetchRequest, Route, TransientNetworkError

POOL_3000 = default_pools()["USDC_WETH_3000"]

STUB_URL = "https://stub.test/api"


@pytest.fixture
def mocked_http():
    """Activated ``responses`` mock; unused registered responses never fail the test."""
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield rsps


def _noop_sleep(_: float) -> None:
    """Injected sleep for fast tests — never actually sleeps."""


class _StubFetcher(BaseHttpFetcher):
    """Minimal concrete fetcher: POSTs the block range, decodes rows / JSON-RPC errors."""

    route = Route.RPC

    def __init__(
        self,
        url: str,
        endpoints: EndpointConfig,
        cache_dir: Path,
        *,
        cache_endpoint: str | None = None,
        sleep=None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(endpoints, cache_dir, sleep=sleep, rng=rng)
        self._url = url
        # Extra cache-key material; lets a test inject a secret-bearing endpoint URL without
        # touching the (safe) request URL. Redacted to host by cache_key.
        self._cache_endpoint = cache_endpoint if cache_endpoint is not None else url

    def _cache_params(self, request: FetchRequest, start: int, end: int) -> dict[str, object]:
        params = super()._cache_params(request, start, end)
        params["endpoint"] = self._cache_endpoint
        return params

    def _fetch_chunk(self, request: FetchRequest, start: int, end: int) -> list[dict]:
        resp = self._request(
            "POST",
            self._url,
            json={"start": start, "end": end},
            context=self._describe(request, start, end),
        )
        payload = resp.json()
        if isinstance(payload, dict) and "error" in payload:
            self._raise_for_jsonrpc(payload, self._describe(request, start, end))
        return payload if isinstance(payload, list) else payload.get("rows", [])


def _maker(
    tmp_path: Path,
    *,
    url: str = STUB_URL,
    max_retries: int = 3,
    max_concurrency: int = 1,
    cache_endpoint: str | None = None,
    seed: int = 1,
) -> _StubFetcher:
    endpoints = EndpointConfig(
        graph_url="https://g.test",
        rpc_url="https://r.test",
        reference_base_url="https://b.test",
        max_concurrency=max_concurrency,
        request_timeout_s=5.0,
        max_retries=max_retries,
    )
    return _StubFetcher(
        url,
        endpoints,
        tmp_path,
        cache_endpoint=cache_endpoint,
        sleep=_noop_sleep,
        rng=random.Random(seed),
    )


def _req(stream: str = "swap", start: int = 0, end: int = 9, pool=None) -> FetchRequest:
    return FetchRequest(
        stream=stream,
        pool=pool,
        start_block=BlockNumber(start) if start is not None else None,
        end_block=BlockNumber(end) if end is not None else None,
    )


# ---------------------------------------------------------------------------
# 1. chunks — exactness, boundaries, empty ranges, size <= 0
# ---------------------------------------------------------------------------


def test_chunks_exact() -> None:
    f = _maker(Path("/tmp/cache"))
    assert list(f.chunks(0, 9, 3)) == [(0, 2), (3, 5), (6, 8), (9, 9)]
    assert list(f.chunks(5, 5, 10)) == [(5, 5)]
    assert list(f.chunks(0, 0, 1)) == [(0, 0)]


def test_chunks_invalid_or_empty_sizes() -> None:
    f = _maker(Path("/tmp/cache"))
    for bad in (0, -1, -10):
        with pytest.raises(ValueError):
            list(f.chunks(0, 9, bad))
    assert list(f.chunks(5, 3, 3)) == []  # empty range yields nothing


def test_chunks_coverage_exact_no_dup_no_gap() -> None:
    f = _maker(Path("/tmp/cache"))
    for start, end in [(0, 9), (1, 1), (3, 20), (10, 60), (7, 12), (0, 0)]:
        for size in (1, 2, 3, 4, 7, 13):
            ranges = list(f.chunks(start, end, size))
            assert ranges, f"expected a non-empty chunk list for {start}..{end}"
            assert ranges[0][0] == start
            assert ranges[-1][1] == end
            flat = [i for lo, hi in ranges for i in range(lo, hi + 1)]
            assert flat == list(range(start, end + 1)), f"coverage wrong for {start}..{end}/{size}"


# ---------------------------------------------------------------------------
# 2. Retry: 429 -> 200, injected sleep, jitter bound, Retry-After override
# ---------------------------------------------------------------------------


def test_retry_429_then_200(mocked_http, tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _maker(tmp_path, max_retries=3)
    f._sleep = sleeps.append
    mocked_http.add(responses.POST, STUB_URL, status=429)
    mocked_http.add(responses.POST, STUB_URL, json=[{"row": 1}], status=200)

    rows, n, from_cache, _warnings = f.fetch_rows(_req(stream="swap", start=10, end=11))
    assert rows == [{"row": 1}]
    assert len(sleeps) == 1  # exactly one retry, one injected sleep
    bound = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2**0))
    assert 0.0 <= sleeps[0] <= bound  # within the full-jitter bound for attempt 0
    assert len(mocked_http.calls) == 2


def test_retry_after_overrides_backoff(mocked_http, tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _maker(tmp_path, max_retries=3)
    f._sleep = sleeps.append
    mocked_http.add(responses.POST, STUB_URL, status=429, headers={"Retry-After": "2"})
    mocked_http.add(responses.POST, STUB_URL, json=[], status=200)

    f.fetch_rows(_req(start=1, end=1))
    assert sleeps == [2.0]  # Retry-After wins over the jittered backoff


def test_retry_after_seconds_float_value(mocked_http, tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _maker(tmp_path, max_retries=3)
    f._sleep = sleeps.append
    mocked_http.add(responses.POST, STUB_URL, status=429, headers={"Retry-After": "0.5"})
    mocked_http.add(responses.POST, STUB_URL, json=[], status=200)
    f.fetch_rows(_req(start=1, end=1))
    assert sleeps == [0.5]


# ---------------------------------------------------------------------------
# 3. max_retries exhausted -> RateLimitError with the block range in the message
# ---------------------------------------------------------------------------


def test_max_retries_exhausted_ratelimit(mocked_http, tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _maker(tmp_path, max_retries=1)
    f._sleep = sleeps.append
    mocked_http.add(responses.POST, STUB_URL, status=429)

    with pytest.raises(RateLimitError) as excinfo:
        f.fetch_rows(_req(start=42, end=42))
    msg = str(excinfo.value)
    assert "42" in msg  # block range surfaced in the final failure
    assert "swap" in msg
    assert len(sleeps) == 1  # one retry, then exhausted


def test_transient_5xx_retries_then_exhausts(mocked_http, tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _maker(tmp_path, max_retries=1)
    f._sleep = sleeps.append
    mocked_http.add(responses.POST, STUB_URL, status=503)

    with pytest.raises(TransientNetworkError):
        f.fetch_rows(_req(start=1, end=2))
    assert len(sleeps) == 1


# ---------------------------------------------------------------------------
# 4. 404 -> PermanentFetchError, zero retries
# ---------------------------------------------------------------------------


def test_404_permanent_no_retry(mocked_http, tmp_path: Path) -> None:
    sleeps: list[float] = []
    f = _maker(tmp_path, max_retries=5)
    f._sleep = sleeps.append
    mocked_http.add(responses.POST, STUB_URL, status=404)

    with pytest.raises(PermanentFetchError) as excinfo:
        f.fetch_rows(_req(start=5, end=6))
    assert "404" in str(excinfo.value)
    assert sleeps == []  # never backed off
    assert len(mocked_http.calls) == 1  # hit exactly once


# ---------------------------------------------------------------------------
# 5. Adaptive shrink on "too many results"
# ---------------------------------------------------------------------------


def test_adaptive_shrink_full_coverage(mocked_http, tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    def cb(request):
        body = json.loads(request.body or "{}")
        lo, hi = int(body["start"]), int(body["end"])
        seen.append((lo, hi))
        if hi - lo + 1 > 4:
            return (
                200,
                {},
                json.dumps(
                    {"error": {"code": -32602, "message": "query returned more than 10000 results"}}
                ),
            )
        return 200, {}, json.dumps({"rows": [{"lo": lo, "hi": hi}]})

    f = _maker(tmp_path, max_retries=3)
    f._initial_chunk_size = 10
    mocked_http.add_callback(responses.POST, STUB_URL, callback=cb)

    rows, n, from_cache, _warnings = f.fetch_rows(_req(start=0, end=9))
    # full range covered exactly once, no overlap, no gap
    covered: set[int] = set()
    for lo, hi in seen:
        for i in range(lo, hi + 1):
            assert i not in covered, f"duplicate coverage of block {i}"
            covered.add(i)
    assert covered == set(range(10))
    # chunk size shrank below the initial width (no request ever wider than 4)
    assert all(hi - lo + 1 <= 4 for lo, hi in seen)
    assert len(rows) == len(seen)
    assert n == 1  # a single outer chunk was fetched from the network
    assert from_cache is False


def test_adaptive_shrink_single_block_still_fails(mocked_http, tmp_path: Path) -> None:
    def cb(request):
        return (
            200,
            {},
            json.dumps(
                {"error": {"code": -32602, "message": "query returned more than 10000 results"}}
            ),
        )

    f = _maker(tmp_path, max_retries=2)
    f._initial_chunk_size = 4
    mocked_http.add_callback(responses.POST, STUB_URL, callback=cb)

    with pytest.raises(PermanentFetchError) as excinfo:
        f.fetch_rows(_req(start=7, end=10))
    assert "7" in str(excinfo.value)  # block number surfaced


def test_permanent_other_than_too_many_not_shrunk(mocked_http, tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    def cb(request):
        body = json.loads(request.body or "{}")
        lo, hi = int(body["start"]), int(body["end"])
        seen.append((lo, hi))
        return 200, {}, json.dumps({"error": {"code": 32000, "message": "invalid params"}})

    f = _maker(tmp_path, max_retries=2)
    f._initial_chunk_size = 10
    mocked_http.add_callback(responses.POST, STUB_URL, callback=cb)
    with pytest.raises(PermanentFetchError):
        f.fetch_rows(_req(start=0, end=9))
    # an unrelated permanent error is surfaced without any shrinking
    assert len(seen) == 1
    assert seen[0] == (0, 9)


# ---------------------------------------------------------------------------
# 6. Cache: cold writes a file; warm makes zero requests
# ---------------------------------------------------------------------------


def test_cache_cold_then_warm(mocked_http, tmp_path: Path) -> None:
    f = _maker(tmp_path, max_retries=3)
    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"a": 1}]}, status=200)

    req = _req(stream="swap", start=100, end=109, pool=POOL_3000)
    r1, n1, fc1, _w = f.fetch_rows(req)
    assert r1 == [{"a": 1}]
    assert n1 == 1 and fc1 is False
    assert len(mocked_http.calls) == 1

    # second identical call: served entirely from the on-disk cache
    r2, n2, fc2, _w = f.fetch_rows(req)
    assert r2 == r1
    assert n2 == 0 and fc2 is True
    assert len(mocked_http.calls) == 1  # no extra network

    # a different stream still goes to the network
    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"x": 2}]}, status=200)
    r3, n3, _fc3, _w = f.fetch_rows(_req(stream="mint", start=100, end=109, pool=POOL_3000))
    assert r3 == [{"x": 2}]
    assert n3 == 1


def test_cache_file_layout(mocked_http, tmp_path: Path) -> None:
    f = _maker(tmp_path, max_retries=2)
    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"a": 1}]}, status=200)
    req = _req(stream="swap", start=5, end=6, pool=POOL_3000)
    f.fetch_rows(req)
    key = f._cache_key(req, 5, 6)
    path = tmp_path / "rpc" / "swap" / f"{key}.json.gz"
    assert path.is_file()
    assert f._cache_path(req, key) == path


def test_clear_cache(mocked_http, tmp_path: Path) -> None:
    f = _maker(tmp_path, max_retries=2)
    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"a": 1}]}, status=200)
    f.fetch_rows(_req(stream="swap", start=5, end=6, pool=POOL_3000))
    f.fetch_rows(_req(stream="mint", start=5, end=6, pool=POOL_3000))
    assert (tmp_path / "rpc" / "swap").is_dir()
    f.clear_cache(stream="swap")
    assert not (tmp_path / "rpc" / "swap").exists()
    assert (tmp_path / "rpc" / "mint").exists()
    f.clear_cache()
    assert not (tmp_path / "rpc").exists()


# ---------------------------------------------------------------------------
# 7. Cache key determinism (dict order, values, nested, stream/route)
# ---------------------------------------------------------------------------


def test_cache_key_determinism() -> None:
    f = _maker(Path("/tmp/cache"))
    a = {"z": 3, "a": 1, "nested": {"k2": 2, "k1": 1}}
    b = {"a": 1, "nested": {"k1": 1, "k2": 2}, "z": 3}  # same content, different insertion order
    assert f.cache_key(Route.RPC, "swap", a) == f.cache_key(Route.RPC, "swap", b)

    changed = dict(a)
    changed["a"] = 2
    assert f.cache_key(Route.RPC, "swap", a) != f.cache_key(Route.RPC, "swap", changed)

    # ints are rendered consistently as decimal strings
    assert f.cache_key(Route.RPC, "swap", {"n": 5}) == f.cache_key(Route.RPC, "swap", {"n": "5"})

    # nested lists canonicalise
    assert f.cache_key(Route.RPC, "swap", {"topic": ["a", "b"]}) == f.cache_key(
        Route.RPC, "swap", {"topic": ("a", "b")}
    )

    assert f.cache_key(Route.RPC, "swap", a) != f.cache_key(Route.RPC, "mint", a)
    assert f.cache_key(Route.RPC, "swap", a) != f.cache_key(Route.THEGRAPH, "swap", a)


def test_cache_key_unhashable_param_rejected() -> None:
    f = _maker(Path("/tmp/cache"))
    with pytest.raises(TypeError):
        f.cache_key(Route.RPC, "swap", {"bad": object()})


# ---------------------------------------------------------------------------
# 8. Secret hygiene
# ---------------------------------------------------------------------------


def test_secret_never_in_key_filename_or_disk(mocked_http, tmp_path: Path, caplog) -> None:
    secret = "SECRETKEY_abc123"
    secret_url = f"https://x.example.test/v2/{secret}"
    f = _maker(tmp_path, max_retries=2, cache_endpoint=secret_url)

    # key + filename must be free of the secret and the full URL
    key = f.cache_key(Route.RPC, "swap", {"rpc_url": secret_url, "block": 12345})
    assert secret not in key
    assert secret_url not in key
    assert "/v2/" not in key

    # exercise a real write and read so the envelope is checked too
    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"a": 1}]}, status=200)
    with caplog.at_level("WARNING"):
        f.fetch_rows(_req(stream="swap", start=5, end=6))
    key2 = f._cache_key(_req(stream="swap", start=5, end=6), 5, 6)
    path = f._cache_path(_req(stream="swap", start=5, end=6), key2)
    raw_bytes = path.read_bytes()
    secret_bytes = secret.encode()
    assert secret_bytes not in raw_bytes, "secret leaked into the cached envelope"
    assert secret not in path.name
    assert secret not in str(path)
    assert "SECRETKEY" not in caplog.text


# ---------------------------------------------------------------------------
# 9. Atomic write: crash mid-write leaves no entry, next read is a miss
# ---------------------------------------------------------------------------


def test_atomic_write_crash_leaves_no_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _maker(tmp_path, max_retries=2)
    req = _req(start=1, end=2)
    key = f._cache_key(req, 1, 2)
    path = f._cache_path(req, key)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr("undertow.data.fetchers.base.gzip.open", boom)
        with pytest.raises(OSError):
            f._cache_write(req, 1, 2, [{"x": 1}])

    assert not path.exists(), "no final .json.gz entry may exist after a crashed write"
    assert list(path.parent.glob("*.tmp")) == [], "temporary file must not linger"
    assert f._cache_read(req, 1, 2) is None  # clean miss afterwards


# ---------------------------------------------------------------------------
# 10. Corrupt cache entry -> miss + WARNING, network path taken
# ---------------------------------------------------------------------------


def test_corrupt_gzip_entry_is_a_miss(mocked_http, tmp_path: Path, caplog) -> None:
    f = _maker(tmp_path, max_retries=2)
    req = _req(start=3, end=4)
    path = f._cache_path(req, f._cache_key(req, 3, 4))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is definitely not gzip data")

    with caplog.at_level("WARNING", logger="undertow.data.fetchers.base"):
        assert f._cache_read(req, 3, 4) is None
    assert "corrupt" in caplog.text

    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"a": 9}]}, status=200)
    rows, _n, from_cache, _w = f.fetch_rows(req)
    assert rows == [{"a": 9}]
    assert from_cache is False


def test_corrupt_envelope_is_a_miss(mocked_http, tmp_path: Path, caplog) -> None:
    f = _maker(tmp_path, max_retries=2)
    req = _req(start=9, end=10)
    path = f._cache_path(req, f._cache_key(req, 9, 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"body": "not-a-list", "fetched_at_utc": "x", "params": {}}, fh)

    with caplog.at_level("WARNING", logger="undertow.data.fetchers.base"):
        assert f._cache_read(req, 9, 10) is None
    assert "corrupt" in caplog.text

    mocked_http.add(responses.POST, STUB_URL, json={"rows": [{"a": 9}]}, status=200)
    rows, _n, _fc, _w = f.fetch_rows(req)
    assert rows == [{"a": 9}]


# ---------------------------------------------------------------------------
# 11. Empty / absent block range -> empty result, zero requests
# ---------------------------------------------------------------------------


def test_empty_range_no_requests(mocked_http, tmp_path: Path) -> None:
    f = _maker(tmp_path, max_retries=2)
    mocked_http.add(responses.POST, STUB_URL, json=[], status=200)  # must never fire

    rows, n, from_cache, warnings = f.fetch_rows(_req(start=10, end=5))
    assert rows == []
    assert n == 0
    assert from_cache is False
    assert len(warnings) == 1
    assert len(mocked_http.calls) == 0


def test_no_block_range_no_requests(mocked_http, tmp_path: Path) -> None:
    f = _maker(tmp_path, max_retries=2)
    mocked_http.add(responses.POST, STUB_URL, json=[], status=200)
    rows, n, _fc, warnings = f.fetch_rows(_req(start=None, end=None))
    assert rows == []
    assert n == 0
    assert warnings


# ---------------------------------------------------------------------------
# Pattern constants / public surface sanity
# ---------------------------------------------------------------------------


def test_pattern_constants_are_extendable_sets() -> None:
    assert isinstance(TRANSIENT_ERROR_PATTERNS, frozenset)
    assert isinstance(TOO_MANY_RESULTS_PATTERNS, frozenset)
    assert "rate limit" in TRANSIENT_ERROR_PATTERNS
    assert "-32602" in TOO_MANY_RESULTS_PATTERNS
    # both stay extendable by union without mutating the frozen base
    assert TRANSIENT_ERROR_PATTERNS | {"custom"} != TRANSIENT_ERROR_PATTERNS
    assert TOO_MANY_RESULTS_PATTERNS | {"custom"} != TOO_MANY_RESULTS_PATTERNS


def test_module_imports_nothing_from_schemas_module() -> None:
    import undertow.data.fetchers.base as base_mod

    source = Path(base_mod.__file__).read_text(encoding="utf-8")
    assert "schemas" not in source
    assert "SCHEMA_REGISTRY" not in source
    assert "validate_table" not in source