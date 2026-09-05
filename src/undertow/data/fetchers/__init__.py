"""Fetcher package for ``undertow.data`` (CONTRACTS.md §5.1, T04).

Exposes the shared contract types (``FetchRequest``, ``FetchResult``, ``Fetcher``) and the
``BaseHttpFetcher`` plumbing (retry/backoff, block-range chunking with adaptive shrink, and the
atomic on-disk raw cache). Concrete fetchers land in later tasks: T05 thegraph, T06 rpc, T07 gas,
T08 reference. Nothing here imports ``schemas.py``.
"""

from undertow.data.fetchers.base import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_CAP_SECONDS,
    DEFAULT_CHUNK_SIZE,
    FetchRequest,
    FetchResult,
    Fetcher,
    TOO_MANY_RESULTS_PATTERNS,
    TRANSIENT_ERROR_PATTERNS,
    BaseHttpFetcher,
)

__all__ = [
    "BaseHttpFetcher",
    "FetchRequest",
    "FetchResult",
    "Fetcher",
    "BACKOFF_BASE_SECONDS",
    "BACKOFF_CAP_SECONDS",
    "DEFAULT_CHUNK_SIZE",
    "TRANSIENT_ERROR_PATTERNS",
    "TOO_MANY_RESULTS_PATTERNS",
]