"""Parquet storage and dataset manifest for ``undertow.data`` (T09).

The snapshot layer of the pipeline: streams move to and from disk as hive-style,
bucket-partitioned parquet, and ``manifest.json`` records enough provenance that a
number in the thesis can be traced back to the query that produced it. No fetching
and no domain math live here — only tables, files, and provenance.
"""

from __future__ import annotations

from undertow.data.storage.manifest import (
    DatasetManifest,
    StreamManifest,
    compute_dataset_id,
    endpoint_hosts,
    git_commit,
    library_versions,
    read_manifest,
    write_manifest,
)
from undertow.data.storage.parquet import content_hash, read_stream, write_stream

__all__ = [
    "DatasetManifest",
    "StreamManifest",
    "compute_dataset_id",
    "content_hash",
    "endpoint_hosts",
    "git_commit",
    "library_versions",
    "read_manifest",
    "read_stream",
    "write_manifest",
    "write_stream",
]