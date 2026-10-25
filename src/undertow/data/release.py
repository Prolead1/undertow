"""Download the validated datasets from the GitHub release assets (``data-v1``).

The parquet datasets are deliberately **not** stored in git: ~1.1 GB of
already-compressed data would bloat every clone, and every future dataset
revision would add another ~1 GB to history. They are published instead as
checksum-pinned release assets, and this module fetches → verifies → extracts
them into ``data/datasets/`` so a fresh clone becomes runnable with

    uv run undertow-data fetch-data

The SHA-256 of every asset is pinned here (not read from the release), so a
tampered or truncated download is rejected before anything is extracted.
Extraction is staged: members are validated after path normalisation (``..``
and absolute names are refused), the archive is unpacked into a temporary
sibling directory, and only a complete ``<dataset>/manifest.json`` tree is
moved into place — a crash mid-extract can never leave a half dataset that the
next run would mistake for complete.
"""

from __future__ import annotations

import hashlib
import logging
import os
import posixpath
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests

from undertow.data.types import ConfigError, FetchError, PermanentFetchError

LOGGER = logging.getLogger("undertow.data.release")

RELEASE_REPO: Final[str] = "Prolead1/undertow"
RELEASE_TAG: Final[str] = "data-v1"
RELEASE_BASE_URL: Final[str] = (
    f"https://github.com/{RELEASE_REPO}/releases/download/{RELEASE_TAG}"
)

_DOWNLOAD_CHUNK_BYTES: Final[int] = 1 << 20  # 1 MiB
_TIMEOUT: Final[tuple[float, float]] = (10.0, 120.0)  # (connect, read)
_MANIFEST_NAME: Final[str] = "manifest.json"
_MAX_ATTEMPTS: Final[int] = 3  # transient download failures are retried this often
_BACKOFF_BASE_SECONDS: Final[float] = 0.5  # first retry delay; doubles per attempt


class _TransientDownloadError(FetchError):
    """Internal: a download failure worth retrying (connection, 429, 5xx)."""


@dataclass(frozen=True, slots=True)
class DatasetAsset:
    """One release asset: the dataset it carries, its pinned digest, display size."""

    dataset: str  # directory name created under the output root
    asset: str  # release asset filename
    sha256: str  # pinned digest of the *archive* as published
    rows: int  # tape rows (display only; provenance lives in the manifest)
    size_bytes: int  # archive size (display only)

    def url(self, base_url: str = RELEASE_BASE_URL) -> str:
        return f"{base_url}/{self.asset}"


ASSETS: Final[tuple[DatasetAsset, ...]] = (
    DatasetAsset(
        dataset="eth_usdc_500",
        asset="eth_usdc_500-data-v1.tar.gz",
        sha256="2e5506e8e005cf50188da964c91c8255026750e636aad242ab242835274d1ba9",
        rows=7_112_899,
        size_bytes=836_007_611,
    ),
    DatasetAsset(
        dataset="eth_usdc_3000",
        asset="eth_usdc_3000-data-v1.tar.gz",
        sha256="cc87d938e9da56d5e1c3c536d7051f54f70184def02630301d7144280211eafc",
        rows=554_854,
        size_bytes=293_932_800,
    ),
    DatasetAsset(
        dataset="eth_usdc_3000_test",
        asset="eth_usdc_3000_test-data-v1.tar.gz",
        sha256="8d9a8bdd0c7c8af2d921bc5645402ee1b8673de828376f8e074a59b0c3e6d192",
        rows=11_190,
        size_bytes=6_684_537,
    ),
)

ProgressFn = Callable[[DatasetAsset, int, int], None]
"""``(asset, bytes_done, bytes_total)`` — called per downloaded chunk."""


@dataclass(frozen=True, slots=True)
class FetchOutcome:
    """What happened for one dataset: ``fetched`` or ``already_present``."""

    dataset: str
    status: str
    path: Path


def _select_assets(
    datasets: Sequence[str] | None, assets: Sequence[DatasetAsset]
) -> list[DatasetAsset]:
    """Resolve the requested dataset names against the pinned asset table.

    Duplicates are collapsed preserving order, so ``--dataset x --dataset x``
    never downloads the same archive twice.
    """
    if not datasets:
        return list(assets)
    by_name = {asset.dataset: asset for asset in assets}
    unknown = [name for name in datasets if name not in by_name]
    if unknown:
        raise ConfigError(
            f"fetch-data: unknown dataset(s) {unknown}; "
            f"available: {sorted(by_name)}"
        )
    return [by_name[name] for name in dict.fromkeys(datasets)]


def _stream_to_file(
    session: requests.Session,
    asset: DatasetAsset,
    destination: Path,
    *,
    base_url: str,
    progress: ProgressFn | None,
) -> None:
    """One download attempt: stream to ``destination`` and verify the digest.

    Raises ``_TransientDownloadError`` for connection drops / 429 / 5xx (the
    caller retries) and ``PermanentFetchError`` for 404, other 4xx and a digest
    mismatch.
    """
    url = asset.url(base_url)
    try:
        response = session.get(url, stream=True, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise _TransientDownloadError(f"request failed: {exc}") from exc
    with response:
        status = response.status_code
        if status == 404:
            raise PermanentFetchError(
                f"fetch-data: {asset.asset}: not found at {url} — is the "
                f"'{RELEASE_TAG}' release published?"
            )
        if status == 429 or status >= 500:
            raise _TransientDownloadError(f"HTTP {status} from {url}")
        if status >= 400:
            raise PermanentFetchError(
                f"fetch-data: {asset.asset}: HTTP {status} from {url}"
            )
        total = int(response.headers.get("Content-Length") or asset.size_bytes)
        digest = hashlib.sha256()
        done = 0
        try:
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(asset, done, total)
        except requests.RequestException as exc:
            raise _TransientDownloadError(f"download interrupted: {exc}") from exc
    if digest.hexdigest() != asset.sha256:
        raise PermanentFetchError(
            f"fetch-data: {asset.asset}: sha256 mismatch — expected "
            f"{asset.sha256}, got {digest.hexdigest()}"
        )


def _download(
    session: requests.Session,
    asset: DatasetAsset,
    destination: Path,
    *,
    base_url: str,
    progress: ProgressFn | None,
    sleep: Callable[[float], None],
) -> None:
    """Download one asset, retrying transient failures with exponential backoff.

    Every attempt restarts the archive from scratch (there is no range resume);
    the caller never sees a partial ``.part`` — a failed attempt is truncated by
    the next ``open("wb")`` and removed by :func:`fetch_datasets`.
    """
    url = asset.url(base_url)
    LOGGER.info(
        "fetch-data: downloading %s (%.0f MB)", url, asset.size_bytes / 1_000_000
    )
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _stream_to_file(
                session, asset, destination, base_url=base_url, progress=progress
            )
            return
        except _TransientDownloadError as exc:
            last_error = exc
            if attempt == _MAX_ATTEMPTS:
                break
            delay = _BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
            LOGGER.warning(
                "fetch-data: %s: %s — retrying in %.1fs (attempt %d/%d)",
                asset.asset,
                exc,
                delay,
                attempt + 1,
                _MAX_ATTEMPTS,
            )
            sleep(delay)
    raise FetchError(
        f"fetch-data: {asset.asset}: download failed after {_MAX_ATTEMPTS} "
        f"attempt(s): {last_error}"
    )


def _extract(archive: Path, output_dir: Path, dataset: str) -> Path:
    """Extract one verified archive into ``output_dir`` via a staging directory.

    Member names are normalised with :func:`posixpath.normpath` *before* the
    ``<dataset>/`` prefix check: ``tarfile``'s ``data`` filter normalises ``..``
    rather than rejecting it, so ``<dataset>/../evil`` would otherwise pass a
    raw-prefix test and write outside the dataset. The ``data`` filter still runs
    as a second line of defence (absolute links, device files, …).
    """
    target = output_dir / dataset
    staging_root: Path | None = None
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = [posixpath.normpath(member.name) for member in tar.getmembers()]
            for name in names:
                if name != dataset and not name.startswith(f"{dataset}/"):
                    raise PermanentFetchError(
                        f"fetch-data: {archive.name}: member {name!r} escapes "
                        f"{dataset}/ — refusing to extract"
                    )
            if f"{dataset}/{_MANIFEST_NAME}" not in names:
                raise PermanentFetchError(
                    f"fetch-data: {archive.name}: archive has no "
                    f"{dataset}/{_MANIFEST_NAME}; refusing to extract an "
                    "incomplete dataset"
                )
            staging_root = Path(
                tempfile.mkdtemp(prefix=f".{dataset}.", dir=output_dir)
            )
            try:
                tar.extractall(staging_root, filter="data")
            except (tarfile.TarError, OSError, ValueError) as exc:
                raise PermanentFetchError(
                    f"fetch-data: {archive.name}: extraction failed: {exc}"
                ) from exc
        staged = staging_root / dataset
        if not (staged / _MANIFEST_NAME).is_file():  # belt-and-braces
            raise PermanentFetchError(
                f"fetch-data: {archive.name}: extracted {dataset}/ without a "
                f"{_MANIFEST_NAME}; the archive is incomplete"
            )
        if target.is_symlink() or target.is_file():
            target.unlink()  # --force over a symlink/file must not rmtree-error
        elif target.is_dir():
            LOGGER.info("fetch-data: replacing existing %s", target)
            shutil.rmtree(target)
        os.replace(staged, target)
        return target
    except tarfile.TarError as exc:
        raise PermanentFetchError(
            f"fetch-data: {archive.name}: not a valid tar archive: {exc}"
        ) from exc
    except OSError as exc:
        raise PermanentFetchError(
            f"fetch-data: {archive.name}: cannot place {dataset}/ at {target}: {exc}"
        ) from exc
    finally:
        # Function-wide cleanup: every failure path after mkdtemp — including the
        # data-filter rejections raised above — must not leak a hidden staging dir.
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)


def fetch_datasets(
    output_dir: Path,
    *,
    datasets: Sequence[str] | None = None,
    force: bool = False,
    base_url: str = RELEASE_BASE_URL,
    assets: Sequence[DatasetAsset] | None = None,
    session: requests.Session | None = None,
    progress: ProgressFn | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[FetchOutcome]:
    """Download, verify and extract the validated datasets.

    ``datasets`` selects by name (``None`` = all). A dataset whose
    ``<output_dir>/<dataset>/manifest.json`` already exists is skipped unless
    ``force``; ``force`` downloads and verifies first, then swaps the completed
    staging tree into place, so the existing dataset survives any failure.

    ``assets`` / ``base_url`` / ``session`` / ``sleep`` override the pinned table,
    the GitHub URL and the retry backoff (mirrors and tests); production callers
    use the defaults.
    """
    selected = _select_assets(datasets, list(assets) if assets is not None else ASSETS)
    output_dir = Path(output_dir)

    outcomes: list[FetchOutcome] = []
    owned_session = session is None
    session = session or requests.Session()
    try:
        for asset in selected:
            target = output_dir / asset.dataset
            if not force and (target / _MANIFEST_NAME).is_file():
                LOGGER.info(
                    "fetch-data: %s already present at %s — skipping (use --force "
                    "to re-download)",
                    asset.dataset,
                    target,
                )
                outcomes.append(
                    FetchOutcome(asset.dataset, "already_present", target)
                )
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{asset.asset}.", suffix=".part", dir=output_dir
            )
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                _download(
                    session,
                    asset,
                    tmp,
                    base_url=base_url,
                    progress=progress,
                    sleep=sleep,
                )
                path = _extract(tmp, output_dir, asset.dataset)
            finally:
                tmp.unlink(missing_ok=True)
            outcomes.append(FetchOutcome(asset.dataset, "fetched", path))
    finally:
        if owned_session:
            session.close()
    return outcomes
