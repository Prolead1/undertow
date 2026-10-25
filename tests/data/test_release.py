"""Tests for the release-asset dataset downloader (T20).

Everything runs offline against ``responses``-mocked HTTPS; the pinned asset
table is replaced with a locally built archive so no dataset is downloaded.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
import requests
import responses

from undertow.data.release import DatasetAsset, fetch_datasets
from undertow.data.types import ConfigError, FetchError, PermanentFetchError

BASE_URL = "https://example.test/assets"


def _archive(*members: tuple[str, bytes]) -> bytes:
    """Build a deterministic-enough ``.tar.gz`` from ``(member_name, bytes)``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _dataset_archive(dataset: str = "eth_usdc_500") -> bytes:
    return _archive(
        (f"{dataset}/manifest.json", b'{"dataset": "x"}'),
        (f"{dataset}/validation_report.md", b"# ok"),
        (f"{dataset}/stream=gas/block_bucket=1/part.parquet", b"PAR1"),
    )


def _asset(dataset: str, archive: bytes, **overrides: object) -> DatasetAsset:
    payload: dict[str, object] = {
        "dataset": dataset,
        "asset": f"{dataset}-fake.tar.gz",
        "sha256": hashlib.sha256(archive).hexdigest(),
        "rows": 1,
        "size_bytes": len(archive),
    }
    payload.update(overrides)
    return DatasetAsset(**payload)  # type: ignore[arg-type]


def _part_files(root: Path) -> list[Path]:
    return [path for path in root.iterdir() if path.name.endswith(".part")]


@responses.activate
def test_download_verify_and_extract(tmp_path: Path) -> None:
    """A good download lands the whole tree, reports progress and leaves no part file."""
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)
    seen: list[tuple[str, int, int]] = []

    outcomes = fetch_datasets(
        tmp_path,
        base_url=BASE_URL,
        assets=[asset],
        progress=lambda a, done, total: seen.append((a.dataset, done, total)),
    )

    assert len(outcomes) == 1
    assert outcomes[0].status == "fetched"
    assert outcomes[0].path == tmp_path / "eth_usdc_500"
    base = tmp_path / "eth_usdc_500"
    assert (base / "manifest.json").is_file()
    assert (base / "validation_report.md").is_file()
    assert (base / "stream=gas" / "block_bucket=1" / "part.parquet").read_bytes() == b"PAR1"
    assert seen and seen[-1] == ("eth_usdc_500", len(archive), len(archive))
    assert _part_files(tmp_path) == []


@responses.activate
def test_checksum_mismatch_rejected_and_cleaned_up(tmp_path: Path) -> None:
    """A tampered/truncated body is refused before extraction; nothing is left behind."""
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive, sha256="0" * 64)
    responses.get(asset.url(BASE_URL), body=archive)

    with pytest.raises(PermanentFetchError, match="sha256 mismatch"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert not (tmp_path / "eth_usdc_500").exists()
    assert _part_files(tmp_path) == []


@responses.activate
def test_missing_asset_raises_permanent_error(tmp_path: Path) -> None:
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), status=404)

    with pytest.raises(PermanentFetchError, match="not found"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert _part_files(tmp_path) == []


@responses.activate
def test_existing_dataset_is_skipped_without_network(tmp_path: Path) -> None:
    """The manifest is the completeness marker: present + no --force => no request."""
    target = tmp_path / "eth_usdc_500"
    target.mkdir()
    (target / "manifest.json").write_text("{}", encoding="utf-8")
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)

    outcomes = fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert outcomes[0].status == "already_present"
    assert outcomes[0].path == target
    assert len(responses.calls) == 0


@responses.activate
def test_force_replaces_the_existing_directory(tmp_path: Path) -> None:
    """--force re-downloads and drops stale files from the previous dataset version."""
    target = tmp_path / "eth_usdc_500"
    target.mkdir()
    (target / "manifest.json").write_text("{}", encoding="utf-8")
    (target / "stale.parquet").write_bytes(b"old")
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)

    outcomes = fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset], force=True)

    assert outcomes[0].status == "fetched"
    assert not (target / "stale.parquet").exists()
    assert (target / "manifest.json").is_file()


def test_unknown_dataset_name_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown dataset"):
        fetch_datasets(tmp_path, datasets=["not_a_dataset"])


@responses.activate
def test_archive_member_outside_dataset_dir_is_rejected(tmp_path: Path) -> None:
    """A malicious archive cannot write outside ``<output>/<dataset>/``."""
    archive = _archive(
        ("evil/payload.txt", b"x"),
        ("eth_usdc_500/manifest.json", b"{}"),
    )
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)

    with pytest.raises(PermanentFetchError, match="escapes"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "eth_usdc_500").exists()


@responses.activate
def test_dotdot_members_cannot_clobber_a_sibling_dataset(tmp_path: Path) -> None:
    """``<dataset>/../x`` normalises outside the dataset and must be refused.

    ``tarfile``'s ``data`` filter *normalises* ``..`` rather than rejecting it, so
    the guard must compare normalised names — otherwise this archive would write
    ``<out>/evil.txt`` and overwrite a sibling dataset's manifest.
    """
    sibling = tmp_path / "eth_usdc_3000"
    sibling.mkdir()
    (sibling / "manifest.json").write_text("original", encoding="utf-8")
    archive = _archive(
        ("eth_usdc_500/manifest.json", b"{}"),
        ("eth_usdc_500/../eth_usdc_3000/manifest.json", b"hostile"),
        ("eth_usdc_500/../evil.txt", b"x"),
    )
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)

    with pytest.raises(PermanentFetchError, match="escapes"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert (sibling / "manifest.json").read_text(encoding="utf-8") == "original"
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "eth_usdc_500").exists()
    assert _part_files(tmp_path) == []


@responses.activate
def test_archive_without_manifest_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive = _archive(("eth_usdc_500/validation_report.md", b"# ok"))
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)

    with pytest.raises(PermanentFetchError, match="manifest.json"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert not (tmp_path / "eth_usdc_500").exists()
    assert _part_files(tmp_path) == []


@responses.activate
def test_all_datasets_in_one_call(tmp_path: Path) -> None:
    """The default selection fetches every requested asset in order."""
    archives = {
        "eth_usdc_500": _dataset_archive("eth_usdc_500"),
        "eth_usdc_3000": _dataset_archive("eth_usdc_3000"),
    }
    assets = [_asset(name, body) for name, body in archives.items()]
    for asset in assets:
        responses.get(asset.url(BASE_URL), body=archives[asset.dataset])

    outcomes = fetch_datasets(tmp_path, base_url=BASE_URL, assets=assets)

    assert [outcome.dataset for outcome in outcomes] == ["eth_usdc_500", "eth_usdc_3000"]
    assert all(outcome.status == "fetched" for outcome in outcomes)
    assert (tmp_path / "eth_usdc_3000" / "manifest.json").is_file()


@responses.activate
def test_dataset_selection_dedupes_and_skips_unselected(tmp_path: Path) -> None:
    """``datasets=[...]`` downloads only what is named, once even if repeated."""
    archives = {
        "eth_usdc_500": _dataset_archive("eth_usdc_500"),
        "eth_usdc_3000": _dataset_archive("eth_usdc_3000"),
    }
    assets = [_asset(name, body) for name, body in archives.items()]
    responses.get(assets[0].url(BASE_URL), body=archives["eth_usdc_500"])

    outcomes = fetch_datasets(
        tmp_path,
        datasets=["eth_usdc_500", "eth_usdc_500"],
        base_url=BASE_URL,
        assets=assets,
    )

    assert [outcome.dataset for outcome in outcomes] == ["eth_usdc_500"]
    assert len(responses.calls) == 1


@responses.activate
def test_force_keeps_existing_dataset_when_the_new_archive_fails(
    tmp_path: Path,
) -> None:
    """The old directory is replaced only after the replacement verified."""
    target = tmp_path / "eth_usdc_500"
    target.mkdir()
    (target / "manifest.json").write_text("{}", encoding="utf-8")
    (target / "keep.parquet").write_bytes(b"keep")
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive, sha256="0" * 64)
    responses.get(asset.url(BASE_URL), body=archive)

    with pytest.raises(PermanentFetchError, match="sha256 mismatch"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset], force=True)

    assert (target / "keep.parquet").read_bytes() == b"keep"
    assert (target / "manifest.json").read_text(encoding="utf-8") == "{}"
    assert _part_files(tmp_path) == []


@responses.activate
def test_transient_http_error_is_retried_then_succeeds(tmp_path: Path) -> None:
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), status=503)
    responses.get(asset.url(BASE_URL), body=archive)
    sleeps: list[float] = []

    outcomes = fetch_datasets(
        tmp_path, base_url=BASE_URL, assets=[asset], sleep=sleeps.append
    )

    assert outcomes[0].status == "fetched"
    assert sleeps == [0.5]
    assert (tmp_path / "eth_usdc_500" / "manifest.json").is_file()


@responses.activate
def test_persistent_transient_error_raises_fetch_error(tmp_path: Path) -> None:
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    for _ in range(3):
        responses.get(asset.url(BASE_URL), status=500)

    with pytest.raises(FetchError, match="download failed after 3"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset], sleep=lambda _: None)

    assert _part_files(tmp_path) == []
    assert not (tmp_path / "eth_usdc_500").exists()


@responses.activate
def test_connection_error_is_retried_then_raises_fetch_error(tmp_path: Path) -> None:
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    for _ in range(3):
        responses.get(asset.url(BASE_URL), body=requests.ConnectionError("boom"))

    with pytest.raises(FetchError, match="download failed after 3"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset], sleep=lambda _: None)

    assert _part_files(tmp_path) == []


def test_midstream_drop_is_retried_then_succeeds(tmp_path: Path) -> None:
    """A connection drop while streaming the body must not corrupt or abort the fetch."""
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)

    class BrokenResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> BrokenResponse:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def iter_content(self, chunk_size: int):  # type: ignore[no-untyped-def]
            yield b"partial"
            raise requests.ConnectionError("drop")

    class GoodResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def __enter__(self) -> GoodResponse:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def iter_content(self, chunk_size: int):  # type: ignore[no-untyped-def]
            yield archive

    class ScriptedSession(requests.Session):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def get(self, url, **kwargs):  # type: ignore[override]
            self.calls += 1
            return BrokenResponse() if self.calls == 1 else GoodResponse()

    session = ScriptedSession()
    outcomes = fetch_datasets(
        tmp_path,
        base_url=BASE_URL,
        assets=[asset],
        session=session,
        sleep=lambda _: None,
    )

    assert outcomes[0].status == "fetched"
    assert session.calls == 2  # first attempt dropped mid-stream, second succeeded
    assert (tmp_path / "eth_usdc_500" / "manifest.json").is_file()
    assert _part_files(tmp_path) == []


@responses.activate
def test_non_tar_asset_is_rejected(tmp_path: Path) -> None:
    body = b"this is not a tar archive"
    asset = _asset("eth_usdc_500", body)
    responses.get(asset.url(BASE_URL), body=body)

    with pytest.raises(PermanentFetchError, match="not a valid tar archive"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert _part_files(tmp_path) == []
    assert not (tmp_path / "eth_usdc_500").exists()


@responses.activate
def test_forbidden_asset_is_permanent_and_not_retried(tmp_path: Path) -> None:
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), status=403)

    with pytest.raises(PermanentFetchError, match="HTTP 403"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset], sleep=lambda _: None)

    assert len(responses.calls) == 1


@responses.activate
def test_filter_rejection_leaves_no_staging_directory(tmp_path: Path) -> None:
    """A member the ``data`` filter refuses must not leak the hidden staging dir.

    An absolute symlink is the classic filter rejection: the pre-check only sees a
    regular member under ``<dataset>/``, so the failure happens *during* extractall
    — exactly the path whose cleanup must be guaranteed.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        manifest = tarfile.TarInfo("eth_usdc_500/manifest.json")
        payload = b"{}"
        manifest.size = len(payload)
        tar.addfile(manifest, io.BytesIO(payload))
        link = tarfile.TarInfo("eth_usdc_500/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    archive = buffer.getvalue()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)

    with pytest.raises(PermanentFetchError, match="extraction failed"):
        fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset])

    assert not (tmp_path / "eth_usdc_500").exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".eth_usdc_500")]
    assert leftovers == [], f"staging leaked: {leftovers}"
    assert _part_files(tmp_path) == []


@responses.activate
def test_force_replaces_a_file_named_like_the_dataset(tmp_path: Path) -> None:
    """--force over a file (not a directory) must unlink, not fail in rmtree."""
    target = tmp_path / "eth_usdc_500"
    target.write_bytes(b"i am a file, not a dataset")
    archive = _dataset_archive()
    asset = _asset("eth_usdc_500", archive)
    responses.get(asset.url(BASE_URL), body=archive)

    outcomes = fetch_datasets(tmp_path, base_url=BASE_URL, assets=[asset], force=True)

    assert outcomes[0].status == "fetched"
    assert target.is_dir()
    assert (target / "manifest.json").is_file()
