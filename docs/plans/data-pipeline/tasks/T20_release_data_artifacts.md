# T20 — Datasets as release artifacts (keep the repo light)

**Wave 6 · size S · depends on: T19 · unblocks: clone-and-run without repo bloat**
**Branch:** `feature-data-release-fetch`

## Why this task exists

Checking the validated datasets into git (the first attempt, PR #97) worked, but
put ~1.1 GB of already-compressed parquet into `main`, with every future dataset
revision (re-pull, repair, new pool) adding another ~1 GB to history. The dataset
is an **artifact**, not source: git is the wrong store for it.

Instead the three datasets are published as checksum-pinned **GitHub release
assets** under the [`data-v1`](https://github.com/Prolead1/undertow/releases/tag/data-v1)
tag, and a small downloader command makes a fresh clone runnable in one step.

## What was built

- **Release `data-v1`** with one asset per dataset (`tar.gz`, extraction root
  `<dataset>/`), built from the verified local parquet:
  - `eth_usdc_500-data-v1.tar.gz` — 836 007 611 B —
    `2e5506e8e005cf50188da964c91c8255026750e636aad242ab242835274d1ba9`
  - `eth_usdc_3000-data-v1.tar.gz` — 293 932 800 B —
    `cc87d938e9da56d5e1c3c536d7051f54f70184def02630301d7144280211eafc`
  - `eth_usdc_3000_test-data-v1.tar.gz` — 6 684 537 B —
    `8d9a8bdd0c7c8af2d921bc5645402ee1b8673de828376f8e074a59b0c3e6d192`
- **`src/undertow/data/release.py`** — `fetch_datasets()` streams each asset,
  verifies the pinned SHA-256, refuses archives whose members escape
  `<dataset>/` or that lack `manifest.json`, and extracts with
  `tarfile.extractall(filter="data")`. Present datasets are skipped unless
  `--force`; a failed download never leaves a `.part` file or a half-extracted
  directory.
- **`undertow-data fetch-data`** — `--dataset` (repeatable), `--out`
  (default `data/datasets`), `--force`, `--json`; throttled progress on stderr.
- README **Dataset** section documents the one-command bootstrap.

The digests are hardcoded in `release.py` (not read from the release), so a
tampered asset is rejected before extraction.

## Cleanup of the first attempt

PR #97 (data in git) was closed and its branch deleted. The local repo was
pruned (`git reflog expire` + `git gc --prune=now`): `.git` went from 942 MB to
**2.3 MB**. The dataset files themselves are untouched on disk and in the
release; `main` stays small.

## Tests

- `tests/data/test_release.py`: download+verify+extract, checksum mismatch,
  404, skip-without-network, force-replaces, unknown name, escaping member,
  missing manifest, multi-asset selection. All offline via `responses`.
- `tests/data/test_cli.py`: `fetch-data` delegation + `--json` output.

## Acceptance

- [x] `main` contains no dataset binaries; clone stays small
- [x] `uv run undertow-data fetch-data` on a fresh clone extracts all three
      datasets, SHA-256-verified
- [x] the 0.05 % eval split loads from the fetched data with no env vars
