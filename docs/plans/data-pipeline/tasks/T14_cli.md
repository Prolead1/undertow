# T14 — CLI + reproducible snapshot pipeline

**Wave 6 · size M · depends on: T05–T13 and T15 · blocks: T16**
**Branch:** `feature-data-cli`

## Why this task exists

§10.6's Week-1 deliverable is *"a validated, reproducible dataset loader (script + a parquet
snapshot)"*. Everything before this task built components; this task is the **script** — one command
that turns a config file into a verified dataset, resumably, without re-querying anything it already
has. §10.7's schedule-risk mitigation ("cache to parquet so you never re-query") is only real if the
top-level entry point is genuinely resumable and idempotent.

## Files you own

```
src/undertow/data/cli.py
src/undertow/data/pipeline.py          (orchestration logic — keep it out of cli.py)
pyproject.toml                          (add [project.scripts] only — coordinate, others touch this file)
tests/data/test_cli.py
tests/data/test_pipeline.py
docs/running_the_pipeline.md
```

**Architecture note the reviewer will check:** `cli.py` does argument parsing, exit codes and human
output — nothing else. The orchestration (which fetchers in which order, how to resume, how to assemble
the `Dataset`) lives in `pipeline.py` as testable functions that never touch `sys.argv` or `sys.exit`.
A CLI module containing the pipeline logic is a god module.

## What to build

### 1. `pipeline.py`
```python
def pull(config: DataConfig, *, streams: Sequence[str] | None = None,
         route: Route | None = None, force: bool = False) -> PullReport
def build(config: DataConfig) -> Dataset                    # assemble tape from cached streams
def verify(config: DataConfig, *, sample_windows: int = 1) -> list[CheckResult]
def snapshot(config: DataConfig, out: Path | None = None) -> SnapshotReport   # pull + build + verify + manifest
def info(config: DataConfig) -> InfoReport
```

- **Resumability.** `pull` skips work already in the T04 cache and already written to parquet. Resuming
  an interrupted three-year pull must not restart it. Track progress at bucket granularity (T09's
  `block_number // 100_000`) so a resume is coarse but reliable, and record completed buckets in a small
  progress file next to the manifest. On resume, re-verify the last completed bucket rather than trusting
  it blindly — an interrupted process may have written a partial bucket (T09's atomic write should
  prevent this; verify anyway, cheaply, via row count and block-range continuity).
- **Idempotence.** A second `pull` with a warm cache must make **zero** network requests and leave the
  parquet content hash unchanged. This is a tested acceptance criterion, not an aspiration.
- **Route policy** (from §10.1.5's pragmatic plan): default bulk source is The Graph for the four log
  streams; RPC for `fee_growth` always; RPC additionally for a **sampled** verification window (size
  from `--sample-windows`) so `crosscheck_routes` has data to compare. `--route rpc` forces RPC for
  everything (slow, for the paranoid full re-derivation). Document the default clearly — a reader must
  know which route produced the numbers in the thesis, and the manifest records it per stream.
- **Fee-growth sampling strategy.** Full per-block snapshots are infeasible (millions of archive calls).
  Sample at: every position lifecycle boundary in the window (mint/burn/collect blocks), plus a regular
  stride (config-driven, default every 1000 blocks), plus every block T13's reconciliation needs. Record
  the strategy in the manifest so the sampling is reproducible and auditable. T10's replay covers the
  gaps; the samples are what prove the replay right.
- **Progress reporting.** Long-running; log progress at INFO with ETA and cumulative request counts.
  Nobody will trust a silent three-hour command.
- **Failure semantics.** A fetch failure after retries aborts the stream but records what completed, so
  a resume continues. Never leave a manifest claiming a stream is complete when it is not — write the
  manifest **last**, after all streams and verification.

### 2. `cli.py`
Four subcommands exactly as `CONTRACTS.md` §9 specifies (`pull`, `verify`, `snapshot`, `info`), with the
documented flags and exit codes: `0` ok, `1` critical check failed, `2` config error, `3` fetch error.

- `--config` is required on all subcommands.
- `verify --fail-on warning|critical` controls the exit-code threshold.
- Human-readable output by default; `--json` for machine output (T16's docs and any future CI want it).
- `argparse` is fine — do not add a CLI framework dependency for four subcommands.
- On `ConfigError` / `FetchError` / `ValidationError`, print a clean message and exit with the mapped
  code. **No tracebacks in normal operation**; `--debug` re-raises for developers.
- Register `[project.scripts] undertow-data = "undertow.data.cli:main"` in `pyproject.toml`. T00
  deliberately left this out (an entry point to a nonexistent module breaks `uv sync`), so adding it now
  is your job. `pyproject.toml` is shared — keep the diff to that one section.

### 3. `docs/running_the_pipeline.md`
The runbook: env vars needed, the three commands to get from nothing to a verified snapshot, expected
wall-clock and request counts for the pinned window, how to resume after an interruption, how to clear
the cache, and how to read `validation_report.md`. Include the actual observed timings if you ran it.

## Tests you must write

No network. Use T11's `tiny_dataset()` and monkeypatched fetchers.

1. `pull` with mocked fetchers writes the expected parquet layout; `PullReport` counts are right.
2. **Idempotence:** second `pull` → `n_requests == 0`, identical `content_hash` per stream.
3. **Resume:** kill `pull` mid-way (raise from the third bucket's fetcher), re-run, and assert the final
   dataset equals the uninterrupted one *and* that the first two buckets were not re-fetched.
4. Partial-bucket recovery: corrupt the last bucket's row count, resume, assert it is re-fetched rather
   than trusted.
5. `--force` re-fetches (request count > 0) and still produces an identical hash.
6. `build` assembles a `Dataset` whose tape matches T11's direct construction.
7. `verify` returns check results and does not raise on failures.
8. Exit codes: a critical failure → `1`; a bad config path → `2`; a mocked exhausted-retry fetch error →
   `3`; a clean run → `0`. Four tests, asserted via `SystemExit.code`.
9. `verify --fail-on warning` exits `1` on a warning-only dataset; the default does not.
10. `snapshot` writes parquet + `manifest.json` + `validation_report.md` to `--out`, and the manifest's
    per-stream routes reflect the route policy actually used.
11. Manifest-last ordering: simulate a failure during verification and assert **no** manifest was written
    (so a failed run can never masquerade as complete).
12. `info` on a written dataset prints/returns correct row counts and block coverage.
13. `--json` output parses and contains the documented keys.
14. No secret appears in any output stream or log for any subcommand (parametrize over all four).

## Acceptance criteria

- `uv run pytest tests/data/test_cli.py tests/data/test_pipeline.py` green; `mypy` clean.
- `uv run undertow-data --help` works after `uv sync`.
- `cli.py` contains no orchestration logic and `pipeline.py` contains no `argparse`/`sys.exit`.
- The runbook is accurate enough that the human can run it without reading the code.

## Handoff notes for your PR body

- Observed wall-clock and request counts for a real pull, if you ran one (even a one-month slice is
  useful — the human needs to know whether the three-year pull is a 2-hour or a 2-day job).
- The fee-growth sampling strategy actually implemented and its call count.
- The route policy as implemented, quotable for the thesis's methods section.
- Anything that had to be stubbed because an upstream task's PR was unmerged.

## Process (mandatory)

Branch (stacked per `PLAN.md` §3.2) → implement → tests green → `code-reviewer` → fix → commit → push →
PR → update `STATUS.md`.
