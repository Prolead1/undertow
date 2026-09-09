"""Validation & cross-check suite for ``undertow.data`` (T13, CONTRACTS.md §8).

The loader's correctness is its own deliverable: every check here is a pass/fail
artifact that either certifies or indicts the dataset. ``run_all_checks`` loops
over everything the dataset has inputs for — and for inputs that are *absent* it
returns an ``info``-severity result saying so, never a silent omission and never
a spurious failure. It **never raises on a failed check** — only T14's CLI decides
the exit code (nonzero on any critical).

The two headline checks live in :mod:`crosscheck`:

- :func:`crosscheck_routes` — The Graph vs RPC field-level agreement (§10.1.5);
  any disagreement is critical because one of the two routes is wrong.
- :func:`reconcile_fees_against_collect` — reconstructed fees vs the chain's
  actual ``Collect`` amounts, principal separated out via the paired-``Burn``
  rule (§10.2.4): *the* acceptance test for the thesis's gap G3.

``render_report`` writes the markdown appendix artifact — human-readable by
someone who has not read the code, tied to the dataset it describes via the
manifest header.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]

from undertow.data.storage.manifest import DatasetManifest
from undertow.data.transforms.align import Dataset
from undertow.data.types import CheckResult

from .checks import (
    CHECK_SEVERITIES,
    check_block_ordering,
    check_key_uniqueness,
    check_liquidity_conservation,
    check_monotonic_timestamps,
    check_no_block_gaps,
    check_no_future_data,
    check_reference_coverage,
    check_regime_labels_complete,
    check_swap_sign_convention,
    check_tick_price_consistency,
    check_ticks_on_spacing_grid,
)
from .crosscheck import (
    check_dune_counts,
    crosscheck_routes,
    reconcile_fees_against_collect,
)

__all__ = [
    "run_all_checks",
    "render_report",
    "CHECK_SEVERITIES",
    "check_block_ordering",
    "check_key_uniqueness",
    "check_monotonic_timestamps",
    "check_no_block_gaps",
    "check_no_future_data",
    "check_swap_sign_convention",
    "check_tick_price_consistency",
    "check_ticks_on_spacing_grid",
    "check_liquidity_conservation",
    "check_reference_coverage",
    "check_regime_labels_complete",
    "crosscheck_routes",
    "reconcile_fees_against_collect",
    "check_dune_counts",
]

# The order checks run in — deterministic, pinned so reports are comparable.
_CHECK_ORDER: tuple[str, ...] = (
    "check_block_ordering",
    "check_key_uniqueness",
    "check_monotonic_timestamps",
    "check_no_future_data",
    "check_no_block_gaps",
    "check_swap_sign_convention",
    "check_tick_price_consistency",
    "check_ticks_on_spacing_grid",
    "check_liquidity_conservation",
    "check_reference_coverage",
    "check_regime_labels_complete",
    "reconcile_fees_against_collect",
    "check_dune_counts",
)


def _skipped(name: str, reason: str) -> CheckResult:
    """An ``info``-severity result for a check whose inputs are absent."""
    return CheckResult(
        name=name,
        passed=True,
        severity="info",
        detail=f"{name}: skipped — {reason}",
        metrics={"skipped": reason},
    )


def _slice(tape: pa.Table, etype: str) -> pa.Table:
    """The tape rows for one log stream (the tape carries the union schema)."""
    mask = pa.compute.equal(tape.column("event_type"), pa.scalar(etype, type=pa.string()))
    return tape.filter(mask)


def _window_start_block(dataset: Dataset) -> int:
    """The block-number form of the window start (date-mode windows fall back to
    the tape's first block; the liquidity seed tolerates either)."""
    if dataset.manifest.window.start_block is not None:
        return int(dataset.manifest.window.start_block)
    tape = dataset.tape
    if tape is not None and tape.num_rows > 0:
        return min(int(v) for v in tape.column("block_number").to_pylist())
    return 0


def run_all_checks(dataset: Dataset) -> list[CheckResult]:
    """Run every check the dataset has inputs for, in a fixed order.

    Checks whose inputs are absent (no tape, no gas, no route samples, Dune
    unavailable, …) return an ``info``-severity result naming the reason — never
    a silent omission and never a spurious failure. ``run_all_checks`` itself
    never raises on a failed check; callers (T14) decide the exit code.
    """
    pool = dataset.manifest.pool
    window = dataset.manifest.window
    results: list[CheckResult] = []

    tape = dataset.tape
    if tape is None:
        for name in (
            "check_block_ordering",
            "check_key_uniqueness",
            "check_monotonic_timestamps",
            "check_no_future_data",
            "check_swap_sign_convention",
            "check_tick_price_consistency",
            "check_ticks_on_spacing_grid",
            "check_liquidity_conservation",
            "reconcile_fees_against_collect",
            "check_dune_counts",
        ):
            results.append(_skipped(name, "event tape absent from this dataset"))
    else:
        swaps = _slice(tape, "swap")
        mints = _slice(tape, "mint")
        burns = _slice(tape, "burn")
        results.append(check_block_ordering(tape))
        results.append(check_key_uniqueness(tape))
        results.append(check_monotonic_timestamps(tape))
        results.append(check_no_future_data(tape, window))
        results.append(check_swap_sign_convention(swaps))
        results.append(check_tick_price_consistency(swaps, pool))
        results.append(check_ticks_on_spacing_grid(mints, burns, pool))
        if dataset.fee_growth is not None:
            start = _window_start_block(dataset)
            results.append(
                check_liquidity_conservation(swaps, mints, burns, dataset.fee_growth, start)
            )
            results.append(reconcile_fees_against_collect(tape, dataset.fee_growth, pool))
        else:
            results.append(_skipped("check_liquidity_conservation", "fee_growth absent"))
            results.append(_skipped("reconcile_fees_against_collect", "fee_growth absent"))
        try:
            results.append(check_dune_counts(tape))
        except Exception as exc:  # noqa: BLE001 — a malformed Dune fixture must not
            # abort the other nine checks; report it as an info skip instead
            results.append(
                _skipped(
                    "check_dune_counts",
                    f"dune_expectations raised {type(exc).__name__}: {exc}",
                )
            )

    if dataset.gas is None:
        results.append(_skipped("check_no_block_gaps", "gas stream absent"))
    else:
        results.append(check_no_block_gaps(dataset.gas, window))

    if dataset.reference is None:
        results.append(_skipped("check_reference_coverage", "reference stream absent"))
    else:
        results.append(check_reference_coverage(dataset.reference, window))

    if dataset.regime is None:
        results.append(_skipped("check_regime_labels_complete", "regime stream absent"))
    else:
        results.append(check_regime_labels_complete(dataset.regime))

    # route agreement: both routes must be present for a stream to be comparable
    by_stream: dict[str, dict[str, pa.Table]] = {}
    for (route, stream), table in dataset.route_samples.items():
        by_stream.setdefault(stream, {})[route] = table
    if not by_stream:
        results.append(
            _skipped(
                "crosscheck_routes",
                "no route samples in the dataset (T14's route policy populates them)",
            )
        )
    else:
        for stream, routes in sorted(by_stream.items()):
            graph = routes.get("thegraph")
            rpc = routes.get("rpc")
            if graph is not None and rpc is not None:
                results.append(crosscheck_routes(graph, rpc, stream))
            else:
                results.append(
                    _skipped(
                        "crosscheck_routes",
                        f"{stream}: only {sorted(routes)} route(s) sampled — need both "
                        "'thegraph' and 'rpc' to compare",
                    )
                )

    # keep the pinned order (unseen names get appended in discovery order)
    order_idx = {name: i for i, name in enumerate(_CHECK_ORDER)}
    results.sort(key=lambda r: (order_idx.get(r.name, 99), r.name))
    return results


# ---------------------------------------------------------------------------
# render_report — the thesis appendix artifact
# ---------------------------------------------------------------------------


def _metric_str(metrics: Mapping[str, float | int | str]) -> str:
    uninteresting = {"rows"}
    parts = [f"{k}={v}" for k, v in metrics.items() if k not in uninteresting]
    if not parts:
        return " ".join(f"{k}={v}" for k, v in metrics.items())
    return ", ".join(parts[:8]) + ("…" if len(parts) > 8 else "")


def render_report(
    results: Sequence[CheckResult],
    path: Path,
    *,
    manifest: DatasetManifest | None = None,
) -> None:
    """Write a human-readable markdown report of the validation run.

    Structure: summary table (check, severity, pass/fail, key metric), then a
    detail section per failure (a reader who has not read the code must be able to
    act on it), then the skipped/info items. When ``manifest`` is given the report
    is tied to the dataset it describes (dataset id, pool, window, git commit,
    schema version) — the header is the point of tying a report to its data.

    The report never contains secrets: the manifest's endpoint provenance is
    host-only by construction (``storage.manifest.endpoint_hosts``), and no URL,
    key or header value is ever rendered here.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] = []
    if manifest is not None:
        w = manifest.window
        window_desc = (
            f"[{w.start_block}, {w.end_block}] (blocks)"
            if w.start_block is not None
            else f"{w.start_utc} → {w.end_utc} (UTC)"
        )
        header = [
            f"- **dataset id**: `{manifest.dataset_id}`",
            f"- **pool**: `{manifest.pool.address}` "
            f"({manifest.pool.token0_symbol}/{manifest.pool.token1_symbol}, "
            f"{manifest.pool.fee_tier.value} bps, spacing {manifest.pool.tick_spacing})",
            f"- **window**: {window_desc}",
            f"- **schema version**: {manifest.schema_version}",
            f"- **git commit**: `{manifest.git_commit}`",
        ]

    failures = [r for r in results if not r.passed and r.severity != "info"]
    info_rows = [r for r in results if r.severity == "info"]
    passes = [r for r in results if r.passed and r.severity != "info"]

    lines: list[str] = [
        "# Validation report — undertow.data",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        *(header or []),
        "",
        f"## Summary — {len(results)} check(s), {len(failures)} failed, "
        f"{len(passes)} passed, {len(info_rows)} skipped",
        "",
        "| Check | Severity | Result | Key metric |",
        "|---|---|---|---|",
    ]
    for r in results:
        result = "FAIL" if not r.passed else ("skip" if r.severity == "info" else "ok")
        lines.append(f"| {r.name} | {r.severity} | {result} | {_metric_str(r.metrics)} |")
    lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.name} — FAIL ({r.severity})")
            lines.append("")
            lines.append(r.detail)
            for k, v in r.metrics.items():
                lines.append(f"- `{k}` = `{v}`")
            lines.append("")

    if info_rows:
        lines.append("## Skipped / informational")
        lines.append("")
        for r in info_rows:
            lines.append(f"- **{r.name}** ({r.severity}): {r.detail}")
        lines.append("")

    lines.append("---")
    lines.append("_Rendered by `undertow.data.validation.render_report` (T13)._")
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
