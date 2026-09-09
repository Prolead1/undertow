"""Single-stream invariants for the validation suite (T13, ``CONTRACTS.md`` §8).

Every check here is **pure and reporting-only**: no I/O, no shared state, never
raises on a failed check — the runner (``__init__.py``) decides what a failure
means and T14's CLI decides the exit code. A validator that crashes on the first
problem cannot report the other nine.

Severity is **per check, never per data** — registered in ``CHECK_SEVERITIES``,
the single source of truth, pinned by ``tests/data/test_checks.py`` so a later
edit that quietly downgrades a Critical to a warning breaks a test. Per-data
nuance lives in the *detail* string and the *metrics*, never in the severity.

Severity rationale (mirrors the T13 brief):

- **critical** — the row content contradicts the protocol or the contract; silent
  acceptance would feed a wrong number downstream. Ordering / unique keys /
  timestamps are tape-level guarantees (a fetcher pagination bug shows up here);
  block gaps is §10.2.3's gas-coverage requirement; swap sign, tick↔price
  consistency and the tick-spacing grid are protocol invariants.
- **warning** — windowed or sampled data with legitimate slack: liquidity
  conservation over a sampled fee-growth seed (delta-based **by design**, see its
  docstring), reference/regime coverage quality, and ``no_future_data`` (a
  windowing defect, not a chain-encoding contradiction).

T11's shared ``tiny_dataset`` discusses the off-grid position fix in its
``tests/data/conftest.py`` docstring: ``check_ticks_on_spacing_grid`` is the one
check the shared fixture originally tripped, because the fixture's out-of-range
position was pinned at non-grid ticks (196300/196360, neither a multiple of 60) —
an off-grid position is impossible on-chain, so the fixture was corrected to
grid-aligned bounds (T13's fix, documented in the PR body).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import pyarrow as pa  # type: ignore[import-untyped]

from undertow.data.config import GLOBAL_TICK_SENTINEL, PoolConfig, WindowConfig
from undertow.data.fixedpoint import sqrt_price_x96_to_tick
from undertow.data.types import CheckResult, Severity

# ---------------------------------------------------------------------------
# Severity registry — the single source of truth; pinned by a test.
# ---------------------------------------------------------------------------

CHECK_SEVERITIES: Final[Mapping[str, Severity]] = {
    "check_block_ordering": "critical",
    "check_key_uniqueness": "critical",
    "check_monotonic_timestamps": "critical",
    "check_no_block_gaps": "critical",
    "check_no_future_data": "warning",
    "check_swap_sign_convention": "critical",
    "check_tick_price_consistency": "critical",
    "check_ticks_on_spacing_grid": "critical",
    "check_liquidity_conservation": "warning",
    "check_reference_coverage": "warning",
    "check_regime_labels_complete": "warning",
}
"""Check name -> severity. Overridden only for ``skip``-flavoured results (``info``)."""

# Reference-coverage thresholds (config-separation rule: named, not inlined).
# A gap-fill share above 2% or a single outage run above 12 bars means the
# exchange feed was down enough that the as-of reference joins degrade.
GAP_FILL_WARNING_SHARE: Final[float] = 0.02
GAP_FILL_WARNING_RUN: Final[int] = 12


def _result(
    name: str,
    passed: bool,
    detail: str,
    metrics: Mapping[str, float | int | str] | None = None,
    severity: Severity | None = None,
    registry: Mapping[str, Severity] | None = None,
) -> CheckResult:
    """Build a :class:`CheckResult` with the registered severity for ``name``
    (override allowed for ``skip``-style results, which are ``info``). Cross-check
    modules pass their own severity registry (each module owns its checks)."""
    sev = severity if severity is not None else (registry or CHECK_SEVERITIES)[name]
    return CheckResult(
        name=name,
        passed=passed,
        severity=sev,
        detail=detail,
        metrics=dict(metrics or {}),
    )


# ---------------------------------------------------------------------------
# Column accessors
# ---------------------------------------------------------------------------


def _col(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _as_int(value: object) -> int:
    """Schema big-integers are decimal strings; coerce int/str/None safely."""
    if value is None:
        raise ValueError("expected a non-null integer value")
    return int(str(value))


def _as_int_col(table: pa.Table, name: str) -> list[int]:
    return [_as_int(v) for v in _col(table, name)]


def _as_bool_col(table: pa.Table, name: str) -> list[bool]:
    return [bool(v) for v in _col(table, name)]


def _epoch_us(table: pa.Table, column: str) -> list[int]:
    """tz-aware Arrow timestamp column as epoch-microseconds ints."""
    out: list[int] = []
    for v in _col(table, column):
        if v is None:
            out.append(0)
        else:
            out.append(int(v.timestamp() * 1_000_000))
    return out


# ---------------------------------------------------------------------------
# Tape-level invariants
# ---------------------------------------------------------------------------


def check_block_ordering(table: pa.Table) -> CheckResult:
    """``(block_number, log_index)`` must be strictly increasing row by row.

    A violation means rows were concatenated out of order or a fetcher's
    pagination interleaved pages. Critical. Reports the first offenders.
    """
    name = "check_block_ordering"
    blocks = _as_int_col(table, "block_number")
    logs = _as_int_col(table, "log_index")
    offending: list[tuple[int, int]] = []
    for i in range(1, len(blocks)):
        if (blocks[i], logs[i]) <= (blocks[i - 1], logs[i - 1]):
            key = (blocks[i], logs[i])
            if key not in offending:
                offending.append(key)
    if not offending:
        return _result(
            name,
            True,
            f"{len(blocks)} rows strictly increasing in (block, log_index)",
            {"rows": len(blocks)},
        )
    return _result(
        name,
        False,
        f"{len(offending)} row(s) not strictly increasing; first: {offending[:5]}",
        {"violations": len(offending)},
    )


def check_key_uniqueness(table: pa.Table) -> CheckResult:
    """No duplicate ``(block_number, log_index)`` merge keys.

    A duplicate means a fetcher paginated wrong and must never be silently
    de-duplicated (``CONTRACTS.md``: "Duplicates ⇒ ``ValidationError``"). Critical.
    """
    name = "check_key_uniqueness"
    blocks = _as_int_col(table, "block_number")
    logs = _as_int_col(table, "log_index")
    seen: set[tuple[int, int]] = set()
    dup: list[tuple[int, int]] = []
    for key in zip(blocks, logs, strict=True):
        key_t = (int(key[0]), int(key[1]))
        if key_t in seen and key_t not in dup:
            dup.append(key_t)
        seen.add(key_t)
    if not dup:
        return _result(name, True, f"{len(seen)} unique (block, log) keys", {"rows": len(seen)})
    return _result(
        name,
        False,
        f"{len(dup)} duplicated (block, log) key(s); first: {dup[:5]} — a duplicate "
        "means a fetcher paginated wrong and must not be silently dropped",
        {"duplicates": len(dup)},
    )


def check_monotonic_timestamps(table: pa.Table) -> CheckResult:
    """``block_timestamp`` must be non-decreasing as ``block_number`` increases.

    A block has a single timestamp, so rows within a block share it and later
    blocks must not be *older*. A timestamp going backwards while the block number
    rises is a decode bug. Critical.
    """
    name = "check_monotonic_timestamps"
    blocks = _as_int_col(table, "block_number")
    stamps = _epoch_us(table, "block_timestamp")
    violations: list[tuple[int, int, int]] = []  # (block, prev_us, this_us)
    for i in range(1, len(blocks)):
        if blocks[i] == blocks[i - 1]:
            if stamps[i] != stamps[i - 1]:
                violations.append((blocks[i], stamps[i - 1], stamps[i]))
        elif stamps[i] < stamps[i - 1]:
            violations.append((blocks[i], stamps[i - 1], stamps[i]))
    if not violations:
        return _result(name, True, "timestamps non-decreasing with block", {"rows": len(blocks)})
    return _result(
        name,
        False,
        f"{len(violations)} timestamp violation(s); first: {violations[:5]}",
        {"violations": len(violations)},
    )


def check_no_future_data(table: pa.Table, window: WindowConfig) -> CheckResult:
    """No row beyond the configured window end (``end_block`` or ``end_utc``).

    ``end_block`` is inclusive. Rows beyond mean the pull overran the window and
    would silently extend the backtest horizon. Warning: a windowing defect, not
    a protocol violation.
    """
    name = "check_no_future_data"
    offender_blocks = 0
    offender_stamps = 0
    if window.end_block is not None:
        end = int(window.end_block)
        offender_blocks = sum(1 for b in _as_int_col(table, "block_number") if b > end)
    if window.end_utc is not None:
        end_us = int(window.end_utc.timestamp() * 1_000_000)
        offender_stamps = sum(1 for t in _epoch_us(table, "block_timestamp") if t > end_us)
    total = offender_blocks + offender_stamps
    if total == 0:
        return _result(name, True, "no rows beyond the configured window end", {"rows_beyond": 0})
    return _result(
        name,
        False,
        f"{offender_blocks} row(s) past end_block and/or {offender_stamps} past "
        "end_utc; the dataset extends beyond the configured window",
        {"rows_beyond": total},
    )


# ---------------------------------------------------------------------------
# Stream-content invariants
# ---------------------------------------------------------------------------


def check_no_block_gaps(gas: pa.Table, window: WindowConfig) -> CheckResult:
    """The gas stream must cover every block in the window.

    Critical: the tape equi-joins gas on ``block_number`` — a missing block
    silently drops the tape row at that block (T11 test pins this). In block mode
    the whole ``[start, end]`` interval must be present; in date mode (blocks
    unknown) every implied block between the first and last gas rows must exist.
    """
    name = "check_no_block_gaps"
    blocks = _as_int_col(gas, "block_number")
    if not blocks:
        return _result(name, False, "gas stream is empty — no block is covered", {"rows": 0})
    expected_start = int(window.start_block) if window.start_block is not None else blocks[0]
    expected_end = int(window.end_block) if window.end_block is not None else blocks[-1]
    present = set(blocks)
    missing = [b for b in range(expected_start, expected_end + 1) if b not in present]
    if not missing:
        return _result(
            name,
            True,
            f"gas covers {len(present)} block(s) from {expected_start} to {expected_end}, no gaps",
            {"rows": len(present), "first": blocks[0], "last": blocks[-1]},
        )
    return _result(
        name,
        False,
        f"{len(missing)} block(s) missing from gas in [{expected_start}, "
        f"{expected_end}]; first: {missing[:5]} — tape rows at these blocks are "
        "silently dropped by the gas equi-join",
        {"gaps": len(missing)},
    )


def check_swap_sign_convention(swaps: pa.Table) -> CheckResult:
    """For every swap, ``amount0`` and ``amount1`` must have opposite signs and
    neither may be zero; report the offending ``tx_hash`` (the positive amount is
    the input, CONTRACTS.md §4.1).

    Critical. The T13 brief allows downgrading to a warning *if real data ever
    shows a degenerate zero-amount swap* — that would be a documented decision
    (the count is reported either way), not a silent tolerance.
    """
    name = "check_swap_sign_convention"
    a0 = _as_int_col(swaps, "amount0")
    a1 = _as_int_col(swaps, "amount1")
    txs = [str(v) for v in _col(swaps, "tx_hash")]
    bad: list[str] = []
    for x, y, tx in zip(a0, a1, txs, strict=True):
        if x == 0 or y == 0 or (x > 0) == (y > 0):
            bad.append(tx)
    if not a0:
        return _result(name, True, "no swaps to check", {"rows": 0})
    if not bad:
        return _result(
            name,
            True,
            f"all {len(a0)} swaps have opposite-signed, non-zero amounts",
            {"rows": len(a0)},
        )
    return _result(
        name,
        False,
        f"{len(bad)} swap(s) where amount0/amount1 are same-signed or zero; "
        f"example tx_hash: {sorted(bad)[:4]}",
        {"violations": len(bad)},
    )


def check_tick_price_consistency(swaps: pa.Table, pool: PoolConfig) -> CheckResult:
    """``sqrt_price_x96_to_tick(row.sqrt_price_x96)`` vs ``row.tick`` per swap —
    a very strong end-to-end check (T02's TickMath port + fetcher decoding +
    schema fidelity at once).

    Allow a deviation of **±1**, documented: after a *downward* tick crossing the
    pool writes ``slot0.tick = tickNext - 1`` which is not always equal to
    ``getTickAtSqrtRatio(sqrtPriceX96)`` at the boundary. |deviation| > 1 is
    Critical. The deviation distribution is always reported so a systematic drift
    toward the tolerance cannot hide a real bug. ``pool`` is accepted for the
    contract signature; the sqrt→tick map is pure TickMath (decimals do not
    enter).
    """
    name = "check_tick_price_consistency"
    del pool  # pure TickMath: decimals don't enter
    ticks = _as_int_col(swaps, "tick")
    raws = _col(swaps, "sqrt_price_x96")
    if not ticks:
        return _result(name, True, "no swaps to check", {"rows": 0})
    distribution: dict[int, int] = {}
    bad: list[tuple[int, int, int]] = []  # (row_tick, computed, deviation)
    for row_tick, raw in zip(ticks, raws, strict=True):
        try:
            computed = sqrt_price_x96_to_tick(int(str(raw)))
        except ValueError:  # outside the TickMath domain — itself a bug
            computed = 0  # guarantees a large deviation, flagged below
        dev = row_tick - computed
        distribution[dev] = distribution.get(dev, 0) + 1
        if abs(dev) > 1 and len(bad) < 5:
            bad.append((row_tick, computed, dev))
    dist_metrics = {f"dev{k}": v for k, v in sorted(distribution.items())}
    dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(distribution.items()))
    if not bad:
        return _result(
            name,
            True,
            f"all {len(ticks)} swaps within ±1 tick of sqrt_price_x96_to_tick "
            f"(distribution: {dist_str})",
            {"rows": len(ticks), **dist_metrics},
        )
    return _result(
        name,
        False,
        f"{len(bad)} swap(s) with |deviation| > 1 (row.tick - computed): {bad} — "
        f"decode/port disagreement; distribution: {dist_str}",
        {"violating_rows": len(bad), "rows": len(ticks), **dist_metrics},
    )


def check_ticks_on_spacing_grid(
    mints: pa.Table | None, burns: pa.Table | None, pool: PoolConfig
) -> CheckResult:
    """Every mint/burn ``tick_lower``/``tick_upper`` must be a multiple of the
    pool's ``tick_spacing``.

    Critical: an off-grid position is impossible on-chain (``Pool.mint`` reverts
    on a non-aligned tick), so a violation means corrupted decoding or the wrong
    fee-tier→spacing mapping for the pool.
    """
    name = "check_ticks_on_spacing_grid"
    spacing = pool.tick_spacing
    offenders: list[str] = []
    for label, table in (("mint", mints), ("burn", burns)):
        if table is None:
            continue
        lowers = _as_int_col(table, "tick_lower")
        uppers = _as_int_col(table, "tick_upper")
        for value in lowers:
            if value % spacing != 0 and len(offenders) < 2 * 8:
                offenders.append(f"{label}.tick_lower={value}")
        for value in uppers:
            if value % spacing != 0 and len(offenders) < 2 * 8:
                offenders.append(f"{label}.tick_upper={value}")
    if not offenders:
        n_m = 0 if mints is None else mints.num_rows
        n_b = 0 if burns is None else burns.num_rows
        return _result(
            name,
            True,
            f"{n_m} mint(s) and {n_b} burn(s), all bounds on the {spacing}-tick grid",
            {"mints": n_m, "burns": n_b, "tick_spacing": spacing},
        )
    return _result(
        name,
        False,
        f"{len(offenders)} off-grid bound(s) for tick_spacing={spacing}: "
        f"{offenders[:5]} — impossible on-chain",
        {"off_grid": len(offenders), "tick_spacing": spacing},
    )


def check_liquidity_conservation(
    swaps: pa.Table,
    mints: pa.Table | None,
    burns: pa.Table | None,
    fee_growth: pa.Table,
    start_block: int,
) -> CheckResult:
    """**Delta-based** active-liquidity conservation, seeded from the snapshot.

    Positions minted before ``start_block`` are invisible in a windowed dataset,
    so absolute liquidity can never be reproduced from in-window events alone —
    only **changes** can. Procedure (T13 brief §3):

    1. Seed absolute active liquidity from the ``fee_growth`` global snapshot's
       ``current_liquidity`` (and ``current_tick``) at or immediately before
       ``start_block``.
    2. Apply in-range mints/burns (+L / -L when the position spans the current
       tick) and, on each swap, the ``liquidity_net`` of every tick the price
       crossed; the swap's reported ``liquidity`` must equal the seeded value
       evolved through those deltas.
    3. Report the max absolute deviation and **the block where drift first
       appears** — a drift starting at a specific block is a decoding bug with a
       findable cause, which is the point of this check.

    Severity: warning by design — tick-crossing bookkeeping over a sampled
    fee-growth seed has legitimate slack. If it comes out exact, the report says
    so: that is a strong signal T06's decoding and T10's replay are both right.
    """
    name = "check_liquidity_conservation"

    # --- 1. seed from the nearest global snapshot at or before start_block ---
    fg_blocks = _as_int_col(fee_growth, "block_number")
    fg_ticks = [int(v) for v in _col(fee_growth, "tick")]
    fg_liq = _as_int_col(fee_growth, "current_liquidity")
    fg_cur_tick = [int(v) for v in _col(fee_growth, "current_tick")]
    seeds: list[tuple[int, int, int]] = []  # (block, current_liquidity, current_tick)
    for b, t, liq, cur in zip(fg_blocks, fg_ticks, fg_liq, fg_cur_tick, strict=True):
        if t == GLOBAL_TICK_SENTINEL:
            seeds.append((b, liq, cur))
    eligible = [(b, liq, cur) for (b, liq, cur) in seeds if b <= start_block]
    if not eligible:
        return _result(
            name,
            True,
            f"no fee-growth snapshot at or before start_block={start_block} — cannot "
            "seed the liquidity trace; skipped",
            {"seed_block": "none"},
            severity="info",
        )
    seed_block, active_liquidity, current_tick = max(eligible)

    # --- 2. walk events; track per-tick net and the in-range deltas ---
    tick_net: dict[int, int] = {}
    events: list[tuple[int, int, str, dict[str, Any]]] = []  # (block, log, kind, row)

    def _feed(kind: str, table: pa.Table | None) -> None:
        if table is None:
            return
        for i in range(table.num_rows):
            events.append(
                (
                    int(table.column("block_number")[i]),
                    int(table.column("log_index")[i]),
                    kind,
                    {
                        "tick_lower": (
                            int(table.column("tick_lower")[i]) if kind != "swap" else None
                        ),
                        "tick_upper": (
                            int(table.column("tick_upper")[i]) if kind != "swap" else None
                        ),
                        "liquidity_amount": (
                            _as_int(table.column("liquidity_amount")[i]) if kind != "swap" else None
                        ),
                        "liquidity": (
                            _as_int(table.column("liquidity")[i]) if kind == "swap" else None
                        ),
                        "tick": int(table.column("tick")[i]) if kind == "swap" else None,
                    },
                )
            )

    _feed("mint", mints)
    _feed("burn", burns)
    _feed("swap", swaps)
    events.sort(key=lambda e: (e[0], e[1]))

    max_abs_dev: float = 0.0
    first_drift_block: int | None = None
    n_swaps_compared = 0
    for _b, _l, kind, row in events:
        if kind in ("mint", "burn"):
            lower = int(row["tick_lower"])
            upper = int(row["tick_upper"])
            delta = (
                int(row["liquidity_amount"]) if kind == "mint" else -int(row["liquidity_amount"])
            )
            if lower <= current_tick < upper:
                active_liquidity += delta
            tick_net[lower] = tick_net.get(lower, 0) + delta
            tick_net[upper] = tick_net.get(upper, 0) - delta
            continue
        # swap
        reported = int(row["liquidity"])
        post_tick = int(row["tick"])
        if current_tick is not None and post_tick != current_tick:
            if post_tick > current_tick:
                crossed = sorted(t for t in tick_net if current_tick < t <= post_tick)
                for t in crossed:
                    active_liquidity += tick_net[t]
            else:
                crossed = sorted(
                    (t for t in tick_net if post_tick <= t < current_tick), reverse=True
                )
                for t in crossed:
                    active_liquidity -= tick_net[t]
        dev = reported - active_liquidity
        n_swaps_compared += 1
        if dev != 0:
            if first_drift_block is None:
                first_drift_block = int(_b)
            max_abs_dev = max(max_abs_dev, abs(dev))
        # subsequent events measure from the reported basis (deltas-only check)
        active_liquidity = reported
        current_tick = post_tick

    if n_swaps_compared == 0:
        return _result(
            name,
            True,
            f"no swap rows to compare (seeded from snapshot block {seed_block}); "
            "liquidity conservation not exercised — skipped",
            {"seed_block": seed_block, "swaps_compared": 0},
            severity="info",
        )
    if first_drift_block is None:
        return _result(
            name,
            True,
            f"liquidity conservation exact (0 deviation over {n_swaps_compared} "
            f"swap(s), seeded from snapshot block {seed_block}) — strong evidence "
            "T06 decoding and T10 replay agree",
            {"max_abs_deviation": 0, "seed_block": seed_block, "swaps_compared": n_swaps_compared},
        )
    return _result(
        name,
        False,
        f"liquidity deviation reaches {max_abs_dev} at block {first_drift_block} "
        f"(seeded from snapshot block {seed_block}); drift starting at a single "
        "block is usually a decoding bug in that block's mints/burns",
        {
            "max_abs_deviation": max_abs_dev,
            "first_drift_block": first_drift_block,
            "seed_block": seed_block,
        },
    )


def check_reference_coverage(reference: pa.Table, window: WindowConfig) -> CheckResult:
    """Reference bars span the window; report the gap-filled count and longest run.

    Gap-filled bars are the forward-filled outage rows the as-of join must land
    on — a few are fine; a large share means the exchange feed was down and the
    reference joins degrade. Date-mode windows span-check explicitly; in block
    mode (no window times) only the structural gap statistics run.
    """
    name = "check_reference_coverage"
    if reference.num_rows == 0:
        return _result(name, False, "no reference bars at all", {"rows": 0})
    closes = [v for v in _col(reference, "close_time")]
    filled = _as_bool_col(reference, "is_gap_filled")
    n_filled = sum(filled)
    longest_run = 0
    run = 0
    for f in filled:
        run = run + 1 if f else 0
        longest_run = max(longest_run, run)
    share = n_filled / len(filled)

    span_detail = ""
    if window.start_utc is not None and window.end_utc is not None:
        first_close = float(min(closes).timestamp())
        last_close = float(max(closes).timestamp())
        lo = window.start_utc.timestamp()
        hi = window.end_utc.timestamp()
        if first_close > lo or last_close < hi:
            return _result(
                name,
                False,
                f"reference bars span [{first_close}, {last_close}] but the window is "
                f"[{lo}, {hi}] — the as-of join starves at the edges",
                {"first_close": first_close, "last_close": last_close},
            )
        span_detail = "spans the window"

    if share <= GAP_FILL_WARNING_SHARE and longest_run <= GAP_FILL_WARNING_RUN:
        return _result(
            name,
            True,
            f"{len(filled)} bars, {n_filled} gap-filled ({share:.1%}), longest run "
            f"{longest_run}; {span_detail}".rstrip(),
            {
                "rows": len(filled),
                "gap_filled": n_filled,
                "longest_gap_run": longest_run,
                "gap_share": share,
            },
        )
    return _result(
        name,
        False,
        f"gap-fill share {share:.1%} ({n_filled}/{len(filled)}) with longest run "
        f"{longest_run}; {span_detail}".rstrip(),
        {
            "rows": len(filled),
            "gap_filled": n_filled,
            "longest_gap_run": longest_run,
            "gap_share": share,
        },
    )


def check_regime_labels_complete(regime: pa.Table) -> CheckResult:
    """Regime labels: none null, ``unknown`` only in the initial lookback
    (non-complete-window) period, and all four real bins non-empty over the full
    window. The empty-bin case is a warning — T12's handoff note says that is a
    finding the human needs to see.
    """
    name = "check_regime_labels_complete"
    if regime.num_rows == 0:
        return _result(name, False, "no regime rows", {"rows": 0})
    labels = [str(v) if v is not None else None for v in _col(regime, "regime")]
    windows = _as_bool_col(regime, "window_complete")

    n_null = sum(1 for x in labels if x is None)
    bad_unknown = sum(1 for x, w in zip(labels, windows, strict=True) if x == "unknown" and w)
    non_warmup = {str(x) for x, w in zip(labels, windows, strict=True) if w and x is not None}
    missing_bins = [b for b in ("bull", "bear", "sideways", "high_vol") if b not in non_warmup]

    problems: list[str] = []
    if n_null:
        problems.append(f"{n_null} null label(s)")
    if bad_unknown:
        problems.append(f"{bad_unknown} 'unknown' label(s) outside the warmup window")
    if missing_bins:
        problems.append(f"never observed over the window: {', '.join(missing_bins)}")

    if not problems:
        return _result(
            name,
            True,
            f"{len(labels)} labels, unknown confined to the warmup, all four bins present",
            {"rows": len(labels)},
        )
    return _result(
        name,
        False,
        "handoff finding: " + "; ".join(problems),
        {"rows": len(labels)},
    )
