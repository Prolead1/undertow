"""Route agreement and fee reconciliation — the two headline checks (T13 §2).

``crosscheck_routes`` is §10.1.5's point made executable: the same chain events
decoded two independent ways must agree field-for-field; any disagreement means
one of the routes is wrong and we do not yet know which.

``reconcile_fees_against_collect`` is the acceptance test for the thesis's gap
**G3**: *can you show that your reconstructed fees equal the fees the chain
actually paid out?* It finds real position lifecycles in the tape (mint →
burn/collect on the same ``(owner, tick_lower, tick_upper)`` key), replays
T10's ``FeeGrowthTracker`` from the nearest fee-growth snapshot, and compares
reconstructed ``uncollected_fees`` with the actual ``Collect`` amounts —
**principal separated out** by the paired-``Burn`` rule, which is the subtlety
that makes naive reconciliations fail (CONTRACTS.md §4.3).

Both checks report, never raise: a replay that trips a tracker invariant
(ValueError on an inconsistent stream) marks that lifecycle failed inside the
result instead of aborting the run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pyarrow as pa  # type: ignore[import-untyped]

from undertow.data.config import GLOBAL_TICK_SENTINEL, PoolConfig
from undertow.data.transforms.feegrowth import (
    FeeGrowthState,
    FeeGrowthTracker,
    PositionKey,
    TickState,
)
from undertow.data.types import CheckResult, Severity
from undertow.data.validation import dune_expectations
from undertow.data.validation.checks import _result


def check_result(
    name: str,
    passed: bool,
    detail: str,
    metrics: Mapping[str, float | int | str] | None = None,
    severity: Severity | None = None,
) -> CheckResult:
    """Build a result whose severity comes from THIS module's registry (the
    cross-checks own their severities; each call site stays registry-driven)."""
    return _result(name, passed, detail, metrics, severity, registry=CHECK_SEVERITIES)


# Severity registry for the cross-checks (same philosophy as ``checks.py``).
CHECK_SEVERITIES: Final[Mapping[str, Severity]] = {
    "crosscheck_routes": "critical",
    "reconcile_fees_against_collect": "warning",
    "check_dune_counts": "warning",
}
"""Any field disagreement between the two routes is **critical** — the two routes
decode the same chain events, so a disagreement means one of them is wrong and we
do not yet know which. The fee reconciliation fails as a **warning**: it is a
reconstruction over sampled snapshots and T10's documented approximations (an
unknown pre-swap price) — the honest signal is "needs human investigation", not
an automatic pipeline failure, and the achieved deltas are always reported so the
human sees exactly how far off it is."""

_LOG_STREAMS: Final[frozenset[str]] = frozenset({"swap", "mint", "burn", "collect"})
_REL_TOLERANCE: Final[float] = 1e-6
_RAW_TOLERANCE: Final[int] = 2


def _py(table: pa.Table, i: int, column: str) -> Any:  # noqa: ANN401 — Arrow scalars are dynamically typed; the whole point of this accessor
    """The Python scalar at row ``i`` of ``column`` (Arrow scalar -> py value)."""
    return table.column(column)[i].as_py()


def _as_int(value: object) -> int:
    return int(str(value))


# ---------------------------------------------------------------------------
# crosscheck_routes
# ---------------------------------------------------------------------------


def crosscheck_routes(graph: pa.Table, rpc: pa.Table, stream: str) -> CheckResult:
    """Field-level equality between The Graph's and the RPC's table for the same
    block range and stream.

    Compare on the merge key ``(block_number, log_index)``, then every column both
    routes carry. Any mismatch is **Critical** — both routes decode the same chain
    events, so a disagreement means one of them is wrong and we do not yet know
    which. Expected asymmetries are handled explicitly rather than by loosening
    the comparison:

    - rows present in one route only → reported as a mismatch (with the count);
    - a column one route does not carry (graph swap rows carry ``liquidity =
      NULL``; it is RPC-only) → excluded from the comparison *and* named in the
      note, so the exemption is visible, not silent;
    - an unsupported stream → ``info``-severity skip naming the reason.
    """
    name = "crosscheck_routes"
    if stream not in _LOG_STREAMS:
        return check_result(
            name,
            True,
            f"stream {stream!r} is not a log stream this check understands; skipped",
            {"stream": stream},
            severity="info",
        )

    def _by_key(
        table: pa.Table,
    ) -> tuple[dict[tuple[int, int], dict[str, Any]], list[str]]:
        cols = table.column_names
        out: dict[tuple[int, int], dict[str, Any]] = {}
        for i in range(table.num_rows):
            key = (int(_py(table, i, "block_number")), int(_py(table, i, "log_index")))
            out[key] = {c: _py(table, i, c) for c in cols}
        return out, cols

    rows_g, cols_g = _by_key(graph)
    rows_r, cols_r = _by_key(rpc)
    keys_g = set(rows_g)
    keys_r = set(rows_r)
    only_graph = sorted(keys_g - keys_r)
    only_rpc = sorted(keys_r - keys_g)
    shared = [k for k in sorted(keys_g & keys_r)]

    exclude = {"block_number", "log_index"}
    shared_cols = [c for c in cols_g if c in cols_r and c not in exclude]
    rpc_only_cols = sorted(c for c in cols_r if c not in cols_g)
    graph_only_cols = sorted(c for c in cols_g if c not in cols_r)

    # a column the graph route does not carry is all-null there — exempt it from
    # the equality and name the exemption in the note
    exempted: list[str] = []
    compared_cols: list[str] = []
    mismatches: dict[str, list[tuple[int, int]]] = {}
    for col in shared_cols:
        if all(rows_g[k][col] is None for k in shared):
            exempted.append(col)
            continue
        compared_cols.append(col)
        for k in shared:
            gv = rows_g[k][col]
            rv = rows_r[k][col]
            if gv != rv:
                mismatches.setdefault(col, []).append(k)

    problems: list[str] = []
    notes: list[str] = []
    if only_graph:
        problems.append(f"rows only in graph: {len(only_graph)} (first {only_graph[:3]})")
    if only_rpc:
        problems.append(f"rows only in rpc: {len(only_rpc)} (first {only_rpc[:3]})")
    for col, keys in mismatches.items():
        problems.append(f"column {col!r}: {len(keys)} mismatching key(s) (first {keys[:3]})")
    if rpc_only_cols:
        notes.append(f"columns rpc-only (excluded from comparison): {rpc_only_cols}")
    if graph_only_cols:
        notes.append(f"columns graph-only (excluded from comparison): {graph_only_cols}")
    if exempted:
        notes.append(f"columns graph does not carry (all-null, exempted): {sorted(exempted)}")

    metrics: dict[str, float | int | str] = {
        "shared_rows": len(shared),
        "graph_only_rows": len(only_graph),
        "rpc_only_rows": len(only_rpc),
        "compared_columns": len(compared_cols),
        "mismatching_columns": len(mismatches),
    }
    tail = ("; " + "; ".join(notes)) if notes else ""
    if not problems:
        return check_result(
            name,
            True,
            f"{stream}: {len(shared)} shared rows identical on "
            f"{len(compared_cols)} shared column(s){tail}",
            metrics,
        )
    return check_result(
        name,
        False,
        f"{stream}: " + "; ".join(problems) + tail,
        metrics,
    )


# ---------------------------------------------------------------------------
# reconcile_fees_against_collect — THE acceptance test for §10.2.4
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Lifecycle:
    """One completed position lifecycle discovered in the tape."""

    key: PositionKey
    mint_block: int
    liquidity: int  # minted L
    burn_block: int | None
    burn_liquidity: int | None
    burn0: int | None
    burn1: int | None
    burn_tx: str | None
    collect_block: int | None
    collect0: int | None
    collect1: int | None
    collect_tx: str | None
    exited_range: bool = False
    unsupported: str | None = None  # set when the lifecycle cannot be replayed safely


def _find_lifecycles(tape: pa.Table) -> tuple[list[_Lifecycle], int]:
    """Discover completed lifecycles (mint + collect on the same key).

    Returns ``(lifecycles, ignored)`` where ``ignored`` counts burn/collect rows
    whose position was minted before the window (no mint in the tape)."""
    mints: dict[PositionKey, dict[str, Any]] = {}
    burns: dict[PositionKey, list[dict[str, Any]]] = {}
    collects: dict[PositionKey, list[dict[str, Any]]] = {}
    ignored = 0
    for i in range(tape.num_rows):
        etype = _py(tape, i, "event_type")
        if etype not in ("mint", "burn", "collect"):
            continue
        owner = _py(tape, i, "owner")
        lower = int(_py(tape, i, "tick_lower"))
        upper = int(_py(tape, i, "tick_upper"))
        row = {
            "block": int(_py(tape, i, "block_number")),
            "log": int(_py(tape, i, "log_index")),
            "tx": _py(tape, i, "tx_hash"),
            "amount0": _as_int(_py(tape, i, "amount0"))
            if _py(tape, i, "amount0") is not None
            else 0,
            "amount1": _as_int(_py(tape, i, "amount1"))
            if _py(tape, i, "amount1") is not None
            else 0,
        }
        if etype == "mint":
            key = PositionKey(owner, lower, upper)
            row["liquidity"] = _as_int(_py(tape, i, "liquidity_amount"))
            mints[key] = row
        else:
            key = PositionKey(owner, lower, upper)
            if key not in mints:
                ignored += 1  # position minted before the window — invisible here
                continue
            if etype == "burn":
                row["liquidity"] = _as_int(_py(tape, i, "liquidity_amount"))
                burns.setdefault(key, []).append(row)
            else:
                collects.setdefault(key, []).append(row)

    lifecycles: list[_Lifecycle] = []
    for key, mint in mints.items():
        collect_rows = collects.get(key, [])
        if not collect_rows:
            continue  # position still open — not a completed lifecycle
        burn_rows = burns.get(key, [])
        if len(collect_rows) > 1 or len(burn_rows) > 1:
            # a partial burn or a second collect would replay against the wrong
            # principal and could false-EXACT — refuse to guess, mark unsupported
            lifecycles.append(
                _Lifecycle(
                    key=key,
                    mint_block=int(mint["block"]),
                    liquidity=int(mint["liquidity"]),
                    collect_block=int(collect_rows[0]["block"]),
                    collect0=int(collect_rows[0]["amount0"]),
                    collect1=int(collect_rows[0]["amount1"]),
                    collect_tx=str(collect_rows[0]["tx"]),
                    unsupported="multiple burn/collect events for one key",
                )
            )
            continue
        collect = collect_rows[0]
        burn = burn_rows[0] if burn_rows else None
        lifecycles.append(
            _Lifecycle(
                key=key,
                mint_block=int(mint["block"]),
                liquidity=int(mint["liquidity"]),
                burn_block=int(burn["block"]) if burn else None,
                burn_liquidity=int(burn["liquidity"]) if burn else None,
                burn0=int(burn["amount0"]) if burn else None,
                burn1=int(burn["amount1"]) if burn else None,
                burn_tx=str(burn["tx"]) if burn else None,
                collect_block=int(collect["block"]),
                collect0=int(collect["amount0"]),
                collect1=int(collect["amount1"]),
                collect_tx=str(collect["tx"]),
                exited_range=_did_exit_range(tape, key, int(collect["block"])),
            )
        )
    return lifecycles, ignored


def _did_exit_range(tape: pa.Table, key: PositionKey, collect_block: int) -> bool:
    """True if the pool tick just before/at the collect block was outside the
    position's range — the "price exited the range" property the brief wants
    represented among the replayed lifecycles."""
    best: int | None = None
    for i in range(tape.num_rows):
        if (
            _py(tape, i, "event_type") == "swap"
            and int(_py(tape, i, "block_number")) <= collect_block
        ):
            best = int(_py(tape, i, "tick"))
    if best is None:
        return False
    return best < key.tick_lower or best >= key.tick_upper


def _snapshot_state(fee_growth: pa.Table, block: int) -> FeeGrowthState | None:
    """The tracker seed at a snapshot block: the global row (accumulators, current
    tick, current liquidity, fee_protocol) plus the per-tick rows. ``None`` when the
    block has no global row (not a usable snapshot)."""
    global_g0: int | None = None
    global_g1: int | None = None
    current_tick: int | None = None
    current_liquidity: int | None = None
    fee_protocol: int = 0
    ticks: dict[int, TickState] = {}
    for i in range(fee_growth.num_rows):
        if int(_py(fee_growth, i, "block_number")) != block:
            continue
        tick = int(_py(fee_growth, i, "tick"))
        if tick == GLOBAL_TICK_SENTINEL:
            global_g0 = _as_int(_py(fee_growth, i, "fee_growth_global_0_x128"))
            global_g1 = _as_int(_py(fee_growth, i, "fee_growth_global_1_x128"))
            current_tick = int(_py(fee_growth, i, "current_tick"))
            current_liquidity = _as_int(_py(fee_growth, i, "current_liquidity"))
            fp_val = _py(fee_growth, i, "fee_protocol")
            fee_protocol = int(fp_val) if fp_val is not None else 0
        else:
            initialized = bool(_py(fee_growth, i, "initialized"))
            if not initialized:
                continue  # uninitialized ticks are NOT in the tracker's dict —
                # the engine only crosses initialized ticks (on-chain semantics)
            ticks[tick] = TickState(
                fee_growth_outside_0_x128=_as_int(_py(fee_growth, i, "fee_growth_outside_0_x128")),
                fee_growth_outside_1_x128=_as_int(_py(fee_growth, i, "fee_growth_outside_1_x128")),
                liquidity_gross=_as_int(_py(fee_growth, i, "liquidity_gross")),
                liquidity_net=_as_int(_py(fee_growth, i, "liquidity_net")),
                initialized=bool(_py(fee_growth, i, "initialized")),
            )
    if global_g0 is None or current_tick is None or current_liquidity is None:
        return None
    return FeeGrowthState(
        block_number=block,
        fee_growth_global_0_x128=global_g0,
        fee_growth_global_1_x128=global_g1,
        current_tick=current_tick,
        current_liquidity=current_liquidity,
        ticks=ticks,
        # The snapshot records the current TICK, not the pre-swap PRICE. Seeding
        # the boundary ratio here would hand the engine a "known" pre-swap price
        # that is really decision-4's approximation, so T10's own detection (fires
        # only when pre_price is None) would never trip and a first-crossing swap
        # would be apportioned from the boundary ratio yet still reported exact.
        # Seed None: the engine flags the first crossing swap exact=False itself.
        current_sqrt_price_x96=None,
        exact=True,
        fee_protocol=fee_protocol,
    )


def _row_mapping(tape: pa.Table, i: int) -> dict[str, Any]:
    """Project tape row ``i`` onto the mapping the tracker consumes. Null payload
    columns become ``"0"``/``0`` (a swap row's ``tick_lower`` is null and vice
    versa; the tracker never reads the wrong side)."""
    return {
        "event_type": _py(tape, i, "event_type"),
        "block_number": int(_py(tape, i, "block_number")),
        "amount0": _py(tape, i, "amount0") or "0",
        "amount1": _py(tape, i, "amount1") or "0",
        "sqrt_price_x96": _py(tape, i, "sqrt_price_x96"),
        "tick": _py(tape, i, "tick") or 0,
        "tick_lower": _py(tape, i, "tick_lower") or 0,
        "tick_upper": _py(tape, i, "tick_upper") or 0,
        "liquidity_amount": _py(tape, i, "liquidity_amount") or "0",
    }


def _replay_one(
    tape: pa.Table,
    tracker: FeeGrowthTracker,
    lifecycle: _Lifecycle,
) -> tuple[int, int, bool] | None:
    """Replay the tape between the tracker's seed block and the lifecycle's
    collect block, accruing the position's fees for both tokens.

    Accrual timing (the protocol's own): ``Position.update`` computes
    ``tokensOwed`` from the ``feeGrowthInside`` evaluated **before** the tick
    update, so the position accrues at its burn row using the pre-burn tracker
    state; the burn is then applied. A position with no burn accrues at the end
    (the fee-only collect's ``Position.update(0)``). Returns ``(fees0, fees1,
    exact)`` or ``None`` when a tracker invariant tripped (ValueError on an
    inconsistent event stream)."""
    seed_block = int(tracker.block_number)
    collect_block = int(lifecycle.collect_block or 0)
    # inside-last snapshots: initialised at the mint (state before the mint row —
    # the position's boundary ticks read as fresh, exactly like the protocol's
    # pre-tick-update computation).
    g0_last = g1_last = 0
    remaining = int(lifecycle.liquidity)
    total0 = total1 = 0
    exact = True
    try:
        for i in range(tape.num_rows):
            ev_block = int(_py(tape, i, "block_number"))
            if ev_block <= seed_block or ev_block > collect_block:
                continue
            etype = _py(tape, i, "event_type")
            if etype == "collect":
                continue
            is_this = False
            if etype in ("mint", "burn"):
                is_this = (
                    _py(tape, i, "owner"),
                    int(_py(tape, i, "tick_lower") or 0),
                    int(_py(tape, i, "tick_upper") or 0),
                ) == (lifecycle.key.owner, lifecycle.key.tick_lower, lifecycle.key.tick_upper)
            if etype == "mint":
                if is_this:
                    # capture inside-last BEFORE the mint row is applied (fresh ticks)
                    accrual = tracker.accrue(lifecycle.key, 0, 0, 0)
                    g0_last, g1_last = accrual.g_inside_0_last, accrual.g_inside_1_last
                tracker.apply_liquidity_event(_row_mapping(tape, i))
            elif etype == "burn":
                if is_this:
                    accrual = tracker.accrue(lifecycle.key, remaining, g0_last, g1_last)
                    total0 += accrual.fees0
                    total1 += accrual.fees1
                    g0_last, g1_last = accrual.g_inside_0_last, accrual.g_inside_1_last
                    exact = exact and accrual.exact
                    remaining -= int(lifecycle.burn_liquidity or 0)
                tracker.apply_liquidity_event(_row_mapping(tape, i))
            elif etype == "swap":
                tracker.apply_swap(_row_mapping(tape, i))
    except (ValueError, KeyError):
        return None

    if remaining > 0:
        accrual = tracker.accrue(lifecycle.key, remaining, g0_last, g1_last)
        total0 += accrual.fees0
        total1 += accrual.fees1
        exact = exact and accrual.exact
    return (total0, total1, exact)


def _key_str(key: PositionKey) -> str:
    return f"{key.owner[:14]}…[{key.tick_lower},{key.tick_upper}]"


def _nearest_snapshot_block(fee_growth: pa.Table, mint_block: int) -> int | None:
    """The latest snapshot block with a global row **strictly before** the mint."""
    best: int | None = None
    for i in range(fee_growth.num_rows):
        if int(_py(fee_growth, i, "tick")) != GLOBAL_TICK_SENTINEL:
            continue
        b = int(_py(fee_growth, i, "block_number"))
        if b < mint_block:
            best = b if best is None else max(best, b)
    return best


def _chain_fee_component(lc: _Lifecycle) -> tuple[int, int]:
    """The fee component of the chain's ``Collect``: ``Collect - Burn`` when the
    burn and the collect share a transaction (the collect disimpenses principal
    *plus* the accrued fees together), else ``Collect`` alone (a fee-only collect,
    e.g. after an earlier burn in a different transaction)."""
    if lc.collect_tx is not None and lc.collect_tx == lc.burn_tx:
        return (
            int(lc.collect0 or 0) - int(lc.burn0 or 0),
            int(lc.collect1 or 0) - int(lc.burn1 or 0),
        )
    return (int(lc.collect0 or 0), int(lc.collect1 or 0))


def reconcile_fees_against_collect(
    tape: pa.Table,
    fee_growth: pa.Table,
    pool: PoolConfig,
    *,
    rel_tolerance: float = _REL_TOLERANCE,
    raw_tolerance: int = _RAW_TOLERANCE,
    max_lifecycles: int = 5,
) -> CheckResult:
    """**THE acceptance test for §10.2.4** — the thesis's credibility on gap G3.

    Method (per the T13 brief):

    1. Find real position lifecycles in the tape: a ``Mint`` at
       ``(owner, tick_lower, tick_upper)`` and a later ``Burn``/``Collect`` at the
       same key. Pick completed lifecycles — short ones first, and **at least one
       where the price exited the range** when any exists.
    2. Replay T10's ``FeeGrowthTracker`` from the fee-growth snapshot at or before
       the mint block to the collect block, obtaining the position's
       ``uncollected_fees`` for both tokens.
    3. **Separate principal from fees**: a ``Collect`` paired with a ``Burn`` in
       the **same transaction** withdraws principal + accrued fees together, so
       the fee component is ``Collect.amountX - Burn.amountX``; a standalone
       ``Collect`` is the fee component itself (fee-only collect). Identified by
       ``tx_hash`` equality — the subtlety that makes naive reconciliations fail.
    4. Compare reconstructed vs chain fee component per token; report absolute and
       relative deltas. Pass when, per token, ``abs_delta <= raw_tolerance`` *or*
       ``rel_delta <= rel_tolerance`` (a raw-unit-level tolerance absorbs the
       per-swap Q128 division floors at small magnitudes, where a relative
       threshold is meaningless; ADR-002 was resolved by T10's exact
       re-simulation, so ``rel_tolerance=1e-6`` is the documented fallback for the
       bounded apportionment residual, never 0.0).
    5. The result **states the tolerance actually achieved per lifecycle** — if
       agreement is exact, the report says so in so many words.

    Never raises: a lifecycle whose replay trips a tracker invariant is reported
    as ``replay_failed`` inside the result; a lifecycle without a usable snapshot
    is ``skipped_no_snapshot`` with the reason named.
    """
    name = "reconcile_fees_against_collect"
    lifecycles, ignored = _find_lifecycles(tape)
    completed = [lc for lc in lifecycles if lc.collect_block is not None]
    if not completed:
        return check_result(
            name,
            True,
            f"no completed position lifecycles in the tape ({ignored} burn/collect "
            "row(s) refer to positions minted before the window); nothing to reconcile",
            {"lifecycles": 0, "ignored_external": ignored},
            severity="info",
        )
    # prefer exited-range lifecycles, then the cheapest to replay
    completed.sort(key=lambda lc: (not lc.exited_range, int(lc.collect_block or 0) - lc.mint_block))

    rows: list[dict[str, float | int | str]] = []
    n_replayed = 0
    n_failed = 0
    n_skipped = 0
    max_rel: float = 0.0
    all_exact = True

    for lc in completed[:max_lifecycles]:
        base: dict[str, float | int | str] = {
            "key": _key_str(lc.key),
            "mint_block": lc.mint_block,
            "collect_block": int(lc.collect_block or 0),
        }
        if lc.unsupported is not None:
            n_skipped += 1
            rows.append({**base, "status": "unsupported", "reason": lc.unsupported})
            continue
        seed = _nearest_snapshot_block(fee_growth, lc.mint_block)
        if seed is None:
            n_skipped += 1
            rows.append({**base, "status": "skipped_no_snapshot"})
            continue
        state = _snapshot_state(fee_growth, seed)
        if state is None:
            n_skipped += 1
            rows.append({**base, "status": "skipped_bad_snapshot"})
            continue
        replay = _replay_one(tape, FeeGrowthTracker(pool, state), lc)
        if replay is None:
            n_failed += 1
            rows.append({**base, "status": "replay_failed"})
            continue
        fees0, fees1, exact = replay
        all_exact = all_exact and exact
        chain0, chain1 = _chain_fee_component(lc)
        abs0 = fees0 - chain0
        abs1 = fees1 - chain1
        rel0 = abs(abs0) / max(abs(chain0), 1)
        rel1 = abs(abs1) / max(abs(chain1), 1)
        max_rel = max(max_rel, rel0, rel1)
        ok = (abs(abs0) <= raw_tolerance or rel0 <= rel_tolerance) and (
            abs(abs1) <= raw_tolerance or rel1 <= rel_tolerance
        )
        n_replayed += 1
        status = (
            "EXACT"
            if abs0 == 0 and abs1 == 0
            else "within_tolerance"
            if ok
            else "OUTSIDE_TOLERANCE"
        )
        rows.append(
            {
                **base,
                "paired_burn": "yes" if lc.collect_tx == lc.burn_tx else "no",
                "chain0": chain0,
                "chain1": chain1,
                "reconstructed0": fees0,
                "reconstructed1": fees1,
                "abs0": abs0,
                "abs1": abs1,
                "rel0": round(rel0, 9),
                "rel1": round(rel1, 9),
                "exact_replay": "yes" if exact else "no",
                "status": status,
            }
        )

    metrics: dict[str, float | int | str] = {
        "lifecycles_found": len(completed),
        "lifecycles_replayed": n_replayed,
        "lifecycles_failed": n_failed,
        "skipped": n_skipped,
        "ignored_external": ignored,
        "max_rel_delta": max_rel,
        "all_replays_exact": "yes" if all_exact else "no",
        "target_rel_tolerance": rel_tolerance,
        "target_raw_tolerance": raw_tolerance,
        "lifecycle_details": json.dumps(rows, sort_keys=True),
    }
    detail_rows = "; ".join(
        f"{r['key']} @{r.get('collect_block', '?')} {r['status']}" for r in rows
    )
    if n_replayed == 0:
        return check_result(
            name,
            True,
            f"{len(completed)} completed lifecycle(s) found, but none could be "
            f"replayed ({n_skipped} skipped, {n_failed} failed); nothing compared",
            metrics,
            severity="info",
        )
    outside = sum(1 for r in rows if r["status"] == "OUTSIDE_TOLERANCE")
    if outside == 0 and n_failed == 0:
        achieved = "EXACT" if all(r["status"] == "EXACT" for r in rows) else "within tolerance"
        return check_result(
            name,
            True,
            f"reconciliation {achieved} over {n_replayed} lifecycle(s); max rel "
            f"delta {max_rel:.3g} vs tolerance {rel_tolerance}; "
            f"lifecycles: {detail_rows}",
            metrics,
        )
    return check_result(
        name,
        False,
        f"{outside} lifecycle(s) OUTSIDE tolerance (max rel delta {max_rel:.3g} vs "
        f"{rel_tolerance}) and/or {n_failed} failed replay(s); "
        f"lifecycles: {detail_rows}",
        metrics,
    )


# ---------------------------------------------------------------------------
# Dune cross-check (T15's expected magnitudes, wired in as a warning)
# ---------------------------------------------------------------------------


def check_dune_counts(
    tape: pa.Table,
    *,
    month: str | None = None,
    fixture_path: Path | None = None,
) -> CheckResult:
    """Compare our monthly row counts against Dune's expected magnitudes.

    ``expected_counts`` is library code owned by T15
    (``undertow.data.validation.dune_expectations``): a missing expectation —
    month not recorded, stream not tracked, or a ``status: "unavailable"``
    fixture — yields an ``info``-severity skip naming the reason, never a pass or
    a fail. Any (month, stream) pair whose count is beyond Dune's
    ``tolerance_pct`` is a warning-level failure.
    """
    name = "check_dune_counts"
    if tape is None or tape.num_rows == 0:
        return check_result(name, True, "no event tape to count; skipped", {}, severity="info")
    if dune_expectations.status(fixture_path=fixture_path) == "unavailable":
        return check_result(
            name,
            True,
            "dune magnitudes fixture is status 'unavailable' (no Dune account at "
            "authoring time); comparison skipped",
            {"skipped_reason": "dune_unavailable"},
            severity="info",
        )
    counts: dict[tuple[str, str], int] = {}
    for i in range(tape.num_rows):
        etype = _py(tape, i, "event_type")
        ts = _py(tape, i, "block_timestamp")
        if etype not in ("swap", "mint", "burn", "collect") or ts is None:
            continue
        key = (ts.strftime("%Y-%m"), etype)
        counts[key] = counts.get(key, 0) + 1
    if month is not None:
        counts = {k: v for k, v in counts.items() if k[0] == month}

    mismatches: list[str] = []
    compared = 0
    for (m, stream), ours in sorted(counts.items()):
        expected = dune_expectations.expected_counts(m, stream, fixture_path=fixture_path)
        if expected is None:
            continue  # no expectation recorded — nothing to compare, not a failure
        exp_count, tol_pct = expected
        compared += 1
        tol = exp_count * tol_pct / 100.0
        dev = ours - exp_count
        if abs(dev) > tol:
            mismatches.append(
                f"{m}/{stream}: ours={ours} vs dune={exp_count} "
                f"(±{tol_pct}% → {tol:.0f}) dev {dev:+.0f}"
            )
    if not mismatches:
        return check_result(
            name,
            True,
            f"monthly counts agree with Dune on {compared} (month, stream) pair(s)",
            {"compared": compared},
        )
    return check_result(
        name,
        False,
        f"{len(mismatches)} (month, stream) pair(s) beyond Dune's tolerance; "
        "first: " + "; ".join(mismatches[:5]),
        {"compared": compared, "mismatches": len(mismatches)},
    )
