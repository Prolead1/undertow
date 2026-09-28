"""Generator for ``docs/data_dictionary.md`` (T16, ``T16_public_api.md`` §3).

The data dictionary is **generated from the canonical schemas**, never hand-written:
every column, its Arrow type, its nullability and its unit/convention string come
straight from ``SCHEMA_REGISTRY`` and the per-column ``unit:<name>`` schema metadata
(``CONTRACTS.md`` §4.9). This is what makes the document impossible to drift from the
code — a column added without a unit string fails the schema builder, and a document
that is not regenerated fails ``tests/data/test_data_dictionary.py``'s freshness test.

Regenerate with::

    uv run python -m undertow.data.dictionary

This module is an internal development tool; it is deliberately **not** part of the
``undertow.data`` public surface (``T16_public_api.md`` §2).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from undertow.data.schemas import POOL_SCOPED, SCHEMA_REGISTRY, SORT_KEYS

# ---------------------------------------------------------------------------
# Provenance: where each stream's bytes come from. Verified against the fetchers
# (``fetchers/thegraph.py``, ``fetchers/rpc.py``, ``fetchers/gas.py``,
# ``fetchers/reference.py``) and the pipeline routing in ``pipeline.py``.
# ---------------------------------------------------------------------------

STREAM_SOURCES: dict[str, str] = {
    "swap": (
        "The Graph subgraph (Route A, the pull-time default) or RPC "
        "``eth_getLogs`` (Route C); the two routes are cross-checked for exact "
        "agreement by T13"
    ),
    "mint": (
        "The Graph subgraph (Route A, the pull-time default) or RPC "
        "``eth_getLogs`` (Route C); cross-checked by T13"
    ),
    "burn": (
        "The Graph subgraph (Route A, the pull-time default) or RPC "
        "``eth_getLogs`` (Route C); cross-checked by T13"
    ),
    "collect": (
        "The Graph subgraph (Route A, the pull-time default) or RPC "
        "``eth_getLogs`` (Route C); cross-checked by T13. ``recipient`` is RPC-only"
    ),
    "flash": (
        "RPC ``eth_getLogs`` — the pinned subgraph does not index flash events, so "
        "Route C is the sole source"
    ),
    "fee_growth": (
        "RPC ``eth_call`` snapshots of ``ticks()`` / ``slot0()`` / ``liquidity()`` "
        "at sampled blocks (T06); no block-range fetch, and not in the default pull"
    ),
    "gas": "RPC ``eth_feeHistory``, base-fee only (ADR-006, ADR-005)",
    "reference": "Binance 1m klines — REST API and bulk monthly archive (T08)",
    "regime": "Derived in T12 from the cached ``reference`` feed (no network)",
    "event_tape": (
        "T11 aligned union of the log streams plus backward as-of joins of the "
        "reference, regime, gas and fee-growth side-streams"
    ),
}

_BANNER = (
    "<!-- GENERATED FILE — do not edit by hand. "
    "Regenerate with `uv run python -m undertow.data.dictionary`. -->"
)

_INTRO = (
    "This dictionary is generated from the canonical schemas in "
    "`src/undertow/data/schemas.py` (`SCHEMA_REGISTRY`) and their per-column "
    "unit/convention metadata. It is committed as generated output and tested for "
    "freshness; do not edit it by hand. Timestamps are `timestamp[us, tz=UTC]` "
    "values; big integers are decimal strings (see *Fixed-point and big integers* "
    "below)."
)

_TABLE_HEADER = "| column | type | unit / convention | nullable |"
_TABLE_SEP = "| --- | --- | --- | --- |"

# ---------------------------------------------------------------------------
# Prose: the non-obvious semantics from T16 §3 and docs/schema_notes.md.
# Kept as a list of already-wrapped lines so the rendered markdown stays inside
# the project's line-length budget.
# ---------------------------------------------------------------------------

_SEMANTICS: tuple[str, ...] = (
    "",
    "## Semantics and conventions",
    "",
    "### Swap signs and which side pays the fee",
    "",
    "`swap.amount0` / `swap.amount1` are **signed** `int256` values in raw token "
    "units.",
    "**Sign rule:** positive = flowed into the pool; negative = flowed out. A real "
    "swap has",
    "opposite signs and neither amount is zero (T13's `check_swap_sign_convention`). "
    "The fee is",
    "taken on the **input** (positive) side: `fee_paid = amount_in * fee_tier / "
    "1_000_000`, in the",
    "input token. Never assume which token is the input — read the sign. On the "
    "`event_tape`,",
    "`amount0`/`amount1` are signed on swap rows and non-negative on "
    "mint/burn/collect rows; the",
    "`event_type` discriminator tells a consumer which convention applies.",
    "",
    "### Fixed-point and big integers",
    "",
    "On-chain fixed-point values are integers, not decimals: `Q64.96` means scaled "
    "by `2**96`",
    "(`sqrt_price_x96 = sqrt(price_raw) * 2**96`) and `Q128` means scaled by `2**128` "
    "(fee growth",
    "per unit of liquidity). `tick` indexes the `1.0001**tick` grid. Values that can "
    "exceed 64 bits",
    "— `uint256`, `int256`, `uint128`, `uint160` — are stored in Parquet as "
    "`pa.string()` holding a",
    "plain **decimal string** and handled in Python as `int`. Parquet has no native "
    "256-bit integer",
    "type, and `float64` preserves only 53 bits of mantissa (`2**128` is ~70 bits "
    "wider), so a Q128",
    "accumulator round-tripped through a float would silently destroy exactly the low "
    "bits this",
    "pipeline exists to preserve. `schemas.encode_uint` / `schemas.decode_uint` are "
    "the only",
    "sanctioned conversions; signed values keep their leading `-`, and hex, exponent "
    "and `+`-signed",
    "forms are rejected.",
    "",
    "### Price orientation",
    "",
    "`price_pool` and `price_reference` are in the **same units**: the price of token1 "
    "denominated",
    "in token0, in human (decimals-adjusted) units — for the pinned USDC/WETH pools, "
    "**USDC per",
    "WETH**, i.e. ~2000–4000 over the study window. This is T02's "
    "`sqrt_price_x96_to_price`",
    "orientation, quoted verbatim: *“Human price of token1 denominated in token0 — "
    "USDC per WETH",
    "for the pinned pools (~3000).”* USDC is token0 (6 dp) and WETH is token1 "
    "(18 dp), and the raw",
    "tick is `1.0001**tick` in raw token1-per-token0 terms, so `price_pool` is "
    "**decreasing** in",
    "`tick` for the pinned pools (a higher tick means fewer USDC per WETH). The "
    "conversion is",
    "`10**(dec1 - dec0) * 2**192 / sqrt_price_x96**2`, evaluated exactly as a "
    "`Decimal`.",
    "",
    "### `Collect` is not accrual; paired `Burn` separates principal",
    "",
    "`collect.amount0` / `collect.amount1` are the amounts **withdrawn**, not the "
    "fees accrued in",
    "the current window: a `Collect` includes fees that accrued *before* the window "
    "began. A `Burn`",
    "immediately followed by a `Collect` in the same transaction withdraws **principal "
    "+ fees",
    "together**, so reconciliation must separate them using the paired "
    "`Burn(amount0, amount1)`",
    "principal: the fee component is `Collect.amountX - Burn.amountX` when the two are "
    "paired in one",
    "transaction, and `Collect.amountX` for a fee-only collect that stands alone. This "
    "is why T13's",
    "`reconcile_fees_against_collect` tolerance is defined on the **fee** component, "
    "not the total.",
    "`burn.liquidity_amount` is unsigned in the event; the sign of the liquidity "
    "*delta* is applied",
    "in T11, not stored in this stream.",
    "",
    "### `fee_growth_source` and why there is no forward-interpolated option",
    "",
    "`event_tape.fee_growth_source` is either `\"exact\"` (a fee-growth snapshot "
    "exists at this exact",
    "block) or `\"stale_prior\"` (the value was forward-filled from the nearest "
    "snapshot at a **prior**",
    "block). There is deliberately **no** `\"interpolated_future\"` value: the only "
    "values knowable at",
    "block *n* come from blocks ≤ *n*, and a value that needs a later snapshot is "
    "look-ahead bias.",
    "All as-of joins (`price_reference`, `regime`, the fee-growth globals) are "
    "backward-only. Inside",
    "the `fee_growth` stream itself, `source` is `\"rpc_call\"` (an exact `eth_call`) "
    "or",
    "`\"interpolated\"`, which is always flagged and never silent.",
    "",
    "### Regime labels and the ADR-001 `μ` definition",
    "",
    "`regime.regime` is one of `bull`, `bear`, `sideways`, `high_vol` or `unknown`, "
    "computed from a",
    "30-day **backward-looking** rolling window of reference minute closes. The "
    "thresholds are",
    "pre-committed in `RegimeConfig` and swept by `sensitivity_sweep`: "
    "`vol_threshold = 0.80`",
    "(annualized `sigma_rv`) and `drift_threshold = 0.05`. The decision order is fixed "
    "and the",
    "comparisons are strict: (1) `sigma_rv > 0.80` → `high_vol`; (2) else `mu > 0.05` "
    "→ `bull`;",
    "(3) else `mu < -0.05` → `bear`; (4) else `sideways`. The first `lookback_days` "
    "of output have",
    "`window_complete = false` and `regime = \"unknown\"`; they are never dropped. Per "
    "**ADR-001**,",
    "`mu` is the **total** window log return `μ₃₀ = Σ log(S_t/S_{t−1}) = "
    "log(S_end/S_start)`, not the",
    "per-step mean the roadmap's formula literally wrote — a per-minute mean "
    "thresholded at ±5% is",
    "dimensionally inconsistent and would label every real window `sideways`. "
    "`sigma_rv` is",
    "`std(log returns, ddof=1) * sqrt(365*1440)`.",
    "",
    "### Reference-feed gaps and the USDT proxy assumption",
    "",
    "`reference` is Binance 1-minute klines: `ETHUSDT` primary, `ETHUSDC` as the "
    "cross-check. Gaps",
    "(exchange outages, thin pairs, a kline that never printed) are **forward-filled "
    "and flagged**,",
    "never interpolated and never silently dropped: a missing bar after at least one "
    "observed bar",
    "copies the prior close into OHLC, zeroes volume and trades, and sets "
    "`is_gap_filled = true`.",
    "A missing **leading** edge is left absent — back-filling it would be look-ahead. "
    "`close_time` is",
    "`open_time + 59.999 s`, and every as-of join uses `close_time <= block_timestamp`. "
    "Prices are",
    "ETH in USD-stable terms, and this pipeline treats **USDT as a USD proxy "
    "(1 USDT ≙ 1 USD)** — an",
    "explicit assumption, not a fact: USDT can deviate from parity under stress (tens "
    "of bps within",
    "the study window), which is second-order for IL/LVR measurement and 30-day regime "
    "labels. The",
    "symbol set is a constructor argument, so the feed can be switched without code "
    "changes.",
    "",
    "### Known limitations",
    "",
    "- **Position attribution** uses `owner` as it appears on-chain. Retail Uniswap V3 "
    "positions are",
    "  owned by the NFT position manager (`0xC364…FE88`), so `mint.owner`, `burn.owner` "
    "and",
    "  `collect.owner` identify the manager, not the end user. Per-user attribution "
    "needs a separate",
    "  NFT-transfer join and is out of scope (`PLAN.md` §7).",
    "- **Fee growth is sampled, not per-block.** `fee_growth` rows are `eth_call` "
    "snapshots at",
    "  selected blocks; between samples the pipeline replays swaps and liquidity events "
    "with T10's",
    "  `FeeGrowthTracker` and verifies that replay against the observed snapshots (T13's",
    "  `reconcile_fees_against_collect`), rather than paying for an `eth_call` at every "
    "block.",
    "  The lifecycle-boundary-plus-stride sampling strategy is not wired into the "
    "default pull;",
    "  `fee_growth` has no block-range fetch and is not in the default pull set.",
    "- **Gas is base-fee only** (ADR-006): per-block timestamps and per-block "
    "priority-fee resolution",
    "  were removed. The tape carries `block_timestamp` from the event logs and "
    "converts gas to USD",
    "  via its `price_reference`; `gas_used`, `gas_limit`, `priority_fee_p50_wei` and "
    "`priority_fee_p90_wei`",
    "  are written as `0` (ADR-005).",
    "- **BigQuery is not implemented.** `Route.BIGQUERY` is an escape hatch reserved "
    "by the contract,",
    "  not a v1 code path.",
    "- **No `adr/002` exists** in this repository, so there is no documented "
    "swap-segment-apportionment",
    "  residual; T13's fee reconciliation targets exact integer agreement within a bounded",
    "  tolerance (raw 2 or relative 1e-6) as a documented fallback.",
)


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell (pipes are the cell separator)."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_sort_key(stream: str) -> str:
    keys = SORT_KEYS[stream]
    suffix = "," if len(keys) == 1 else ""
    return f"({', '.join(keys)}{suffix})"


def _stream_section(stream: str, schema: pa.Schema) -> list[str]:
    """Render one stream's heading, provenance line and column table."""
    try:
        source = STREAM_SOURCES[stream]
    except KeyError:  # pragma: no cover - guarded by a test, but fail loudly anyway
        raise ValueError(
            f"dictionary: no source route registered for stream {stream!r}; "
            "add it to STREAM_SOURCES"
        ) from None

    scoped = "yes" if stream in POOL_SCOPED else "no"
    lines = [
        f"### `{stream}`",
        "",
        f"**Source route:** {source}",
        "",
        f"**Sort key:** `{_format_sort_key(stream)}` · **Pool-scoped:** {scoped}",
        "",
        _TABLE_HEADER,
        _TABLE_SEP,
    ]

    for field in schema:
        unit = schema.metadata.get(f"unit:{field.name}".encode())
        if not unit:
            raise ValueError(
                f"dictionary: schema {stream!r} field {field.name!r} has no "
                "'unit:<name>' metadata; the data dictionary cannot be generated"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(f"`{field.name}`"),
                    _cell(f"`{field.type}`"),
                    _cell(unit.decode("utf-8")),
                    "yes" if field.nullable else "no",
                )
            )
            + " |"
        )

    lines.append("")
    return lines


def render_data_dictionary() -> str:
    """Render the entire ``docs/data_dictionary.md`` document deterministically.

    Column lists are never hand-maintained: they are read from
    ``SCHEMA_REGISTRY`` in insertion order, and every unit/convention cell is the
    schema's ``unit:<column>`` metadata value.
    """
    parts: list[str] = [
        _BANNER,
        "",
        "# Undertow data dictionary",
        "",
        _INTRO,
        "",
        "## Streams",
        "",
    ]
    for stream, schema in SCHEMA_REGISTRY.items():
        parts.extend(_stream_section(stream, schema))
    parts.extend(_SEMANTICS)
    return "\n".join(parts).rstrip("\n") + "\n"


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory containing ``pyproject.toml``."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(
        f"dictionary: could not locate the repository root (no pyproject.toml above {start})"
    )


def main() -> Path:
    """Regenerate ``docs/data_dictionary.md`` and return the written path."""
    root = _find_repo_root(Path(__file__).resolve())
    destination = root / "docs" / "data_dictionary.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_data_dictionary(), encoding="utf-8")
    print(f"wrote {destination}")
    return destination


if __name__ == "__main__":
    main()