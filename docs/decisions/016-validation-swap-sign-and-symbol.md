# ADR 016 — Validation: downgrade the swap-sign check, filter coverage checks by reference symbol

**Status:** accepted (implementation and export)
**Affects:** `src/undertow/data/validation/checks.py` · `src/undertow/data/validation/__init__.py` · `src/undertow/data/pipeline.py` · data `CONTRACTS.md` §8
**Raised by:** training-readiness follow-up (`bug-data-validation-fixes`)

## Context

Three validation checks were written before the pipeline saw the full real
dataset and before the reference symbol was pinned (ADR-014):

- `check_swap_sign_convention` requires every swap's `amount0`/`amount1` to be
  non-zero and oppositely signed, at **critical** severity. Its own T13 brief
  allows downgrading to a warning "if real data ever shows a degenerate
  zero-amount swap".
- `check_reference_coverage` and `check_regime_labels_complete` run over the raw
  `reference`/`regime` streams, which (since ADR-014) carry **both** the primary
  symbol and its cross-check.

## Problem

1. **Zero-amount swaps are real.** On the full `eth_usdc_3000` swap stream,
   `check_swap_sign_convention` reports **26 / 447 714** swaps with
   `amount0 = 0, amount1 > 0`. These are on-chain Uniswap V3 swaps whose
   one-sided amount rounds to zero at a tick boundary — not a decoding bug. At
   critical severity the check blocks `snapshot` entirely (no manifest is
   written), which is exactly the case the brief anticipated.

2. **Pooled symbol statistics are misleading.** `check_reference_coverage`
   pools `ETHUSDT` and `ETHUSDC`. The primary's 80-minute outage and the
   cross-check's 236 450-bar outage (cross-filled since the cross-fill fix) then
   average into a single share and a single longest-run, so the report cannot
   describe the series the simulator actually consumes. `check_regime_labels_complete`
   has the same problem.

## Decision

- **Downgrade `check_swap_sign_convention` to `warning`.** The check still flags
  every same-signed or zero-amount swap and reports the count and example
  `tx_hash`es; it no longer blocks the pipeline. Same-sign violations (a real
  decoding bug) therefore stay visible without a hard stop.
- **Filter the two reference/regime checks to `reference_symbol`.** A private
  `_select_symbol(table, symbol)` restricts the table before the check runs;
  `run_all_checks(dataset, *, reference_symbol=DEFAULT_REFERENCE_SYMBOL)` gains a
  keyword-only argument, and `pipeline.verify(config)` passes
  `config.reference_symbol`. A feed that does not contain the configured symbol
  produces an empty selection, so the check reports "no reference bars at all"
  with the symbol in its metrics rather than silently describing the wrong
  series. Hand-built tables without a `symbol` column are unchanged.

## Consequences

- `snapshot`/`verify` no longer fail critically on the 26 real zero-amount swaps;
  the severity registry (`CHECK_SEVERITIES`) and its pinning test are updated.
- `check_reference_coverage` now reports per-series gap statistics (e.g. the
  primary's 80-bar run, not a pooled 236 450), and includes `symbol` in its
  metrics. `check_regime_labels_complete` likewise.
- `verify(config)` and `run_all_checks` gain a defaulted, keyword-only argument;
  the `build_event_tape`-style additive extension is recorded in data
  `CONTRACTS.md` §8.
- Rejected alternatives:
  - *Drop the zero-amount check.* Loses the same-sign signal; the brief wants the
    count reported, only the severity relaxed.
  - *Filter to the primary by hard-coding `ETHUSDT` inside the checks.* Breaks the
    single source of truth (`DataConfig.reference_symbol`) that ADR-014 established.
