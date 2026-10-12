# ADR 014 — One reference series: explicit primary-symbol selection for the tape and the sim

**Status:** accepted (implementation and export)
**Affects:** `src/undertow/data/config.py` · `src/undertow/data/transforms/align.py` · `src/undertow/data/pipeline.py` · `src/undertow/sim/marketview.py` · `configs/eth_usdc_3000.toml` · `configs/eth_usdc_500.toml` · data `CONTRACTS.md` §2/§6.2 · sim `CONTRACTS.md` §4.1
**Raised by:** bug fix on `bug-sim-reference-symbol` (training-readiness review)

## Context

The data plan pins **two** Binance symbols for the reference feed (data `PLAN.md` §7: "Binance
`ETHUSDT` 1m klines primary, `ETHUSDC` 1m as cross-check"). `ReferenceFetcher` therefore fetches
both by default (`fetchers/reference.py`, `symbols=("ETHUSDT", "ETHUSDC")`), and the stored
`reference` and `regime` streams each carry one row per symbol per minute.

The frozen contracts, however, describe a *single* series:

- `REFERENCE_SCHEMA` has a `symbol` column, but data `CONTRACTS.md` §6.2's `build_event_tape` joins
  `reference` as-is, and sim `CONTRACTS.md` §4.1 invariant 4 calls `reference` "the *entire* feed"
  and gives `reference_close` a singular return.
- The sim's `MarketView.reference_bars` / `reference_close` and `regime_at` were written for that
  single-series reading.

## Problem

With two symbols present, every "single series" consumer received **two rows per minute**:

1. `build_event_tape`'s backward `join_asof` on `close_time` had duplicate right keys for every bar,
   so `join_asof` silently kept whichever row sorted last — the tape's `price_reference` (and,
   identically, `regime`) was chosen by **symbol sort order**, not by intent. For the pinned pair
   (`ETHUSDC` < `ETHUSDT`) that happened to be `ETHUSDT`; any other symbol set flips it with no
   error.
2. `MarketView.reference_bars` returned both symbols interleaved, so the replay price process
   alternated series every step and reported duplicate `close_time`s. In the 176-day ETHUSDC
   gap-fill (flat `1338.12`) versus live ETHUSDT (~`1297`) this produced an artificial ~3%
   oscillation every minute, which also poisoned the MRSJD calibration fit.
3. Because the tape and `MarketView` could name different series, one run could carry two reference
   prices and two regime labelings.

## Decision

Make the primary symbol an explicit, configured property and select it at every seam:

- Add `DEFAULT_REFERENCE_SYMBOL = "ETHUSDT"` and an additive `DataConfig.reference_symbol` field
  (default `"ETHUSDT"`), overridable through an optional `[reference] symbol` TOML section. The
  value is normalised to uppercase; an empty/non-alphanumeric value, an unknown key, a non-table
  section, or a present-but-empty section is a `ConfigError`.
- `build_event_tape` gains a keyword-only `reference_symbol: str = DEFAULT_REFERENCE_SYMBOL` and,
  through the shared `_select_symbol_frame` helper, restricts **both** the `reference` and `regime`
  feeds to that symbol before their backward `join_asof`. A feed that does not contain the symbol is
  a `ValidationError`, never a silent fallback.
- `pipeline.build` threads `config.reference_symbol` into the tape.
- `MarketView` (`build_market_view`) restricts `reference` and `regimes` to
  `DataConfig.reference_symbol` at the one door to history, and raises `MarketViewError` when the
  symbol is absent — including a single-symbol feed whose symbol is not the configured primary.
- The two shipped configs declare `[reference] symbol = "ETHUSDT"`; the shared sim test fixture
  declares `reference_symbol="ETHUSDC"` to match its synthetic feed.

This is an **additive** change in the sense of `PLAN.md` §0.2 (a new defaulted config field and a new
keyword-only argument with a default); the data and sim `CONTRACTS.md` passages were updated in
place to record it.

## Consequences

- `build_event_tape` is pure and the tape is derived at load time, so **no re-snapshot is required**
  for the tape fix: the next `build`/`load_dataset` reads the stored streams and rebuilds it. The
  stored `regime` stream still contains both symbols; the sim's `_select_primary_symbol` restricts
  it at load. Computing regimes for the primary only (`pipeline._compute_regime`) is a mechanical
  follow-up that would halve that stream's size.
- `snapshot(out=...)` rebuilds the config with a new output directory through
  `pipeline._config_with_output_dir`, which copies **every** field; previously it reconstructed the
  dataclass by hand and silently dropped `reference_symbol` (and, pre-existing, `log_dir` and
  `tip_surcharge_pct`), so a `--out` snapshot could build an ETHUSDT tape while the same config named
  ETHUSDC for the sim.
- `load_dataset(<dataset directory>)` infers a `DataConfig` from the manifest, which does not record
  the reference symbol, so that path assumes the pinned primary (`ETHUSDT`); pass a `DataConfig`/TOML
  config for any other primary. Recording `reference_symbol` in the manifest is a follow-up.
- The reference-fetch symbol set (`ReferenceFetcher`, independent of `DataConfig`) is unchanged, so
  `[reference] symbol = "ETHBTC"` fails at `build`/`build_market_view` time rather than at
  `load_config`. The data validation checks (`check_reference_coverage`, `check_regime_labels_complete`)
  now filter to `reference_symbol` — see ADR-016. Deriving the pull's symbols from the config remains a
  follow-up.
- Consumers must not assume the stored reference/regime streams are single-symbol. `MarketView`
  is the enforced seam.
- Rejected alternative: filter only in `MarketView` and leave the tape joining all symbols. That
  keeps the tape's `price_reference`/`regime` ambiguous, and the backtester reads those tape columns;
  the sim's replay path and the backtester would then disagree on the series — the exact
  two-price failure this ADR removes.
