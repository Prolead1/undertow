# S02 — `undertow.data` public-API extension

**Wave 1 · size S · depends on: S00 · blocks: S03, S04, S07, S11**
**Branch:** `feature-sim-data-api`

## Why this task exists

The sim package may only import from `undertow.data`'s top-level `__init__.py`. Currently that
surface is a stub. The data plan's T16 owns the final surface, but S04 and S07 need position
math and fee-growth primitives *now* — they cannot wait for T16 (which is the last data-pipeline
task, wave 7). This task adds the **minimum additive exports** the sim needs, without touching
anything T16 will later own.

**Scope is strictly additive**: add import lines + `__all__` entries. Do not rename, retype,
reorder, or remove anything in the existing data files. Do not create new data modules.

## Files you own

```
src/undertow/data/__init__.py              (appends exports — does NOT replace the stub)
tests/sim/test_data_api.py                 (asserts the surface is importable)
```

## What to build

### 1. Extend `src/undertow/data/__init__.py`

Append imports and `__all__` entries for the names listed in CONTRACTS.md §3. Specifically:

**From `undertow.data.types`:**
`Address`, `BlockNumber`, `Tick` (import as `DataTick` — the sim has its own `Tick`), `FeeTier`,
`Regime` (import as `DataRegime`), `EventType`, `CheckResult`, `Severity`

**From `undertow.data.config`:**
`DataConfig`, `PoolConfig`, `WindowConfig`, `RegimeConfig`, `EndpointConfig`,
`load_config` (import as `load_data_config`), `default_pools`

**From `undertow.data.schemas`:**
All schema constants: `SWAP_SCHEMA`, `MINT_SCHEMA`, `BURN_SCHEMA`, `COLLECT_SCHEMA`, `FLASH_SCHEMA`,
`FEE_GROWTH_SCHEMA`, `GAS_SCHEMA`, `REFERENCE_SCHEMA`, `REGIME_SCHEMA`, `EVENT_TAPE_SCHEMA`,
`SCHEMA_REGISTRY`, `SCHEMA_VERSION`, `SORT_KEYS`, `encode_uint`, `decode_uint`, `validate_table`,
`empty_table`

**From `undertow.data.fixedpoint`:**
`Q96`, `Q128`, `TICK_BASE`, `MIN_TICK`, `MAX_TICK`,
`sqrt_price_x96_to_price`, `price_to_sqrt_price_x96`,
`price_to_tick`, `tick_to_price`, `tick_to_sqrt_price`

**New (S02 writes these — they live directly in `__init__.py` or a tiny `loader.py`):**
`load_dataset`, `load_tape`, `load_reference_feed`, `load_gas_feed`, `load_regime_labels`

The `load_*` functions are bridge adapters. For now, they can be thin wrappers that import and
delegate to the internal data modules — but the sim only calls the public versions.

If a data internal module doesn't yet exist (because the data plan's T16 hasn't shipped), write a
stub that raises `NotImplementedError("Pending T16 — data public API not yet complete")` and mark
it with a comment referencing T16. The test for that function should `xfail` with the same message.
The sim tasks that depend on these loaders (S03, S11) will use synthesized test fixtures, not the
real loaders, until T16 ships.

### 2. Verify the boundary

After your changes, run this check (and make it a test):

```bash
rg "from undertow.data\." src/undertow/sim/ --no-filename
```

This must produce zero hits. If it finds anything, your task is incomplete — the sim must only
reach `undertow.data`'s public surface.

## Tests you must write

1. `import undertow.data` succeeds
2. Every name listed in CONTRACTS.md §3 is in `undertow.data.__all__` and is importable
3. `undertow.data.__all__` does not contain any fetcher names (`BaseHttpFetcher`, `FeeGrowthTracker`,
   `GraphFetcher`, `RpcFetcher`, `GasFetcher`, `ReferenceFetcher`, `AbiFetcher`)
4. `from undertow.data import Q96, Q128, sqrt_price_x96_to_price, price_to_tick, tick_to_price`
   works
5. The boundary grep (above) returns zero hits in `src/undertow/sim/`

## Definition of done

- `src/undertow/data/__init__.py` has the extensions
- `uv run pytest tests/sim/test_data_api.py -q` green
- Boundary grep clean
- `code-reviewer` APPROVE
- PR open against `main` from `feature-sim-data-api`
- `STATUS.md` updated