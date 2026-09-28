# ADR 008 — Data ⇄ sim public-API surface reconciliation

**Status:** accepted
**Affects:** `src/undertow/data/__init__.py` · `src/undertow/data/loader.py` · `tests/data/test_public_api.py` · `tests/sim/test_data_api.py` · `docs/plans/data-pipeline/CONTRACTS.md` §9 · `docs/plans/sim/CONTRACTS.md` §3 · supersedes in part ADR-007
**Raised by:** S02 (`feature-sim-data-api`)

## Context

Two frozen contracts disagree about the surface `undertow.data` exposes to `undertow.sim`.

- **Data plan `CONTRACTS.md` §9** documents a frozen **18-name** `__all__` and states it is *"the
  only surface `undertow.sim` may use … Nothing more."* T16's brief repeats *"Exactly the `__all__`
  … Nothing more."*
- **Sim plan `CONTRACTS.md` §3** requires a much larger surface (all `undertow.data.types`, the
  schema constants, the fixed-point primitives, config types, and the `load_*` bridge adapters), and
  sim `PLAN.md` §0.7/§2.1 says explicitly: *"S02 exists precisely to widen that public surface where
  the sim needs it."*

T16 shipped in PR #25 and is merged. Its as-built `__init__.py` exports **16 names that already
deviate from §9's documented 18** — it renamed `load_config` to `load_data_config`, dropped
`default_pools`, `RegimeConfig`, `EventType`, `SCHEMA_REGISTRY` and the four error classes, and added
`EVENT_TAPE_SCHEMA`, `Route` and five fixed-point helpers. That T16 deviation was never recorded in an
ADR either. The guard T16's brief demanded — `tests/data/test_public_api.py` asserting set equality —
was never committed, nor were `docs/data_dictionary.md` or `tests/data/test_data_dictionary.py`. So
nothing enforced the "Nothing more" clause, and §9's documented list was not the as-built surface.

S02 (this branch) followed the sim plan and appended the names sim `CONTRACTS.md` §3 requires to the
same `__all__`. The suite stays green only because the data-side guard is absent.

Separately, sim `CONTRACTS.md` §3 lists a fixed-point name, `tick_to_sqrt_price`, that **does not
exist anywhere in `undertow.data`** — the data module exposes only the exact-integer
`tick_to_sqrt_price_x96(tick) -> int`. ADR-007 stubbed it as "pending T16", but T16 has shipped and
never contained that name; the stub's premise is void.

## Problem

1. The two frozen contracts cannot both hold. The sim genuinely needs the wider surface; the data
   contract forbids it; and with no enforcing test the contradiction is silent.
2. `tick_to_sqrt_price` is a phantom name. No sim task consumes it — human-unit sqrt-price belongs to
   S04's `calc_sqrt_price_a` (sim `CONTRACTS.md` §5), and the exact integer form is already exported.
3. The data surface has no regression guard, so any future export (or accidental internal leak) goes
   unnoticed.
4. T16's own deviation from §9 (18 documented vs 16 as-built names) is unrecorded; this ADR records
   both deviations together so the settled surface has a single, honest lineage.

## Decision

**The sim-facing surface is a deliberate, documented superset of both §9's documented list and T16's
as-built 16-name surface.** Data `CONTRACTS.md` §9 is amended by this ADR: the "Nothing more" clause is
superseded for the names the sim contract requires. The authoritative list is the `undertow.data.__all__`
on this branch, pinned by a restored `tests/data/test_public_api.py` (set-equality) and cross-checked by
`tests/sim/test_data_api.py` (required-subset).

Concretely:

1. **Keep** T16's as-built 16 names.
2. **Add** the sim `CONTRACTS.md` §3 names (types, config, schemas, fixed-point, `load_*` adapters) —
   as implemented by S02, minus the phantom below.
3. **Drop** `tick_to_sqrt_price` from both the data surface and sim `CONTRACTS.md` §3. The sim uses
   S04's `calc_sqrt_price_a` for human-unit sqrt-price and data's exact `tick_to_sqrt_price_x96` for
   the integer form. The `NotImplementedError` stub and its `strict=True` xfail are removed.
4. **Restore the missing guard**: `tests/data/test_public_api.py` asserts `undertow.data.__all__` set
   equality against the settled surface and that no fetcher/tracker/internal name leaks.

The 54-name settled surface (grouped):

- entry point: `load_dataset`
- loaders: `load_tape`, `load_reference_feed`, `load_gas_feed`, `load_regime_labels`
- core/dataset: `Dataset`, `DatasetManifest`, `DataConfig`, `PoolConfig`, `WindowConfig`,
  `RegimeConfig`, `EndpointConfig`, `default_pools`, `load_data_config`
- types: `Address`, `BlockNumber`, `DataTick`, `DataRegime`, `EventType`, `CheckResult`, `Severity`,
  `FeeTier`, `Route`, `Regime`
- schemas: `SWAP_SCHEMA`, `MINT_SCHEMA`, `BURN_SCHEMA`, `COLLECT_SCHEMA`, `FLASH_SCHEMA`,
  `FEE_GROWTH_SCHEMA`, `GAS_SCHEMA`, `REFERENCE_SCHEMA`, `REGIME_SCHEMA`, `EVENT_TAPE_SCHEMA`,
  `SCHEMA_REGISTRY`, `SCHEMA_VERSION`, `SORT_KEYS`, `encode_uint`, `decode_uint`, `validate_table`,
  `empty_table`
- fixed-point: `Q96`, `Q128`, `TICK_BASE`, `MIN_TICK`, `MAX_TICK`, `sqrt_price_x96_to_price`,
  `price_to_sqrt_price_x96`, `price_to_tick`, `tick_to_price`, `tick_to_sqrt_price_x96`,
  `sqrt_price_x96_to_tick`, `liquidity_for_amounts`, `amounts_for_liquidity`

## Consequences

- `docs/plans/data-pipeline/CONTRACTS.md` §9 gains an amendment block pointing here; its documented
  18-name list is kept for historical accuracy, and the amendment records that T16's as-built 16
  names already deviated from it.
- `docs/plans/sim/CONTRACTS.md` §3 no longer lists `tick_to_sqrt_price`; the note points here.
- `src/undertow/data/loader.py` no longer carries a stub; `src/undertow/data/__init__.py` drops
  `tick_to_sqrt_price` and documents the superset.
- `tests/data/test_public_api.py` is restored, restoring T16's intended regression guard.
- ADR-007 is corrected: its `TICK_BASE`-from-`.config` decision stands, but its "T16 pending" premise
  and its `tick_to_sqrt_price` stub are withdrawn by this ADR.
- **For the data plan:** the still-missing T16 deliverables (`docs/data_dictionary.md`,
  `tests/data/test_data_dictionary.py`) remain outstanding and are not addressed here; they belong to
  the data plan.
- **Rejected alternative:** leaving the §9/T16 surface frozen and having sim import data internals
  (`undertow.data.fixedpoint`, `.schemas`, …). That directly violates sim `PLAN.md` §4/§0.7 and the
  boundary rule, and would make every sim task depend on unstable internals — exactly what T16 exists
  to prevent.
- Vault edits are **not** made here; only repo files change.
