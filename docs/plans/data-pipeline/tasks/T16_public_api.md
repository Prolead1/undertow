# T16 — Public API surface + data dictionary

**Wave 7 · size M · depends on: T09, T11, T12, T13 (`load_dataset(validate=True)` runs its checks), T14 · blocks: nothing (final task)**
**Branch:** `feature-data-public-api`

## Why this task exists

`undertow/AGENTS.md` requires *"clean `__init__.py` exports"* and *"`undertow.data` and `undertow.sim`
communicate only through stable public interfaces"*. Fifteen tasks have built internals; this task
decides what the rest of the project — and specifically the simulator and backtester that come next — is
allowed to see. Get the surface small and right, because `undertow.sim` will be written against it and
every extra export becomes a thing you can never change.

The data dictionary is the other half: §10.3's evidence standard and §10.4's metrics all depend on
knowing exactly what each column means, in what unit, with what sign. A reader of the thesis must be able
to look up `amount0` and learn that positive means *into the pool*.

## Files you own

```
src/undertow/data/__init__.py          (replace T00's stub with the real surface)
src/undertow/data/loader.py            (load_dataset — the single entry point)
docs/data_dictionary.md
README.md                               (Status checklist + a Quickstart section)
tests/data/test_public_api.py
tests/data/test_data_dictionary.py
```

## What to build

### 1. `loader.py` — `load_dataset`
```python
def load_dataset(config: DataConfig | str | Path, *,
                 streams: Sequence[str] | None = None,
                 validate: bool = True) -> Dataset
```
- Reads a **cached snapshot from disk only**. It must **never** hit the network — fetching is
  `undertow-data pull`'s job. This separation is deliberate: a training run that silently starts making
  archive-node calls mid-episode is a debugging nightmare and a reproducibility hole. Assert it with a
  test that patches the network layer to explode and confirms `load_dataset` still works.
- Reads the manifest, checks `schema_version` compatibility, loads the requested streams via T09, and
  returns T11's `Dataset`.
- `validate=True` runs T13's checks and raises `ValidationError` on any Critical. Include the failing
  check names in the message.
- Missing dataset → a `ValidationError`/`ConfigError` whose message tells the user the exact
  `undertow-data pull` command to run. Small touch, large time saving.
- `streams=None` loads everything; a subset loads only those (fast path for a caller that only wants the
  reference feed and regimes).

### 2. `__init__.py` — the surface
Exactly the `__all__` from `CONTRACTS.md` §9. Nothing more. In particular do **not** export fetchers,
`FeeGrowthTracker`, `BaseHttpFetcher`, or any `transforms` internals — if `undertow.sim` needs
fee-growth math later, that is a deliberate contract change with an ADR, not an accident of a broad
export.

Module docstring states the package boundary rule (no `undertow.sim` import, ever) and points at
`docs/data_dictionary.md`.

Also add `__version__` sourced from package metadata, and re-export `SCHEMA_VERSION` — a caller must be
able to check compatibility without importing internals.

### 3. `docs/data_dictionary.md` — generated, not hand-written
Generate it from T03's schema metadata (that is why every field carries a unit string) so it can never
drift from the code. Structure: one section per stream, a table of
`column | type | unit / convention | nullable | source route`, followed by the non-obvious semantics
spelled out in prose:

- `swap.amount0/amount1`: signed, **positive = flowed into the pool**; fee is taken on the *input*
  (positive) side.
- Q128 / Q64.96 fixed-point scales, and why big integers are stored as strings.
- `price_pool` / `price_reference` orientation — quote T02's sentence verbatim: USDC per WETH for the
  pinned pools.
- The `Collect` ≠ accrual distinction, and the paired-`Burn` principal separation (from T13's work).
- `fee_growth_source` values and why there is no forward-interpolated option (look-ahead).
- Regime labels, the pre-committed thresholds, and the ADR-001 `μ` definition.
- Reference-feed gap policy (forward-fill and flag) and the USDT-vs-USD assumption from T08.
- Known limitations, stated plainly: position attribution is by NFT-manager `owner` not end user
  (`PLAN.md` §7); fee-growth snapshots are sampled, not per-block, with the replay-plus-verify argument;
  any residual from `adr/002` if it exists.

Provide the generator as a small function (`docs`-writing script or a `python -m` entry) and a test that
asserts the committed file is **up to date** — regenerate in the test and compare, failing with "run X to
regenerate". That is what keeps it honest after you leave.

### 4. `README.md`
Update the Status checklist (data pipeline scaffolding → done, with what "done" covers) and add a
Quickstart: env vars, `uv sync`, the three `undertow-data` commands, then a five-line Python snippet
using `load_dataset`. Link to `docs/running_the_pipeline.md` and `docs/data_dictionary.md`.

## Tests you must write

1. `__all__` matches `CONTRACTS.md` §9 exactly — assert the **set equality**, so both a missing and an
   extra export fail. This is the test that keeps the surface stable.
2. Every name in `__all__` is importable from `undertow.data` and is not `None`.
3. Nothing in `__all__` is a fetcher, a tracker, or a `transforms` internal (assert by module origin of
   each exported object).
4. `load_dataset` on a `tiny_dataset()`-backed snapshot returns a `Dataset` with the expected streams.
5. **No-network guarantee:** patch `requests.Session.request` (and `requests.get`) to raise, and assert
   `load_dataset` still succeeds. This is the important one.
6. `streams=["reference", "regime"]` loads only those: the requested fields are tables, the others are
   **`None`** (not empty tables — `CONTRACTS.md` §6.2), and no parquet outside them was read (assert via
   a read spy, or by deleting the other stream directories and confirming success).
7. `validate=True` on a corrupted snapshot raises `ValidationError` naming the failing checks;
   `validate=False` returns the dataset without raising.
8. Missing dataset → error message contains the literal `undertow-data pull` command.
9. Schema-version mismatch → `ValidationError` telling the user to re-pull.
10. Data dictionary freshness: regenerate from the schemas in-test and assert byte equality with the
    committed `docs/data_dictionary.md`.
11. Every column in every schema appears in the dictionary with a non-empty unit/convention cell
    (parametrize over `SCHEMA_REGISTRY` — catches a column added without documentation).
12. The architecture guard from T00 still passes (no `undertow.sim` import) — re-assert it here since this
    is the task that touches the package root.

## Acceptance criteria

- `uv run pytest` **fully** green (the whole suite, not just your module — you are the last task and
  you own the final state).
- `mypy src/undertow/data` clean; `ruff check .` clean.
- `docs/data_dictionary.md` committed, generated, and freshness-tested.
- README Quickstart actually works if followed literally.

## Handoff notes for your PR body — write these carefully

You are the last task in the plan, so your PR body is the plan's closing report. Include:

- The final public surface, listed.
- The **plan-level definition of done** from `PLAN.md` §6, item by item, each marked met or not met with
  evidence (the fee-reconciliation result from T13, the route-agreement result, the block-gap result, the
  Dune dashboard URL).
- Total suite size and runtime.
- The honest list of what `undertow.data` does **not** yet do — sampled rather than per-block fee-growth,
  position attribution, BigQuery route, any `adr/002` residual — so that the `undertow.sim` plan starts
  from an accurate picture rather than an optimistic one.

## Process (mandatory)

Branch (stacked per `PLAN.md` §3.2) → implement → **full** suite green → `code-reviewer` → fix → commit →
push → PR → update `STATUS.md` and mark the plan complete.
