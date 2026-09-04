# T01 — `types.py` + `config.py`

**Wave 1 · size S · depends on: T00 · blocks: T03, T04, T12, T15**
**Branch:** `feature-data-config`

## Why this task exists

Fifteen other modules need to agree on what a pool is, what a fee tier is, which exceptions are
retryable, and where the thresholds live. `undertow/AGENTS.md` requires config and constants to be
separate from logic — this task is that separation, created up front so nobody is tempted to hardcode.

## Files you own

```
src/undertow/data/types.py
src/undertow/data/config.py
configs/eth_usdc_3000.toml            (fill in the skeleton T00 created)
configs/eth_usdc_500.toml             (the 0.05% contrast pool)
tests/data/test_types.py
tests/data/test_config.py
```

## What to build

Implement **exactly** the interfaces in `CONTRACTS.md` §1 and §2. Nothing in this module does I/O
beyond reading a TOML file, and nothing in it does math.

### `types.py`
All the enums, `NewType` aliases and the full exception hierarchy from `CONTRACTS.md` §1, plus
`TICK_SPACING`, **plus `Severity` and the `CheckResult` dataclass**. `CheckResult` lives here rather than
in `validation/` on purpose: T08 (wave 3) returns one, but `validation/` is not created until T13
(wave 5). Read the comment above it in `CONTRACTS.md` §1 before moving it. `EventType`, `FeeTier`, `Regime`, `Route` are `str`/`int` mixin enums so they serialize
into Parquet string/int columns without conversion helpers.

### `config.py`
The five frozen dataclasses from `CONTRACTS.md` §2, plus:

- `load_config(path)` — TOML → `DataConfig`. Env-var expansion `${VAR}` applies **only** to
  `endpoints.*` string fields. Missing var ⇒ `ConfigError` naming the variable (not its value).
- `default_pools()` → the two pinned pools from `PLAN.md` §7, keyed `"USDC_WETH_3000"` /
  `"USDC_WETH_500"`. Token metadata to get right: USDC is **token0** (6 decimals), WETH is **token1**
  (18 decimals) for both — Uniswap orders tokens by address and `0xA0b8…eB48` (USDC) sorts below
  `0xC02a…6Cc2` (WETH). Getting this backwards silently inverts every price in the thesis, so state the
  ordering in a comment and assert it in a test.
- Pinned constants: `Q96`, `Q128`, `TICK_BASE`, `MIN_TICK = -887272`, `MAX_TICK = 887272`,
  `GLOBAL_TICK_SENTINEL` (int32 minimum, used by the fee-growth schema for global-only rows).
- Gas-unit constants for the friction model, as a frozen mapping with a sourced comment:
  `GAS_UNITS = {"mint": 400_000, "burn": 250_000, "collect": 150_000, "swap": 180_000}`. These are
  order-of-magnitude estimates for the NonfungiblePositionManager path — say so in the comment, and
  make them config-overridable so T14/`undertow.sim` can calibrate them later against real receipts.
- `PoolConfig.__post_init__` validation: address lowercase + 42 chars, `tick_spacing ==
  TICK_SPACING[fee_tier]`, decimals in `0..18`, `deployment_block > 0`. Raise `ConfigError`.
- `WindowConfig.__post_init__`: exactly one of (blocks, dates) fully specified; `start <= end`; dates
  timezone-aware UTC. Naive datetime ⇒ `ConfigError`.
- `WindowConfig` also carries **optional** `train_end_utc` / `eval_start_utc`. `undertow.data` never acts
  on them — they exist so a walk-forward split (§10.3) is recorded in the manifest rather than living in
  someone's notebook (`PLAN.md` §1). Validate only that, when both are given, they are UTC-aware and
  `train_end_utc <= eval_start_utc`, and that both fall inside the window.
- `RegimeConfig` defaults exactly as pinned in `PLAN.md` §7.

### Config files
`configs/eth_usdc_3000.toml` uses the pinned window `2022-01-01` → `2024-12-31` UTC, and
`${GRAPH_API_KEY}` / `${ETH_RPC_URL}` placeholders. `configs/eth_usdc_500.toml` is the same window on
the 0.05% pool.

## Tests you must write

- Every `__post_init__` rejection path, one test each, asserting the exception **type and message
  content** (not just that something raised).
- `TICK_SPACING` covers all four `FeeTier` members and matches the §5.4 table (1/10/60/200).
- `default_pools()` returns USDC as token0 with 6 decimals and WETH as token1 with 18 — the
  price-orientation guard.
- `load_config` round-trip on both real config files, with env vars set via `monkeypatch`.
- Missing env var ⇒ `ConfigError` whose message names the variable and **does not** contain any value.
- A config with `fee_tier = 3000, tick_spacing = 10` is rejected.
- Frozen-ness: mutating a `PoolConfig` field raises.
- `CheckResult` is constructible with each `Severity` value, is frozen, and defaults `metrics` to an
  empty mapping (not a shared mutable default — test that two instances do not share it).
- `WindowConfig` with `eval_start_utc < train_end_utc` raises; with both omitted, constructs fine; with
  a naive datetime for either, raises.
- `MIN_TICK`/`MAX_TICK` are exactly ±887272 (an off-by-one here silently corrupts T02's tick math).

## Acceptance criteria

- `uv run pytest` green; `mypy` clean.
- No I/O other than reading the TOML path handed to `load_config`.
- No secret value can appear in any log line, exception message or `repr`. Add a test that
  `repr(EndpointConfig(...))` with an expanded secret does not leak it — implement `__repr__` masking if
  needed, and if you do, note it in the PR because it changes the contract's spirit (mask, don't rename).

## Handoff notes for your PR body

- The final field list of each dataclass (T03/T04/T12/T15 code against it).
- Whether you added `__repr__` masking.
- Anything in `CONTRACTS.md` §2 you found underspecified — as an ADR, not a silent change.

## Process (mandatory)

Branch off `main` (or off `chore-data-scaffold` if T00's PR is unmerged — see `PLAN.md` §3.2) →
implement → tests green → `code-reviewer` → fix → commit → push → PR → update `STATUS.md`.
