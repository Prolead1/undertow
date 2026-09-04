# Undertow — `undertow.data` Pipeline: Multi-Agent Execution Plan

**Plan id:** `data-pipeline-v1`
**Scope:** the Week-1 critical-path deliverable of the thesis roadmap — a *validated, reproducible*
Uniswap V3 dataset loader for `undertow.data`.
**Source requirement:** `~/Documents/fyp/lesson_plan/10_data_evaluation_roadmap.md` §10.1 (data routes,
five+ streams), §10.1.6 (per-position stream inventory), §10.1.7 (external reference feed), §10.2.4
(fee-growth accumulators), §10.3 (regime labels), §10.6 (Week 1–2 milestones), §10.7 (risks).
Supporting: `09_research_gap_problem_statement.md` §9.2 (G1–G4), `05_concentrated_liquidity_lp_problem.md`
§5.4/§5.7 (tick lattice, fee tiers → tick spacing).

> **Roadmap quote that defines "done":** *"A validated, reproducible dataset loader (script + a parquet
> snapshot), plus a Dune dashboard sanity-checking volume/positions."* — §10.6, Week 1, feeds gap **G3**.

---

## 0. Read this first (all agents)

1. This plan is **normative**. Do not redesign it. If you believe a task's design is wrong, write an
   ADR note in `.pi/plans/data-pipeline/adr/` and flag it in your PR description — do not silently
   deviate, because other agents are coding against your interface *as written here*.
2. **`CONTRACTS.md` in this directory is frozen.** Every column name, type, unit, sign convention and
   function signature in it is a cross-task contract. Implement exactly that. Extending is allowed
   (add a column, add a keyword-only argument with a default); renaming, retyping, or reordering is not.
3. **ADRs live in two places, deliberately.** `.pi/` is never committed (`undertow/AGENTS.md`), so an ADR
   kept only here is invisible to anyone reading the repo — and code that cites `adr/002-…` in a
   docstring would point at nothing. So when you raise an ADR:
   - write the full decision record to `.pi/plans/data-pipeline/adr/NNN-slug.md` (planning history, stays
     in the vault), **and**
   - commit a short version to `undertow/docs/decisions/NNN-slug.md` in the repo, and cite *that* path
     from any code comment or docstring.
   The vault copy may reference this plan; the repo copy must stand alone (a reader of the repo has never
   seen `PLAN.md`). `docs/decisions/` is owned by whichever task raises the ADR.
4. Obey `undertow/AGENTS.md` without exception: branch first → write code + tests → `uv run pytest`
   green → **`code-reviewer` subagent review** → address findings → commit → push → open PR against
   `main`. **Never** commit to `main`, **never** merge your own PR, **never** commit `.pi/`.
5. One task = one branch = one PR. Your task file lists the **files you own**. Do not touch files owned
   by another task — that is how parallel PRs stay conflict-free.
6. Update `STATUS.md` (in this directory) when you start and when you finish. It is the handoff board.

---

## 1. What is being built

`undertow.data` ingests, verifies, and caches the seven data streams the backtester and simulator need,
and hands them over as one block-ordered, look-ahead-free event tape plus reference-price and regime
side-tables.

| # | Stream | Why it exists (roadmap §) | Primary route | Owning task |
|---|---|---|---|---|
| 1 | `swap` | realized price path, volume, fee flow | The Graph, verify via RPC | T05 / T06 |
| 2 | `mint` | where/when liquidity was deployed | The Graph, verify via RPC | T05 / T06 |
| 3 | `burn` | where/when liquidity was removed | The Graph, verify via RPC | T05 / T06 |
| 4 | `collect` | when fees were actually withdrawn (≠ accrued) | The Graph, verify via RPC | T05 / T06 |
| 5 | `tick` / fee-growth | **exact** per-position fee accrual + IL (§10.2.4) | archive RPC `eth_call` | T06 / T10 |
| 6 | `gas` | the friction `G_t` that makes RL hard (G1) | RPC blocks + receipts | T07 |
| 7 | `reference` price | "true" price for IL/LVR + regime labels (§10.1.7) | Binance klines archive | T08 |
| 8 | `regime` labels | per-regime evaluation (G4, §10.3) | derived from #7 | T12 |

**Out of scope for this plan** (do not build): the AMM simulator, the backtest loop, the RL environment,
metrics, baselines. Those live in `undertow.sim` and in a later plan. `undertow.data` must not import
`undertow.sim`, and vice versa.

**The walk-forward train/eval split is deliberately out of scope too, with one hook.** §10.3's first
evaluation practice (train on `[t0,t1]`, evaluate on a later non-overlapping `[t2,t3]`) is an
*experiment-design* concern, not an ingestion concern: the same parquet snapshot serves every split, and
baking boundaries into the dataset would make the data artifact experiment-specific. So this plan ships
the full contiguous window and `undertow.sim`'s plan owns the split. The hook: `WindowConfig` carries
optional `train_end_utc` / `eval_start_utc` fields (T01) that `undertow.data` records in the manifest and
otherwise ignores — so a split, once chosen, is provenance-tracked rather than a number in a notebook.

---

## 2. Target architecture (file ownership map)

```
undertow/
├── pyproject.toml                        T00
├── src/undertow/__init__.py              T00
├── src/undertow/data/
│   ├── __init__.py                       T16   (public API surface; stub created by T00)
│   ├── types.py                          T01   enums, aliases, exception hierarchy
│   ├── config.py                         T01   pool registry, endpoints, windows, thresholds
│   ├── fixedpoint.py                     T02   Q64.96 / Q128 / tick ↔ price math
│   ├── schemas.py                        T03   canonical pyarrow schemas + row dataclasses
│   ├── fetchers/
│   │   ├── base.py                       T04   HTTP session, retry, chunking, raw cache
│   │   ├── thegraph.py                   T05   Route A (GraphQL)
│   │   ├── rpc.py                        T06   Route C (eth_getLogs + eth_call, ABI decode)
│   │   ├── abi.py                        T06   topic0 constants + event decoders
│   │   ├── gas.py                        T07   per-block base fee, gas used, tx receipts
│   │   ├── reference.py                  T08   Binance kline archive
│   │   └── queries/*.graphql             T05   GraphQL queries as files, not inline strings
│   ├── transforms/
│   │   ├── feegrowth.py                  T10   eqs (3)–(6) + tick-crossing state machine
│   │   ├── align.py                      T11   block-ordered merge → event tape
│   │   └── regimes.py                    T12   σ_rv / μ rolling labels
│   ├── storage/
│   │   ├── parquet.py                    T09   partitioned read/write
│   │   └── manifest.py                   T09   provenance / dataset versioning
│   ├── validation/
│   │   ├── __init__.py                   T13
│   │   ├── checks.py                     T13   invariants, gap detection
│   │   ├── crosscheck.py                 T13   subgraph↔RPC, fees↔Collect reconciliation
│   │   └── dune_expectations.py          T15   expected magnitudes (library, so T13 can import it)
│   ├── cli.py                            T14   `undertow-data pull|verify|snapshot|info`
│   ├── pipeline.py                       T14   orchestration, kept out of cli.py
│   └── loader.py                         T16   `load_dataset` — the single entry point
├── src/undertow/sim/__init__.py          T00   empty placeholder (makes the split visible)
├── uv.lock, .gitignore                   T00
├── tests/conftest.py                     T00   incl. the --run-network flag/skip hook
├── tests/data/conftest.py                T11   the shared `tiny_dataset()` factory
├── sql/dune/README.md                    T15
├── configs/eth_usdc_3000.toml            T00 skeleton → T01 fills  (shared, §2.1)
├── configs/eth_usdc_500.toml             T01
├── sql/dune/*.sql                        T15   exploration queries (not Python)
├── docs/schema_notes.md                  T03
├── docs/decisions/NNN-*.md               whoever raises the ADR (§0.3); T12 owns 001
├── docs/running_the_pipeline.md          T14
├── docs/data_dictionary.md               T16   generated from T03's schema metadata
└── tests/data/...                        each task ships its own tests + fixtures
                                                (fixture ownership table: CONTRACTS.md §10)
```

Sub-package `__init__.py` files belong to the first task that creates the package: `fetchers/` → T04,
`transforms/` → T10, `storage/` → T09, `validation/` → T13 (T15 may have to create an empty one first if
it lands earlier; its brief says so).

Rules the reviewer will enforce: no god modules (a module doing I/O *and* math *and* config is a
rejection), config/constants out of logic, type hints on every public function, big integers never
silently coerced to float.

### 2.1 The deliberately-shared paths

Every other path above has exactly one owner. These six are touched by more than one task, so they are
the only places a merge conflict is possible. Rules per path:

| File | Created by | Also edited by | Rule |
|---|---|---|---|
| `pyproject.toml` | T00 (whole file) | T14 (adds `[project.scripts]` only) | T14's diff must touch nothing but that one table. T00 deliberately omits the entry point because pointing it at a module that does not exist yet breaks `uv sync`. |
| `src/undertow/data/__init__.py` | T00 (stub: docstring + `__all__: list[str] = []`) | T16 (replaces it with the real surface) | Nobody between T00 and T16 adds an export. If your module needs to be importable, it is importable by its full path. |
| `README.md` | T00 (Status checklist) | T16 (Status + Quickstart) | Status-checklist lines and the Quickstart section only. |
| `configs/eth_usdc_3000.toml` | T00 (empty section skeleton) | T01 (fills the real field set) | T01 owns the field semantics; T00 only creates the sections so the file exists. |
| `tests/data/fixtures/collect_reconciliation/` | T06 (raw captures) | T13 (adds `cases.json` only) | T06 captures the three real position lifecycles; T13 adds only the expected-value case file. Neither edits the other's files. |
| `src/undertow/data/validation/__init__.py` | T13 | T15 (creates it empty if it lands first) | T15 owns `dune_expectations.py` inside T13's package; whoever arrives first creates the `__init__.py`. |

Everything else: if a path is not in your task's "Files you own", do not create or edit it. Need a
change there? Ask the orchestrator, or write an ADR.

---

## 3. Task graph

Read this as "an arrow means *needs the interface of*". Waves are the parallel batches; §9 has the
launch order.

```
wave 0   T00  scaffold
              │
wave 1   T01  config/types            T02  fixedpoint
              │                            │
wave 2   T03  schemas   T04  fetcher-base  │   T10  feegrowth
              │              │             │        │
wave 3   T05 thegraph   T06 rpc   T07 gas   T08 ref   T09 storage   T12 regimes   T15 dune
              │              │        │         │          │             │            │
wave 4        └──────────────┴────────┴─────────┴──► T11  align ◄────────┘            │
                                                          │                           │
wave 5                                               T13  validation ◄────────────────┘
                                                          │
wave 6                                               T14  cli / snapshot
                                                          │
wave 7                                               T16  public API + data dictionary
```

Arrows into T11 come from T02/T03/T09/T10 (code) and T05–T08/T12 (data). T15 bypasses T11 entirely and
feeds T13. T13 additionally needs T05, T06 and T10.

Edges the diagram flattens, spelled out:

- T02 → T05, T06, T10, T11 (everyone who decodes or converts a fixed-point number)
- T03 → T05, T06, T07, T08, T09, T11, T12 (everyone who emits a table)
- T04 → T05, T06, T07, T08 (all four fetchers)
- T08 → T12 (regimes are computed from the reference feed, not the pool)
- T06 → T10 (RPC supplies the fee-growth snapshots T10 reconciles against)
- T15 → T13 (T13 imports T15's `validation/dune_expectations.py`; the magnitudes themselves are a fixture)

### 3.1 Dependency table (authoritative)

| Task | Title | Depends on | Blocks | Wave | Est. size |
|---|---|---|---|---|---|
| **T00** | uv project scaffold + test harness | — | all | 0 | S |
| **T01** | `types.py` + `config.py` (pool registry, endpoints, windows, regime thresholds) | T00 | T03,T04,T12,T15 | 1 | S |
| **T02** | Fixed-point & tick math | T00 | T05,T06,T10,T11 | 1 | M |
| **T03** | Canonical schemas (the data contract) | T00,T01 | T05,T06,T07,T08,T09,T11,T12 | 2 | M |
| **T04** | Fetcher base layer (retry/chunk/cache) | T00,T01 | T05,T06,T07,T08 | 2 | M |
| **T10** | Fee-growth engine (eqs 3–6 + tick state machine) | T00,T02 | T11,T13 | 2 | L |
| **T05** | The Graph fetcher (Route A) | T02,T03,T04 | T11,T13,T14 | 3 | L |
| **T06** | RPC fetcher (Route C) + ABI decoders | T02,T03,T04 | T10 (data),T13,T14 | 3 | L |
| **T07** | Gas / block fetcher | T01,T03,T04 | T11,T14 | 3 | M |
| **T08** | Reference price feed fetcher (Binance) | T01,T03,T04 | T11,T12,T14 | 3 | M |
| **T09** | Parquet storage + dataset manifest | T03 | T11,T14,T16 | 3 | M |
| **T12** | Regime labeller | T01,T03 (+T08 *data*) | T11,T14 | 3 | M |
| **T15** | Dune exploration queries + dashboard | T01 | T13 (imports `dune_expectations`) | 3 | S |
| **T11** | Stream alignment → event tape | T02,T03,T09,T10 | T13,T14,T16 | 4 | L |
| **T13** | Validation & cross-check suite | T02,T05,T06,T10,T11,T15 | T14,T16 | 5 | L |
| **T14** | CLI + reproducible snapshot pipeline | T05–T13,T15 | T16 | 6 | M |
| **T16** | Public API surface + data dictionary | T09,T11,T12,**T13**,T14 | — | 7 | M |

Waves 1–2 and 3 are the parallel ones. Waves 4–7 are serial by nature (each consumes everything before).

**Code dependency vs data dependency — why T12 sits in wave 3.** The graph in §3 shows an edge
T08 → T12, but the table lists T12's dependencies as T01 and T03. Both are right, and the distinction
matters for scheduling:

- A **code** dependency means you import the other task's module and need its interface to exist. It
  forces you into a later wave.
- A **data** dependency means you consume a *table* the other task produces. You code against the
  **schema** (T03), not the producer, so you can be built in parallel and tested on fixtures.

T12 reads `REFERENCE_SCHEMA` tables; it never imports `fetchers/reference.py`. So it is testable against
its own fixtures in wave 3 alongside T08 and only meets real T08 output at T14. Same pattern for
T06 → T10: T10 consumes `FEE_GROWTH_SCHEMA` rows, whoever fetched them. T04 → T03 was *removed* the same
way — `BaseHttpFetcher` returns raw rows and never imports `schemas.py`, so both sit in wave 2
(`CONTRACTS.md` §5.1 records the layering rule).

This is the main reason wave 3 can run seven agents wide: schemas-as-contracts decouple producers from
consumers. Two consequences:

1. **Your fixtures are load-bearing.** If they misrepresent the real payload, your module is green and
   still wrong. Trim field *values* from genuine responses; synthesize only *envelopes* (pagination
   wrappers and the like), and note which you did.
2. **A consumer in an earlier wave than its producer owns its own fixture.** T10 (wave 2) does not wait
   for T06's captures; T12 does not wait for T08's klines. `CONTRACTS.md` §10 assigns every fixture to
   the earliest task that needs it.

### 3.2 Stacked-branch rule (how to work before a dependency is merged)

The human merges PRs, so a dependency's PR may still be open when you start.

- If your dependency's PR is **merged**: branch off `main`.
- If it is **open**: branch off the dependency branch, and put this line at the top of your PR body:
  `Stacked on #<dep PR> (branch <dep-branch>). Merge that first.`
- Never copy a dependency's files into your branch. Never re-implement a dependency's function
  "temporarily" — if you are blocked, say so in `STATUS.md` and stop.

---

## 4. Conventions every task must follow

- **Big integers stay integers.** `uint256`/`int256`/`uint128`/`uint160` values (`amount0`, `amount1`,
  `liquidity`, `sqrtPriceX96`, all `feeGrowth*X128`, `baseFeePerGas`) are stored in Parquet as
  **decimal-free strings** and handled in Python as `int`. Converting a fee-growth accumulator to
  `float` before the final division is a **Critical** review finding — `2**128 ≈ 3.4e38` destroys
  float64 precision.
- **Wrapping arithmetic.** Uniswap V3 computes fee-growth deltas with Solidity *unchecked* subtraction,
  so accumulators legitimately wrap. All accumulator subtraction goes through
  `fixedpoint.wrapping_sub_256`. Never assert `inside_now >= inside_last`.
- **Ordering key is `(block_number, log_index)`**, never `timestamp` (multiple events share a
  timestamp; block timestamps are not strictly increasing across reorged history).
- **No look-ahead, ever.** Any transform that produces a row for block *n* may only read data from
  blocks ≤ *n*. Rolling windows are backward-looking and closed on the right. This is the §10.7 risk
  the reviewer is told to hunt for.
- **Timestamps** are timezone-aware UTC (`timestamp[us, tz=UTC]`). No naive datetimes anywhere.
- **Determinism / reproducibility.** Same config + same block range ⇒ byte-identical parquet content
  (sort before write, no wall-clock or dict-order leakage into data columns). Wall-clock and library
  versions go in the *manifest*, not in the data.
- **Secrets** (`GRAPH_API_KEY`, `ETH_RPC_URL`, …) come from environment variables only, read in
  `config.py`. No key ever appears in code, tests, fixtures, logs or the manifest.
- **Network in tests is forbidden.** Every fetcher test runs against recorded fixtures
  (`tests/data/fixtures/…`, small hand-trimmed JSON; ownership in `CONTRACTS.md` §10). Mark any
  genuinely-live test `@pytest.mark.network`; T00's `conftest.py` skips those unless `--run-network` is
  passed. A test that silently reaches the network is a Critical review finding.
- **Logging** via `logging.getLogger("undertow.data.<module>")`. No `print` in library code.
- **Type hints** on all public functions; `from __future__ import annotations` at the top of modules.

---

## 5. Definition of done (per task)

A task is done when **all** of these hold:

1. Files listed in "Files you own" exist, are typed, and contain no TODO stubs on the public path.
2. Tests exist in the specified test module and assert **real behavior** — values, invariants, error
   paths, boundary cases — not "does not raise".
3. `uv run pytest` is green (paste the summary line in the PR body).
4. `code-reviewer` has reviewed the change and returned **APPROVE**, or its findings were fixed and it
   re-approved. The review output is summarized in the PR body.
5. The public interface matches `CONTRACTS.md` exactly.
6. A PR is open against `main` from a `feature-data-*` branch; `STATUS.md` is updated.

---

## 6. Definition of done (whole plan / Week-1 deliverable)

- `uv run undertow-data snapshot --config configs/eth_usdc_3000.toml` produces, from a cold cache:
  a partitioned parquet dataset for all 7 streams + regime labels, a `manifest.json` with full
  provenance, and a `validation_report.md`.
- The validation report shows, for at least one sampled 30-day window:
  - subgraph vs RPC event-level agreement (counts and field-level equality) — **exact**;
  - reconstructed uncollected fees vs at least 3 real on-chain `Collect` amounts — within tolerance,
    with the tolerance stated and justified;
  - zero block gaps in the gas stream, zero duplicate `(block_number, log_index)` keys, monotone
    ordering, and liquidity-conservation invariants satisfied.
- `sql/dune/` contains the exploration queries and the dashboard URL, with expected magnitudes recorded
  so any drift is detectable.
- `docs/data_dictionary.md` documents every column: meaning, unit, fixed-point scale, sign convention,
  source route, and nullability.
- The single entry point `undertow.data.load_dataset(...)` returns the tape + side tables and is what
  `undertow.sim` will consume — no `undertow.sim` import in `undertow.data`.

---

## 7. Pinned decisions (so agents don't have to choose)

| Decision | Value | Rationale |
|---|---|---|
| Primary pool | `0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8` (USDC/WETH, 0.30%, spacing 60) | the roadmap's worked example |
| Secondary pool | `0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640` (USDC/WETH, 0.05%, spacing 10) | fee-tier contrast for RQ1 |
| Study window | `2022-01-01` → `2024-12-31` UTC | fully post-EIP-1559 (no legacy `gasPrice` path); spans bull/bear/sideways/crash |
| Reference feed | Binance `ETHUSDT` 1m klines primary, `ETHUSDC` 1m as cross-check | ETHUSDT is the deepest, longest 1m history; USDC pair is thinner |
| Regime window | 30 days rolling, backward-looking, minute closes | §10.3 |
| Regime thresholds | high-vol if σ_rv > 0.80; else bull if μ₃₀ > +0.05, bear if μ₃₀ < −0.05, else sideways | §10.3, pre-committed in `config.py`, sweepable |
| μ definition | **total** log return over the window: `μ₃₀ = log(S_t / S_{t−30d})` | §10.3's per-step mean reading is dimensionally inconsistent with a ±5% threshold; see `CONTRACTS.md` §6 |
| tick-from-price | **floor**, not round | matches the protocol (`tick` = greatest tick with price ≤ current); §10.1.1's "round" is a simplification |
| Position attribution | out of scope — we replay *synthetic* positions | real retail `Mint.owner` is the NFT position manager `0xC364…FE88`, so per-user attribution needs a separate NFT-event join. Recorded as future work. |
| Fee-growth history | sampled by archive `eth_call` at chosen blocks (exact), with optional interpolation flagged and error-bounded against samples | subgraph `Tick` entities expose only *current* values — see T10 §pitfalls |

---

## 8. Risk register (mapped to §10.7)

| Risk | Owner task | Mitigation baked into the plan |
|---|---|---|
| Data engineering eats the schedule | T14 | cache-first, resumable, idempotent pulls; parquet snapshot so nothing is ever re-queried |
| Fee numbers silently wrong by an order of magnitude | T10, T13 | exact integer Q128 math + reconciliation against real `Collect` events as a *blocking* acceptance criterion |
| Look-ahead bias | T11, T12 | `(block_number, log_index)` ordering, backward-only windows, an explicit look-ahead unit test per transform |
| Archive-node access / rate limits | T04, T06 | provider-agnostic RPC client, block-range chunking with backoff, raw-response cache on disk |
| Non-reproducible regime labels | T12 | thresholds pre-committed in config + a sensitivity-sweep API |
| Float precision loss on Q128 | all | strings in parquet, `int` in Python, reviewer instruction to reject float coercion |
| Two agents editing one file | all | the file-ownership map in §2 + stacked-branch rule in §3.2 |

---

## 9. Launch instructions (for the orchestrating agent / human)

Each task file `tasks/T<nn>_*.md` is a **self-contained brief** — a fresh agent needs nothing but that
file, `CONTRACTS.md`, and `PLAN.md`. Launch pattern per task:

```
Agent(subagent_type="general-purpose",
      prompt="Read /home/dev/Documents/fyp/.pi/plans/data-pipeline/PLAN.md and CONTRACTS.md, then
              execute tasks/T05_thegraph_fetcher.md end to end in /home/dev/Documents/fyp/undertow,
              including the mandatory code-reviewer gate and the PR. Report the branch, PR, pytest
              summary and reviewer verdict.")
```

Sequencing:

- Wave 0: T00 alone (blocks everything — run it to completion first).
- Wave 1: T01, T02 in parallel.
- Wave 2: T03, T04, T10 in parallel.
- Wave 3: T05, T06, T07, T08, T09, T12, T15 in parallel (7 agents; disjoint files).
- Wave 4: T11.
- Wave 5: T13.
- Wave 6: T14.
- Wave 7: T16.

Do not start a wave until every task in the previous wave has an open, reviewer-approved PR. If the
human has not merged yet, dependent agents use the stacked-branch rule (§3.2).
