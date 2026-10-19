# Undertow — Live Policy Cockpit (`undertow.ui`): Multi-Agent Execution Plan

**Plan id:** `ui-v1`
**Scope:** a live, dark-themed web dashboard for the RL training pipeline — watch a policy being
trained, inspect what it does, and rank the best policies discovered so far. Delivered as a third
top-level package, `undertow.ui`, plus an isolated React frontend under `web/`.
**Source requirement:** the design discussion captured in `docs/plans/ui/README.md` §"Origin". There
is no external roadmap section for this; **this directory is the specification.** It exists to make
the thesis results legible and reproducible, not to change the science.

> **Do not re-litigate the stack in a task.** The choice (FastAPI + React/Vite + Plotly.js, third
> package, isolated Node toolchain) was made deliberately and is pinned in §7. If you think it is
> wrong, write `docs/decisions/NNN-<slug>.md` and flag it in the PR — do not silently deviate.

---

## 0. Read this first (all agents)

1. This plan is **normative**. `CONTRACTS.md` in this directory is **frozen**: every schema,
   endpoint, dataclass field, event envelope and unit in it is a cross-task contract. Implement
   exactly that. Extending is allowed (add an optional field with a default, add a keyword-only
   argument with a default); renaming, retyping, reordering or removing is not.
2. **ADRs live in `docs/decisions/`** (numbering continues from the highest existing number; `023` is
   the next free slot at the time of writing). Cite the ADR path from any code comment or docstring
   that depends on it. U01 owns the stack ADR.
3. Obey `undertow/AGENTS.md` without exception: **branch first → code + tests → `uv run pytest`
   green → `code-reviewer` subagent review → address findings → commit → push → open PR against
   `main`.** Never commit to `main`, never merge your own PR, never commit `.pi/`.
4. One task = one branch = one PR. **Work in your own git worktree** so parallel agents never edit
   the same files on disk. Your task file lists the **files you own**; do not touch files owned by
   another task. Remove your worktree once the branch is pushed.
5. Update `STATUS.md` (in this directory) when you start and when you finish. It is the handoff board.
6. **Import boundary is law.** The dashboard is a pure, **artifact-only** consumer: `undertow.ui`
   must **not** import `undertow.sim` or `undertow.data` at runtime. Shared wire schemas are mirrored
   in `undertow/ui/schemas.py` (see `CONTRACTS.md` §4) so the `ui` extra stays free of the training
   stack (`stable-baselines3`). The reverse is likewise forbidden: the strings `from undertow.ui` /
   `import undertow.ui` must not appear anywhere under `src/undertow/data/` or `src/undertow/sim/`.
   Both directions are AST-checkable and enforced by a test.
7. **The dashboard never trains, backtests or evaluates.** It reads artifacts (`manifest.json`,
   `telemetry.jsonl`, `status.json`, rollout parquet, results CSV/MD) and serves them. All compute
   that produces those artifacts already lives in `undertow.sim`; U00/U06 only *record* what the
   existing loops already compute.
8. **No secrets, no network, no GPU in tests.** The server binds `127.0.0.1` by default. Tests use
   fixture run directories only.

---

## 1. What is being built

A single-page dashboard with three linked views. Read `README.md` §"The three views" for the user
story; this section is the engineering summary.

| View | Answers | Primary data | Task |
|---|---|---|---|
| **Live Cockpit** | "What is the policy doing *right now*?" | `telemetry.jsonl`, `status.json` | U08 |
| **Policy Replay** | "What did *this* checkpoint actually do, step by step?" | rollout parquet | U09 |
| **Leaderboard / Compare** | "Which policies are best so far, and how do they trade off?" | `manifest.json`, results | U10 |

Design principles the visuals must satisfy:

- **Dark by default, no light mode.** One token set, defined once in `CONTRACTS.md` §9.
- **Domain-first.** The hero visual is the *position range over price* (the thing that makes
  concentrated liquidity legible), not a generic loss curve.
- **Live, not polled-to-death.** Server pushes deltas over WebSocket; the client merges without
  losing scroll/zoom/selection state.
- **Cross-linked.** Hover/brush a time cursor on one chart and every chart in the view snaps to it.
- **Provenance always visible.** `config_hash` + `git_commit` shown on every run card; a result you
  cannot trace is a bug.
- **Fast.** Charts render from pre-aggregated payloads; the server never ships millions of rows to
  the browser for a sparkline.

### 1.1 What already exists (do not rebuild it)

- `RunManifest` + learning curves written to `runs/<exp>/seed_<n>/manifest.json` (S13).
- `training_summary.json` at the run root (S16).
- Backtest artifacts: `equity_curve.csv`, `decision_log.csv`, `cost_ledger.csv`, `summary.json` (S11/S16).
- Evaluation artifacts: `ablation_table.{md,csv}`, `regime_matrix_*.md`, `regime_comparison.md`,
  `gap_report.{md,csv}`, `results_summary.md` (S15/S16).
- SB3 training internals on `model.logger.name_to_value` (policy/value loss, entropy, approx_kl,
  clip_fraction, explained_variance, fps) — computed but **not persisted today**.
- The env's `step()` `info` dict already carries everything the replay view needs
  (`reward_breakdown`, `action`, `tick_lower/upper`, `equity`, fees, gas, slippage, IL, regime) —
  computed but **not persisted today**.

So the plan's real unlock is **instrumentation (U00, U06) + a read-only server (U02–U05) + a thin
React client (U07–U10)**.

### 1.2 Out of scope (do not build)

- Training, evaluation, backtesting, hyperparameter search, or any mutation of a run.
- Authentication / multi-user / remote hosting (localhost only for v1).
- A database (the filesystem *is* the store; a scan index is built in memory on startup and
  refreshed on demand).
- 3D rendering, embeddings, attention interpretability for non-MLP policies (feature-sensitivity is
  a stretch task, see §7).
- Reworking `undertow.sim` evaluation output formats.

---

## 2. Target architecture (file ownership map)

```
undertow/
├── pyproject.toml                         U01  (adds `ui` extra, dev deps, `undertow-ui` script)
├── .gitignore                             U01  (adds node_modules/, web/dist/, *.tsbuildinfo)
├── .github/workflows/ui.yml               U11  (Node typecheck + vitest + build; non-blocking at first)
│
├── src/undertow/sim/                      # core wins; UI only reads what these write
│   ├── config.py                          U00  (+ training.telemetry / record_rollouts flags — additive)
│   ├── train/ppo_runner.py                U00  (wires the telemetry callback + status heartbeat)
│   ├── train/telemetry.py                 U00  (TelemetryCallback → telemetry.jsonl + status.json)
│   ├── train/rollouts.py                  U06  (RolloutRecorder → rollouts/episode_<k>.parquet)
│   ├── train/runs.py                      U00  (status "done" write; no signature changes)
│   └── train/__init__.py                  U00  (+ re-exports)
│
├── src/undertow/ui/                       # NEW Python package — FastAPI backend
│   ├── __init__.py                        U01
│   ├── __main__.py                        U01  (`python -m undertow.ui`)
│   ├── config.py                          U01  (UiConfig: runs_dir, host, port, cors)
│   ├── cli.py                             U05  (`undertow-ui serve`) — created U01 as stub, filled U05
│   ├── paths.py                           U02  (run-directory layout discovery)
│   ├── schemas.py                         U02  (pydantic models — the wire contract)
│   ├── registry.py                        U02  (scan runs/ → RunRecord index; refresh)
│   ├── telemetry.py                       U03  (read/tail telemetry.jsonl + status.json)
│   ├── streaming.py                       U03  (file-tail → PubSub hub for WebSocket)
│   ├── rollouts.py                        U06  (read rollout parquet → RolloutSeries)
│   ├── results.py                         U02  (parse results/ markdown+CSV into eval metrics)
│   ├── figures/                           U04
│   │   ├── __init__.py                    U04
│   │   ├── theme.py                       U04  (Plotly dark template — uses CONTRACTS §9 tokens)
│   │   ├── learning.py                    U04  (learning curve + CI band + checkpoints)
│   │   ├── diagnostics.py                 U04  (PPO health small-multiples)
│   │   ├── actions.py                     U04  (action-grid heatmap, optionally animated)
│   │   ├── ribbon.py                      U04  (price + position range + costs — the hero chart)
│   │   ├── leaderboard.py                 U04  (fingerprint radar / parallel coords / bars)
│   │   └── compare.py                     U04  (multi-policy overlay payloads)
│   └── app.py                             U05  (FastAPI factory: REST + WS + static mount)
│
├── web/                                   # NEW React app — isolated Node toolchain
│   ├── package.json  vite.config.ts  tsconfig.json  tailwind.config.ts  U07
│   ├── index.html                          U07
│   └── src/
│       ├── main.tsx  App.tsx  router.tsx   U07
│       ├── theme/tokens.css               U07
│       ├── api/client.ts  api/types.ts    U07  (generated from OpenAPI)
│       ├── api/useLiveStream.ts           U08  (WebSocket store)
│       ├── components/                    U07  (Panel, Stat, Sparkline, Toolbar, EmptyState, RunBadge)
│       ├── views/LiveCockpit.tsx          U08
│       ├── views/PolicyReplay.tsx         U09
│       ├── views/Leaderboard.tsx          U10
│       └── views/Compare.tsx              U10
│
├── tests/ui/                              # pytest for the Python backend
│   ├── conftest.py                        U01  (fixture run-dir factory)
│   ├── test_registry.py                   U02
│   ├── test_telemetry_reader.py           U03
│   ├── test_streaming.py                  U03
│   ├── test_figures.py                    U04
│   ├── test_api.py                        U05  (FastAPI TestClient)
│   └── test_rollouts.py                   U06
│
└── docs/
    ├── decisions/023-ui-stack-and-live-server.md   U01
    ├── running_the_dashboard.md                    U11
    └── plans/ui/                                   this directory
```

Sub-package `__init__.py` ownership: `figures/__init__.py` → U04 (creator); `ui/__init__.py` and
`ui/__main__.py` → U01.

### 2.1 The deliberately-shared paths

| File | Created by | Also edited by | Rule |
|---|---|---|---|
| `pyproject.toml` | exists | U01 only | U01 adds the `ui` extra + dev deps + script; no other UI task touches it. |
| `src/undertow/sim/config.py` | S01 | U00 (two additive flags), U06 (`rollout_max_episodes`) | Additive fields with defaults only; field order of existing dataclasses is frozen. |
| `src/undertow/sim/train/__init__.py` | S13 | U00 (re-exports), U06 (re-exports) | Append only. U00 edits `__all__`; U06 appends its own. |
| `src/undertow/ui/cli.py` | U01 (stub) | U05 (fills the `serve` command) | U01 creates the parser + `info`; U05 adds `serve`. |
| `src/undertow/ui/__init__.py` | U01 | U05 (appends `create_app`, `serve`) | Append to `__all__` + imports only. |
| `tests/ui/conftest.py` | U01 | U02 (adds the `run_dir` factory fixture) | U01 ships seeds/markers; U02 appends the shared fixture. |
| `web/src/api/types.ts` | U07 (generated) | regenerated whenever U05's OpenAPI changes | Generated, committed; never hand-edited. |

Everything else: if a path is not in your task's "Files you own", **do not create or edit it**. Need
a change there? Write an ADR or ask the orchestrator.

---

## 3. Task graph

Read an arrow as "needs the interface of".

```
wave 0   U00  sim telemetry instrumentation        U01  ui package scaffold + ADR
              │                                      │
wave 1   U02 registry + schemas   U03 live stream   U06 rollout recorder + reader
              │        │              │              │
wave 2   U04 figures ◄─┴──────────────┘              │
              │                                      │
wave 3   U05 FastAPI app + WS ◄──────────────────────┘
              │
wave 4   U07 React scaffold + design system + API client
              │
wave 5   U08 Live Cockpit   U09 Policy Replay   U10 Leaderboard/Compare
              │                    │                    │
wave 6        └────────────► U11 docs + CI + integration smoke ◄──────┘
```

### 3.1 Dependency table (authoritative)

| Task | Title | Depends on | Blocks | Wave | Size |
|---|---|---|---|---|---|
| **U00** | sim telemetry instrumentation (`telemetry.jsonl`, `status.json`, flags) | — | U02,U03,U04,U05,U08,U09 | 0 | M |
| **U01** | `undertow.ui` scaffold + repo plumbing + stack ADR | — | all UI | 0 | S |
| **U02** | run registry, path discovery, schemas, results parsing | U00,U01 | U04,U05,U08,U10 | 1 | M |
| **U03** | telemetry reader + file-tail streaming hub | U00,U01 | U04,U05,U08 | 1 | M |
| **U06** | rollout recorder (sim) + reader (ui) | U00,U01 | U05,U09 | 1 | M |
| **U04** | figure builders + Plotly dark theme | U02,U03 (contracts) | U05 | 2 | L |
| **U05** | FastAPI app: REST + WebSocket + static + CLI `serve` | U02,U03,U04,U06 | U07,U08 | 3 | L |
| **U07** | React/Vite scaffold, tokens, layout, API client | U05 (OpenAPI) | U08,U09,U10 | 4 | M |
| **U08** | Live Cockpit view | U07 | U11 | 5 | L |
| **U09** | Policy Replay / filmstrip view | U07 | U11 | 5 | L |
| **U10** | Leaderboard + Compare view | U07 | U11 | 5 | M |
| **U11** | docs, CI workflow, integration smoke, polish | U08,U09,U10 | — | 6 | M |

**Code dependency vs data dependency.** U04 builds figures against the `CONTRACTS.md` schemas using
fixtures, *not* against live U02/U03 modules — so it can start as soon as the contracts are read.
U02/U03 own the loaders; U04 consumes plain dicts/rows. Keep that decoupling: **figure builders take
data, never readers.**

### 3.2 Stacked-branch rule

Dependency merged → branch off `main`. Dependency PR still open → branch off the dependency branch and
put `Stacked on #<dep PR> (branch <dep-branch>). Merge that first.` at the top of the PR body. Never
copy a dependency's files; never re-implement a dependency's function "temporarily". If blocked, say
so in `STATUS.md` and stop.

---

## 4. Conventions every task must follow

- **Boundary.** §0.6. UI → sim/data public API only. No reverse import.
- **Read-only.** The UI never writes into a run directory. U00/U06 (inside `sim`) are the only
  writers, and they only append their own sidecar files.
- **Schemas are frozen.** All wire types live in `undertow/ui/schemas.py` (U02) and match
  `CONTRACTS.md` exactly. Pydantic models are the single source of truth; TypeScript types are
  generated from OpenAPI, never hand-written.
- **Determinism & ordering.** Registry scan order is deterministic (sort by `updated_at` desc, then
  `run_id`). JSON responses are stable so the client can diff.
- **No blocking I/O in the event loop.** File reads are sync and small; the WS hub tails with a
  bounded async task and `asyncio.to_thread` for stat/read. Never `time.sleep`.
- **Bounded payloads.** Any endpoint that could return a huge series takes a `max_points` (default
  2000) and downsamples deterministically (LTTB or uniform stride — pick one, document it, test it).
- **Errors are data.** The API returns typed error bodies (`{"detail": ...}`) and the UI renders an
  `EmptyState`, never a blank screen. A missing run is a `404`, a malformed artifact a `422` with the
  path.
- **Accessibility.** Every chart has a text summary; interactive controls are keyboard reachable;
  color is never the only encoding (pair with shape/label).
- **Type hints** on all public Python functions; `from __future__ import annotations` everywhere.
  Frontend is strictly `tsc --noEmit` clean.
- **Tests are non-negotiable.** Python: `uv run pytest tests/ui -q`. Frontend: `npm run typecheck &&
  npm run test`. A UI task with no tests for its logic is a Critical review finding.
- **Coverage.** `--cov=undertow` now includes `undertow.ui`. Keep logic in pure, tested modules;
  `# pragma: no cover` is allowed only on the ASGI/CLI composition root (`app.py` wiring,
  `__main__.py`) and must be justified in the PR. The 90% floor is not lowered.
- **No `print` in library code.** `logging.getLogger("undertow.ui.<module>")`.

---

## 5. Definition of done (per task)

1. Owned files exist, typed, no TODO stubs on the public path.
2. Tests assert real behavior — values, invariants, error paths, edge cases (empty run dir, corrupt
   JSON line, missing parquet, out-of-order telemetry) — not "does not raise".
3. `uv run pytest tests/ui -q` green (plus the full `uv run pytest` for U00/U06, which touch `sim`);
   summary pasted in the PR body. For frontend tasks, `npm run typecheck && npm run test` green.
4. `code-reviewer` **APPROVE** (after fixes if needed), verdict quoted in the PR body.
5. Public interface matches `CONTRACTS.md` exactly.
6. PR open against `main` from a `feature-ui-*` (or `feature-sim-telemetry` for U00) branch;
   `STATUS.md` updated; worktree removed.

---

## 6. Definition of done (whole plan)

- `uv run undertow-ui serve --runs runs/` starts a server on `127.0.0.1:8050`; opening the UI shows
  the run list with no runs, then picks up a run in progress within ~2 s of a new telemetry line.
- **Live Cockpit** streams learning curve + PPO diagnostics + action heatmap while `undertow-sim
  train` is running, and the "live" badge flips to "finished" when `status.json` says `done`.
- **Policy Replay** draws the price path with the agent's position bands, rebalances, in/out-of-range
  shading, equity vs HODL, and a cost decomposition for a selected episode.
- **Leaderboard** ranks every run by held-out eval Sharpe (falling back to `best_reward` when no eval
  artifact exists), shows provenance, and pastes two-plus policies onto one comparison canvas.
- `uv run pytest` (including `tests/ui`) is green at ≥90% coverage; `ruff` + `mypy src/undertow` clean.
- `docs/running_the_dashboard.md` walks a fresh clone from `uv sync` to a populated dashboard.
- Boundary intact: no `undertow.ui` import anywhere in `data/` or `sim/`; the dashboard works with
  `undertow.ui` uninstalled (training still writes telemetry sidecars).
- `web/` toolchain is invisible to the Python package: `uv run pytest` passes with `web/node_modules`
  absent.

---

## 7. Pinned decisions (so agents don't have to choose)

| Decision | Value | Rationale |
|---|---|---|
| Package shape | third top-level package `undertow.ui` (`src/undertow/ui/`) | mirrors `data`/`sim`; one distribution, one `pyproject.toml`; importable as `undertow.ui` |
| Frontend location | top-level `web/` | keeps JS out of `src/`; clearly not a Python package |
| Backend | **FastAPI + uvicorn** | OpenAPI → generated TS types (the biggest iteration win); pydantic already present |
| Live transport | **WebSocket** at `/api/stream` | bidirectional-ready; SSE is an acceptable fallback only if WS is unavailable |
| Frontend | **React 18 + Vite + TypeScript**, strict | HMR; component reuse for linked views |
| Styling | **Tailwind v4 + CSS-variable tokens** | fast, consistent dark theme; tokens live in `CONTRACTS.md` §9 |
| Charts | **Plotly.js via `react-plotly.js`** | one lib covers curves+CI, heatmaps, subplots, log axes, rangeslider; backend already speaks Plotly JSON |
| Figure ownership | **server-side Plotly figure specs** for v1; raw typed series also exposed for client-side state | maximum speed to first value; no fig-spec duplication in JS |
| Frontend tests | **vitest** for logic; Playwright smoke is a stretch (U11) | fast, no browser needed for unit logic |
| Node toolchain | isolated in `web/`; own CI job; never required by `uv run pytest` | core Python gate unaffected |
| "Best" ranking | **held-out eval Sharpe**, tiebreak worst-regime PnL; fall back to `best_reward` when no eval exists | matches the thesis' honest-evaluation stance; ranking is always explainable |
| Telemetry default | **on** (`training.telemetry = true`) | cheap JSONL; required for the live view |
| Rollout recording | **off** (`training.record_rollouts = false`) | parquet per episode is not free; opt in for replay runs |
| Downsampling | uniform stride by default; LTTB optional | simple, testable, deterministic |
| Host binding | `127.0.0.1` only | research tool; no auth in v1 |
| Polling fallback | REST `GET .../telemetry?since=` mirrors the WS payload | works behind proxies / when WS blocked |

**Stretch (explicitly not in v1 scope, do not silently add):** feature-sensitivity/permutation
importance over the 18 observation dims, checkpoint timelapse video export, `lightweight-charts` for
the ribbon, multi-run live comparison.

---

## 8. Risk register

| Risk | Owner | Mitigation baked into the plan |
|---|---|---|
| Telemetry JSONL interleaved/partial lines while training writes | U00, U03 | U00 writes one `json.dumps` line + `\n` in a single `write` call; U03 skips unparseable trailing line and advances only past complete lines |
| Tail loop hammers the disk | U03 | bounded poll interval + `asyncio.to_thread`; stop on `status.json: done` |
| Huge rollout/telemetry payloads freeze the browser | U03,U05 | server-side `max_points` downsampling, contract-tested |
| React toolchain makes Python CI flaky/slow | U01,U11 | separate `ui.yml` job; Python job unchanged; `web/node_modules` gitignored |
| Figure builders drift from wire schemas | U02,U04 | U04 takes plain dicts (`CONTRACTS` §5/§6); schema round-trip tests in both |
| Coverage floor breaks when `undertow.ui` lands | all | pure logic + tested loaders; `# pragma: no cover` only on wiring, justified per PR |
| Charts look generic, not "modern" | U04,U07 | one Plotly dark template + Tailwind tokens; hero ribbon is custom-shaped |
| "Which run is best" is ambiguous | U02,U10 | ranking rule pinned in §7 and shown in the UI as a tooltip |
| Two agents edit the same file | all | file-ownership map (§2) + per-task worktrees + stacked-branch rule (§3.2) |
| Sim package accidentally depends on UI | reviewer | grep rule in §0.6 is a Critical review finding |

---

## 9. Subagent dispatch protocol (orchestrator)

Implementation is executed by subagents. For **each** task the orchestrator performs, in order:

1. **Worktree.** `git worktree add ../undertow-wt-<task> -b <branch> <base>` where `<base>` is `main`
   unless the dependency is unmerged, in which case it is the dependency branch (stacked rule §3.2).
   Run all subagents for that task with `cwd = ../undertow-wt-<task>`.
2. **Implement.** Dispatch the `worker` subagent with the task file:
   > Read `docs/plans/ui/PLAN.md`, `docs/plans/ui/CONTRACTS.md`, then execute
   > `docs/plans/ui/tasks/<task>.md` end to end in this worktree. Write the code **and** its tests,
   > run `uv run pytest tests/ui -q` (and `uv run pytest` if you touch `sim`), and fix until green.
   > Do **not** commit. Report files changed, test summary, and anything the reviewer should know.
3. **Review.** Dispatch the **project** `code-reviewer` agent (`agentScope: "both"`) in the same
   worktree with the task file and the worker's report. It must return `APPROVE` or
   `REQUESTS CHANGES` per its own format. Do not commit before `APPROVE`.
4. **Fix loop.** On `REQUESTS CHANGES`, re-dispatch `worker` with the reviewer's findings; repeat
   step 3 until `APPROVE`. Never edit around the reviewer.
5. **Commit + PR.** Commit on the task branch, push, and open the PR against `main` (stacked base for
   stacked tasks). Body contains: task id, summary, the **pytest summary**, the **reviewer verdict**,
   and `STATUS.md` link. Do not merge.
6. **Handoff.** Update `STATUS.md` (state, branch, PR, pytest, reviewer), remove the worktree.

Parallelism: dispatch all tasks in a wave concurrently **only if their owned file sets are disjoint**
(they are, per §2). A task may start only when every task it depends on has an open, reviewer-approved
PR (or is merged).

---

## 10. Launch order

- **Wave 0:** U00, U01 (parallel; disjoint files).
- **Wave 1:** U02, U03, U06 (parallel).
- **Wave 2:** U04.
- **Wave 3:** U05.
- **Wave 4:** U07.
- **Wave 5:** U08, U09, U10 (parallel; disjoint view files).
- **Wave 6:** U11.

Do not start a wave until every task in the previous wave has an open, reviewer-approved PR. If the
human has not merged yet, dependent agents use the stacked-branch rule (§3.2).
