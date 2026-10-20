# Running the Dashboard (`undertow.ui`)

The live policy cockpit is a **read-only** web dashboard over the artifacts the
simulator already writes. It ships as a third package, `undertow.ui` (FastAPI
backend) plus an isolated React frontend under `web/`. It **never trains,
backtests or mutates a run** — it only reads `runs/` and `results/`.

This guide walks a fresh clone from `uv sync` to a populated dashboard. Every
command is copy-pasteable; there are no secrets and no network calls in the
dashboard itself.

- Backend/API contract: [`docs/plans/ui/CONTRACTS.md`](plans/ui/CONTRACTS.md)
- Frontend notes: [`web/README.md`](../web/README.md)
- Training/backtesting: [`docs/running_experiments.md`](running_experiments.md)

---

## 1. Prerequisites

- Python 3.14 and [`uv`](https://docs.astral.sh/uv/)
- A data snapshot (for training; see
  [`running_experiments.md` §"Build a data snapshot"](running_experiments.md#build-a-data-snapshot))
- Node 22 only for **developer mode** or building the frontend (`web/`)

```bash
git clone <repo>
cd undertow
uv sync
```

`uv sync` installs the `ui` backend stack (FastAPI, uvicorn, plotly) and the dev
group, so `uv run undertow-ui …` and `uv run pytest` both work without extra
flags. The React toolchain is **not** installed by `uv sync` — it lives entirely
under `web/`.

## 2. Produce some run artifacts

Training writes the sidecars the dashboard reads. If you already have a
snapshot, one command is enough:

```bash
uv run undertow-sim train --output runs/experiment_1/
```

This runs PPO once per configured seed (default `0–4`) and writes
`runs/experiment_1/seed_<n>/`. Telemetry is **on by default**
(`training.telemetry = true`), so each seed gets `telemetry.jsonl` +
`status.json` while it trains.

Evaluations enrich the Leaderboard (held-out Sharpe, per-regime PnL). Run one and
point the dashboard at the same `results/` directory:

```bash
uv run undertow-sim evaluate --checkpoint runs/experiment_1/seed_0/ppo_seed_0.zip \
    --output results/
```

See [`running_experiments.md`](running_experiments.md) for the full data →
backtest → train → evaluate flow.

## 3. Start the API server

```bash
uv run undertow-ui serve --runs runs/ --results results/
```

Then open **http://127.0.0.1:8050** (the server binds loopback only — there is no
auth in v1).

| View | Route | Answers |
|---|---|---|
| **Live Cockpit** | `/live` | What is the policy doing *right now*? Learning curve, PPO diagnostics, action heatmap. |
| **Policy Replay** | `/replay` | What did this checkpoint do, step by step? Price path + position bands + costs. |
| **Leaderboard / Compare** | `/leaderboard`, `/compare` | Which policies are best, and how do they trade off? |

Useful flags:

```bash
uv run undertow-ui info --runs runs/        # print resolved config + run count, exit 0
uv run undertow-ui serve --help             # all serve options
uv run undertow-ui serve --runs runs/ --results results/ \
    --host 127.0.0.1 --port 8050 --web-dist web/dist
```

`info` is the fastest "is my runs directory right?" check: it prints the resolved
`runs_dir`, `results_dir`, ports and the number of discovered run directories.

## 4. Developer mode (two terminals)

Terminal A — backend with autoreload:

```bash
uv run undertow-ui serve --runs runs/ --results results/ --reload
```

Terminal B — Vite dev server with HMR:

```bash
cd web
npm install        # first time only; node_modules is gitignored
npm run dev        # http://127.0.0.1:5173
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` (including the WebSocket
upgrade) and `/openapi.json` to the backend on `127.0.0.1:8050` (see
`web/vite.config.ts`), so the browser is same-origin and no CORS is needed.

Regenerate the typed API client whenever the Python wire contract changes:

```bash
cd web
npm run gen:api          # from the committed web/openapi.json (offline/reproducible)
npm run gen:api:live     # from a running server at 127.0.0.1:8050
```

`src/api/types.ts` is generated — never hand-edit it. Commit both
`web/openapi.json` and the regenerated `types.ts` in the same change.

## 5. Enabling replay data (`rollouts`)

Live and leaderboard views need only telemetry/manifests. **Policy Replay** needs
per-episode rollouts, which are recorded **off by default** because a parquet per
evaluation episode is not free. Turn them on in the training config:

```toml
# configs/sim_default.toml
[training]
telemetry = true
record_rollouts = true          # write rollouts/episode_<k>.parquet
rollout_max_episodes = 5        # cap on recorded evaluation episodes
```

Re-run training, then open `/replay` and pick a seed/episode. A run without
rollouts renders the replay panel's empty state instead of failing.

## 6. Artifact layout

Everything the dashboard reads lives under the run directory; nothing else is
required.

```
runs/
└── experiment_1/
    └── seed_0/
        ├── manifest.json                       # provenance + learning_curve
        ├── telemetry.jsonl                     # one TelemetryPoint per line (live)
        ├── status.json                         # heartbeat: running|done|failed
        ├── ppo_seed_0.zip                       # checkpoint
        └── rollouts/
            └── episode_0.parquet               # replay rows (opt-in)
results/
└── …                                           # ablation/regime/gap artifacts
```

| File | Written by | Schema |
|---|---|---|
| `telemetry.jsonl` | `undertow.sim` training callback | [`CONTRACTS.md` §1.1](plans/ui/CONTRACTS.md) |
| `status.json` | `undertow.sim` training callback | [`CONTRACTS.md` §1.2](plans/ui/CONTRACTS.md) |
| `rollouts/episode_<k>.parquet` | `undertow.sim` `RolloutRecorder` | [`CONTRACTS.md` §2.1](plans/ui/CONTRACTS.md) |
| `manifest.json` / `results/*` | `undertow.sim` train/evaluate | existing sim artifacts |

Every artifact carries `"schema_version": 1`. Readers **tolerate extra fields**
but **reject a higher major version with a typed `SchemaVersionError`** (`422`
over REST) per [`CONTRACTS.md` §0](plans/ui/CONTRACTS.md). Upgrading the UI never
requires rewriting old runs.

## 7. Application smoke test

`scripts/smoke_ui.sh` starts the server against the committed synthetic run
under `scripts/smoke_fixture/`, curls the contract endpoints (`/api/health`,
`/api/runs`, `…/curve`, `…/figures/learning`, `…/telemetry`), asserts the
expected keys, and shuts down:

```bash
bash scripts/smoke_ui.sh
```

Requires only `uv`, `curl` and `python3`. Override the port with
`SMOKE_UI_PORT=9999` if the default is busy.

## 8. Troubleshooting

- **"No runs discovered yet"** — the server found no run directories under
  `--runs`. A directory is a run when it directly contains `manifest.json`,
  `telemetry.jsonl` or `status.json`; both `runs/<experiment>/seed_<n>/` and
  `runs/seed_<n>/` layouts work. Confirm with
  `uv run undertow-ui info --runs runs/`. New files are picked up within a couple
  of seconds (the index refreshes lazily); force it with
  `GET /api/runs?refresh=true`.
- **WebSocket blocked / "Degraded · polling"** — corporate proxies sometimes
  strip the `/api/stream` upgrade. The client automatically falls back to
  `GET /api/runs/<id>/telemetry?since=<offset>` polling, so the cockpit keeps
  updating. If you run the server behind a proxy, allow WebSocket upgrades for
  `/api/stream` (or accept the polling fallback).
- **Port already in use** — `error while attempting to bind on address
  ('127.0.0.1', 8050)` means something else owns 8050. Pick another port:
  `uv run undertow-ui serve --runs runs/ --port 8051`, then open
  `http://127.0.0.1:8051`.
- **Stale `status.json` (stuck "running")** — a run killed mid-training may leave
  a `running` heartbeat. A missing or malformed `status.json` is treated as
  `unknown`, never `failed`; a *stale* one is still read as `running`. Re-run
  training so the terminal `done`/`failed` status is written, or delete the
  stale sidecar to fall back to the artifact mtime.
- **`web_dist` not built (root shows a JSON hint)** — `undertow-ui serve` mounts
  a built frontend only when `--web-dist` points at a directory containing
  `index.html`. Build it first, then restart with the flag:
  `cd web && npm install && npm run build`, then
  `uv run undertow-ui serve --runs runs/ --results results/ --web-dist web/dist`.
  The bare JSON body at `/` is the expected hint when no dist is mounted.
- **Newer schema rejected (`422` / `SchemaVersionError`)** — the artifact was
  written by a newer `undertow.ui` major. Upgrade the UI package rather than
  rewriting run data; older runs keep working unchanged.
