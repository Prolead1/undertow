# CONTRACTS.md — frozen cross-task interfaces for `undertow.ui`

**Status: FROZEN.** Every schema, field, type, endpoint, event and token below is a contract between
tasks. Implement exactly this. You may **add** (an optional field with a default, a keyword-only
argument with a default, a new event `type` with a documented consumer fallback); you may not
**rename, retype, reorder or remove**. If something here is genuinely wrong, write
`docs/decisions/NNN-<slug>.md`, flag it in the PR, and notify the orchestrator — other agents are
already coding against it.

---

## 0. Global conventions

- **Times.** All timestamps are ISO-8601 UTC strings ending in `Z` on the wire (pydantic serialises
  `datetime` with `tzinfo=UTC`); parquet stores `datetime64[us, UTC]`. Never machine-local.
- **Units.** Wealth/PnL/fees/gas/slippage/IL are USDC (`float`). `price` is human USDC-per-WETH
  (ADR-009). `tick`, `tick_lower`, `tick_upper`, `liquidity` are raw protocol quantities. `center_offset`
  and `width` are in tick-spacings. Regimes are the data module's labels (`bull|bear|sideways|high_vol|unknown`).
- **Signs (cost honesty).** In rollout records, `fees >= 0`; `gas`, `slippage`, `risk_penalty <= 0`;
  `il` is the signed change (usually `<= 0`); `reward` is the env's normalised step reward (capital
  units). Net = `fees + gas + slippage + il + risk_penalty`.
- **Schema version.** Every persisted artifact carries `"schema_version": 1`. Readers reject a higher
  major version with a typed error; they tolerate extra fields.
- **JSON discipline.** Sidecars are newline-delimited JSON, one complete object per line, written with
  a single `write()` + trailing `\n` so a concurrent reader never sees a torn line. `status.json` is
  written atomically (temp file + `os.replace`).
- **Determinism.** Registry scan order: `updated_at` descending, then `run_id` ascending. Series are
  ordered by `timestep` / `step` ascending. Downsampling is deterministic (uniform stride, then forced
  first/last points).
- **Boundary.** `undertow.ui` imports only public `undertow.sim` / `undertow.data`. No reverse import.
- **`from __future__ import annotations`** in every Python file; type hints on all public functions.

---

## 1. Telemetry contract (U00) — `undertow.sim.train.telemetry`

Added to `TrainingConfig` (additive, defaults preserve current behaviour except telemetry which is on):

```python
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    ...
    telemetry: bool = True            # write telemetry.jsonl + status.json per seed
    record_rollouts: bool = True      # write rollouts/*.parquet (U06; on by default)
    rollout_max_episodes: int = 5     # U06; only read when record_rollouts is True
```

Public API:

```python
# src/undertow/sim/train/telemetry.py
SCHEMA_VERSION: int = 1
TELEMETRY_FILENAME: str = "telemetry.jsonl"
STATUS_FILENAME: str = "status.json"

class TelemetryCallback(BaseCallback):
    """SB3 callback appending one TelemetryPoint per log_freq_steps and
    refreshing status.json. Also writes the terminal `done` status."""
    def __init__(self, run_id: str, seed: int, total_timesteps: int,
                 log_freq_steps: int, status_path: Path, telemetry_path: Path) -> None: ...

def write_status(path: Path, status: RunStatus) -> None:
    """Atomically write status.json."""

@dataclass(frozen=True, slots=True)
class RunStatus:
    schema_version: int
    run_id: str
    seed: int
    state: Literal["running", "done", "failed"]
    timestep: int
    total_timesteps: int
    updated_at: datetime
    pid: int
    error: str | None = None
```

### 1.1 `telemetry.jsonl` — one point per line

```jsonc
{
  "schema_version": 1,
  "run_id": "ppo_s0_ab12cd34ef56aa77",
  "seed": 0,
  "timestep": 20000,
  "wall_time_s": 18.42,
  "ep_rew_mean": -0.0512,          // null until the first episode finishes
  "ep_len_mean": 4320.0,           // null until known
  "episodes": 12,                  // null until known
  "train": {                       // any subset; missing keys are null
    "policy_loss": 0.0123,
    "value_loss": 0.2011,
    "entropy_loss": -1.2010,
    "approx_kl": 0.0081,
    "clip_fraction": 0.121,
    "explained_variance": 0.402,
    "learning_rate": 0.0003,
    "std": 0.501,
    "n_updates": 10
  },
  "time": {
    "fps": 128.4,
    "iterations": 5,
    "time_elapsed": 18.42
  }
}
```

Rules: `schema_version`, `run_id`, `seed`, `timestep` are always present. Everything else is
nullable. Keys under `train`/`time` come from SB3's `model.logger.name_to_value` with the
`train/` / `time/` prefix stripped; unknown keys are permitted and preserved.

### 1.2 `status.json`

```jsonc
{
  "schema_version": 1,
  "run_id": "ppo_s0_ab12cd34ef56aa77",
  "seed": 0,
  "state": "running",
  "timestep": 20000,
  "total_timesteps": 1000000,
  "updated_at": "2026-10-19T18:12:00Z",
  "pid": 51234,
  "error": null
}
```

`state` transitions `running → done` (normal finish) or `running → failed` (exception; `error` holds
the message). The reader treats a missing/stale `status.json` as `unknown`, never as `failed`.

---

## 2. Rollout contract (U06) — `undertow.sim.train.rollouts`

```python
# src/undertow/sim/train/rollouts.py
ROLLOUT_DIRNAME: str = "rollouts"
SCHEMA_VERSION: int = 1

class RolloutRecorder:
    """Records env `info` + observation for up to `max_episodes` evaluation episodes
    of a trained policy, writing one parquet per episode. Evaluation is offline and
    deterministic (`deterministic=True`); it never perturbs training RNG."""
    def __init__(self, run_id: str, seed: int, out_dir: Path, max_episodes: int = 5) -> None: ...
    def record_episode(self, episode_index: int, rows: Sequence[RolloutRow]) -> Path: ...
    def close(self) -> None: ...
```

### 2.1 `rollouts/episode_<k>.parquet` schema (exact column names/dtypes)

| Column | dtype | Notes |
|---|---|---|
| `schema_version` | `Int16` | 1 |
| `run_id` | `Utf8` | |
| `seed` | `Int64` | |
| `episode` | `Int32` | 0-based |
| `step` | `Int32` | 0-based decision index |
| `timestamp` | `Datetime("us", "UTC")` | decision time |
| `price` | `Float64` | human USDC/WETH |
| `tick` | `Int64` | raw |
| `tick_lower` | `Int64` | current position |
| `tick_upper` | `Int64` | current position |
| `in_range` | `Boolean` | `tick_lower <= tick < tick_upper` |
| `position_value` | `Float64` | USDC |
| `equity` | `Float64` | USDC |
| `action_index` | `Int32` | discrete env action |
| `action_type` | `Utf8` | `hold` \| `rebalance` \| `rebalance_rejected` (did not execute: unaffordable or market exhausted) |
| `center_offset` | `Int32` | tick-spacings (0 for hold) |
| `width` | `Int32` | tick-spacings (0 for hold) |
| `fees` | `Float64` | step fee income, >= 0 |
| `gas` | `Float64` | step gas, <= 0 |
| `slippage` | `Float64` | step slippage, <= 0 |
| `il` | `Float64` | signed step IL change |
| `risk_penalty` | `Float64` | <= 0 |
| `reward` | `Float64` | normalised env reward |
| `regime` | `Utf8` | as-of regime label |
| `gas_price_wei` | `Float64` | recent base + priority |
| `pool_active_liquidity` | `Float64` | pool raw `L` (ADR-028 §2.1 extension) |
| `position_liquidity` | `Float64` | agent position raw `L` |
| `pool_fee_earned_0` | `Float64` | pool fees generated this step, token0 (raw) |
| `pool_fee_earned_1` | `Float64` | pool fees generated this step, token1 (raw) |

Empty episodes are invalid: `record_episode` raises `ValueError` on an empty row sequence.

---

## 3. `undertow.ui` package surface (U01)

```python
# src/undertow/ui/__init__.py
__all__ = ["UiConfig"]   # U01 starts here; U05 appends "create_app", "serve"

# src/undertow/ui/config.py
@dataclass(frozen=True, slots=True)
class UiConfig:
    runs_dir: Path = Path("runs")
    results_dir: Path = Path("results")
    host: str = "127.0.0.1"
    port: int = 8050
    telemetry_poll_seconds: float = 1.0
    max_points: int = 2000
    cors_origins: tuple[str, ...] = ()     # empty = same-origin only (dev uses the Vite proxy)
    web_dist: Path | None = None           # built web/dist; None = API only
```

CLI (`undertow-ui`):

```
undertow-ui info   [--runs DIR]
undertow-ui serve  [--runs DIR] [--results DIR] [--host H] [--port P] [--web-dist DIR] [--reload]
```

`info` prints the resolved config + discovered run count and exits `0`. `serve` starts uvicorn.
Exit codes match `undertow-sim`: `0` ok, `1` runtime error, `2` config error.

---

## 4. Wire models (U02) — `undertow/ui/schemas.py`

Pydantic v2 models. These are the OpenAPI source and the generated TypeScript source.

```python
class TrainMetrics(BaseModel):
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy_loss: float | None = None
    approx_kl: float | None = None
    clip_fraction: float | None = None
    explained_variance: float | None = None
    learning_rate: float | None = None
    std: float | None = None
    n_updates: float | None = None

class TimeMetrics(BaseModel):
    fps: float | None = None
    iterations: float | None = None
    time_elapsed: float | None = None

class TelemetryPoint(BaseModel):
    schema_version: int = 1
    run_id: str
    seed: int
    timestep: int
    wall_time_s: float | None = None
    ep_rew_mean: float | None = None
    ep_len_mean: float | None = None
    episodes: int | None = None
    train: TrainMetrics = TrainMetrics()
    time: TimeMetrics = TimeMetrics()

class RunStatusModel(BaseModel):
    """Mirrors `undertow.sim.train.telemetry.RunStatus` (CONTRACTS §1); not imported from sim."""
    schema_version: int = 1
    run_id: str
    seed: int
    state: Literal["running", "done", "failed"]
    timestep: int
    total_timesteps: int
    updated_at: datetime
    pid: int
    error: str | None = None

class EvalMetrics(BaseModel):
    """Parsed from a results/ directory when present."""
    sharpe: float | None = None
    total_pnl: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    worst_regime_pnl: float | None = None
    per_regime: dict[str, float] = {}      # regime -> mean total_pnl
    source: str | None = None              # relative path the numbers came from

class RunManifestSummary(BaseModel):
    run_id: str
    seed: int
    config_hash: str
    git_commit: str
    algorithm: str
    total_timesteps: int
    best_reward: float
    checkpoint_path: str
    library_versions: dict[str, str] = {}

class RunRecord(BaseModel):
    run_id: str
    experiment: str                 # parent directory name, e.g. "experiment_1"
    seed: int
    state: Literal["running", "done", "failed", "unknown"]
    updated_at: datetime
    total_timesteps: int
    latest_timestep: int
    best_reward: float | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    manifest: RunManifestSummary | None = None
    eval: EvalMetrics | None = None
    has_telemetry: bool = False
    has_rollouts: bool = False
    telemetry_points: int = 0

class LearningCurve(BaseModel):
    run_id: str
    timesteps: list[int]
    mean: list[float]
    std: list[float]
    per_seed: dict[str, list[float]]      # seed str -> values aligned to `timesteps`
    downsampled: bool = False

class RolloutSummary(BaseModel):
    seed: int
    episode: int
    n_steps: int
    regime: str
    start: datetime
    end: datetime

class RolloutSeries(BaseModel):
    run_id: str
    seed: int
    episode: int
    columns: list[str]
    rows: list[dict[str, object]]          # already downsampled to <= max_points
    downsampled: bool = False
```

`undertow-sim evaluate` additionally writes a policy-keyed
`summary.json` (`{"policies": {name: {"sharpe": float, "total_pnl": float,
"annualized_return": float, "max_drawdown": float}}}`); the UI reads it into
`EvalMetrics` when a run's eval artifacts exist. Its schema is versioned by
`schema_version` like every other sidecar.

`RunStatusModel` mirrors the `RunStatus` schema of §1 **field-for-field** but is defined here (not
imported from `undertow.sim`) so the `ui` extra never pulls in the training stack. Readers parse
`status.json` into it directly.

---

## 5. Streaming contract (U03)

### 5.1 REST tail (also the polling fallback)

`GET /api/runs/{run_id}/telemetry?since=<offset>&limit=<n>` returns:

```json
{
  "run_id": "ppo_s0_ab12cd34ef56aa77",
  "status": { /* RunStatus */ },
  "points": [ /* TelemetryPoint[] */ ],
  "next_offset": 128,
  "done": false
}
```

`since` is the **line offset** (not timestep) so gaps and duplicates are impossible; the client
stores `next_offset`. `limit` defaults to 1000.

### 5.2 WebSocket `GET /api/stream?run_id=<id>`

Text frames, one JSON object each. Envelope is always `{"type": <str>, "run_id": <str|null>, "data": <obj|null>, "ts": <iso8601>}`.

| `type` | `data` | Consumer fallback |
|---|---|---|
| `hello` | `{"server_time": ..., "runs": [RunRecord, ...]}` | initial full refresh |
| `telemetry` | `TelemetryPoint` | append to the live buffer |
| `status` | `RunStatus` | update the badge; stop tailing on `done`/`failed` |
| `run_added` | `RunRecord` | insert in the run list |
| `error` | `{"message": str}` | show a toast; keep the last good state |

Rules: the server sends `hello` immediately on connect, then deltas. Only the requested `run_id` is
streamed (`run_id` omitted = server status only). Heartbeat ping every 20 s. On `done`/`failed` the
hub sends the final `status` and stops the tail task.

---

## 6. Figure spec contract (U04) — `undertow/ui/figures/`

Every builder is **pure**: it takes plain data (dicts/lists per §1–§4, never a reader object) and
returns a JSON-serialisable Plotly figure dict with the shape:

```python
FigureSpec = TypedDict("FigureSpec", {
    "data": list[dict[str, object]],
    "layout": dict[str, object],
    "config": dict[str, object],   # displayModeBar etc.
})
```

Required builders (signatures frozen):

```python
def learning_curve_figure(curve: Mapping[str, object], *, log_y: bool = False) -> FigureSpec: ...
def diagnostics_figure(points: Sequence[Mapping[str, object]], metric: str) -> FigureSpec: ...
def action_heatmap_figure(counts: Mapping[tuple[int, int], float],
                          *, offsets: Sequence[int], widths: Sequence[int],
                          title: str = "Action distribution") -> FigureSpec: ...
def ribbon_figure(rows: Sequence[Mapping[str, object]], *, show_hodl: bool = True) -> FigureSpec: ...
def leaderboard_figure(records: Sequence[Mapping[str, object]], *, metric: str) -> FigureSpec: ...
def compare_equity_figure(series: Mapping[str, Sequence[float]]) -> FigureSpec: ...
```

`theme.py` exposes `DARK_TEMPLATE: dict` and `apply_theme(fig: dict) -> dict` built from the §9
tokens; every builder MUST apply it. All figures set `layout.template = "undertow_dark"` (registered
at import). Charts never rely on color alone: line traces set `name`, heatmaps set a colorbar title,
ribbons carry a legend entry.

---

## 7. REST API contract (U05)

Base path `/api`. All responses JSON (UTC datetimes as ISO strings). Errors use FastAPI's
`{"detail": ...}`.

| Method | Path | Query/body | Returns |
|---|---|---|---|
| `GET` | `/api/health` | — | `{"status":"ok","schema_version":1}` |
| `GET` | `/api/runs` | `?state=` optional filter | `RunRecord[]` |
| `GET` | `/api/runs/{run_id}` | — | `RunRecord` (404 if absent) |
| `GET` | `/api/runs/{run_id}/status` | — | `RunStatus` |
| `GET` | `/api/runs/{run_id}/telemetry` | `since`, `limit` | §5.1 payload |
| `GET` | `/api/runs/{run_id}/curve` | — | `LearningCurve` |
| `GET` | `/api/runs/{run_id}/figures/{name}` | `seed?`, `episode?`, `log_y?` | `FigureSpec` |
| `GET` | `/api/runs/{run_id}/rollouts` | — | `RolloutSummary[]` |
| `GET` | `/api/rollouts/{run_id}/{seed}/{episode}` | `max_points?` | `RolloutSeries` |
| `POST` | `/api/compare` | `{"run_ids":[str,...],"metric?":str}` | `{"figures": {...}, "records": RunRecord[]}` |
| `WS` | `/api/stream` | `run_id?` | §5.2 |

`name` ∈ the figure registry (U04): `learning`, `diagnostics`, `actions`, `ribbon`, `leaderboard`.
Unknown name → `404`. `run_id` is the manifest `run_id`; the registry also accepts a
`<experiment>/seed_<n>` path segment via the `experiment` field if a `run_id` collides.

Static assets: when `UiConfig.web_dist` is set, mount it at `/` **after** the API routes, with an SPA
fallback (`index.html`) for non-`/api` paths. When unset, `/` returns a JSON hint telling the
developer to run the Vite dev server.

---

## 8. Frontend contract (U07–U10)

- **Routes:** `/` → redirect `/live`; `/live` → Live Cockpit; `/replay` → Policy Replay;
  `/leaderboard` → Leaderboard; `/compare` → Compare. Query params carry selection
  (`?run_id=…&seed=…&episode=…`) so views are shareable/restorable.
- **API client:** generated from `GET /openapi.json` via `openapi-typescript` (U07). A thin
  `api/client.ts` wraps `fetch` with the generated types; `api/useLiveStream.ts` owns the WebSocket
  and reconnects with exponential backoff (max 30 s), falling back to the REST tail.
- **State:** TanStack Query for REST caches (keyed by run/query), a small Zustand store for the live
  buffer, pinned comparison set, and the shared time cursor. No prop-drilling of raw series.
- **Live merge rule:** telemetry deltas append to the buffer keyed by `(run_id, timestep)`; duplicates
  are ignored; the chart re-renders only the last window plus a full re-render every N points.
- **Empty/error states:** every panel renders `EmptyState` (no runs / no telemetry / no rollouts) or
  an inline error with the failing path — never a blank canvas.
- **Perf:** memoise figure payloads by JSON signature; `PlotlyFigure` calls `Plotly.react` directly and preserves view state via the layout's `uirevision` (ADR-028).

---

## 9. Design tokens (U07 owns implementation; U04 imports the same values)

Dark theme, single source of truth (exported as CSS variables in `web/src/theme/tokens.css` and as a
Python dict `undertow/ui/figures/theme.py:TOKENS`):

| Token | Value | Use |
|---|---|---|
| `bg-base` | `#0B0E14` | page background |
| `bg-surface` | `#11161F` | panels/cards |
| `bg-elevated` | `#161B26` | popovers, hover |
| `border` | `#232A38` | 1px hairlines |
| `text-primary` | `#E6EAF2` | body |
| `text-muted` | `#8B93A7` | secondary |
| `grid` | `#1A2130` | plot gridlines |
| `accent` | `#22D3EE` | agent / primary series |
| `accent-2` | `#A78BFA` | checkpoints / pinned policies |
| `bull` | `#34D399` | bull regime, positive |
| `bear` | `#F87171` | bear regime, negative |
| `sideways` | `#94A3B8` | sideways regime |
| `high-vol` | `#FBBF24` | high-vol regime |
| `hold` | `#64748B` | hold action |
| `rebalance` | `#38BDF8` | rebalance action |

Typography: UI = `Inter`; numerics = `JetBrains Mono` with `tabular-nums`. Plotly font family
`"Inter, system-ui, sans-serif"`, size 12; gridcolor `grid`; paper/plot bg `bg-surface`. No shadows
heavier than `0 1px 0 rgba(255,255,255,0.03)`; radii 8px (panels) / 6px (controls).
