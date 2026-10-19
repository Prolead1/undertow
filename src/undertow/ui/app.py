"""FastAPI application factory for the ``undertow.ui`` dashboard (U05, CONTRACTS §5/§7).

``create_app`` is the composition root that turns the artifact readers
(:mod:`undertow.ui.registry`, :mod:`undertow.ui.telemetry`,
:mod:`undertow.ui.rollouts`) and the pure figure builders
(:mod:`undertow.ui.figures`) into the frozen REST + WebSocket surface.  It is a
**read-only** consumer of a ``runs/`` tree (`docs/plans/ui/PLAN.md` §0.6/§0.7):
no route writes into a run directory and nothing here imports ``undertow.sim`` or
``undertow.data``.

Design notes
------------
* **Scan once, refresh lazily.**  The :class:`RunRegistry` is scanned once when
  the app is built; subsequent requests reuse the in-memory index and only
  re-scan after a short TTL (or an explicit ``?refresh=true``), never per
  request.
* **Sync handlers.**  Every REST handler is a plain ``def`` so FastAPI runs it in
  the threadpool, keeping parquet/JSON reads off the event loop.  Only the
  WebSocket route is ``async`` and it offloads its blocking reads explicitly.
* **No chart logic.**  ``GET .../figures/{name}`` delegates data assembly to
  :mod:`undertow.ui.assembly`, which calls the pure builder.
* **Split by responsibility.**  WebSocket plumbing lives in
  :mod:`undertow.ui.ws` and the SPA mount in :mod:`undertow.ui.static`; this
  module is route wiring plus the U05 wire models.

``CONTRACTS.md`` §7 is frozen; any extension below (the optional ``metric``
query on the figure route, the ``refresh`` query on the run list, the heartbeat
``ping`` and ``run_added`` frames) is additive and documented in ADR-025.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from undertow.ui.assembly import (
    UnknownMetricError,
    build_compare_figure,
    build_figure,
    rank_compare,
)
from undertow.ui.config import UiConfig, UiConfigError
from undertow.ui.figures import FIGURE_REGISTRY
from undertow.ui.paths import RunDir
from undertow.ui.registry import RunRegistry
from undertow.ui.rollouts import list_rollouts, read_rollout
from undertow.ui.schemas import (
    LearningCurve,
    RolloutSeries,
    RolloutSummary,
    RunRecord,
    RunStatusModel,
    TelemetryPoint,
)
from undertow.ui.static import mount_spa
from undertow.ui.streaming import TERMINAL_STATES, build_frame
from undertow.ui.telemetry import read_points, read_status
from undertow.ui.ws import TelemetryHubs, heartbeat_loop, now_iso, pump_frames

__all__ = [
    "CONFIG_ENV_VAR",
    "config_payload",
    "create_app",
    "create_app_from_env",
]

logger = logging.getLogger("undertow.ui.app")

#: An index older than this is re-scanned on the next request (never per-request).
_REGISTRY_TTL_SECONDS: float = 2.0

#: Environment variable carrying the serialised :class:`UiConfig` for
#: ``uvicorn --reload``.  A reloading worker is a fresh interpreter, so the
#: config cannot be passed as an app instance.
CONFIG_ENV_VAR = "UNDERTOW_UI_CONFIG_JSON"

#: Filter values accepted by ``GET /api/runs?state=``.
RunState = Literal["running", "done", "failed", "unknown"]


# ---------------------------------------------------------------------------
# Wire models not already owned by ``schemas.py`` (U02)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """``GET /api/health`` (CONTRACTS §7)."""

    status: Literal["ok"] = "ok"
    schema_version: int = 1


class TelemetryTail(BaseModel):
    """``GET /api/runs/{run_id}/telemetry`` payload (CONTRACTS §5.1)."""

    run_id: str
    status: RunStatusModel | None = None
    points: list[TelemetryPoint]
    next_offset: int
    done: bool = False


class FigureSpecModel(BaseModel):
    """JSON-serialisable Plotly spec returned by every figure route (§6)."""

    data: list[dict[str, Any]]
    layout: dict[str, Any]
    config: dict[str, Any]


class CompareRequest(BaseModel):
    """``POST /api/compare`` body (CONTRACTS §7)."""

    run_ids: list[str] = Field(min_length=1, max_length=6)
    metric: str = "sharpe"


class CompareResponse(BaseModel):
    """``POST /api/compare`` response (CONTRACTS §7)."""

    figures: dict[str, FigureSpecModel]
    records: list[RunRecord]


class RootHint(BaseModel):
    """``GET /`` when no built frontend is mounted."""

    message: str
    openapi: str = "/openapi.json"


# ---------------------------------------------------------------------------
# App-state helpers
# ---------------------------------------------------------------------------
def _refresh_registry(app: FastAPI, *, force: bool = False) -> None:
    """Re-scan the registry when stale, or when explicitly forced.

    A short TTL keeps a live training run visible without thrashing the disk; a
    per-request scan is never performed (CONTRACTS §7, PLAN §4).  The scan
    returns both records and run directories in one pass, so a resolved run id
    can never point at two different directories.
    """
    state = app.state
    with state.registry_lock:
        if not force and time.monotonic() - state.registry_scanned_at < _REGISTRY_TTL_SECONDS:
            return
        state.records = state.registry.refresh()
        state.registry_scanned_at = time.monotonic()


def _require_run(app: FastAPI, run_id: str) -> tuple[RunRecord, RunDir]:
    """Resolve a run id to its record and directory, or raise ``404``."""
    resolved = app.state.registry.resolve(run_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return resolved


# ---------------------------------------------------------------------------
# Config serialisation for ``uvicorn --reload``
# ---------------------------------------------------------------------------
def config_payload(config: UiConfig) -> dict[str, object]:
    """Serialise a :class:`UiConfig` for :data:`CONFIG_ENV_VAR`."""
    return {
        "runs_dir": str(config.runs_dir),
        "results_dir": str(config.results_dir),
        "host": config.host,
        "port": config.port,
        "telemetry_poll_seconds": config.telemetry_poll_seconds,
        "max_points": config.max_points,
        "cors_origins": list(config.cors_origins),
        "web_dist": str(config.web_dist) if config.web_dist is not None else None,
    }


def create_app_from_env() -> FastAPI:
    """Build an app from :data:`CONFIG_ENV_VAR` (used by ``--reload``).

    A uvicorn reload worker is a fresh process, so it cannot receive the app
    instance.  :func:`undertow.ui.cli.serve` serialises the resolved
    :class:`UiConfig` into the environment and points uvicorn at this factory.
    A malformed payload raises :class:`UiConfigError` rather than leaking a
    ``KeyError``/``JSONDecodeError`` from a worker process.
    """
    raw = os.environ.get(CONFIG_ENV_VAR)
    if not raw:
        return create_app(UiConfig())
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise UiConfigError(f"invalid {CONFIG_ENV_VAR}: not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UiConfigError(f"invalid {CONFIG_ENV_VAR}: expected a JSON object")
    try:
        config = UiConfig(
            runs_dir=Path(payload["runs_dir"]),
            results_dir=Path(payload["results_dir"]),
            host=str(payload["host"]),
            port=int(payload["port"]),
            telemetry_poll_seconds=float(payload["telemetry_poll_seconds"]),
            max_points=int(payload["max_points"]),
            cors_origins=tuple(payload.get("cors_origins") or ()),
            web_dist=Path(payload["web_dist"]) if payload.get("web_dist") else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UiConfigError(f"invalid {CONFIG_ENV_VAR}: {exc}") from exc
    return create_app(config)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(config: UiConfig) -> FastAPI:
    """Compose the FastAPI app implementing CONTRACTS §7 over ``config``.

    The registry is scanned once here; routes refresh it lazily.  A per-run
    telemetry hub is created on first WebSocket subscribe and closed/evicted
    when its last subscriber disconnects (or on shutdown).
    """
    hubs = TelemetryHubs(config.telemetry_poll_seconds)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        await application.state.hubs.close()

    app = FastAPI(
        title="Undertow UI",
        version="1.0.0",
        description="Read-only live cockpit for Undertow RL training runs.",
        lifespan=lifespan,
    )

    registry = RunRegistry(config.runs_dir, config.results_dir, config.max_points)
    app.state.config = config
    app.state.registry = registry
    app.state.records = registry.scan()
    app.state.hubs = hubs
    app.state.registry_lock = threading.Lock()
    app.state.registry_scanned_at = time.monotonic()

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

    # -- system ----------------------------------------------------------
    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse()

    # -- runs ------------------------------------------------------------
    @app.get("/api/runs", response_model=list[RunRecord], tags=["runs"])
    def list_runs(
        state: Annotated[
            RunState | None, Query(description="only runs in this state")
        ] = None,
        refresh: bool = Query(default=False, description="force a registry re-scan"),
    ) -> list[RunRecord]:
        _refresh_registry(app, force=refresh)
        runs = list(app.state.records)
        if state is not None:
            runs = [record for record in runs if record.state == state]
        return runs

    # Suffix routes use ``{run_id:path}`` so a colliding manifest run_id can be
    # disambiguated by the CONTRACTS §7 ``<experiment>/seed_<n>`` path, whether
    # the client sends it multi-segment or URL-encoded.  They must be registered
    # before the plain ``/api/runs/{run_id}`` detail route.
    @app.get(
        "/api/runs/{run_id:path}/status", response_model=RunStatusModel, tags=["runs"]
    )
    def get_status(run_id: str) -> RunStatusModel:
        _refresh_registry(app)
        _, run_dir = _require_run(app, run_id)
        status = read_status(run_dir.status_path) if run_dir.status_path else None
        if status is None:
            raise HTTPException(status_code=404, detail=f"no status.json for run {run_id!r}")
        return status

    @app.get(
        "/api/runs/{run_id:path}/telemetry", response_model=TelemetryTail, tags=["telemetry"]
    )
    def get_telemetry(
        run_id: str,
        since: int = Query(default=0, ge=0, description="telemetry line offset"),
        limit: int = Query(default=1000, ge=0, description="maximum points to return"),
    ) -> TelemetryTail:
        _refresh_registry(app)
        record, run_dir = _require_run(app, run_id)
        if run_dir.telemetry_path is None:
            points: list[TelemetryPoint] = []
            next_offset = since
        else:
            points, next_offset = read_points(
                run_dir.telemetry_path, since=since, limit=limit
            )
        status = read_status(run_dir.status_path) if run_dir.status_path else None
        done = status is not None and status.state in TERMINAL_STATES
        return TelemetryTail(
            run_id=record.run_id,
            status=status,
            points=points,
            next_offset=next_offset,
            done=done,
        )

    @app.get("/api/runs/{run_id:path}/curve", response_model=LearningCurve, tags=["runs"])
    def get_curve(run_id: str) -> LearningCurve:
        _refresh_registry(app)
        # Pass the *requested* id so an alias aggregates its own directory's
        # experiment; the route must not re-canonicalise by ``record.run_id``.
        # ``curve`` resolves one coherent snapshot itself, so a successful
        # response never pairs a directory from one scan with a curve from
        # another; the extra resolve is only paid on the error path below.
        curve = app.state.registry.curve(run_id)
        if curve is None:
            # Distinguish an unknown run from a run that simply has no curve.
            _require_run(app, run_id)
            raise HTTPException(
                status_code=404, detail=f"no learning curve for run {run_id!r}"
            )
        return curve

    @app.get(
        "/api/runs/{run_id:path}/figures/{name}",
        response_model=FigureSpecModel,
        tags=["figures"],
    )
    def get_figure(
        run_id: str,
        name: str,
        seed: int | None = Query(default=None, description="rollout seed"),
        episode: int | None = Query(default=None, ge=0, description="rollout episode"),
        log_y: bool = Query(default=False, description="log scale the value axis"),
        metric: str | None = Query(
            default=None, description="diagnostic/leaderboard metric"
        ),
    ) -> FigureSpecModel:
        if name not in FIGURE_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"unknown figure {name!r}; expected one of {sorted(FIGURE_REGISTRY)}",
            )
        _refresh_registry(app)
        record, run_dir = _require_run(app, run_id)
        try:
            spec = build_figure(
                name,
                run_id=run_id,
                record=record,
                run_dir=run_dir,
                registry=app.state.registry,
                records=app.state.records,
                seed=seed,
                episode=episode,
                log_y=log_y,
                metric=metric,
                max_points=config.max_points,
            )
        except UnknownMetricError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FigureSpecModel.model_validate(spec)

    @app.get(
        "/api/runs/{run_id:path}/rollouts",
        response_model=list[RolloutSummary],
        tags=["rollouts"],
    )
    def get_rollouts(run_id: str) -> list[RolloutSummary]:
        _refresh_registry(app)
        _, run_dir = _require_run(app, run_id)
        if run_dir.rollout_dir is None:
            return []
        return list(list_rollouts(run_dir.rollout_dir))

    @app.get("/api/runs/{run_id:path}", response_model=RunRecord, tags=["runs"])
    def get_run(run_id: str) -> RunRecord:
        _refresh_registry(app)
        record, _ = _require_run(app, run_id)
        return record

    @app.get(
        "/api/rollouts/{run_id:path}/{seed}/{episode}",
        response_model=RolloutSeries,
        tags=["rollouts"],
    )
    def get_rollout(
        run_id: str,
        seed: int,
        episode: int,
        max_points: int | None = Query(default=None, ge=1),
    ) -> RolloutSeries:
        _refresh_registry(app)
        record, run_dir = _require_run(app, run_id)
        if run_dir.rollout_dir is None:
            raise HTTPException(status_code=404, detail=f"no rollouts for run {run_id!r}")
        path = run_dir.rollout_dir / f"episode_{episode}.parquet"
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"rollout {record.run_id}/seed_{seed}/episode_{episode} not found",
            )
        try:
            series = read_rollout(path, max_points=max_points or config.max_points)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"{path}: {exc}") from exc
        if series.seed != seed:
            raise HTTPException(
                status_code=404,
                detail=f"rollout {path.name} belongs to seed {series.seed}, not {seed}",
            )
        return series

    # -- compare ---------------------------------------------------------
    @app.post("/api/compare", response_model=CompareResponse, tags=["compare"])
    def compare(request: CompareRequest) -> CompareResponse:
        _refresh_registry(app)
        runs: list[tuple[RunRecord, RunDir]] = []
        # De-dupe on the *resolved directory*, not the request string: a raw id
        # and one of its aliases can name the same directory, and keeping both
        # would put two ``records`` behind one overlay trace (and tie on
        # ``run_id`` in ``compare_series``).
        seen_paths: set[str] = set()
        for run_id in request.run_ids:
            resolved = app.state.registry.resolve(run_id)
            if resolved is None:
                raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
            path = str(resolved[1].path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            runs.append(resolved)
        try:
            ordered = rank_compare(
                app.state.registry, [record for record, _ in runs], request.metric
            )
        except UnknownMetricError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # ``runs`` can hold two distinct directories that share a manifest
        # ``run_id`` (the CONTRACTS §7 alias escape hatch).  ``rank`` returns the
        # same record objects, so map them back by identity rather than by
        # ``run_id`` — otherwise both aliases collapse onto one directory and the
        # overlay silently plots only one of them.
        runs_by_identity = {id(record): (record, run_dir) for record, run_dir in runs}
        ordered_runs = [runs_by_identity[id(record)] for record in ordered]
        figure = build_compare_figure(ordered_runs, max_points=config.max_points)
        return CompareResponse(
            figures={"compare": FigureSpecModel.model_validate(figure)},
            records=ordered,
        )

    # -- streaming -------------------------------------------------------
    @app.websocket("/api/stream")
    async def stream(websocket: WebSocket, run_id: str | None = None) -> None:
        await websocket.accept()
        # REST handlers are sync so FastAPI's threadpool offloads them; this
        # async route must offload its own blocking filesystem scan.  The hub is
        # also acquired *before* ``hello`` so its tail offset is fixed before a
        # client can append a line in response to connecting.
        await asyncio.to_thread(_refresh_registry, app)
        snapshot = list(app.state.records)

        resolved = app.state.registry.resolve(run_id) if run_id is not None else None
        hub = None
        if resolved is not None:
            hub = await asyncio.to_thread(app.state.hubs.acquire, *resolved)

        hello = build_frame(
            "hello",
            None,
            {
                "server_time": now_iso(),
                "runs": [record.model_dump(mode="json") for record in snapshot],
            },
        )
        await websocket.send_text(hello)

        if run_id is None:
            def refresh_records() -> list[RunRecord]:
                _refresh_registry(app)
                return list(app.state.records)

            await heartbeat_loop(
                websocket,
                run_id=None,
                refresh=refresh_records,
                known_run_ids={record.run_id for record in snapshot},
            )
            return

        if resolved is None or hub is None:
            await websocket.send_text(
                build_frame("error", run_id, {"message": f"unknown run {run_id!r}"})
            )
            await websocket.close()
            return

        record, run_dir = resolved
        active_hub = hub

        def start_tail() -> None:
            # Run once the pump has registered its subscriber queue, so the
            # initial status frame cannot be published into an empty fan-out.
            app.state.hubs.start_tail_if_idle(active_hub, record.run_id)

        try:
            await pump_frames(
                websocket,
                active_hub.subscribe(on_subscribe=start_tail),
                run_id=record.run_id,
            )
        finally:
            # Last subscriber: stop the tail and evict the idle hub.  The
            # disconnect itself may cancel this task; cleanup must still finish.
            # The lease key is the resolved directory, matching ``acquire``.
            with suppress(asyncio.CancelledError):
                await app.state.hubs.release(app.state.hubs.hub_key(run_dir))

    # -- static / root ---------------------------------------------------
    if config.web_dist is not None:
        mount_spa(app, config.web_dist)
    else:

        @app.get("/", response_model=RootHint, tags=["system"])
        def root() -> RootHint:
            return RootHint(
                message=(
                    "undertow.ui API is running. Set UiConfig.web_dist to mount the "
                    "built frontend, or run the Vite dev server under web/."
                )
            )

    return app
