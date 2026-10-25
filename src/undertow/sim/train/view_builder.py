"""Build the training ``MarketView`` once, in a short-lived subprocess.

PR #103 stopped every ``SubprocVecEnv`` worker from rebuilding the multi-GB
view by publishing it once (as memory-mappable Arrow IPC) and pointing the
pickled factories at the artifact.  The *parent* still built it in-process,
which leaves the build's arena resident for the whole run: measured on the
primary lane, RSS sat at ~13 GB after the load while the live view was ~1.5 GB,
and most of the difference is jemalloc pages that polars' statically linked
allocator never returns (Arrow's half can be purged, polars' cannot).

This module moves the one-off build into a ``spawn`` subprocess.  The child
loads the parquet streams, fits the calibrated process when configured, writes
the shared-view IPC directory (plus a small params sidecar) and exits, taking
its arenas with it.  The parent then only ever maps the artifact, so its
resident cost is the mapped pages plus Python — no build peak, no arena.

The artifact contract is exactly :mod:`undertow.sim.train.envs`': frames in
``share_view``, params in :data:`PARAMS_FILENAME`.  A missing sidecar means
replay mode (the child always removes a stale one before publishing).
"""

from __future__ import annotations

import multiprocessing
import pickle
from pathlib import Path

from undertow.sim.config import SimConfig
from undertow.sim.marketview import MarketView, load_market_view
from undertow.sim.prices.regime_jump import MRSJDParams
from undertow.sim.train.envs import fit_calibrated, share_view
from undertow.sim.types import PriceMode, UndertowSimError

__all__ = [
    "PARAMS_FILENAME",
    "ViewBuilderError",
    "build_shared_view",
    "read_builder_params",
    "spawn_view_builder",
]

#: Sidecar written next to the IPC frames when the price process is calibrated.
PARAMS_FILENAME = "train_params.pkl"


class ViewBuilderError(UndertowSimError):
    """Raised when the builder subprocess fails to publish the shared view."""


def _publish_view(view: MarketView, config: SimConfig, directory: Path) -> None:
    """Write ``view``'s IPC frames (and calibrated params) under ``directory``.

    A stale params sidecar is removed even in replay mode, otherwise the parent
    would reconstruct a calibrated factory from a previous run's fit.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    params_path = directory / PARAMS_FILENAME
    if params_path.exists():
        params_path.unlink()
    if config.price_mode is PriceMode.CALIBRATED:
        params, initial_price = fit_calibrated(view)
        params_path.write_bytes(
            pickle.dumps({"mrsjd_params": params, "initial_price": initial_price})
        )
    share_view(view, directory)


def build_shared_view(
    config: SimConfig,
    data_config_path: Path,
    split: str,
    directory: Path,
) -> None:
    """Subprocess entry point: load the view once and publish it, then exit.

    ``config`` travels as a pickle through ``spawn`` (it is plain dataclasses),
    so the child needs no TOML parsing and the parent needs no view.
    """
    view = load_market_view(
        Path(data_config_path),
        config.split,
        split=split,
        episode_config=config.episode,
    )
    _publish_view(view, config, Path(directory))


def _spawn(args: tuple[SimConfig, Path, str, Path]) -> int:
    """Run :func:`build_shared_view` in a fresh interpreter; return its exit code.

    ``spawn`` (not ``fork``) is deliberate: the child gets a clean allocator and
    thread state rather than inheriting polars/torch pools from the parent.
    """
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=build_shared_view, args=args)
    process.start()
    process.join()
    return int(process.exitcode or 0)


def spawn_view_builder(
    config: SimConfig,
    data_config_path: Path,
    split: str,
    directory: Path,
) -> Path:
    """Build the shared view out-of-process and return its directory.

    Raises :class:`ViewBuilderError` on a non-zero child exit; the child's own
    traceback is on stderr.  The caller owns the directory's lifetime.
    """
    code = _spawn((config, Path(data_config_path), split, Path(directory)))
    if code != 0:
        raise ViewBuilderError(
            f"market-view builder subprocess exited with code {code}; "
            "its traceback on stderr names the cause"
        )
    return Path(directory)


def read_builder_params(directory: Path) -> tuple[MRSJDParams | None, float | None]:
    """Read the calibrated params sidecar; ``(None, None)`` for replay mode."""
    path = Path(directory) / PARAMS_FILENAME
    if not path.is_file():
        return None, None
    try:
        payload = pickle.loads(path.read_bytes())
    except (pickle.UnpicklingError, EOFError) as exc:
        raise ViewBuilderError(f"{path} is not a readable builder sidecar: {exc}") from exc
    if not isinstance(payload, dict):
        raise ViewBuilderError(f"{path} does not carry a builder payload")
    params = payload.get("mrsjd_params")
    initial_price = payload.get("initial_price")
    if not isinstance(params, MRSJDParams):
        raise ViewBuilderError(
            f"{path} does not carry MRSJDParams; refusing to train with a "
            "corrupt builder artifact"
        )
    if initial_price is None:
        return params, None
    if isinstance(initial_price, bool) or not isinstance(initial_price, (int, float)):
        raise ViewBuilderError(f"{path} carries a non-numeric initial_price")
    return params, float(initial_price)
