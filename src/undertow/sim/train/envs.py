"""Picklable environment factories for process-parallel PPO training.

``stable_baselines3``'s ``SubprocVecEnv`` cloudpickles each factory into a worker
process.  A real training :class:`~undertow.sim.marketview.MarketView` is a
multi-gigabyte polars frame, so a factory that closed over it would ship a copy
to every worker and exhaust memory.  :class:`EnvFactory` therefore stores the
data-config path, the sim config and (when the parent has preloaded the view) a
**shared-view directory**: :meth:`EnvFactory.publish_shared_view` writes each
frame once as an uncompressed Arrow IPC file and workers reopen it with
``memory_map=True``.  The mapped pages are file-backed and shared through the
page cache, so four workers cost one copy of the data instead of four eager
rebuilds of the event tape (each of which peaks at many gigabytes of join
intermediates on the primary 0.05 % lane).  A factory without a shared directory
falls back to rebuilding the view from disk, and
:meth:`EnvFactory.__getstate__` always drops the in-process frame so a pickled
factory never carries it through a pipe.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import polars as pl

from undertow.sim.config import SimConfig
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.evaluate.protocol import build_replay_env
from undertow.sim.marketview import MarketView, load_market_view
from undertow.sim.prices import CalibratedPriceProcess
from undertow.sim.prices.regime_jump import MRSJDParams
from undertow.sim.types import PriceMode, SimConfigError

__all__ = ["EnvFactory", "fit_calibrated", "load_shared_view", "share_view"]

#: ``MarketView`` frame fields persisted by :func:`share_view`.
_VIEW_FRAMES: tuple[str, ...] = (
    "train",
    "eval",
    "reference",
    "regimes",
    "gas",
    "pool_state",
    "warmup",
    "funding",
)

#: Sidecar holding the view's non-frame fields (split bounds, config dataclasses).
_VIEW_META_FILENAME = "view_meta.pkl"

#: ``MarketView`` non-frame fields persisted in the sidecar.
_VIEW_META_FIELDS: tuple[str, ...] = (
    "train_start_utc",
    "train_end_utc",
    "eval_start_utc",
    "eval_end_utc",
    "is_train",
    "episode_config",
    "pool",
)


def _view_meta(view: MarketView) -> dict[str, Any]:
    """Non-frame ``MarketView`` fields persisted next to the IPC frames."""
    return {name: getattr(view, name) for name in _VIEW_META_FIELDS}


def share_view(view: MarketView, directory: Path) -> Path:
    """Persist ``view`` under ``directory`` for zero-copy reuse by workers.

    Uncompressed Arrow IPC is used deliberately: it is the only format polars can
    memory-map without copying, which is what lets subprocess workers share the
    pages through the page cache.  ``directory`` must not be in use by workers
    yet — stale frame files are removed so a reused directory cannot leak frames
    from an earlier view.
    """
    covered = set(_VIEW_FRAMES) | set(_VIEW_META_FIELDS)
    expected = {f.name for f in fields(MarketView) if f.init}
    if covered != expected:
        # Fail loudly instead of silently dropping a field workers would then
        # reconstruct with a default — a new MarketView field must be persisted.
        raise SimConfigError(
            f"share_view: MarketView field(s) {sorted(expected - covered)} are not "
            "persisted; update _VIEW_FRAMES/_VIEW_META_FIELDS"
        )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.arrow"):
        stale.unlink()
    meta_path = directory / _VIEW_META_FILENAME
    if meta_path.exists():
        meta_path.unlink()

    for name in _VIEW_FRAMES:
        frame = getattr(view, name)
        if frame is not None:
            frame.write_ipc(directory / f"{name}.arrow")
    meta_path.write_bytes(pickle.dumps(_view_meta(view)))
    return directory


def load_shared_view(directory: Path) -> MarketView:
    """Reconstruct a view written by :func:`share_view` with zero-copy IPC maps.

    Raises :class:`SimConfigError` when the sidecar is missing (the directory was
    deleted or never published) rather than silently rebuilding from disk, which
    would hide a lifecycle bug in the caller.
    """
    directory = Path(directory)
    meta_path = directory / _VIEW_META_FILENAME
    if not meta_path.is_file():
        raise SimConfigError(
            f"shared market view is missing at {directory}; it must outlive every "
            "SubprocVecEnv worker (publish it before training starts)"
        )
    meta: dict[str, Any] = pickle.loads(meta_path.read_bytes())
    frames: dict[str, Any] = {}
    for name in _VIEW_FRAMES:
        path = directory / f"{name}.arrow"
        if path.is_file():
            frames[name] = pl.read_ipc(path, memory_map=True)
    # ``train``/``eval`` are required constructor arguments; exactly one is
    # non-None, and the absent one is written as no file, so default it back to
    # None explicitly.
    for split_frame in ("train", "eval"):
        frames.setdefault(split_frame, None)
    return MarketView(**meta, **frames)


def fit_calibrated(market_view: MarketView) -> tuple[MRSJDParams, float]:
    """Fit the calibrated MRSJD on ``market_view`` and resolve its entry price.

    The caller must reject an eval view first (fitting on eval data would be
    look-ahead); the CLI does that and reports it as a config error.  Returns the
    fitted parameters and the first tape row's as-of price.
    """
    params = CalibratedPriceProcess.fit(market_view)
    # ``active_split_data`` already guarantees seq order (it sorts only when a
    # frame is not sorted), so an explicit sort here was a full tape copy.
    tape = market_view.active_split_data()
    first_row = tape.row(0, named=True)
    initial_price = first_row.get("price_reference") or first_row.get("price_pool")
    if initial_price is None:
        initial_price = market_view.reference_close(first_row["block_timestamp"])
    if initial_price is None:
        raise SimConfigError("could not resolve an initial price for the MRSJD process")
    return params, float(initial_price)


@dataclass
class EnvFactory:
    """Zero-argument ``LpEnvironment`` factory shared across worker processes.

    ``_view`` is an in-process cache that the CLI sets so a single-process run
    does not reload the snapshot.  It is excluded from pickles, so every
    ``SubprocVecEnv`` worker rebuilds the view from ``data_config_path`` unless
    the parent has published one with :meth:`publish_shared_view`, in which case
    workers memory-map those IPC frames instead.
    """

    data_config_path: Path
    split: str
    config: SimConfig
    mrsjd_params: MRSJDParams | None = None
    initial_price: float | None = None
    _view: MarketView | None = field(default=None, repr=False)
    #: Directory of frames written by :meth:`publish_shared_view`.  Workers (whose
    #: pickled copy has ``_view`` dropped) memory-map it instead of rebuilding;
    #: ``None`` keeps the previous rebuild-from-disk behaviour.
    _shared_dir: Path | None = field(default=None, repr=False)

    def load_view(self) -> MarketView:
        """Return the process-local view, loading it on first use.

        Precedence: an in-process frame (the parent), then the shared IPC
        directory (workers), then a rebuild from the parquet dataset (the
        fallback).
        """
        if self._view is not None:
            return self._view
        if self._shared_dir is not None:
            self._view = load_shared_view(self._shared_dir)
        else:
            self._view = load_market_view(
                self.data_config_path,
                self.config.split,
                split=self.split,
                episode_config=self.config.episode,
            )
        return self._view

    def publish_shared_view(self, directory: Path) -> Path:
        """Write the in-process view once for every subprocess worker to map.

        Called by the training CLI before ``SubprocVecEnv`` starts.  Returns the
        directory so the caller can delete it after training; deleting early
        would break workers mid-run.
        """
        view = self.load_view()
        self._shared_dir = share_view(view, directory)
        return self._shared_dir

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_view"] = None  # workers rebuild; never ship the polars frame
        return state

    def __call__(self) -> LpEnvironment:
        view = self.load_view()
        if self.config.price_mode is PriceMode.CALIBRATED:
            if self.mrsjd_params is None or self.initial_price is None:
                raise SimConfigError(
                    "calibrated EnvFactory requires fitted MRSJD params "
                    "(call fit_calibrated before building environments)"
                )
            process = CalibratedPriceProcess(
                self.mrsjd_params, initial_price=self.initial_price
            )
            return build_replay_env(view, self.config, price_process=process)
        return build_replay_env(view, self.config)
