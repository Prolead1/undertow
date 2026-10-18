"""Picklable environment factories for process-parallel PPO training.

``stable_baselines3``'s ``SubprocVecEnv`` cloudpickles each factory into a worker
process.  A real training :class:`~undertow.sim.marketview.MarketView` is a
multi-gigabyte polars frame, so a factory that closed over it would ship a copy
to every worker and exhaust memory.  :class:`EnvFactory` stores only the
data-config path and the sim config and rebuilds the view *inside* each worker on
first use; :meth:`EnvFactory.__getstate__` drops the in-process cache so a
pickled factory never carries the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from undertow.sim.config import SimConfig
from undertow.sim.env.lp_env import LpEnvironment
from undertow.sim.evaluate.protocol import build_replay_env
from undertow.sim.marketview import MarketView, load_market_view
from undertow.sim.prices import CalibratedPriceProcess
from undertow.sim.prices.regime_jump import MRSJDParams
from undertow.sim.types import PriceMode, SimConfigError

__all__ = ["EnvFactory", "fit_calibrated"]


def fit_calibrated(market_view: MarketView) -> tuple[MRSJDParams, float]:
    """Fit the calibrated MRSJD on ``market_view`` and resolve its entry price.

    The caller must reject an eval view first (fitting on eval data would be
    look-ahead); the CLI does that and reports it as a config error.  Returns the
    fitted parameters and the first tape row's as-of price.
    """
    params = CalibratedPriceProcess.fit(market_view)
    tape = market_view.active_split_data().sort("seq")
    first_row = tape.row(0, named=True)
    initial_price = first_row.get("price_reference") or first_row.get("price_pool")
    if initial_price is None:
        initial_price = market_view.reference_close(first_row["block_timestamp"])
    if initial_price is None:
        raise SimConfigError("could not resolve an initial price for the MRSJD process")
    return params, float(initial_price)


@dataclass
class EnvFactory:
    """Zero-argument ``LpEnvironment`` factory that reloads its view per process.

    ``_view`` is an in-process cache that the CLI sets so a single-process run
    does not reload the snapshot.  It is excluded from pickles, so every
    ``SubprocVecEnv`` worker rebuilds the view from ``data_config_path`` instead
    of receiving the frame.
    """

    data_config_path: Path
    split: str
    config: SimConfig
    mrsjd_params: MRSJDParams | None = None
    initial_price: float | None = None
    _view: MarketView | None = field(default=None, repr=False)

    def load_view(self) -> MarketView:
        """Return the process-local view, loading it from disk on first use."""
        if self._view is None:
            self._view = load_market_view(
                self.data_config_path,
                self.config.split,
                split=self.split,
                episode_config=self.config.episode,
            )
        return self._view

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
