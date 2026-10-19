"""Tests for the picklable process-safe env factory (ADR-022).

``SubprocVecEnv`` cloudpickles the factory into each worker.  These pin the two
properties that make that safe: the cached ``MarketView`` is dropped from the
pickle (workers rebuild it from disk), and ``load_view`` is loaded once per
process.  Both replay and calibrated construction go through
``build_replay_env``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest

import undertow.sim.train.envs as envs
from undertow.sim.config import SimConfig
from undertow.sim.marketview import MarketView
from undertow.sim.prices import CalibratedPriceProcess
from undertow.sim.types import PriceMode, SimConfigError


def test_env_factory_drops_the_view_when_pickled(tiny_market_view: MarketView) -> None:
    """A pickled factory must not carry the multi-gigabyte polars frame."""
    factory = envs.EnvFactory(
        Path("data.toml"), "train", SimConfig(), _view=tiny_market_view
    )

    restored = pickle.loads(pickle.dumps(factory))

    assert restored._view is None
    assert restored.data_config_path == Path("data.toml")
    assert restored.split == "train"


def test_env_factory_loads_the_view_once(
    monkeypatch: pytest.MonkeyPatch, tiny_market_view: MarketView
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_load(*args: object, **kwargs: object) -> MarketView:
        calls.append(dict(kwargs))
        return tiny_market_view

    monkeypatch.setattr(envs, "load_market_view", fake_load)
    factory = envs.EnvFactory(Path("data.toml"), "train", SimConfig())

    assert factory.load_view() is tiny_market_view
    assert factory.load_view() is tiny_market_view
    assert len(calls) == 1
    assert calls[0]["split"] == "train"


def test_env_factory_builds_a_replay_env(
    monkeypatch: pytest.MonkeyPatch, tiny_market_view: MarketView
) -> None:
    monkeypatch.setattr(envs, "load_market_view", lambda *a, **k: tiny_market_view)

    env = envs.EnvFactory(Path("data.toml"), "train", SimConfig())()

    assert env.is_replay
    # build_replay_env's parity width (ADR-022), not lp_env's standalone default
    assert env._initial_width == 2


def test_env_factory_builds_a_calibrated_env(
    monkeypatch: pytest.MonkeyPatch, tiny_market_view: MarketView
) -> None:
    monkeypatch.setattr(envs, "load_market_view", lambda *a, **k: tiny_market_view)
    params, initial_price = envs.fit_calibrated(tiny_market_view)
    config = SimConfig(price_mode=PriceMode.CALIBRATED)

    env = envs.EnvFactory(
        Path("data.toml"),
        "train",
        config,
        mrsjd_params=params,
        initial_price=initial_price,
    )()

    assert not env.is_replay
    assert isinstance(env.price_process, CalibratedPriceProcess)
    # Calibrated now shares build_replay_env, so it deploys the parity width 2
    # (it previously used lp_env's standalone default of 10) - ADR-022.
    assert env._initial_width == 2


def test_env_factory_calibrated_requires_fitted_params(
    monkeypatch: pytest.MonkeyPatch, tiny_market_view: MarketView
) -> None:
    monkeypatch.setattr(envs, "load_market_view", lambda *a, **k: tiny_market_view)
    config = SimConfig(price_mode=PriceMode.CALIBRATED)

    with pytest.raises(SimConfigError, match="fitted MRSJD"):
        envs.EnvFactory(Path("data.toml"), "train", config)()
