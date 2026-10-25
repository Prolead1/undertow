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
    # build_replay_env derives the width from the config spacing so it matches
    # the backtester's ±120-tick band; the default config is spacing 10 → 12.
    assert env._initial_width == 12


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
    # Calibrated shares build_replay_env, so it deploys the same derived width
    # (spacing 10 → 12; it previously used lp_env's standalone default of 10).
    assert env._initial_width == 12


def test_env_factory_calibrated_requires_fitted_params(
    monkeypatch: pytest.MonkeyPatch, tiny_market_view: MarketView
) -> None:
    monkeypatch.setattr(envs, "load_market_view", lambda *a, **k: tiny_market_view)
    config = SimConfig(price_mode=PriceMode.CALIBRATED)

    with pytest.raises(SimConfigError, match="fitted MRSJD"):
        envs.EnvFactory(Path("data.toml"), "train", config)()


# ---------------------------------------------------------------------------
# Shared-view IPC artifact (subproc workers map instead of rebuilding)
# ---------------------------------------------------------------------------


def test_shared_view_field_sets_cover_every_marketview_init_field() -> None:
    """A new ``MarketView`` field must land in ``_VIEW_FRAMES``/``_VIEW_META_FIELDS``
    or workers would silently reconstruct it with a default."""
    import dataclasses

    init_fields = {f.name for f in dataclasses.fields(MarketView) if f.init}
    assert set(envs._VIEW_FRAMES) | set(envs._VIEW_META_FIELDS) == init_fields


def test_shared_view_round_trips_optional_frames(
    tmp_path: Path, tiny_market_view: MarketView
) -> None:
    import dataclasses

    import polars as pl

    funding = pl.DataFrame(
        {
            "funding_time": pl.Series([1], dtype=pl.Datetime("us", "UTC")),
            "funding_rate": pl.Series([0.0001], dtype=pl.Float64),
        }
    )
    pool_state = pl.DataFrame({"block_number": pl.Series([9], dtype=pl.Int64)})
    view = dataclasses.replace(
        tiny_market_view, funding=funding, pool_state=pool_state
    )

    loaded = envs.load_shared_view(
        envs.share_view(view, tmp_path / "shared_optional")
    )

    assert loaded.funding is not None and loaded.funding.equals(funding)
    assert loaded.pool_state is not None and loaded.pool_state.equals(pool_state)


def test_shared_view_round_trips_frames_meta_and_sampling(
    tmp_path: Path, tiny_market_view: MarketView
) -> None:
    directory = envs.share_view(tiny_market_view, tmp_path / "shared")

    loaded = envs.load_shared_view(directory)

    assert loaded.is_train is tiny_market_view.is_train
    assert loaded.train_start_utc == tiny_market_view.train_start_utc
    assert loaded.train_end_utc == tiny_market_view.train_end_utc
    assert loaded.eval_start_utc == tiny_market_view.eval_start_utc
    assert loaded.eval_end_utc == tiny_market_view.eval_end_utc
    assert loaded.episode_config == tiny_market_view.episode_config
    assert loaded.pool == tiny_market_view.pool
    for name in ("train", "reference", "regimes", "gas"):
        original = getattr(tiny_market_view, name)
        assert getattr(loaded, name).equals(original)
    # Sampling is a function of the frames + bounds, so equal windows prove the
    # reconstructed view is behaviourally identical to the original.
    assert loaded.build_episode(seed=7) == tiny_market_view.build_episode(seed=7)


def test_share_view_removes_stale_frames_from_a_reused_directory(
    tmp_path: Path, tiny_market_view: MarketView
) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()
    (directory / "warmup.arrow").write_bytes(b"stale")

    envs.share_view(tiny_market_view, directory)

    # The fixture has no warmup frame, so a stale one must not resurrect it.
    assert not (directory / "warmup.arrow").exists()
    assert envs.load_shared_view(directory).warmup is None


def test_load_shared_view_rejects_a_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(SimConfigError, match="shared market view is missing"):
        envs.load_shared_view(tmp_path / "never_published")


def test_env_factory_worker_maps_the_shared_view_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_market_view: MarketView
) -> None:
    """A worker's pickled factory has no in-process frame; it must read the
    published IPC files, never rebuild the event tape from parquet."""

    def boom(*args: object, **kwargs: object) -> MarketView:
        raise AssertionError("worker rebuilt the view instead of mapping the artifact")

    monkeypatch.setattr(envs, "load_market_view", boom)
    directory = envs.share_view(tiny_market_view, tmp_path / "shared")
    factory = envs.EnvFactory(
        Path("data.toml"), "train", SimConfig(), _shared_dir=directory
    )

    view = factory.load_view()

    assert view.train.equals(tiny_market_view.train)
    assert factory.load_view() is view  # cached per process


def test_env_factory_pickle_keeps_shared_dir_and_drops_view(
    tmp_path: Path, tiny_market_view: MarketView
) -> None:
    factory = envs.EnvFactory(
        Path("data.toml"),
        "train",
        SimConfig(),
        _view=tiny_market_view,
        _shared_dir=tmp_path,
    )

    restored = pickle.loads(pickle.dumps(factory))

    assert restored._view is None
    assert restored._shared_dir == tmp_path


def test_publish_shared_view_writes_the_artifact_and_points_the_factory(
    tmp_path: Path, tiny_market_view: MarketView
) -> None:
    factory = envs.EnvFactory(
        Path("data.toml"), "train", SimConfig(), _view=tiny_market_view
    )

    directory = factory.publish_shared_view(tmp_path / "published")

    assert directory == tmp_path / "published"
    assert factory._shared_dir == directory
    assert (directory / envs._VIEW_META_FILENAME).is_file()
    assert envs.load_shared_view(directory).train.equals(tiny_market_view.train)


def test_publish_shared_view_loads_the_view_when_not_preloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_market_view: MarketView
) -> None:
    monkeypatch.setattr(envs, "load_market_view", lambda *a, **k: tiny_market_view)
    factory = envs.EnvFactory(Path("data.toml"), "train", SimConfig())

    directory = factory.publish_shared_view(tmp_path / "published")

    assert envs.load_shared_view(directory).train.equals(tiny_market_view.train)
