"""Tests for the out-of-process training view builder (PR #103 follow-up).

The builder exists so the long-lived training parent never pays the event-tape
build's allocator arenas: a ``spawn`` child loads the view, publishes the IPC
frames (and the calibrated params sidecar) and exits.  These tests pin the
artifact contract, the calibrated/replay sidecar behaviour, and the spawn
orchestration's failure handling.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from undertow.sim.config import SimConfig
from undertow.sim.marketview import MarketView
from undertow.sim.train import view_builder
from undertow.sim.train.envs import fit_calibrated, load_shared_view
from undertow.sim.train.view_builder import (
    PARAMS_FILENAME,
    ViewBuilderError,
    build_shared_view,
    read_builder_params,
    spawn_view_builder,
)
from undertow.sim.types import PriceMode


def test_build_shared_view_publishes_replay_frames(
    tmp_path: Path, tiny_market_view: MarketView, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        view_builder, "load_market_view", lambda *args, **kwargs: tiny_market_view
    )
    directory = tmp_path / "shared"
    directory.mkdir()
    # A previous calibrated run's sidecar must not survive a replay publish.
    (directory / PARAMS_FILENAME).write_bytes(pickle.dumps({"stale": True}))

    build_shared_view(SimConfig(), Path("data.toml"), "train", directory)

    assert load_shared_view(directory).train.equals(tiny_market_view.train)
    assert not (directory / PARAMS_FILENAME).exists()
    assert read_builder_params(directory) == (None, None)


def test_build_shared_view_writes_calibrated_params(
    tmp_path: Path, tiny_market_view: MarketView, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_params, expected_price = fit_calibrated(tiny_market_view)
    monkeypatch.setattr(
        view_builder, "load_market_view", lambda *args, **kwargs: tiny_market_view
    )
    directory = tmp_path / "shared"

    build_shared_view(
        SimConfig(price_mode=PriceMode.CALIBRATED),
        Path("data.toml"),
        "train",
        directory,
    )

    params, initial_price = read_builder_params(directory)
    assert params is not None
    assert params.state_names == expected_params.state_names
    np.testing.assert_allclose(
        params.transition_matrix, expected_params.transition_matrix
    )
    assert initial_price == pytest.approx(expected_price)
    # The frames are published alongside the params for calibrated runs too.
    assert load_shared_view(directory).train.equals(tiny_market_view.train)


def test_spawn_view_builder_forwards_the_build_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[object, ...]] = []

    def fake_spawn(args: tuple[object, ...]) -> int:
        seen.append(args)
        return 0

    monkeypatch.setattr(view_builder, "_spawn", fake_spawn)
    directory = tmp_path / "shared"

    result = spawn_view_builder(SimConfig(), Path("data.toml"), "train", directory)

    assert result == directory
    assert len(seen) == 1
    config, data_path, split, out_dir = seen[0]
    assert isinstance(config, SimConfig)
    assert data_path == Path("data.toml")
    assert split == "train"
    assert out_dir == directory


def test_spawn_view_builder_raises_on_child_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(view_builder, "_spawn", lambda args: 3)

    with pytest.raises(ViewBuilderError, match="exited with code 3"):
        spawn_view_builder(SimConfig(), Path("data.toml"), "train", tmp_path / "x")


def test_spawn_view_builder_reports_a_real_child_failure(tmp_path: Path) -> None:
    """Exercise the real ``multiprocessing`` invocation: a child that cannot
    load the (missing) dataset must surface as ``ViewBuilderError``, not hang or
    silently succeed."""
    with pytest.raises(ViewBuilderError, match="exited with code"):
        spawn_view_builder(
            SimConfig(),
            tmp_path / "missing.toml",
            "train",
            tmp_path / "shared",
        )


def test_read_builder_params_rejects_a_corrupt_sidecar(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()
    (directory / PARAMS_FILENAME).write_bytes(
        pickle.dumps({"mrsjd_params": "not-params", "initial_price": 1.0})
    )

    with pytest.raises(ViewBuilderError, match="MRSJDParams"):
        read_builder_params(directory)


def test_read_builder_params_validates_the_price(
    tmp_path: Path, tiny_market_view: MarketView
) -> None:
    params, _ = fit_calibrated(tiny_market_view)
    directory = tmp_path / "shared"
    directory.mkdir()
    path = directory / PARAMS_FILENAME

    path.write_bytes(pickle.dumps({"mrsjd_params": params, "initial_price": None}))
    loaded, price = read_builder_params(directory)
    assert loaded is not None
    assert loaded.state_names == params.state_names
    np.testing.assert_allclose(loaded.transition_matrix, params.transition_matrix)
    assert price is None

    for bad_price in ("x", True):
        path.write_bytes(
            pickle.dumps({"mrsjd_params": params, "initial_price": bad_price})
        )
        with pytest.raises(ViewBuilderError, match="non-numeric"):
            read_builder_params(directory)


def test_read_builder_params_rejects_unreadable_payloads(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    directory.mkdir()
    path = directory / PARAMS_FILENAME

    path.write_bytes(b"not a pickle")
    with pytest.raises(ViewBuilderError, match="not a readable"):
        read_builder_params(directory)

    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ViewBuilderError, match="builder payload"):
        read_builder_params(directory)
