"""S03 — ``MarketView`` + walk-forward split.

Every test asserts real behavior: look-ahead enforcement, boundary errors,
fixture values, and the ``build_episode`` invariants from ``CONTRACTS.md`` §4.1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from undertow.data import DataConfig
from undertow.sim.config import EpisodeConfig, SplitConfig
from undertow.sim.marketview import MarketView, build_market_view
from undertow.sim.types import LookAheadError, MarketViewError

_TINY_TRAIN_START = datetime(2022, 1, 1, tzinfo=UTC)
_TINY_TRAIN_END = datetime(2022, 2, 1, tzinfo=UTC)
_TINY_EVAL_START = datetime(2024, 1, 1, tzinfo=UTC)
_TINY_EVAL_END = datetime(2024, 12, 31, tzinfo=UTC)

_TINY_SPLIT = SplitConfig(
    train_start_utc=_TINY_TRAIN_START,
    train_end_utc=_TINY_TRAIN_END,
    eval_start_utc=_TINY_EVAL_START,
    eval_end_utc=_TINY_EVAL_END,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_loaders(monkeypatch: pytest.MonkeyPatch, view: MarketView) -> None:
    """Make ``build_market_view`` read this fixture's frames as pyarrow tables."""
    import undertow.sim.marketview as mv

    monkeypatch.setattr(mv, "load_tape", lambda cfg: view.train.to_arrow())
    monkeypatch.setattr(mv, "load_reference_feed", lambda cfg: view.reference.to_arrow())
    monkeypatch.setattr(mv, "load_gas_feed", lambda cfg: view.gas.to_arrow())
    monkeypatch.setattr(mv, "load_regime_labels", lambda cfg: view.regimes.to_arrow())


def _train_view_with_bounds(view: MarketView, start: datetime, end: datetime) -> MarketView:
    """Re-bound the fixture's train view over ``[start, end]``."""
    tape = view.train.filter(
        (pl.col("block_timestamp") >= start) & (pl.col("block_timestamp") <= end)
    )
    return MarketView(
        train=tape,
        eval=None,
        reference=view.reference,
        regimes=view.regimes,
        gas=view.gas,
        train_start_utc=start,
        train_end_utc=end,
        eval_start_utc=_TINY_EVAL_START,
        eval_end_utc=_TINY_EVAL_END,
        is_train=True,
    )


def _eval_view_from_halves(view: MarketView, bound: datetime) -> MarketView:
    """Build an eval view over ``[bound, Feb 1]`` from the fixture's tape."""
    tape = view.train.filter(pl.col("block_timestamp") >= bound)
    return MarketView(
        train=None,
        eval=tape,
        reference=view.reference,
        regimes=view.regimes,
        gas=view.gas,
        train_start_utc=_TINY_TRAIN_START,
        train_end_utc=bound,
        eval_start_utc=bound,
        eval_end_utc=_TINY_TRAIN_END,
        is_train=False,
    )


# ---------------------------------------------------------------------------
# __post_init__ invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_rejects_both_splits_active(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="both"):
            MarketView(
                train=tiny_market_view.train,
                eval=tiny_market_view.train,
                reference=tiny_market_view.reference,
                regimes=tiny_market_view.regimes,
                gas=tiny_market_view.gas,
                train_start_utc=_TINY_TRAIN_START,
                train_end_utc=_TINY_TRAIN_END,
                eval_start_utc=_TINY_EVAL_START,
                eval_end_utc=_TINY_EVAL_END,
                is_train=True,
            )

    def test_rejects_no_split_active(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="exactly one"):
            MarketView(
                train=None,
                eval=None,
                reference=tiny_market_view.reference,
                regimes=tiny_market_view.regimes,
                gas=tiny_market_view.gas,
                train_start_utc=_TINY_TRAIN_START,
                train_end_utc=_TINY_TRAIN_END,
                eval_start_utc=_TINY_EVAL_START,
                eval_end_utc=_TINY_EVAL_END,
                is_train=True,
            )

    def test_rejects_empty_active_split(self, tiny_market_view: MarketView) -> None:
        empty = tiny_market_view.train.clear()
        with pytest.raises(MarketViewError, match="no tape rows"):
            MarketView(
                train=empty,
                eval=None,
                reference=tiny_market_view.reference,
                regimes=tiny_market_view.regimes,
                gas=tiny_market_view.gas,
                train_start_utc=_TINY_TRAIN_START,
                train_end_utc=_TINY_TRAIN_END,
                eval_start_utc=_TINY_EVAL_START,
                eval_end_utc=_TINY_EVAL_END,
                is_train=True,
            )

    def test_rejects_is_train_mismatch(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="is_train"):
            MarketView(
                train=tiny_market_view.train,
                eval=None,
                reference=tiny_market_view.reference,
                regimes=tiny_market_view.regimes,
                gas=tiny_market_view.gas,
                train_start_utc=_TINY_TRAIN_START,
                train_end_utc=_TINY_TRAIN_END,
                eval_start_utc=_TINY_EVAL_START,
                eval_end_utc=_TINY_EVAL_END,
                is_train=False,
            )

    def test_rejects_reference_not_spanning_split(self, tiny_market_view: MarketView) -> None:
        short_reference = tiny_market_view.reference.filter(
            pl.col("close_time") < _TINY_TRAIN_END - timedelta(days=1)
        )
        with pytest.raises(MarketViewError, match="does not span"):
            MarketView(
                train=tiny_market_view.train,
                eval=None,
                reference=short_reference,
                regimes=tiny_market_view.regimes,
                gas=tiny_market_view.gas,
                train_start_utc=_TINY_TRAIN_START,
                train_end_utc=_TINY_TRAIN_END,
                eval_start_utc=_TINY_EVAL_START,
                eval_end_utc=_TINY_EVAL_END,
                is_train=True,
            )


# ---------------------------------------------------------------------------
# Look-ahead enforcement — the whole point of S03
# ---------------------------------------------------------------------------


class TestLookAhead:
    def test_lookahead_query_past_train_end_raises(self, tiny_market_view: MarketView) -> None:
        """Dedicated LOOK-AHEAD test: a time past the train wall raises."""
        wall = tiny_market_view.train_end_utc
        assert tiny_market_view.is_train
        with pytest.raises(LookAheadError):
            tiny_market_view.reference_close(wall + timedelta(seconds=1))
        with pytest.raises(LookAheadError):
            tiny_market_view.reference_bars(wall + timedelta(seconds=1), wall + timedelta(days=1))
        with pytest.raises(LookAheadError):
            tiny_market_view.regime_at(wall + timedelta(seconds=1))

    def test_lookahead_after_wall_but_eval_data_exists(
        self, tiny_market_view: MarketView
    ) -> None:
        """Train view whose tape stops mid-month still refuses to read past the wall."""
        mid = _TINY_TRAIN_START + timedelta(days=15)
        train = _train_view_with_bounds(tiny_market_view, _TINY_TRAIN_START, mid)
        with pytest.raises(LookAheadError):
            train.reference_close(mid + timedelta(days=1))
        # Within the wall is fine and returns the last close at/before `mid`.
        assert train.reference_close(mid) is not None

    def test_slice_beyond_split_raises(self, tiny_market_view: MarketView) -> None:
        last = int(tiny_market_view.active_split_data()["seq"].max())
        with pytest.raises(LookAheadError):
            tiny_market_view.slice(0, last + 2)

    def test_slice_before_split_raises(self, tiny_market_view: MarketView) -> None:
        min_seq = int(tiny_market_view.active_split_data()["seq"].min())
        if min_seq == 0:
            pytest.skip("fixture starts at seq 0; use an eval view for the before-wall case")
        with pytest.raises(LookAheadError):
            tiny_market_view.slice(min_seq - 1, min_seq + 1)

    def test_tape_at_out_of_split_raises(self, tiny_market_view: MarketView) -> None:
        last = int(tiny_market_view.active_split_data()["seq"].max())
        with pytest.raises(LookAheadError):
            tiny_market_view.tape_at(last + 1)
        with pytest.raises(LookAheadError):
            tiny_market_view.tape_at(-1)

    def test_eval_view_cannot_read_train_seq(self, tiny_market_view: MarketView) -> None:
        """On an eval view, a seq that belongs to the train window is out of range."""
        bound = _TINY_TRAIN_START + timedelta(days=15)
        eval_view = _eval_view_from_halves(tiny_market_view, bound)
        assert not eval_view.is_train
        eval_min = int(eval_view.active_split_data()["seq"].min())
        assert eval_min > 0
        with pytest.raises(LookAheadError):
            eval_view.tape_at(0)
        with pytest.raises(LookAheadError):
            eval_view.slice(0, eval_min)


# ---------------------------------------------------------------------------
# Query methods
# ---------------------------------------------------------------------------


class TestQueries:
    def test_active_split_data_and_step_count(self, tiny_market_view: MarketView) -> None:
        assert tiny_market_view.active_split_data().equals(tiny_market_view.train)
        assert tiny_market_view.step_count() == 1000

    def test_tape_at_returns_row(self, tiny_market_view: MarketView) -> None:
        expected = tiny_market_view.active_split_data().row(5, named=True)
        result = tiny_market_view.tape_at(5)
        assert result["seq"] == 5
        assert result["block_timestamp"] == expected["block_timestamp"]
        assert result["price_pool"] == pytest.approx(expected["price_pool"])

    def test_reference_close_exactly_at_close_time(self, tiny_market_view: MarketView) -> None:
        row = tiny_market_view.reference.row(5, named=True)
        assert tiny_market_view.reference_close(row["close_time"]) == pytest.approx(row["close"])

    def test_reference_close_returns_last_bar_before(self, tiny_market_view: MarketView) -> None:
        row0 = tiny_market_view.reference.row(5, named=True)
        row1 = tiny_market_view.reference.row(6, named=True)
        at_time = row1["close_time"] - timedelta(microseconds=1)
        assert tiny_market_view.reference_close(at_time) == pytest.approx(row0["close"])

    def test_reference_close_none_before_feed_starts(self, tiny_market_view: MarketView) -> None:
        before = _TINY_TRAIN_START - timedelta(days=1)
        assert tiny_market_view.reference_close(before) is None

    def test_reference_bars_clamped_to_split(self, tiny_market_view: MarketView) -> None:
        bars = tiny_market_view.reference_bars(
            _TINY_TRAIN_START - timedelta(days=5),
            _TINY_TRAIN_END + timedelta(days=5),
        )
        assert bars.height > 0
        assert bars["open_time"].min() >= _TINY_TRAIN_START
        assert bars["open_time"].max() < _TINY_TRAIN_END

    def test_reference_bars_empty_window(self, tiny_market_view: MarketView) -> None:
        bars = tiny_market_view.reference_bars(
            _TINY_TRAIN_START + timedelta(days=10),
            _TINY_TRAIN_START + timedelta(days=10),
        )
        assert bars.height == 0
        assert set(bars.columns) == set(tiny_market_view.reference.columns)

    def test_regime_at_correct_label(self, tiny_market_view: MarketView) -> None:
        row = tiny_market_view.regimes.row(30, named=True)
        assert tiny_market_view.regime_at(row["timestamp"]) == row["regime"]

    def test_regime_at_unknown_before_first_label(self, tiny_market_view: MarketView) -> None:
        before = _TINY_TRAIN_START - timedelta(days=1)
        assert tiny_market_view.regime_at(before) == "unknown"

    def test_reference_bars_excludes_straddling_bar(self, tiny_market_view: MarketView) -> None:
        """A bar whose open precedes the wall but close exceeds it must not leak."""
        row = tiny_market_view.reference.row(5, named=True)
        wall = row["open_time"] + timedelta(seconds=30)
        assert row["close_time"] > wall
        tape = tiny_market_view.train.filter(pl.col("block_timestamp") < wall)
        view = MarketView(
            train=tape,
            eval=None,
            reference=tiny_market_view.reference,
            regimes=tiny_market_view.regimes,
            gas=tiny_market_view.gas,
            train_start_utc=_TINY_TRAIN_START,
            train_end_utc=wall,
            eval_start_utc=_TINY_EVAL_START,
            eval_end_utc=_TINY_EVAL_END,
            is_train=True,
        )
        bars = view.reference_bars(_TINY_TRAIN_START, wall)
        assert bars.height > 0
        assert bars["close_time"].max() <= wall
        assert row["close_time"] not in bars["close_time"].to_list()

    def test_gas_at_block(self, tiny_market_view: MarketView) -> None:
        expected = tiny_market_view.gas.row(7, named=True)
        result = tiny_market_view.gas_at_block(7)
        assert result is not None
        assert result["base_fee_per_gas"] == expected["base_fee_per_gas"]
        assert result["gas_used"] == expected["gas_used"]
        assert tiny_market_view.gas_at_block(10_000) is None

    def test_slice_preserves_feeds_and_narrows_bounds(self, tiny_market_view: MarketView) -> None:
        sub = tiny_market_view.slice(100, 200)
        assert sub.step_count() == 100
        assert sub.reference.equals(tiny_market_view.reference)
        assert sub.regimes.equals(tiny_market_view.regimes)
        assert sub.gas.equals(tiny_market_view.gas)
        assert sub.is_train
        assert sub.active_start_utc == sub.train["block_timestamp"].min()
        assert sub.active_end_utc == sub.train["block_timestamp"].max()

    def test_slice_invalid_range_raises(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="start_seq"):
            tiny_market_view.slice(10, 10)
        with pytest.raises(MarketViewError, match="start_seq"):
            tiny_market_view.slice(10, 5)


# ---------------------------------------------------------------------------
# build_episode
# ---------------------------------------------------------------------------


class TestBuildEpisode:
    def test_returns_valid_window(self, tiny_market_view: MarketView) -> None:
        start, end = tiny_market_view.build_episode(seed=0, max_steps=137)
        assert 0 <= start < end <= tiny_market_view.step_count()
        assert end - start == 137

    def test_deterministic_for_same_seed(self, tiny_market_view: MarketView) -> None:
        assert tiny_market_view.build_episode(3, max_steps=200) == tiny_market_view.build_episode(
            3, max_steps=200
        )

    def test_never_crosses_split_boundary(self, tiny_market_view: MarketView) -> None:
        last_seq = int(tiny_market_view.active_split_data()["seq"].max())
        for seed in range(200):
            start, end = tiny_market_view.build_episode(seed, max_steps=91)
            assert start <= last_seq
            assert end <= last_seq + 1
            assert start < end

    def test_never_crosses_slice_boundary(self, tiny_market_view: MarketView) -> None:
        sub = tiny_market_view.slice(100, 500)
        for seed in range(100):
            start, end = sub.build_episode(seed, max_steps=50)
            assert start >= 100
            assert end <= 500

    def test_default_config_caps_to_available_rows(self, tiny_market_view: MarketView) -> None:
        # Default EpisodeConfig is 30 days @ 10 min == 4320 steps > 1000 rows.
        assert tiny_market_view.build_episode(0) == (0, 1000)

    def test_explicit_episode_config_is_used(self, tiny_market_view: MarketView) -> None:
        ep = EpisodeConfig(duration_days=0, step_minutes=10)
        # duration_days=0 -> config_steps 0 -> error (desired must be > 0)
        with pytest.raises(MarketViewError, match="desired window"):
            tiny_market_view.build_episode(0, episode_config=ep)

    def test_window_is_contiguous_rows(self, tiny_market_view: MarketView) -> None:
        start, end = tiny_market_view.build_episode(7, max_steps=50)
        seqs = tiny_market_view.active_split_data()["seq"].to_list()
        window = [s for s in seqs if start <= s < end]
        assert window == list(range(start, end))


# ---------------------------------------------------------------------------
# build_market_view
# ---------------------------------------------------------------------------


class TestBuildMarketView:
    def test_partitions_train_and_converts_pyarrow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        _patch_loaders(monkeypatch, tiny_market_view)
        view = build_market_view(tiny_data_config, _TINY_SPLIT)
        assert view.is_train
        assert isinstance(view.train, pl.DataFrame)
        assert isinstance(view.reference, pl.DataFrame)
        assert isinstance(view.regimes, pl.DataFrame)
        assert isinstance(view.gas, pl.DataFrame)
        assert view.step_count() == tiny_market_view.step_count()
        assert view.train_start_utc == _TINY_TRAIN_START

    def test_empty_train_split_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        _patch_loaders(monkeypatch, tiny_market_view)
        split = SplitConfig(
            train_start_utc=datetime(2021, 1, 1, tzinfo=UTC),
            train_end_utc=datetime(2021, 2, 1, tzinfo=UTC),
        )
        with pytest.raises(MarketViewError, match="no tape rows"):
            build_market_view(tiny_data_config, split, split="train")

    def test_eval_split_is_active_and_disjoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        _patch_loaders(monkeypatch, tiny_market_view)
        bound = _TINY_TRAIN_START + timedelta(days=15)
        split = SplitConfig(
            train_start_utc=_TINY_TRAIN_START,
            train_end_utc=bound,
            eval_start_utc=bound,
            eval_end_utc=_TINY_TRAIN_END,
        )
        view = build_market_view(tiny_data_config, split, split="eval")
        assert not view.is_train
        assert view.eval is not None
        assert view.train is None
        assert view.step_count() > 0

    def test_unknown_split_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        _patch_loaders(monkeypatch, tiny_market_view)
        with pytest.raises(MarketViewError, match="unknown split"):
            build_market_view(tiny_data_config, _TINY_SPLIT, split="trian")  # type: ignore[arg-type]

    def test_episode_config_is_propagated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        _patch_loaders(monkeypatch, tiny_market_view)
        ep = EpisodeConfig(duration_days=1, step_minutes=10)
        view = build_market_view(tiny_data_config, _TINY_SPLIT, episode_config=ep)
        assert view.episode_config is ep
        start, end = view.build_episode(0)
        assert end - start == 144  # 1 day @ 10 min

    def test_equal_boundary_splits_are_disjoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        """``train_end == eval_start`` must place the boundary row in eval only."""
        _patch_loaders(monkeypatch, tiny_market_view)
        bound = _TINY_TRAIN_START + timedelta(days=10)
        split = SplitConfig(
            train_start_utc=_TINY_TRAIN_START,
            train_end_utc=bound,
            eval_start_utc=bound,
            eval_end_utc=_TINY_TRAIN_END,
        )
        train = build_market_view(tiny_data_config, split, split="train")
        ev = build_market_view(tiny_data_config, split, split="eval")
        train_seqs = set(train.active_split_data()["seq"].to_list())
        eval_seqs = set(ev.active_split_data()["seq"].to_list())
        assert train_seqs and eval_seqs
        assert train_seqs.isdisjoint(eval_seqs)

    def test_loader_not_implemented_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
    ) -> None:
        """A loader's NotImplementedError must surface unchanged, not be swallowed."""
        import undertow.sim.marketview as mv

        def _boom(cfg: object) -> object:
            raise NotImplementedError("pending T16")

        monkeypatch.setattr(mv, "load_tape", _boom)
        with pytest.raises(NotImplementedError, match="pending T16"):
            build_market_view(tiny_data_config, _TINY_SPLIT)


# ---------------------------------------------------------------------------
# Fixture contract
# ---------------------------------------------------------------------------


class TestTinyFixture:
    def test_fixture_is_usable(self, tiny_market_view: MarketView) -> None:
        assert tiny_market_view.is_train
        assert tiny_market_view.step_count() > 0
        assert tiny_market_view.train is not None
        assert tiny_market_view.eval is None
        assert tiny_market_view.step_count() == 1000
        assert tiny_market_view.reference.height == 100
        assert tiny_market_view.gas.height == 100
        assert tiny_market_view.regimes.height == 100

    def test_fixture_data_config(self, tiny_data_config: DataConfig) -> None:
        assert tiny_data_config.pool.token0_symbol == "USDC"
        assert tiny_data_config.pool.token1_symbol == "WETH"