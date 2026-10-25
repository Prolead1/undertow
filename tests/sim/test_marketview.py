"""S03 — ``MarketView`` + walk-forward split.

Every test asserts real behavior: look-ahead enforcement, boundary errors,
fixture values, and the ``build_episode`` invariants from ``CONTRACTS.md`` §4.1.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from undertow.data import FUNDING_SCHEMA, DataConfig, ValidationError
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


class _FakeDataset:
    """Structural stand-in for the one ``undertow.data.Dataset`` build.

    Mirrors ``Dataset.require`` for the five streams ``build_market_view`` reads
    (the tape + reference/gas/regime required, fee_growth optional).
    """

    _FIELDS = {
        "event_tape": "tape",
        "tape": "tape",
        "reference": "reference",
        "gas": "gas",
        "regime": "regime",
    }

    def __init__(
        self, view: MarketView, *, fee_growth: pl.DataFrame | None = None
    ) -> None:
        assert view.train is not None  # the fixtures patched in are train views
        self.tape = view.train.to_arrow()
        self.reference = view.reference.to_arrow()
        self.gas = view.gas.to_arrow()
        self.regime = view.regimes.to_arrow()
        self.fee_growth = (
            fee_growth.to_arrow() if fee_growth is not None else None
        )

    def require(self, name: str) -> object:
        table = getattr(self, self._FIELDS[name])
        if table is None:
            raise ValidationError(f"stream {name!r} is absent from this dataset")
        return table


def _patch_loaders(
    monkeypatch: pytest.MonkeyPatch,
    view: MarketView,
    *,
    fee_growth: pl.DataFrame | None = None,
) -> None:
    """Make ``build_market_view`` read this fixture's frames from one dataset build."""
    import undertow.sim.marketview as mv

    monkeypatch.setattr(
        mv, "load_dataset", lambda cfg: _FakeDataset(view, fee_growth=fee_growth)
    )
    # No funding stream in the synthetic fixtures → empty, `funding=None`.
    monkeypatch.setattr(mv, "load_funding", lambda cfg: FUNDING_SCHEMA.empty_table())


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


def _two_symbol_view(
    view: MarketView, *, primary: str = "ETHUSDT", delta: float = 100.0
) -> MarketView:
    """Copy ``view`` with reference/regime feeds carrying a second symbol.

    The fixture feed is the cross-check (``ETHUSDC``); the added ``primary``
    series has a clearly different close (``+delta``) so a test can prove the
    primary was selected rather than the two being mixed.
    """
    reference = pl.concat(
        [
            view.reference.with_columns(
                pl.lit(primary).alias("symbol"),
                (pl.col("close") + delta).alias("close"),
            ),
            view.reference,
        ]
    ).sort("open_time")
    regimes = pl.concat(
        [view.regimes.with_columns(pl.lit(primary).alias("symbol")), view.regimes]
    ).sort("timestamp")
    return replace(view, reference=reference, regimes=regimes)


def _window_times(view: MarketView, start: int, end: int) -> list[datetime]:
    """Block timestamps of the active-split rows in ``[start, end)``."""
    sub = view.active_split_data().filter(
        (pl.col("seq") >= start) & (pl.col("seq") < end)
    ).sort("seq")
    return sub["block_timestamp"].to_list()


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

    def test_gas_at_block_binary_searches_a_sorted_feed(
        self, tiny_market_view: MarketView
    ) -> None:
        """``gas_at_block`` binary-searches, so ``MarketView.gas`` must be sorted
        by ``block_number`` (``build_market_view`` sorts; hand-built views must
        too).  Pins the exact lookup with an interior gap, both bounds and a
        duplicate block (the first occurrence wins, as with the old filter)."""
        sparse = tiny_market_view.gas.head(3).with_columns(
            pl.Series("block_number", [10, 20, 30])
        )
        view = replace(tiny_market_view, gas=sparse)

        assert view.gas_at_block(10) is not None
        assert view.gas_at_block(20) is not None
        assert view.gas_at_block(30) is not None
        assert view.gas_at_block(5) is None  # below the first
        assert view.gas_at_block(15) is None  # interior gap
        assert view.gas_at_block(31) is None  # above the last

        first = sparse.with_columns(pl.lit("111").alias("base_fee_per_gas"))
        second = sparse.with_columns(pl.lit("222").alias("base_fee_per_gas"))
        duplicated = pl.concat([first, second]).sort(
            "block_number", maintain_order=True
        )
        result = replace(tiny_market_view, gas=duplicated).gas_at_block(20)
        assert result is not None
        assert result["base_fee_per_gas"] == "111"

    def test_regime_at_is_inclusive_and_as_of_between_labels(
        self, tiny_market_view: MarketView
    ) -> None:
        rows = tiny_market_view.regimes.sort("timestamp")
        first = rows.row(0, named=True)
        mid = rows.row(5, named=True)
        nxt = rows.row(6, named=True)

        assert tiny_market_view.regime_at(first["timestamp"]) == first["regime"]
        assert (
            tiny_market_view.regime_at(first["timestamp"] - timedelta(seconds=1))
            == "unknown"
        )
        between = mid["timestamp"] + (nxt["timestamp"] - mid["timestamp"]) / 2
        assert tiny_market_view.regime_at(between) == mid["regime"]

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

    def test_window_matches_slice_active_rows(self, tiny_market_view: MarketView) -> None:
        """ADR-033: ``window`` is ``slice(...).active_split_data()`` without warmup."""
        total = tiny_market_view.step_count()
        for start, end in ((0, 1), (0, 10), (5, 20), (total - 5, total), (0, total)):
            window = tiny_market_view.window(start, end)
            expected = tiny_market_view.slice(start, end).active_split_data()
            assert window.equals(expected)
            assert int(window["seq"].min()) >= start
            assert int(window["seq"].max()) < end

    def test_window_invalid_range_raises(self, tiny_market_view: MarketView) -> None:
        with pytest.raises(MarketViewError, match="start_seq"):
            tiny_market_view.window(5, 5)
        # Out-of-range bounds are a look-ahead error, exactly like ``slice``.
        with pytest.raises(LookAheadError, match="outside"):
            tiny_market_view.window(10_000, 10_001)
        with pytest.raises(LookAheadError, match="outside"):
            tiny_market_view.window(-5, 10)

    def test_active_split_data_sorts_an_unsorted_seq_frame(
        self, tiny_market_view: MarketView
    ) -> None:
        """The defensive sort still runs when a hand-built frame is unordered."""
        reversed_train = tiny_market_view.train.reverse()
        view = replace(tiny_market_view, train=reversed_train)
        out = view.active_split_data()
        assert out["seq"].is_sorted()
        assert out["seq"].to_list() == sorted(reversed_train["seq"].to_list())

    def test_episode_cache_is_not_copied_by_dataclasses_replace(
        self, tiny_market_view: MarketView
    ) -> None:
        """ADR-033: ``replace`` must not serve the parent's cached tape index."""
        tiny_market_view.build_episode(0, max_steps=5)  # warm the cache
        shifted = tiny_market_view.train.with_columns(
            (pl.col("seq") + 10_000).alias("seq")
        )
        swapped = replace(tiny_market_view, train=shifted)
        start, end = swapped.build_episode(0, max_steps=5)
        assert start >= 10_000


# ---------------------------------------------------------------------------
# build_episode
# ---------------------------------------------------------------------------


class TestBuildEpisode:
    def test_returns_valid_window(self, tiny_market_view: MarketView) -> None:
        window = timedelta(minutes=137 * 10)  # max_steps is decision steps, not rows
        start, end = tiny_market_view.build_episode(seed=0, max_steps=137)
        assert 0 <= start < end <= tiny_market_view.step_count()
        times = _window_times(tiny_market_view, start, end)
        assert times
        assert times[-1] - times[0] < window
        # Maximal: the next row starts at/after the requested window end.
        after = tiny_market_view.active_split_data().filter(pl.col("seq") == end)
        if after.height:
            assert after["block_timestamp"][0] - times[0] >= window

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

    def test_default_config_samples_duration_days_window(
        self, tiny_market_view: MarketView
    ) -> None:
        # Default EpisodeConfig is 30 days @ 10 min; the split is 31 days, so a
        # 30-day wall-clock window is sampled (not the whole split as before).
        start, end = tiny_market_view.build_episode(0)
        assert 0 <= start < end <= tiny_market_view.step_count()
        times = _window_times(tiny_market_view, start, end)
        assert timedelta(days=29) < times[-1] - times[0] < timedelta(days=30)

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

    def test_matches_reference_bisect_sampling(self, tiny_market_view: MarketView) -> None:
        """ADR-033: the cached numpy index reproduces the original bisect sampling."""
        active = tiny_market_view.active_split_data()
        times = active["block_timestamp"].to_list()
        seqs = [int(s) for s in active["seq"].to_list()]
        window = timedelta(minutes=91 * 10)
        limit = tiny_market_view.active_end_utc - window
        n_candidates = bisect.bisect_right(times, limit)
        assert n_candidates > 0
        for seed in range(30):
            idx = int(np.random.default_rng(seed).integers(0, n_candidates))
            end_idx = bisect.bisect_left(times, times[idx] + window)
            expected = (seqs[idx], seqs[end_idx - 1] + 1)
            assert tiny_market_view.build_episode(seed, max_steps=91) == expected


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
        times = _window_times(view, start, end)
        span = times[-1] - times[0]
        assert timedelta(hours=23) < span < timedelta(days=1)  # 1 day @ 10 min

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

        monkeypatch.setattr(mv, "load_dataset", _boom)
        with pytest.raises(NotImplementedError, match="pending T16"):
            build_market_view(tiny_data_config, _TINY_SPLIT)

    def test_missing_required_stream_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        """A required stream goes through ``Dataset.require``, not the optional
        path: only ``fee_growth`` may be absent (and then becomes an empty table)."""
        import undertow.sim.marketview as mv

        fake = _FakeDataset(tiny_market_view)
        fake.gas = None
        monkeypatch.setattr(mv, "load_dataset", lambda cfg: fake)
        monkeypatch.setattr(mv, "load_funding", lambda cfg: FUNDING_SCHEMA.empty_table())
        with pytest.raises(ValidationError, match="gas"):
            build_market_view(tiny_data_config, _TINY_SPLIT)

    def test_dataset_is_built_once_for_all_streams(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        """Perf regression: the five streams come from a single dataset build.

        Calling one loader per stream rebuilt the whole event tape five times
        (~5x view-load time and most of the peak RSS).
        """
        import undertow.sim.marketview as mv

        calls = {"count": 0}
        fake = _FakeDataset(tiny_market_view)

        def counting(cfg: object) -> object:
            calls["count"] += 1
            return fake

        monkeypatch.setattr(mv, "load_dataset", counting)
        monkeypatch.setattr(mv, "load_funding", lambda cfg: FUNDING_SCHEMA.empty_table())
        view = build_market_view(tiny_data_config, _TINY_SPLIT)
        assert calls["count"] == 1
        assert view.step_count() == tiny_market_view.step_count()


class TestReferenceSymbolSelection:
    """The reference/regime feeds carry two symbols; MarketView must pick one."""

    def test_multi_symbol_feed_is_restricted_to_configured_primary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        cfg = replace(tiny_data_config, reference_symbol="ETHUSDT")
        _patch_loaders(monkeypatch, _two_symbol_view(tiny_market_view))
        view = build_market_view(cfg, _TINY_SPLIT)
        assert view.reference["symbol"].unique().to_list() == ["ETHUSDT"]
        assert view.regimes["symbol"].unique().to_list() == ["ETHUSDT"]

    def test_selected_series_is_one_bar_per_minute(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        cfg = replace(tiny_data_config, reference_symbol="ETHUSDT")
        two = _two_symbol_view(tiny_market_view)
        _patch_loaders(monkeypatch, two)
        view = build_market_view(cfg, _TINY_SPLIT)
        # Both symbols share close_time; mixing them doubles the bar count. After
        # selection there is exactly one bar per timestamp.
        assert view.reference.height == tiny_market_view.reference.height
        assert view.reference["close_time"].n_unique() == view.reference.height
        primary = two.reference.filter(pl.col("symbol") == "ETHUSDT").sort("open_time")
        assert view.reference["close"].to_list() == primary["close"].to_list()

    def test_reference_close_uses_primary_not_cross_check(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        cfg = replace(tiny_data_config, reference_symbol="ETHUSDT")
        two = _two_symbol_view(tiny_market_view)
        _patch_loaders(monkeypatch, two)
        view = build_market_view(cfg, _TINY_SPLIT)
        at = two.reference["close_time"].min() + timedelta(seconds=1)
        expected = float(
            two.reference.filter(
                (pl.col("symbol") == "ETHUSDT") & (pl.col("close_time") <= at)
            ).sort("close_time")["close"][-1]
        )
        cross = float(
            two.reference.filter(
                (pl.col("symbol") == "ETHUSDC") & (pl.col("close_time") <= at)
            ).sort("close_time")["close"][-1]
        )
        assert expected != cross  # the synthetic delta makes the series distinguishable
        assert view.reference_close(at) == pytest.approx(expected)

    def test_single_symbol_feed_is_selected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        # The fixture's config names its single ETHUSDC series, so it is selected.
        _patch_loaders(monkeypatch, tiny_market_view)
        view = build_market_view(tiny_data_config, _TINY_SPLIT)
        assert view.reference["symbol"].unique().to_list() == ["ETHUSDC"]
        assert view.regimes["symbol"].unique().to_list() == ["ETHUSDC"]

    def test_single_symbol_feed_missing_configured_symbol_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        # A single-symbol feed is not a licence to use whatever series is present:
        # the configured primary must be the one actually in the feed.
        cfg = replace(tiny_data_config, reference_symbol="ETHUSDT")
        _patch_loaders(monkeypatch, tiny_market_view)
        with pytest.raises(MarketViewError, match="reference_symbol"):
            build_market_view(cfg, _TINY_SPLIT)

    def test_multi_symbol_feed_missing_configured_symbol_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        _patch_loaders(monkeypatch, _two_symbol_view(tiny_market_view))
        cfg = replace(tiny_data_config, reference_symbol="ETHBTC")
        with pytest.raises(MarketViewError, match="reference_symbol"):
            build_market_view(cfg, _TINY_SPLIT)


# ---------------------------------------------------------------------------
# Fixture contract
# ---------------------------------------------------------------------------


class TestFundingAt:
    """S20: the as-of funding lookup the hedge overlay consumes."""

    @staticmethod
    def _funded_view(
        view: MarketView,
        *,
        rates: tuple[float, ...] = (0.001, -0.002),
        marks: tuple[float | None, ...] | None = None,
        symbol: str = "ETHUSDC",
    ) -> MarketView:
        closes = view.reference.sort("close_time")["close_time"].to_list()
        times = [closes[i] for i in range(len(rates))]
        funding = pl.DataFrame(
            {
                "funding_time": times,
                "symbol": [symbol] * len(rates),
                "funding_rate": list(rates),
                "mark_price": list(marks) if marks is not None else [None] * len(rates),
                "is_gap_filled": [False] * len(rates),
            }
        )
        return replace(view, funding=funding)

    def test_view_without_funding_returns_none(
        self, tiny_market_view: MarketView
    ) -> None:
        assert tiny_market_view.funding is None
        assert tiny_market_view.funding_at(tiny_market_view.active_start_utc) is None

    def test_as_of_lookup_is_backward_and_signed(
        self, tiny_market_view: MarketView
    ) -> None:
        view = self._funded_view(tiny_market_view, marks=(3020.0, 3050.0))
        closes = view.reference.sort("close_time")["close_time"].to_list()
        assert view.funding_at(closes[0] - timedelta(minutes=1)) is None
        assert view.funding_at(closes[0]) == (0.001, 3020.0)
        # Between the two settlements the first one is still the as-of row.
        assert view.funding_at(closes[1] - timedelta(minutes=1)) == (0.001, 3020.0)
        assert view.funding_at(closes[1]) == (-0.002, 3050.0)
        assert view.funding_at(closes[2]) == (-0.002, 3050.0)

    def test_null_mark_price_is_nan(self, tiny_market_view: MarketView) -> None:
        import math

        view = self._funded_view(tiny_market_view, marks=(None, None))
        closes = view.reference.sort("close_time")["close_time"].to_list()
        settlement = view.funding_at(closes[0])
        assert settlement is not None
        rate, mark = settlement
        assert rate == 0.001
        assert math.isnan(mark)

    def test_past_the_wall_raises(self, tiny_market_view: MarketView) -> None:
        view = self._funded_view(tiny_market_view)
        with pytest.raises(LookAheadError):
            view.funding_at(
                tiny_market_view.active_end_utc + timedelta(minutes=1)
            )

    def test_slice_propagates_funding(self, tiny_market_view: MarketView) -> None:
        view = self._funded_view(tiny_market_view)
        sub = view.slice(0, 10)
        assert sub.funding is view.funding

    def test_build_market_view_loads_and_selects_primary_funding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_data_config: DataConfig,
        tiny_market_view: MarketView,
    ) -> None:
        import undertow.sim.marketview as mv

        _patch_loaders(monkeypatch, tiny_market_view)
        closes = tiny_market_view.reference.sort("close_time")["close_time"].to_list()
        frame = pl.DataFrame(
            {
                "funding_time": [closes[0]],
                "symbol": ["ETHUSDC"],
                "funding_rate": [0.0007],
                "mark_price": [None],
                "is_gap_filled": [False],
            }
        )
        monkeypatch.setattr(mv, "load_funding", lambda cfg: frame)
        view = build_market_view(tiny_data_config, _TINY_SPLIT)
        assert view.funding is not None
        assert view.funding["symbol"].unique().to_list() == ["ETHUSDC"]
        assert view.funding_at(closes[0]) is not None
        assert view.funding_at(closes[0])[0] == pytest.approx(0.0007)


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

def test_slice_propagates_pool_state(tiny_market_view: MarketView) -> None:
    state = pl.DataFrame({"block_number": [0], "tick": [0]})
    view = replace(tiny_market_view, pool_state=state)
    sub = view.slice(0, 100)
    assert sub.pool_state is not None
    assert sub.pool_state.equals(state)


def _fee_growth_frame(blocks: list[int], pool_address: str) -> pl.DataFrame:
    count = len(blocks)
    return pl.DataFrame(
        {
            "block_number": blocks,
            "tick": [-(2**31)] * count,
            "fee_growth_outside_0_x128": [None] * count,
            "fee_growth_outside_1_x128": [None] * count,
            "liquidity_gross": ["0"] * count,
            "liquidity_net": ["0"] * count,
            "initialized": [False] * count,
            "fee_growth_global_0_x128": ["0"] * count,
            "fee_growth_global_1_x128": ["0"] * count,
            "current_tick": [0] * count,
            "current_liquidity": ["0"] * count,
            "source": ["rpc_call"] * count,
            "pool_address": [pool_address] * count,
            "fee_protocol": [0] * count,
        }
    )


def test_build_market_view_selects_the_seed_block_before_the_first_event(
    monkeypatch: pytest.MonkeyPatch,
    tiny_data_config: DataConfig,
    tiny_market_view: MarketView,
) -> None:
    first = int(tiny_market_view.train["block_number"].min())
    frame = _fee_growth_frame([first, first - 1], str(tiny_data_config.pool.address))
    _patch_loaders(monkeypatch, tiny_market_view, fee_growth=frame)
    view = build_market_view(tiny_data_config, _TINY_SPLIT)
    assert view.pool_state is not None
    assert view.pool_state["block_number"].to_list() == [first - 1]


def test_build_market_view_warns_when_only_the_wrong_seed_block_is_present(
    monkeypatch: pytest.MonkeyPatch,
    tiny_data_config: DataConfig,
    tiny_market_view: MarketView,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first = int(tiny_market_view.train["block_number"].min())
    frame = _fee_growth_frame([first], str(tiny_data_config.pool.address))
    _patch_loaders(monkeypatch, tiny_market_view, fee_growth=frame)
    with caplog.at_level(logging.WARNING, logger="undertow.sim.marketview"):
        view = build_market_view(tiny_data_config, _TINY_SPLIT)
    assert view.pool_state is None
    assert any("no snapshot" in record.message for record in caplog.records)


def test_slice_composes_warmup_across_nested_slices(tiny_market_view: MarketView) -> None:
    sub = tiny_market_view.slice(100, 300).slice(150, 250)
    assert sub.warmup is not None
    assert sub.warmup.height == 150
    assert sub.warmup["seq"].to_list() == list(range(150))


def test_load_market_view_disables_endpoint_expansion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The simulator never fetches: the data config loads without credentials."""
    import undertow.sim.marketview as mv
    from undertow.sim.config import SimConfig

    seen: dict[str, object] = {}
    sentinel = object()

    def fake_load(path: object, **kwargs: object) -> object:
        seen["path"] = path
        seen.update(kwargs)
        return sentinel

    def fake_build(
        data_config: object,
        split_config: object,
        *,
        split: str = "train",
        episode_config: object = None,
    ) -> str:
        assert data_config is sentinel
        seen["split"] = split
        return "VIEW"

    monkeypatch.setattr(mv, "load_data_config", fake_load)
    monkeypatch.setattr(mv, "build_market_view", fake_build)

    result = mv.load_market_view(tmp_path / "data.toml", SimConfig().split, split="train")

    assert result == "VIEW"
    assert seen == {
        "path": tmp_path / "data.toml",
        "expand_endpoints": False,
        "split": "train",
    }
