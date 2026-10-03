"""Look-ahead-safe window onto the historical dataset (S03).

Implements ``docs/plans/sim/CONTRACTS.md`` §4: ``MarketView`` is the *only* door
through which ``undertow.sim`` reads historical data. It owns the walk-forward
split balled in ``SplitConfig`` and makes it impossible for a consumer building a
training view to read evaluation-window rows.

The data loaders shipped by S02 return pyarrow ``Table`` objects; the frozen
contract for this module declares polars ``DataFrame`` fields, so every table is
converted with ``polars.from_arrow`` at the boundary. Historical data is reached
only through the top-level ``undertow.data`` public API
(``docs/decisions/008-data-sim-public-api-reconciliation.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl

from undertow.data import (
    DataConfig,
    PoolConfig,
    load_gas_feed,
    load_reference_feed,
    load_regime_labels,
    load_tape,
)
from undertow.sim.config import EpisodeConfig, SplitConfig
from undertow.sim.types import LookAheadError, MarketViewError, Regime

__all__ = ["MarketView", "build_market_view"]

SplitName = Literal["train", "eval"]


def _to_polars(table: object) -> pl.DataFrame:
    """Convert a pyarrow ``Table`` (or pass through a polars ``DataFrame``)."""
    if isinstance(table, pl.DataFrame):
        return table
    return pl.from_arrow(table)  # type: ignore[arg-type]


def _sort(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Sort ``df`` by ``columns`` when they are present and the frame is non-empty."""
    if df.height == 0:
        return df
    present = [c for c in columns if c in df.columns]
    if not present:
        return df
    return df.sort(present)


def _between(df: pl.DataFrame, column: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Rows with ``start <= df[column] < end`` (half-open, UTC).

    Half-open on the upper bound keeps the train/eval splits disjoint even when
    ``SplitConfig.train_end_utc == eval_start_utc`` (the boundary row belongs to
    the eval split only).
    """
    if df.height == 0:
        return df
    return df.filter((pl.col(column) >= start) & (pl.col(column) < end))


@dataclass(frozen=True, slots=True)
class MarketView:
    """A look-ahead-safe view of the train OR eval split. Never both.

    See ``CONTRACTS.md`` §4/§4.1. ``reference`` is always the *entire* reference
    feed; ``reference_close`` / ``reference_bars`` enforce the active split's
    upper wall so an evaluation-window row can never be read while building a
    training view.
    """

    # -- Splits (disjoint; exactly one is active) --
    train: pl.DataFrame | None  # train-split event tape, seq-sorted
    eval: pl.DataFrame | None  # eval-split event tape, seq-sorted
    reference: pl.DataFrame  # reference feed, sorted by open_time
    regimes: pl.DataFrame  # regime labels, sorted by timestamp
    gas: pl.DataFrame  # gas feed, sorted by block_number

    # -- Pinned split boundaries (UTC) --
    train_start_utc: datetime
    train_end_utc: datetime
    eval_start_utc: datetime
    eval_end_utc: datetime
    is_train: bool  # True if this view was built for training

    # -- Extension (additive, defaulted; PLAN §0.2) --
    #: Episode length driver for :meth:`build_episode`. Kept out of the frozen
    #: CONTRACTS signature, which only passes ``seed``/``max_steps``.
    episode_config: EpisodeConfig | None = None
    #: The ``PoolConfig`` the active tape was pulled from (ADR-013 additive
    #: extension). ``build_market_view`` populates it from ``DataConfig.pool``;
    #: hand-built views may leave it ``None`` and the backtester falls back to
    #: a pool derived from ``SimConfig.episode``.
    pool: PoolConfig | None = None

    # ------------------------------------------------------------------
    # Invariants (CONTRACTS §4.1)
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.train is not None and self.eval is not None:
            raise MarketViewError(
                "MarketView: train and eval cannot both be active (look-ahead risk)"
            )
        if self.train is None and self.eval is None:
            raise MarketViewError("MarketView: exactly one of train/eval must be active")
        if self.train is not None and not self.is_train:
            raise MarketViewError("MarketView: train data present but is_train is False")
        if self.eval is not None and self.is_train:
            raise MarketViewError("MarketView: eval data present but is_train is True")

        active = self.active_split_data()
        if active.height == 0:
            raise MarketViewError("MarketView: active split contains no tape rows")
        if "seq" not in active.columns:
            raise MarketViewError("MarketView: active split has no 'seq' column")

        self._check_reference_spans_split()

    def _check_reference_spans_split(self) -> None:
        """Invariant 3: the reference feed covers the active split's bounds."""
        if self.reference.height == 0:
            raise MarketViewError("MarketView: reference feed is empty")
        for column in ("open_time", "close_time"):
            if column not in self.reference.columns:
                raise MarketViewError(
                    f"MarketView: reference feed has no '{column}' column"
                )
        ref_start = self.reference["open_time"].min()
        ref_end = self.reference["close_time"].max()
        if ref_start is None or ref_end is None:
            raise MarketViewError("MarketView: reference feed has no timestamps")
        if ref_start > self.active_start_utc or ref_end < self.active_end_utc:
            raise MarketViewError(
                "MarketView: reference feed does not span the active split "
                f"[{self.active_start_utc}, {self.active_end_utc}] "
                f"(feed covers [{ref_start}, {ref_end}])"
            )

    # ------------------------------------------------------------------
    # Active-split helpers
    # ------------------------------------------------------------------
    @property
    def active_start_utc(self) -> datetime:
        """Lower time bound of the active split."""
        return self.train_start_utc if self.is_train else self.eval_start_utc

    @property
    def active_end_utc(self) -> datetime:
        """Upper time bound of the active split (data rows are ``<`` this bound)."""
        return self.train_end_utc if self.is_train else self.eval_end_utc

    def _active_range(self) -> tuple[int, int]:
        """Inclusive ``(min_seq, max_seq)`` of the active split."""
        active = self.active_split_data()
        return int(active["seq"].min()), int(active["seq"].max())

    def _check_time(self, at_time: datetime) -> None:
        """Raise ``LookAheadError`` when ``at_time`` is past the active wall."""
        if at_time > self.active_end_utc:
            raise LookAheadError(
                f"time {at_time} is past the active split wall {self.active_end_utc}"
            )

    # ------------------------------------------------------------------
    # Query methods (CONTRACTS §4)
    # ------------------------------------------------------------------
    def slice(self, start_seq: int, end_seq: int) -> MarketView:
        """Return a sub-view over ``[start_seq, end_seq)``.

        Raises ``LookAheadError`` if the requested range extends beyond the
        active split's seq range (or starts before it).
        """
        active = self.active_split_data()
        min_seq, last_seq = self._active_range()
        if start_seq < min_seq or end_seq > last_seq + 1:
            raise LookAheadError(
                f"slice [{start_seq}, {end_seq}) is outside the active split "
                f"seq range [{min_seq}, {last_seq}]"
            )
        if start_seq >= end_seq:
            raise MarketViewError(
                f"slice start_seq ({start_seq}) must be < end_seq ({end_seq})"
            )

        sub = active.filter((pl.col("seq") >= start_seq) & (pl.col("seq") < end_seq))
        if sub.height == 0:
            raise MarketViewError(
                f"slice [{start_seq}, {end_seq}) produced an empty view"
            )

        sub_start = sub["block_timestamp"].min()
        sub_end = sub["block_timestamp"].max()
        if sub_start is None or sub_end is None:
            raise MarketViewError("slice produced rows with no block_timestamp")

        if self.is_train:
            return MarketView(
                train=sub,
                eval=None,
                reference=self.reference,
                regimes=self.regimes,
                gas=self.gas,
                train_start_utc=sub_start,
                train_end_utc=sub_end,
                eval_start_utc=self.eval_start_utc,
                eval_end_utc=self.eval_end_utc,
                is_train=True,
                episode_config=self.episode_config,
                pool=self.pool,
            )
        return MarketView(
            train=None,
            eval=sub,
            reference=self.reference,
            regimes=self.regimes,
            gas=self.gas,
            train_start_utc=self.train_start_utc,
            train_end_utc=self.train_end_utc,
            eval_start_utc=sub_start,
            eval_end_utc=sub_end,
            is_train=False,
            episode_config=self.episode_config,
            pool=self.pool,
        )

    def tape_at(self, seq: int) -> dict[str, object]:
        """All columns for the event at ``seq`` as a dict.

        Raises ``LookAheadError`` for a seq outside the active split's range.
        """
        min_seq, last_seq = self._active_range()
        if seq < min_seq or seq > last_seq:
            raise LookAheadError(
                f"seq {seq} is outside the active split seq range [{min_seq}, {last_seq}]"
            )
        sub = self.active_split_data().filter(pl.col("seq") == seq)
        if sub.height == 0:
            raise MarketViewError(f"no tape row at seq={seq}")
        return dict(sub.row(0, named=True))

    def reference_close(self, at_time: datetime) -> float | None:
        """Last reference close with ``close_time <= at_time``.

        Returns ``None`` if no bar precedes ``at_time`` (before the feed starts).
        Raises ``LookAheadError`` when ``at_time`` is past the active wall.
        """
        self._check_time(at_time)
        sub = self.reference.filter(pl.col("close_time") <= at_time)
        if sub.height == 0:
            return None
        return float(sub.sort("close_time")["close"][-1])

    def reference_bars(self, start_time: datetime, end_time: datetime) -> pl.DataFrame:
        """Reference bars in ``[start_time, end_time)``, clamped to the split.

        The upper bound is clamped to the active wall (it can never leak an
        evaluation bar into a training view). A ``start_time`` at or beyond the
        wall is a look-ahead read and raises ``LookAheadError``.
        """
        if start_time >= self.active_end_utc:
            raise LookAheadError(
                f"reference_bars start {start_time} is past the active split "
                f"wall {self.active_end_utc}"
            )
        clamped_start = max(start_time, self.active_start_utc)
        clamped_end = min(end_time, self.active_end_utc)
        if clamped_start >= clamped_end:
            return self.reference.clear()
        # No look-ahead: a bar may only be used once its close_time is known,
        # so exclude any bar whose close falls after the wall. Filtering on
        # open_time alone would leak a bar that straddles the boundary.
        return self.reference.filter(
            (pl.col("open_time") >= clamped_start) & (pl.col("close_time") <= clamped_end)
        ).sort("open_time")

    def regime_at(self, at_time: datetime) -> str:
        """Regime label at ``at_time`` (as-of join).

        Returns ``"unknown"`` if no label covers ``at_time``. Raises
        ``LookAheadError`` when ``at_time`` is past the active wall.
        """
        self._check_time(at_time)
        if self.regimes.height == 0:
            return Regime.UNKNOWN.value
        sub = self.regimes.filter(pl.col("timestamp") <= at_time).sort("timestamp")
        if sub.height == 0:
            return Regime.UNKNOWN.value
        return str(sub["regime"][-1])

    def gas_at_block(self, block_number: int) -> dict[str, object] | None:
        """Gas row for ``block_number``. ``None`` if no gas data for that block.

        The gas feed is chain-wide (not split-scoped), so this takes no time/seq
        argument and needs no wall check.
        """
        sub = self.gas.filter(pl.col("block_number") == block_number)
        if sub.height == 0:
            return None
        return dict(sub.row(0, named=True))

    def active_split_data(self) -> pl.DataFrame:
        """The event tape for the active split (train or eval), sorted by seq."""
        active = self.train if self.is_train else self.eval
        if active is None:  # pragma: no cover - __post_init__ guarantees this
            raise MarketViewError("MarketView: no active split data")
        if "seq" in active.columns:
            return active.sort("seq")
        return active

    def step_count(self) -> int:
        """Number of tape events in the active split."""
        return self.active_split_data().height

    def build_episode(
        self,
        seed: int,
        max_steps: int | None = None,
        *,
        episode_config: EpisodeConfig | None = None,
    ) -> tuple[int, int]:
        """Sample a contiguous episode window of the active split.

        Returns ``(start_seq, end_seq)`` where ``end_seq`` is exclusive. The
        window is config-driven (``EpisodeConfig.duration_days → steps`` at
        ``step_minutes`` cadence), capped by ``max_steps`` when supplied and by
        the active split's length. It never crosses the train/eval boundary.

        ``seq`` is the dense canonical tape step index (EVENT_TAPE_SCHEMA), so
        ``end_seq - start_seq`` equals the number of sampled rows.
        """
        ep = episode_config or self.episode_config or EpisodeConfig()
        if ep.step_minutes <= 0:
            raise MarketViewError("build_episode: step_minutes must be > 0")
        n = self.step_count()
        config_steps = (ep.duration_days * 24 * 60) // ep.step_minutes
        desired = config_steps if max_steps is None else int(max_steps)
        if desired <= 0:
            raise MarketViewError(f"build_episode: desired window must be > 0, got {desired}")
        window_len = min(desired, n)

        seqs = self.active_split_data()["seq"].to_numpy()
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, n - window_len + 1))
        start_seq = int(seqs[start_idx])
        end_seq = int(seqs[start_idx + window_len - 1]) + 1
        return start_seq, end_seq


def build_market_view(
    data_config: DataConfig,
    split_config: SplitConfig,
    *,
    split: SplitName = "train",
    episode_config: EpisodeConfig | None = None,
) -> MarketView:
    """Load the dataset and partition it into a look-ahead-safe ``MarketView``.

    Historical tables are read only through the top-level ``undertow.data``
    public API and converted from pyarrow to polars. Raises ``MarketViewError``
    if the requested split window contains no tape rows.
    """
    if split not in ("train", "eval"):
        raise MarketViewError(f"build_market_view: unknown split {split!r}")

    tape = _sort(_to_polars(load_tape(data_config)), ["block_number", "log_index"])
    reference = _sort(_to_polars(load_reference_feed(data_config)), ["open_time"])
    gas = _sort(_to_polars(load_gas_feed(data_config)), ["block_number"])
    regimes = _sort(_to_polars(load_regime_labels(data_config)), ["timestamp"])

    if split == "train":
        start, end = split_config.train_start_utc, split_config.train_end_utc
        active = _between(tape, "block_timestamp", start, end)
        if active.height == 0:
            raise MarketViewError(
                f"train split [{start}, {end}] contains no tape rows"
            )
        return MarketView(
            train=active,
            eval=None,
            reference=reference,
            regimes=regimes,
            gas=gas,
            train_start_utc=split_config.train_start_utc,
            train_end_utc=split_config.train_end_utc,
            eval_start_utc=split_config.eval_start_utc,
            eval_end_utc=split_config.eval_end_utc,
            is_train=True,
            episode_config=episode_config,
            pool=data_config.pool,
        )

    start, end = split_config.eval_start_utc, split_config.eval_end_utc
    active = _between(tape, "block_timestamp", start, end)
    if active.height == 0:
        raise MarketViewError(f"eval split [{start}, {end}] contains no tape rows")
    return MarketView(
        train=None,
        eval=active,
        reference=reference,
        regimes=regimes,
        gas=gas,
        train_start_utc=split_config.train_start_utc,
        train_end_utc=split_config.train_end_utc,
        eval_start_utc=split_config.eval_start_utc,
        eval_end_utc=split_config.eval_end_utc,
        is_train=False,
        episode_config=episode_config,
        pool=data_config.pool,
    )