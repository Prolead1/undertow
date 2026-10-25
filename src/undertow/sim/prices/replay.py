"""Deterministic reference-feed replay price process (S06, ``CONTRACTS.md`` §7).

:class:`ReplayPriceProcess` replays the human USDC/WETH closes of the
``MarketView`` reference feed over a ``(start_seq, end_seq)`` episode window.
Every returned triplet is mutually consistent per ADR-009.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from undertow.sim.marketview import MarketView
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.types import (
    MarketViewError,
    Price,
    PriceMode,
    SqrtPrice,
    Tick,
)

__all__ = ["ReplayPriceProcess"]


def _resample_decision_bars(bars: pl.DataFrame, step_minutes: int) -> pl.DataFrame:
    """Aggregate 1-minute reference bars onto ``step_minutes`` decision boundaries.

    The reference feed is 1-minute, but the pinned decision cadence is
    ``EpisodeConfig.step_minutes`` (ADR-010). A decision bar is the **last** bar
    whose ``close_time`` falls in each ``step_minutes`` bucket, so its close is
    observed exactly at (or just after) the decision time — still backward-only,
    never a bar from the next interval. ``step_minutes <= 1`` is the identity, so
    callers that already pass decision-cadence bars (the shared fixtures, the
    backtester's own cadence) are unaffected.
    """
    if step_minutes <= 1 or bars.height == 0:
        return bars
    return (
        bars.sort("close_time")
        .with_columns(
            pl.col("close_time").dt.truncate(f"{int(step_minutes)}m").alias("_decision_bucket")
        )
        .group_by("_decision_bucket", maintain_order=True)
        .agg(pl.all().last())
        .drop("_decision_bucket")
        .sort("close_time")
    )


class ReplayPriceProcess:
    """Replays reference-feed close prices. Deterministic given the feed slice.

    Constructed from a :class:`~undertow.sim.marketview.MarketView` plus a
    ``(start_seq, end_seq)`` episode window. The window's tape timestamps are
    mapped to the reference bars covering it (as-of, look-ahead-safe). When
    ``step_minutes > 1`` the 1-minute feed is aggregated onto ``step_minutes``
    decision boundaries (ADR-010) so each :meth:`step` is one **decision**, not
    one minute; the default ``step_minutes=1`` leaves the bars untouched. The
    process never reads a bar whose ``close_time`` falls after the window or
    outside the active split.
    """

    def __init__(
        self,
        market_view: MarketView,
        start_seq: int,
        end_seq: int,
        *,
        step_minutes: int = 1,
    ) -> None:
        self._dec0 = 6
        self._dec1 = 18
        self._start_seq = int(start_seq)
        self._end_seq = int(end_seq)
        self._step_minutes = int(step_minutes)
        if self._step_minutes <= 0:
            raise MarketViewError(
                f"ReplayPriceProcess: step_minutes must be > 0, got {step_minutes!r}"
            )
        if self._start_seq >= self._end_seq:
            raise MarketViewError(
                f"ReplayPriceProcess: start_seq ({self._start_seq}) must be < "
                f"end_seq ({self._end_seq})"
            )

        # ``window`` validates the window against the active split with the same
        # bounds as ``slice`` but returns the window's own rows without
        # materialising the lane's pre-window rows into ``warmup`` — the price
        # process never reads that history, and copying it on every reset cost
        # a transient copy of up to the whole lane.
        tape = market_view.window(self._start_seq, self._end_seq)
        if tape.height == 0:
            raise MarketViewError(
                f"ReplayPriceProcess: no tape rows in [{self._start_seq}, {self._end_seq})"
            )
        start_time_raw = tape["block_timestamp"].min()
        end_time_raw = tape["block_timestamp"].max()
        if start_time_raw is None or end_time_raw is None:
            raise MarketViewError("ReplayPriceProcess: episode window has no timestamps")

        # ``reference_bars`` clamps to the parent's active split, which is a
        # superset of the window: the bars are identical to clamping against the
        # temporary sliced view (the window's own bounds).
        bars = market_view.reference_bars(
            cast(datetime, start_time_raw), cast(datetime, end_time_raw)
        )
        if bars.height == 0:
            raise MarketViewError(
                "ReplayPriceProcess: no reference bars cover the episode window"
            )
        bars = _resample_decision_bars(bars, self._step_minutes)

        self._bars: pl.DataFrame = bars
        self._closes: np.ndarray = bars["close"].to_numpy().astype(float)
        self._close_times: list[datetime] = bars["close_time"].to_list()
        self._index = 0

    # -- Introspection (read-only) ---------------------------------------
    @property
    def start_seq(self) -> int:
        """Inclusive start of the episode window."""
        return self._start_seq

    @property
    def end_seq(self) -> int:
        """Exclusive end of the episode window."""
        return self._end_seq

    @property
    def n_bars(self) -> int:
        """Number of reference bars in the episode window."""
        return int(self._closes.size)

    @property
    def close_times(self) -> list[datetime]:
        """Close timestamps of the stored bars (in replay order)."""
        return list(self._close_times)

    # -- PriceProcess ----------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        """Restart from the first bar."""
        self._index = 0

    def step(self, rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]:
        """Return the next bar's consistent ``(price, sqrt_price, tick)`` triplet."""
        if self._index >= self._closes.size:
            raise StopIteration("ReplayPriceProcess exhausted its reference bars")
        price = float(self._closes[self._index])
        self._index += 1
        return human_price_to_triplet(price, self._dec0, self._dec1)

    @property
    def mode(self) -> str:
        """Always ``"replay"``."""
        return PriceMode.REPLAY.value
