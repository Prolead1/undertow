"""Deterministic reference-feed replay price process (S06, ``CONTRACTS.md`` §7).

:class:`ReplayPriceProcess` replays the human USDC/WETH closes of the
``MarketView`` reference feed over a ``(start_seq, end_seq)`` episode window.
Every returned triplet is mutually consistent per ADR-009.
"""

from __future__ import annotations

from datetime import datetime

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


class ReplayPriceProcess:
    """Replays reference-feed close prices. Deterministic given the feed slice.

    Constructed from a :class:`~undertow.sim.marketview.MarketView` plus a
    ``(start_seq, end_seq)`` episode window. The window's tape timestamps are
    mapped to the reference bars covering it (as-of, look-ahead-safe); each
    :meth:`step` advances one bar. The process never reads a bar whose
    ``close_time`` falls after the window or outside the active split.
    """

    def __init__(self, market_view: MarketView, start_seq: int, end_seq: int) -> None:
        self._dec0 = 6
        self._dec1 = 18
        self._start_seq = int(start_seq)
        self._end_seq = int(end_seq)
        if self._start_seq >= self._end_seq:
            raise MarketViewError(
                f"ReplayPriceProcess: start_seq ({self._start_seq}) must be < "
                f"end_seq ({self._end_seq})"
            )

        # ``slice`` validates the window against the active split and raises
        # ``LookAheadError`` if it extends past the wall, so the replayed bars
        # can never come from an out-of-window or future row.
        window_view = market_view.slice(self._start_seq, self._end_seq)
        tape = window_view.active_split_data()
        if tape.height == 0:
            raise MarketViewError(
                f"ReplayPriceProcess: no tape rows in [{self._start_seq}, {self._end_seq})"
            )
        start_time = tape["block_timestamp"].min()
        end_time = tape["block_timestamp"].max()
        if start_time is None or end_time is None:
            raise MarketViewError("ReplayPriceProcess: episode window has no timestamps")

        bars = window_view.reference_bars(start_time, end_time)
        if bars.height == 0:
            raise MarketViewError(
                "ReplayPriceProcess: no reference bars cover the episode window"
            )

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
