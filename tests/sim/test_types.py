"""Tests for ``undertow.sim.types``."""

from __future__ import annotations

import pytest

from undertow.sim.types import (
    Action,
    BacktestError,
    EnvError,
    LookAheadError,
    MarketViewError,
    ParityError,
    PositionError,
    PriceMode,
    Regime,
    SimConfigError,
    Tick,
    TickSpacing,
    UndertowSimError,
)


class TestAction:
    def test_is_hold(self) -> None:
        assert Action("hold").is_hold()
        assert not Action("rebalance", center_offset=0, width=5).is_hold()

    def test_lower_tick_raises_for_hold(self) -> None:
        action = Action("hold")
        with pytest.raises(PositionError, match="hold"):
            action.lower_tick(Tick(100), TickSpacing(60))

    def test_upper_tick_raises_for_hold(self) -> None:
        action = Action("hold")
        with pytest.raises(PositionError, match="hold"):
            action.upper_tick(Tick(100), TickSpacing(60))

    def test_lower_tick_centered(self) -> None:
        """Center on current tick, width=2 spacings. lower = current - width*spacing."""
        action = Action("rebalance", center_offset=0, width=2)
        result = action.lower_tick(Tick(0), TickSpacing(60))
        assert result == Tick(-120)  # 0 + 0*60 - 2*60 = -120, already aligned

    def test_lower_tick_offset(self) -> None:
        """Offset -2, width=5. lower = current + (-2)*60 - 5*60 = -420."""
        action = Action("rebalance", center_offset=-2, width=5)
        result = action.lower_tick(Tick(100), TickSpacing(60))
        # 100 + (-2*60) - (5*60) = 100 - 120 - 300 = -320 → floor to spacing
        # -320 is already aligned to 60 (-320/60 = -5.33, floor=-6, -6*60 = -360)
        # Actually -320 // 60 = -6 (Python floor division), -6*60 = -360
        assert result == Tick(-360)

    def test_upper_tick_centered(self) -> None:
        """Center on current tick, width=2 spacings."""
        action = Action("rebalance", center_offset=0, width=2)
        result = action.upper_tick(Tick(0), TickSpacing(60))
        assert result == Tick(120)  # 0 + 0*60 + 2*60 = 120, already aligned

    def test_upper_tick_offset(self) -> None:
        """Offset 1, width=3."""
        action = Action("rebalance", center_offset=1, width=3)
        result = action.upper_tick(Tick(-50), TickSpacing(60))
        # -50 + 1*60 + 3*60 = 190 → ceil to spacing → 240
        assert result == Tick(240)

    def test_lower_upper_are_ordered(self) -> None:
        """Lower tick is always strictly below upper tick for width > 0."""
        action = Action("rebalance", center_offset=-1, width=5)
        lower = action.lower_tick(Tick(1000), TickSpacing(60))
        upper = action.upper_tick(Tick(1000), TickSpacing(60))
        assert lower < upper
        assert (upper - lower) >= 2 * int(action.width) * 60  # at least width*spacing each side


class TestRegime:
    def test_members(self) -> None:
        assert Regime.BULL.value == "bull"
        assert Regime.BEAR.value == "bear"
        assert Regime.SIDEWAYS.value == "sideways"
        assert Regime.HIGH_VOL.value == "high_vol"
        assert Regime.UNKNOWN.value == "unknown"

    def test_is_string_enum(self) -> None:
        assert isinstance(Regime.BULL, str)


class TestPriceMode:
    def test_members(self) -> None:
        assert PriceMode.REPLAY.value == "replay"
        assert PriceMode.CALIBRATED.value == "calibrated"


class TestExceptionHierarchy:
    def test_sim_error_is_base(self) -> None:
        with pytest.raises(UndertowSimError):
            raise UndertowSimError("base")

    def test_config_error_catches(self) -> None:
        with pytest.raises(UndertowSimError):
            raise SimConfigError("bad config")

    def test_look_ahead_is_market_view_error(self) -> None:
        err = LookAheadError("past the wall")
        assert isinstance(err, MarketViewError)
        assert isinstance(err, UndertowSimError)

    def test_market_view_error_not_look_ahead(self) -> None:
        err = MarketViewError("generic market error")
        assert not isinstance(err, LookAheadError)

    def test_all_concrete_errors_are_raisable(self) -> None:
        for cls in (
            SimConfigError,
            MarketViewError,
            LookAheadError,
            PositionError,
            EnvError,
            BacktestError,
            ParityError,
        ):
            with pytest.raises(UndertowSimError):
                raise cls("test")
