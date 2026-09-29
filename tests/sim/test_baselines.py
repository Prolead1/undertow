"""Behavioural tests for the S10 baseline ladder (CONTRACTS.md §10).

These assert real action sequences over synthetic observations: hold vs
rebalance, correct centre/width, tick snapping, the τ-reset exit trigger, the
cost-aware interval + fee-vs-gas gate, determinism, and the absence of any
learning API on the frozen ``Policy`` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from undertow.data import MAX_TICK, MIN_TICK, tick_to_price
from undertow.sim.core.position import DEFAULT_DEC0, DEFAULT_DEC1
from undertow.sim.policies import (
    FULL_RANGE_WIDTH,
    CostAwareRebalancePolicy,
    FullRangeV2Policy,
    HODLPolicy,
    PassiveNarrowPolicy,
    PassiveWidePolicy,
    Policy,
    TauResetPolicy,
)
from undertow.sim.policies.base import HOLDAction

ENTRY_TICK = 196242
ENTRY_PRICE = float(tick_to_price(ENTRY_TICK, DEFAULT_DEC0, DEFAULT_DEC1))
TICK_SPACING = 60


@dataclass(frozen=True, slots=True)
class _StubObservation:
    """The CONTRACTS.md §11 observation shape, restricted to the fields S10 reads.

    Deliberately **not** named ``Observation`` — S12 owns that type; this is only a
    test-local stand-in.
    """

    price: float = ENTRY_PRICE
    tick: int = ENTRY_TICK
    position_in_range: bool = True
    uncollected_fees_token0: float = 0.0
    step_in_episode: int = 0
    gas_price_recent_wei: float = 21e9
    eth_usd_price_recent: float = 3000.0


class _FakeGasModel:
    """Minimal ``GasModel``-shaped stub returning a fixed USDC cost per action."""

    def __init__(self, cost_usdc: float) -> None:
        self.cost_usdc = cost_usdc
        self.calls: list[str] = []

    def gas_cost_usdc(
        self,
        action_type: str,
        block_number: int,
        base_fee_per_gas_wei: int,
        priority_fee_p50_wei: int,
        eth_usd_price: float,
    ) -> float:
        self.calls.append(action_type)
        return self.cost_usdc


def _obs(**kwargs: float | int) -> _StubObservation:
    return _StubObservation(**kwargs)  # type: ignore[arg-type]


def _all_policies() -> list[object]:
    return [
        HODLPolicy(),
        PassiveNarrowPolicy(),
        PassiveWidePolicy(),
        FullRangeV2Policy(),
        TauResetPolicy(120),
        CostAwareRebalancePolicy(120, 144, _FakeGasModel(5.0)),
    ]


# ---------------------------------------------------------------------------
# HODL
# ---------------------------------------------------------------------------


def test_hodl_always_holds() -> None:
    policy = HODLPolicy()
    for step in range(3):
        action = policy.act(_obs(step_in_episode=step, tick=ENTRY_TICK + step * 500))
        assert action.is_hold()
        assert action == HOLDAction
    assert policy.name == "hodl"


def test_hodl_accepts_no_rng() -> None:
    assert HODLPolicy().act(_obs(), rng=None).is_hold()


# ---------------------------------------------------------------------------
# Passive narrow / wide
# ---------------------------------------------------------------------------


def test_passive_narrow_deploys_then_holds() -> None:
    policy = PassiveNarrowPolicy()
    first = policy.act(_obs())
    assert not first.is_hold()
    assert first.center_offset == 0
    assert first.width == 8  # ±5 % at the ADR-009 entry price, spacing 60
    assert first.lower_tick(ENTRY_TICK, TICK_SPACING) == 195_720
    assert first.upper_tick(ENTRY_TICK, TICK_SPACING) == 196_740
    assert policy.act(_obs()).is_hold()
    assert policy.name == "passive_narrow"


def test_passive_wide_is_wider_than_narrow() -> None:
    narrow = PassiveNarrowPolicy().act(_obs())
    wide = PassiveWidePolicy().act(_obs())
    assert wide.width > narrow.width
    assert wide.width == 33  # ±20 %
    assert wide.lower_tick(ENTRY_TICK, TICK_SPACING) == 194_220
    assert wide.upper_tick(ENTRY_TICK, TICK_SPACING) == 198_240
    assert PassiveWidePolicy().name == "passive_wide"


def test_passive_range_is_tick_snapped() -> None:
    action = PassiveNarrowPolicy().act(_obs(tick=ENTRY_TICK + 13))
    lower = action.lower_tick(ENTRY_TICK + 13, TICK_SPACING)
    upper = action.upper_tick(ENTRY_TICK + 13, TICK_SPACING)
    assert lower % TICK_SPACING == 0
    assert upper % TICK_SPACING == 0
    assert lower < upper


def test_passive_reset_redeploys() -> None:
    policy = PassiveNarrowPolicy()
    assert not policy.act(_obs()).is_hold()
    assert policy.act(_obs()).is_hold()
    policy.reset()
    assert not policy.act(_obs()).is_hold()


# ---------------------------------------------------------------------------
# Full-range V2
# ---------------------------------------------------------------------------


def test_full_range_deploys_at_protocol_bounds() -> None:
    policy = FullRangeV2Policy()
    observation = _obs(tick=ENTRY_TICK)
    action = policy.act(observation)
    assert not action.is_hold()
    assert action.width == FULL_RANGE_WIDTH
    assert policy.name == "full_range_v2"

    # Exercise the consumer path: the returned Action must resolve to a valid,
    # near-full range that stays inside the protocol bounds (ADR-011).
    lower = action.lower_tick(ENTRY_TICK, TICK_SPACING)
    upper = action.upper_tick(ENTRY_TICK, TICK_SPACING)
    assert MIN_TICK <= lower < upper <= MAX_TICK
    assert upper - lower >= 0.999 * (MAX_TICK - MIN_TICK)
    assert lower <= ENTRY_TICK <= upper
    assert policy.act(observation).is_hold()


@pytest.mark.parametrize("current_tick", [MIN_TICK + 1, 0, ENTRY_TICK, MAX_TICK - 1])
def test_full_range_resolved_bounds_stay_valid(current_tick: int) -> None:
    action = FullRangeV2Policy().act(_obs(tick=current_tick))
    lower = action.lower_tick(current_tick, TICK_SPACING)
    upper = action.upper_tick(current_tick, TICK_SPACING)
    assert MIN_TICK <= lower < upper <= MAX_TICK
    assert lower % TICK_SPACING == 0
    assert upper % TICK_SPACING == 0


def test_full_range_respects_tick_spacing_override() -> None:
    action = FullRangeV2Policy(tick_spacing=30).act(_obs(tick=ENTRY_TICK))
    lower = action.lower_tick(ENTRY_TICK, 30)
    upper = action.upper_tick(ENTRY_TICK, 30)
    assert MIN_TICK <= lower < upper <= MAX_TICK
    assert lower % 30 == 0
    assert upper % 30 == 0


def test_full_range_rejects_spacing_too_large_for_a_range() -> None:
    with pytest.raises(ValueError, match="too large"):
        FullRangeV2Policy(tick_spacing=MAX_TICK + 1).act(_obs(tick=ENTRY_TICK))


def test_full_range_reset_redeploys() -> None:
    policy = FullRangeV2Policy()
    policy.act(_obs())
    policy.reset()
    assert not policy.act(_obs()).is_hold()


# ---------------------------------------------------------------------------
# Tau reset
# ---------------------------------------------------------------------------


def test_tau_reset_deploys_on_first_call() -> None:
    policy = TauResetPolicy(120)
    action = policy.act(_obs())
    assert not action.is_hold()
    assert action.center_offset == 0
    assert action.width == 2  # 120 ticks / 60 spacing
    assert policy.name == "tau_reset"


def test_tau_reset_holds_while_in_range() -> None:
    policy = TauResetPolicy(120)
    policy.act(_obs(tick=ENTRY_TICK))
    # The snapped band is [196080, 196380]; interior ticks must hold.
    for tick in (196_080, 196_200, 196_380):
        assert policy.act(_obs(tick=tick, step_in_episode=1)).is_hold()


def test_tau_reset_recenters_on_exit() -> None:
    policy = TauResetPolicy(120)
    policy.act(_obs(tick=ENTRY_TICK))
    exit_tick = 196_500  # above the snapped upper bound 196380
    action = policy.act(_obs(tick=exit_tick, step_in_episode=1))
    assert not action.is_hold()
    assert action.width == 2
    assert action.lower_tick(exit_tick, TICK_SPACING) == 196_380
    assert action.upper_tick(exit_tick, TICK_SPACING) == 196_620
    # Once recentred, an interior tick holds again.
    assert policy.act(_obs(tick=196_500, step_in_episode=2)).is_hold()
    # Exiting below the new lower bound also triggers a recenter.
    assert not policy.act(_obs(tick=196_300, step_in_episode=3)).is_hold()


def test_tau_reset_reset_redeploys() -> None:
    policy = TauResetPolicy(120)
    policy.act(_obs(tick=ENTRY_TICK))
    policy.reset()
    assert not policy.act(_obs(tick=ENTRY_TICK)).is_hold()


# ---------------------------------------------------------------------------
# Cost-aware rebalance
# ---------------------------------------------------------------------------


def test_cost_aware_deploys_unconditionally_on_first_call() -> None:
    gas = _FakeGasModel(1_000_000.0)  # prohibitive gas; first deploy still happens
    policy = CostAwareRebalancePolicy(120, 144, gas)
    action = policy.act(_obs(uncollected_fees_token0=0.0))
    assert not action.is_hold()
    assert policy.name == "cost_aware"


def test_cost_aware_respects_min_interval() -> None:
    gas = _FakeGasModel(1.0)
    policy = CostAwareRebalancePolicy(120, 144, gas)
    policy.act(_obs(step_in_episode=0))
    action = policy.act(
        _obs(tick=196_500, step_in_episode=10, uncollected_fees_token0=10_000.0)
    )
    assert action.is_hold()
    assert gas.calls == []  # gas never even consulted before the interval elapses


def test_cost_aware_holds_when_fees_below_gas() -> None:
    gas = _FakeGasModel(5.0)
    policy = CostAwareRebalancePolicy(120, 0, gas)
    policy.act(_obs(step_in_episode=0))
    action = policy.act(
        _obs(tick=196_500, step_in_episode=100, uncollected_fees_token0=4.99)
    )
    assert action.is_hold()
    assert gas.calls == ["rebalance"]


def test_cost_aware_rebalances_when_both_conditions_met() -> None:
    gas = _FakeGasModel(5.0)
    policy = CostAwareRebalancePolicy(120, 0, gas)
    policy.act(_obs(step_in_episode=0))
    exit_tick = 196_500
    action = policy.act(
        _obs(tick=exit_tick, step_in_episode=100, uncollected_fees_token0=5.01)
    )
    assert not action.is_hold()
    assert action.width == 2
    assert action.lower_tick(exit_tick, TICK_SPACING) == 196_380
    assert action.upper_tick(exit_tick, TICK_SPACING) == 196_620
    assert gas.calls == ["rebalance"]


def test_cost_aware_rejects_in_range_before_gas_check() -> None:
    gas = _FakeGasModel(0.0)
    policy = CostAwareRebalancePolicy(120, 0, gas)
    policy.act(_obs(step_in_episode=0))
    action = policy.act(
        _obs(tick=ENTRY_TICK + 10, step_in_episode=1, uncollected_fees_token0=1e9)
    )
    assert action.is_hold()
    assert gas.calls == []


def test_cost_aware_reset_resets_state() -> None:
    gas = _FakeGasModel(1.0)
    policy = CostAwareRebalancePolicy(120, 144, gas)
    policy.act(_obs(step_in_episode=0))
    # Not enough interval elapsed -> hold.
    assert policy.act(_obs(tick=196_500, step_in_episode=5)).is_hold()
    policy.reset()
    # After reset the next call is an unconditional deploy again.
    assert not policy.act(_obs(step_in_episode=100)).is_hold()


# ---------------------------------------------------------------------------
# Frozen-protocol invariants
# ---------------------------------------------------------------------------


def test_all_baselines_satisfy_policy_protocol() -> None:
    for policy in _all_policies():
        assert isinstance(policy, Policy)


def test_policy_protocol_rejects_incomplete_classes() -> None:
    class OnlyAct:
        def act(self, observation: object, rng: object = None) -> object:
            return None

    class OnlyName:
        @property
        def name(self) -> str:
            return "x"

    assert not isinstance(OnlyAct(), Policy)
    assert not isinstance(OnlyName(), Policy)


def test_no_policy_has_an_update_method() -> None:
    """The protocol is frozen: no policy may expose a learning/update API."""
    assert "update" not in Policy.__dict__
    for policy in _all_policies():
        assert not hasattr(policy, "update")


def test_baselines_are_deterministic() -> None:
    observations = [
        _obs(tick=ENTRY_TICK, step_in_episode=0),
        _obs(tick=ENTRY_TICK + 700, step_in_episode=1),
        _obs(tick=ENTRY_TICK + 700, step_in_episode=2),
    ]
    for factory in (
        lambda: TauResetPolicy(120),
        lambda: CostAwareRebalancePolicy(120, 0, _FakeGasModel(0.0)),
    ):
        first = [factory().act(ob) for ob in observations]
        second = [factory().act(ob) for ob in observations]
        assert first == second


def test_baselines_do_not_mutate_the_observation() -> None:
    observation = _obs(tick=ENTRY_TICK + 900, uncollected_fees_token0=10.0, step_in_episode=1)
    for policy in _all_policies():
        policy.act(observation)
        assert observation == _obs(
            tick=ENTRY_TICK + 900, uncollected_fees_token0=10.0, step_in_episode=1
        )


def test_act_accepts_injected_rng() -> None:
    rng = np.random.default_rng(0)
    for policy in _all_policies():
        action = policy.act(_obs(), rng=rng)
        assert action.action_type in {"hold", "rebalance"}


# ---------------------------------------------------------------------------
# Shared conftest fixture
# ---------------------------------------------------------------------------


def test_dummy_policy_fixture_holds(dummy_policy: object) -> None:
    action = dummy_policy.act(_obs())  # type: ignore[attr-defined]
    assert action.is_hold()


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        TauResetPolicy(0)
    with pytest.raises(ValueError):
        TauResetPolicy(120, tick_spacing=0)
    with pytest.raises(ValueError):
        CostAwareRebalancePolicy(120, -1, _FakeGasModel(1.0))
    with pytest.raises(ValueError):
        CostAwareRebalancePolicy(0, 10, _FakeGasModel(1.0))
    with pytest.raises(ValueError):
        FullRangeV2Policy(tick_spacing=0)


def test_tick_spacing_override_changes_width() -> None:
    # 120 ticks at spacing 30 -> 4 half-width spacings (vs 2 at spacing 60).
    action = TauResetPolicy(120, tick_spacing=30).act(_obs())
    assert action.width == 4
    assert action.lower_tick(ENTRY_TICK, 30) % 30 == 0
    assert action.upper_tick(ENTRY_TICK, 30) % 30 == 0
