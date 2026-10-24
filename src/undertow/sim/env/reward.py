"""Reward function and per-step decomposition (S09).

Implements `CONTRACTS.md` §9 exactly — eq (1) of §9.2 of the roadmap:

    R_t = F_t − G_t − S_t + ΔIL_t − λ·σ_PnL,t

where ``ΔIL_t`` is the **signed** step change in the strategy's excess over its
fixed initial HODL basket (ADR-029); it is negative on most steps because the
LP position underperforms the basket. It is added, not subtracted — the older
``− ΔIL_t`` notation implicitly treated ``ΔIL_t`` as a positive loss magnitude,
which is the opposite sign to the ``il_change`` variable passed in here.

Each friction term is gated by an ablation flag on
:class:`undertow.sim.config.RewardConfig`. A disabled term is zero in the
returned :class:`RewardBreakdown`, but the input is never interpreted as
"free": the ungated net economics always survive in ``pnl_net`` so the
evaluation runner (S15) can decompose any ablation run and no cost term is
silently dropped.  :func:`compute_reward_economics` exposes the same values as
a full ungated breakdown (S20) so the parity harness compares true economics
rather than the agent's ablated objective.

Sign convention (do not double-negate)
--------------------------------------

* ``fees_earned`` is entered **positive** and kept positive.
* ``gas_cost`` and ``slippage_cost`` are entered as **positive cost
  magnitudes** (as returned by the S08 friction models); this module stores
  them **negative** (``gas = -gas_cost``).
* ``il_change`` is the signed change in the strategy's excess over its fixed
  initial HODL basket (usually negative — an IL/LVR loss) and is passed
  through unchanged.
* ``reward`` is the sum of the gated terms, so cost terms contribute
  ``≤ 0`` and ``risk_penalty ≤ 0``.

Normalization
-------------

:func:`compute_reward` always returns **raw USDC** values. Capital
normalization (``RewardConfig.normalize_by_capital`` → ``reward = R_t / W_0``)
is applied at the environment boundary (S12), consistent with the pinned
"reward scale" decision in ``PLAN.md`` §7. :func:`normalize_reward` is the
flag-aware scaler S12 can call; it is deliberately a separate function so the
frozen :func:`compute_reward` signature stays exact and the raw decomposition
remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from undertow.sim.config import RewardConfig


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Per-step reward decomposition.

    Every field is in USDC, or in units of initial capital
    (``USDC / initial_capital``) once :func:`normalize_reward` has been applied.

    ``fees`` is ``≥ 0``; ``gas``, ``slippage`` and ``risk_penalty`` are ``≤ 0``
    (costs stored as negatives); ``il_change`` is the signed ΔIL and is usually
    ``≤ 0``. ``reward`` is the sum of the gated terms above. ``pnl_net`` is the
    **ungated** net PnL change — the true economic change the ledger records —
    and is independent of the ablation flags.
    """

    fees: float = 0.0
    gas: float = 0.0
    slippage: float = 0.0
    il_change: float = 0.0
    risk_penalty: float = 0.0
    reward: float = 0.0
    # Reference values for debugging / the ledger.
    pnl_net: float = 0.0
    equity: float = 0.0
    # --- S20 hedge extensions (defaulted; appended after the frozen fields) ---
    #: Signed mark PnL the step's re-hedge crystallised / the interval moved.
    hedge_pnl: float = 0.0
    #: Signed funding income accrued this step (positive = a short is paid).
    hedge_funding: float = 0.0
    #: Signed perp trading cost this step (``<= 0``).
    hedge_fee: float = 0.0


def compute_reward(
    fees_earned: float,
    gas_cost: float,
    slippage_cost: float,
    il_change: float,
    pnl_volatility: float,
    config: RewardConfig,
    *,
    hedge_pnl: float = 0.0,
    hedge_funding: float = 0.0,
    hedge_fee: float = 0.0,
) -> RewardBreakdown:
    """Compute the step reward per eq (1) of §9.2.

    ``fees_earned``, ``gas_cost`` and ``slippage_cost`` are entered as positive
    magnitudes; the cost terms are stored negative in the returned breakdown.
    Each term is gated by its ``*_enabled`` flag in ``config`` — a disabled term
    is ``0.0`` in the breakdown, but ``pnl_net`` is always the ungated economic
    total, including the hedge terms (S20).

    ``hedge_pnl`` is the signed mark PnL contribution, ``hedge_funding`` the
    signed funding income (positive = a short is paid), and ``hedge_fee`` the
    **signed** perp trading cost (``<= 0``) — unlike ``gas_cost``/
    ``slippage_cost`` it is already a negative USDC amount.  All three are
    entered directly; the hedge terms are gated by ``config.hedge_enabled``
    for the RQ1 ablation.

    ``equity`` is left at ``0.0``; the caller (S12/S11) fills it with the
    current equity after this step. Normalization by initial capital is applied
    at the environment boundary via :func:`normalize_reward`.
    """
    fees = fees_earned if config.fees_enabled else 0.0
    gas = -gas_cost if config.gas_enabled else 0.0
    slippage = -slippage_cost if config.slippage_enabled else 0.0
    il = il_change if config.il_enabled else 0.0
    risk = (
        -config.risk_penalty_lambda * pnl_volatility
        if config.risk_penalty_enabled
        else 0.0
    )

    hedge_pnl_gated = hedge_pnl if config.hedge_enabled else 0.0
    hedge_funding_gated = hedge_funding if config.hedge_enabled else 0.0
    hedge_fee_gated = hedge_fee if config.hedge_enabled else 0.0

    reward = (
        fees
        + gas
        + slippage
        + il
        + risk
        + hedge_pnl_gated
        + hedge_funding_gated
        + hedge_fee_gated
    )
    # Ungated on purpose: the ablation flags change the objective the agent
    # optimizes, never the economic record.
    pnl_net = (
        fees_earned
        - gas_cost
        - slippage_cost
        + il_change
        + hedge_pnl
        + hedge_funding
        + hedge_fee
    )

    return RewardBreakdown(
        fees=fees,
        gas=gas,
        slippage=slippage,
        il_change=il,
        risk_penalty=risk,
        reward=reward,
        pnl_net=pnl_net,
        equity=0.0,
        hedge_pnl=hedge_pnl_gated,
        hedge_funding=hedge_funding_gated,
        hedge_fee=hedge_fee_gated,
    )


def compute_reward_economics(
    fees_earned: float,
    gas_cost: float,
    slippage_cost: float,
    il_change: float,
    *,
    equity: float = 0.0,
    hedge_pnl: float = 0.0,
    hedge_funding: float = 0.0,
    hedge_fee: float = 0.0,
) -> RewardBreakdown:
    """The **ungated** economic decomposition for audit and parity (S20).

    :func:`compute_reward` applies the RQ1 ablation flags to the *learning*
    objective; the underlying economics are independent of those flags, so the
    environment additionally exposes them unablated in
    ``info["reward_breakdown_economics"]``.  Without that payload an ablation
    (e.g. ``reward.hedge_enabled = false`` with ``hedge.enabled = true``) would
    make the gated term read ``0`` every step while the backtester still
    records the economic magnitude, turning a correct match into a false
    parity failure.

    Signs and definitions mirror the ungated terms of :func:`compute_reward`
    exactly: ``gas``/``slippage`` are negative cost magnitudes, ``hedge_fee``
    is signed (``<= 0``), ``fees``/``hedge_funding`` are signed income and
    ``il_change`` is the signed excess-vs-HODL change.  ``reward`` and
    ``pnl_net`` are both the ungated cash sum (the risk penalty is a
    preference, not an economic cash flow, so it is excluded); ``equity`` is
    whatever the caller supplies.
    """
    cash = (
        float(fees_earned)
        - float(gas_cost)
        - float(slippage_cost)
        + float(il_change)
        + float(hedge_pnl)
        + float(hedge_funding)
        + float(hedge_fee)
    )
    return RewardBreakdown(
        fees=float(fees_earned),
        gas=-float(gas_cost),
        slippage=-float(slippage_cost),
        il_change=float(il_change),
        risk_penalty=0.0,
        reward=cash,
        pnl_net=cash,
        equity=float(equity),
        hedge_pnl=float(hedge_pnl),
        hedge_funding=float(hedge_funding),
        hedge_fee=float(hedge_fee),
    )


def normalize_reward(
    breakdown: RewardBreakdown,
    initial_capital: float,
    config: RewardConfig,
) -> RewardBreakdown:
    """Scale a breakdown to units of initial capital.

    When ``config.normalize_by_capital`` is ``True``, every monetary field is
    divided by ``initial_capital`` so PPO value targets stay ``O(1)`` without
    hiding any cost term (PLAN.md §7 "Reward scale"). When the flag is
    ``False`` the breakdown is returned unchanged.

    Raises:
        ValueError: if normalization is requested with a non-positive
            ``initial_capital`` (division by zero / meaningless unit).
    """
    if not config.normalize_by_capital:
        return breakdown
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be > 0 to normalize the reward")
    inv = 1.0 / initial_capital
    return RewardBreakdown(
        fees=breakdown.fees * inv,
        gas=breakdown.gas * inv,
        slippage=breakdown.slippage * inv,
        il_change=breakdown.il_change * inv,
        risk_penalty=breakdown.risk_penalty * inv,
        reward=breakdown.reward * inv,
        pnl_net=breakdown.pnl_net * inv,
        equity=breakdown.equity * inv,
        hedge_pnl=breakdown.hedge_pnl * inv,
        hedge_funding=breakdown.hedge_funding * inv,
        hedge_fee=breakdown.hedge_fee * inv,
    )