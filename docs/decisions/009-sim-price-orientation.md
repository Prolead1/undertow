# ADR 009 — Sim price orientation: raw sqrt-price, human amounts/price (USDC numeraire)

**Status:** accepted
**Affects:** `src/undertow/sim/core/position.py` · sim `CONTRACTS.md` §0/§5/§7/§19 · tasks S04, S06, S07, S11, S12
**Raised by:** S04 (`feature-sim-position-math`), orchestrator

## Context

The sim plan needs one coherent price/sqrt-price/amount convention. The frozen docs disagreed:

- sim `CONTRACTS.md` §0: `price` = human token0 per token1 (USDC per WETH, ~2000–4000).
- sim `CONTRACTS.md` §5: `calc_sqrt_price_a(tick) = sqrt(1.0001**tick)` — this is the **raw** Uniswap
  sqrt price (`sqrt_price_x96 / Q96`), ≈ 18244 at tick 196242.
- sim `CONTRACTS.md` §19 golden: `sqrt_price_x96 at ETH=$3000 = 1446501726624926496477173928747177`
  — also the **raw** value.
- S06/S07 briefs: `sqrt_price = sqrt(price)` (≈ 54.8) and S06 `tick = log(price)/log(1.0001)`.

Data `fixedpoint.py` is explicit (`tick_to_price` docstring, `test_tick_to_price_direction`): for the
pinned pools `dec0=6` (USDC), `dec1=18` (WETH),

```
price (human, USDC per WETH) = 10**(dec1 - dec0) / sqrt_price_raw**2
```

which is **decreasing in tick** — a higher tick means a higher raw token1/token0 ratio, i.e. a *lower*
USDC-per-WETH price.

## Problem

1. `Position.value(sqrt_price, price) = amount0 + amount1*price` yields USDC only if the amounts and
   `price` are human. But `calc_sqrt_price_a` and the §19 golden are raw, so mixing them with human
   `price` gives a value off by `10**dec` factors.
2. S06's `tick = log(price)/log(1.0001)` gives **80082** at $3004, not the correct **196242**; and
   `sqrt_price = sqrt(price)` gives 54.8, not the raw 18244 the golden and the pool lattice expect.
3. **Human `sqrt_price = sqrt(price)` cannot be used with the standard V3 range formulas at all.**
   Because human price is decreasing in tick, `tick_lower < tick_upper` implies
   `sqrt_price(tick_lower) > sqrt_price(tick_upper)`. The standard forms
   `amount0 = L(1/s_l - 1/s_u)`, `amount1 = L(s_u - s_l)` assume `s_l < s_u`; with the human values they
   give **negative** token amounts (even for the full range), and every range/in-range test inverts.

S04's first implementation made its IL goldens pass only by using the unit-normalised convention
`price = 1/sqrt_price**2` (`dec0 = dec1 = 0`), where `price ≈ 3e-9` — mathematically self-consistent but
economically meaningless: 100,000 USDC capital becomes nonsense.

## Decision

**Keep the protocol quantities raw; humanise only the money.** This is the realisation of the "USDC
numeraire" goal that preserves the standard V3 math:

- `sqrt_price` is the **raw** Uniswap sqrt price (`sqrt(1.0001**tick)`), i.e. `sqrt_price_x96 / Q96`,
  monotonically increasing in tick. `calc_sqrt_price_a/b(tick)` keep their existing definition.
- `tick`, and `liquidity` `L`, are the **raw protocol quantities**. `Position.liquidity` is therefore
  directly comparable to the tape's recorded active liquidity (no unit conversion for fee shares).
- `price` is **human** `10**(dec1-dec0) / sqrt_price**2` (USDC per WETH), decreasing in tick, obtained
  from data's `tick_to_price` / `sqrt_price_x96_to_price`.
- `Position.amount0/amount1` return **human** token units: the raw amount `L·(…)` divided by
  `10**dec0` / `10**dec1`.
- `value = amount0 + amount1 * price` is USDC (eq 11).
- `Position` gains `dec0: int = 6`, `dec1: int = 18` fields; `initial_deposit` / `position_from_amounts`
  gain the same keyword-only defaults. `10**dec` factors cancel in IL, so `il_vs_hodl` / `il_fraction`
  are unchanged in sign and magnitude.
- Fee accrual stays `fees_raw = L · Δfee_growth` (raw, Q128 engine), then divided by `10**dec0/1` to
  report human token fees; the training sim derives fee income from swap volume × fee tier × raw
  liquidity share, so it never touches the accumulators.
- Conversion helpers (documented, used by S06/S07/S11):
  - `sqrt_price_raw = 10**((dec1-dec0)/2) / sqrt(price)` = `sqrt_price_x96/Q96`
  - `price_human = 10**(dec1-dec0) / sqrt_price_raw**2`
  - `tick = price_to_tick(price, dec0, dec1)` (data, floor semantics; **never** `log(price)/log(1.0001)`)
- S06's `ReplayPriceProcess`/`CalibratedPriceProcess` therefore return
  `(price_human, sqrt_price_raw, tick)` with the relations above; `CalibratedPriceProcess` simulates the
  human price (log-space) and converts for output.
- S07's lattice stays in raw tick/sqrt/liquidity; the §19 `sqrt_price_x96` golden is the raw value and
  equals `calc_sqrt_price_a(tick)**2 * Q96` (to rounding). The §19 tick↔price golden uses
  `tick_to_price(tick, 6, 18)`.

Also correct the `CONTRACTS.md` §19 arithmetic slip: V2 IL at `r = 1.2` is
`2√1.2/2.2 − 1 = −0.004141` (−0.41%), not −0.00454 (the `r = 0.5` row is exact).

## Consequences

- `src/undertow/sim/core/position.py`: keep raw `calc_sqrt_price_a/b`; add `dec0/dec1` to `Position`
  and the factories; divide raw amounts by `10**dec` in `amount0/amount1` and fee outputs. Add module
  constants `DEFAULT_DEC0=6`, `DEFAULT_DEC1=18`.
- `docs/plans/sim/CONTRACTS.md` §0 (`SqrtPrice` = raw unless `*_x96`), §5 (decimals on `Position`),
  §7 (price/sqrt/tick relations), §19 (typo) updated.
- S06 and S07 briefs updated to the raw-sqrt/tick-with-decimals relations.
- **S11 (backtester)** already works in raw integers; it now matches `Position(dec0, dec1)` without
  conversion. **S12 (env)** must pass human `price` to `Position.value` and raw `sqrt_price`/`liquidity`.
- **Rejected — human `sqrt_price = sqrt(price)`:** inverts the tick/price monotonicity and makes the
  standard range formulas produce negative amounts (see Problem 3). Rejected despite the briefs' wording.
- **Rejected — raw end-to-end with value in raw token0:** forces every P&L number to be divided by
  `10**dec0` at the reporting boundary and contradicts §0's USDC numeraire; humanising at the amount
  boundary is one clear conversion instead of many.
- Vault edits are **not** made here; only repo files change.
