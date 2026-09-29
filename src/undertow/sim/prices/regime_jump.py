"""Calibrated Markov-regime-switching jump-diffusion price process (S06).

Implements ``CONTRACTS.md`` §7. The process is fitted once on the **train**
split of a :class:`~undertow.sim.marketview.MarketView` and frozen; at
simulation time :meth:`CalibratedPriceProcess.step` generates prices forward
from the fitted model and never reads from any split (no runtime look-ahead).

Model (per 10-minute step, annualised parameters):

    dlogP = mu_s * dt + sigma_s * sqrt(dt) * eps + sum(jump sizes)
    jumps ~ StudentT(df_s, loc_s, scale_s), count ~ Poisson(lambda_s * dt)

``s`` is a 4-state latent regime matched post-hoc to the data module's regime
labels (``bull``, ``bear``, ``sideways``, ``high_vol``). The transition matrix
is row-stochastic; the stationary distribution seeds the initial regime.

The Gaussian-emission HMM is fitted with a small, pure-numpy Baum-Welch EM
(the plan explicitly prefers clarity over SOTA; ``hmmlearn`` is not installed
on the Python 3.14 project venv).

Orientation (ADR-009)
---------------------
Internally the process simulates the **human** USDC/WETH log price; the output
triplet ``(price, sqrt_price_raw, tick)`` is produced by
:func:`undertow.sim.prices.base.human_price_to_triplet`, i.e. the raw Uniswap
sqrt price and the floor tick from ``undertow.data.price_to_tick`` — never
``log(price) / log(1.0001)``.

Validation note
---------------
The S06 brief's ``__post_init__`` rule "``sigma > 0``" conflicts with its own
required deterministic tests (``sigma = 0``, no jumps ⇒ constant price; known
drift with ``sigma = 0``). We therefore require ``sigma >= 0`` (negative
diffusion is rejected) and allow ``sigma = 0`` as the degenerate pure-drift /
pure-jump limit. All fitted values are strictly positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from undertow.sim.marketview import MarketView
from undertow.sim.prices.base import human_price_to_triplet
from undertow.sim.types import (
    Price,
    PriceMode,
    SqrtPrice,
    Tick,
)

__all__ = ["CalibratedPriceProcess", "MRSJDParams"]

#: Default decision cadence: 10 minutes expressed in years (6 per hour × 24 h).
#:
#: NOTE: ``CONTRACTS.md`` §7 literally writes ``1 / (6 * 365.25)``, which is
#: 4 hours — not the "10 minutes in annualized units" that the same sentence and
#: the pinned plan decisions require (10-minute decision cadence, 144 steps/day,
#: ``EpisodeConfig.step_minutes = 10``). The missing ``* 24`` is an arithmetic
#: slip in the frozen text; the corrected value below is the coherent 10-minute
#: cadence and matches the dt derived from a 1-minute reference feed.
DEFAULT_DT: float = 1.0 / (6.0 * 24.0 * 365.25)

#: Reference-feed cadence used to build 10-minute log-returns when fitting.
_REFERENCE_STEP_MINUTES: int = 1
_DECISION_STEP_MINUTES: int = 10
_DOWNSAMPLE_STRIDE: int = _DECISION_STEP_MINUTES // _REFERENCE_STEP_MINUTES

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class MRSJDParams:
    """Parameters of a Markov-regime-switching jump-diffusion.

    Fitted on the train split only (no look-ahead). Four latent states matching
    the regime labels, Student-t jump sizes. See module docstring for units.
    """

    state_names: tuple[str, ...]
    transition_matrix: np.ndarray
    drift: tuple[float, ...]
    diffusion: tuple[float, ...]
    jump_intensity: tuple[float, ...]
    jump_loc: tuple[float, ...]
    jump_scale: tuple[float, ...]
    jump_dof: tuple[float, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.transition_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("transition_matrix must be a non-empty square matrix")
        n = matrix.shape[0]

        if not np.all(np.isfinite(matrix)):
            raise ValueError("transition_matrix must be finite")
        if np.any(matrix < -1e-9) or np.any(matrix > 1.0 + 1e-9):
            raise ValueError("transition_matrix entries must lie in [0, 1]")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"transition_matrix rows must sum to 1 (got {row_sums.tolist()})"
            )

        if len(self.state_names) != n:
            raise ValueError(
                f"state_names has {len(self.state_names)} entries, expected {n}"
            )
        for name in (
            "drift",
            "diffusion",
            "jump_intensity",
            "jump_loc",
            "jump_scale",
            "jump_dof",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries, expected {n}"
                )

        if any(not math.isfinite(d) for d in self.drift):
            raise ValueError("drift must be finite")
        if any(not math.isfinite(s) for s in self.diffusion):
            raise ValueError("diffusion must be finite")
        if any(s < 0.0 for s in self.diffusion):
            raise ValueError("diffusion (sigma) must be >= 0 for all states")
        if any(not math.isfinite(lam) or lam < 0.0 for lam in self.jump_intensity):
            raise ValueError("jump_intensity must be finite and >= 0")
        if any(not math.isfinite(loc) for loc in self.jump_loc):
            raise ValueError("jump_loc must be finite")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in self.jump_scale):
            raise ValueError("jump_scale must be finite and > 0")
        if any(not math.isfinite(dof) or dof <= 2.0 for dof in self.jump_dof):
            raise ValueError("jump_dof must be finite and > 2 (finite variance)")

    @property
    def n_states(self) -> int:
        """Number of latent regimes."""
        return len(self.state_names)


def _stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Solve ``pi @ P = pi`` with ``sum(pi) = 1`` by least squares."""
    matrix = np.asarray(transition_matrix, dtype=float)
    n = matrix.shape[0]
    if n == 1:
        return np.ones(1, dtype=float)
    a = np.vstack([matrix.T - np.eye(n), np.ones((1, n))])
    b = np.zeros(n + 1, dtype=float)
    b[-1] = 1.0
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    solution = np.clip(solution, 0.0, None)
    total = solution.sum()
    if total <= 0.0:  # pragma: no cover - impossible for a valid transition matrix
        return np.full(n, 1.0 / n)
    return solution / total


def _decision_dt_years(close_times: list[datetime], stride: int) -> float:
    """Years spanned by ``stride`` consecutive reference bars (median spacing).

    The reference feed's cadence is derived from the bars themselves rather
    than assumed, so the annualised MRSJD parameters stay correct on a feed
    that is not 1-minute spaced. Falls back to :data:`DEFAULT_DT` when the
    timestamps cannot supply a positive spacing.
    """
    if len(close_times) >= 2:
        deltas = [
            (close_times[i + 1] - close_times[i]).total_seconds()
            for i in range(len(close_times) - 1)
        ]
        positive = [delta for delta in deltas if delta > 0.0]
        if positive:
            spacing = float(np.median(positive))
            dt = spacing * max(stride, 1) / (365.25 * 24.0 * 3600.0)
            if math.isfinite(dt) and dt > 0.0:
                return dt
    return DEFAULT_DT


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable ``log(sum(exp(values)))`` along ``axis``."""
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    summed = np.sum(np.exp(values - peak), axis=axis, keepdims=True)
    return (peak + np.log(summed)).squeeze(axis=axis)


def _log_gaussian_emissions(
    returns: np.ndarray, means: np.ndarray, sigmas: np.ndarray
) -> np.ndarray:
    """Log N(return | mean, sigma) for every (time, state) pair."""
    z = (returns[:, None] - means[None, :]) / sigmas[None, :]
    return -0.5 * z * z - np.log(sigmas)[None, :] - 0.5 * _LOG_2PI


def _forward_backward(
    values: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    transition: np.ndarray,
    pi: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Baum-Welch forward/backward pass.

    Returns ``(log_likelihood, gamma, xi_sum)`` for the given parameters.
    """
    t_len, n_states = values.shape[0], means.shape[0]
    log_emission = _log_gaussian_emissions(values, means, sigmas)
    log_p = np.log(np.clip(transition, 1e-300, None))
    log_pi = np.log(np.clip(pi, 1e-300, None))

    log_alpha = np.empty((t_len, n_states), dtype=float)
    log_alpha[0] = log_pi + log_emission[0]
    for t in range(1, t_len):
        log_alpha[t] = log_emission[t] + _logsumexp(
            log_alpha[t - 1][:, None] + log_p, axis=0
        )
    log_likelihood = float(_logsumexp(log_alpha[-1], axis=0))

    log_beta = np.zeros((t_len, n_states), dtype=float)
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = _logsumexp(
            log_p + (log_emission[t + 1] + log_beta[t + 1])[None, :], axis=1
        )

    gamma = np.exp(log_alpha + log_beta - log_likelihood)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi_sum = np.zeros((n_states, n_states), dtype=float)
    for t in range(t_len - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_p
            + (log_emission[t + 1] + log_beta[t + 1])[None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi)
    return log_likelihood, gamma, xi_sum


def _fit_gaussian_hmm(
    returns: np.ndarray,
    n_states: int,
    *,
    n_iter: int = 150,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian-emission HMM with Baum-Welch EM (pure numpy).

    Returns ``(pi, transition, means, sigmas, gamma)`` where ``gamma`` is the
    ``(T, n_states)`` posterior responsibility matrix recomputed from the final
    parameters. Robust to tiny samples: variances are floored and empty states
    fall back to the global scale.
    """
    values = np.asarray(returns, dtype=float)
    t_len = values.shape[0]
    if t_len == 0 or n_states <= 0:
        raise ValueError("need at least one return and one state to fit an HMM")

    global_std = float(np.std(values))
    if not math.isfinite(global_std) or global_std <= 0.0:
        global_std = 1e-4
    var_floor = (global_std * 1e-3) ** 2

    quantiles = np.linspace(0.1, 0.9, n_states)
    means = np.quantile(values, quantiles).astype(float)
    sigmas = np.full(n_states, global_std, dtype=float)
    transition = np.full((n_states, n_states), 1.0 / n_states, dtype=float)
    pi = np.full(n_states, 1.0 / n_states, dtype=float)

    prev_ll = -np.inf
    for _ in range(n_iter):
        log_likelihood, gamma, xi_sum = _forward_backward(
            values, means, sigmas, transition, pi
        )

        pi = gamma[0] / max(gamma[0].sum(), 1e-300)
        xi_rows = xi_sum.sum(axis=1, keepdims=True)
        transition = xi_sum / np.maximum(xi_rows, 1e-300)

        weights = np.maximum(gamma.sum(axis=0), 1e-300)
        means = (gamma * values[:, None]).sum(axis=0) / weights
        variance = (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
        sigmas = np.sqrt(np.maximum(variance, var_floor))

        if abs(log_likelihood - prev_ll) <= tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = log_likelihood

    # Recompute responsibilities from the final parameters so the returned
    # gamma is consistent with the means/sigmas/transition that are returned.
    _, gamma, _ = _forward_backward(values, means, sigmas, transition, pi)
    return pi, transition, means, sigmas, gamma


def _fit_student_t(sample: np.ndarray) -> tuple[float, float, float]:
    """Method-of-moments Student-t fit ``(loc, scale, dof)`` with ``dof > 2``.

    ``dof`` is inferred from the sample excess kurtosis (``6 / (dof - 4)`` for
    ``dof > 4``), clipped to ``[2.5, 100]`` so the variance is always finite.
    """
    values = np.asarray(sample, dtype=float)
    if values.size == 0:
        return 0.0, 1e-3, 4.0
    loc = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if not math.isfinite(std) or std <= 0.0:
        std = 1e-3
    z = (values - loc) / std
    m4 = float(np.mean(z**4))
    excess = max(m4 - 3.0, 1e-3)
    dof = 4.0 + 6.0 / excess
    dof = float(np.clip(dof, 2.5, 100.0))
    scale = std * math.sqrt((dof - 2.0) / dof)
    return loc, max(scale, 1e-8), dof


class CalibratedPriceProcess:
    """A calibrated MRSJD. :meth:`step` advances ``dt`` and may switch/jump."""

    def __init__(
        self,
        params: MRSJDParams,
        dt: float = DEFAULT_DT,
        *,
        initial_price: float = 3000.0,
        dec0: int = 6,
        dec1: int = 18,
    ) -> None:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt!r}")
        if not math.isfinite(initial_price) or initial_price <= 0.0:
            raise ValueError(
                f"initial_price must be finite and > 0, got {initial_price!r}"
            )
        self._params = params
        self._dt = float(dt)
        self._dec0 = int(dec0)
        self._dec1 = int(dec1)
        self._initial_log_price = math.log(float(initial_price))
        self._log_price = self._initial_log_price
        self._stationary = _stationary_distribution(params.transition_matrix)
        # No RNG is available at construction time, so start deterministically
        # at the most probable stationary state; reset() resamples.
        self._state = int(np.argmax(self._stationary))

    # -- Introspection (read-only) ---------------------------------------
    @property
    def params(self) -> MRSJDParams:
        """The frozen fitted parameters."""
        return self._params

    @property
    def dt(self) -> float:
        """Step size in annualised units."""
        return self._dt

    @property
    def state(self) -> int:
        """Current latent regime index."""
        return self._state

    @property
    def log_price(self) -> float:
        """Current human log-price."""
        return self._log_price

    # -- PriceProcess ----------------------------------------------------
    def reset(self, rng: np.random.Generator) -> None:
        """Reset the log-price and resample the initial regime."""
        self._log_price = self._initial_log_price
        self._state = int(rng.choice(self._stationary.size, p=self._stationary))

    def step(self, rng: np.random.Generator) -> tuple[Price, SqrtPrice, Tick]:
        """Advance one ``dt`` step and return the ADR-009 triplet."""
        params = self._params
        n_states = params.n_states

        row = np.asarray(params.transition_matrix[self._state], dtype=float)
        row = row / row.sum()
        self._state = int(rng.choice(n_states, p=row))
        state = self._state

        dlog = params.drift[state] * self._dt
        sigma = params.diffusion[state]
        if sigma > 0.0:
            dlog += sigma * math.sqrt(self._dt) * float(rng.standard_normal())

        intensity = params.jump_intensity[state]
        if intensity > 0.0:
            n_jumps = int(rng.poisson(intensity * self._dt))
            if n_jumps > 0:
                jumps = params.jump_loc[state] + params.jump_scale[state] * rng.standard_t(
                    params.jump_dof[state], size=n_jumps
                )
                dlog += float(np.sum(jumps))

        self._log_price += dlog
        price = math.exp(self._log_price)
        return human_price_to_triplet(price, self._dec0, self._dec1)

    @property
    def mode(self) -> str:
        """Always ``"calibrated"``."""
        return PriceMode.CALIBRATED.value

    # -- Fitting ---------------------------------------------------------
    @staticmethod
    def fit(market_view: MarketView) -> MRSJDParams:
        """Fit the MRSJD on the train split only (no look-ahead).

        Raises ``ValueError`` when ``market_view.is_train`` is False: fitting on
        evaluation data would leak the future. The fitted HMM discovers the four
        regimes unsupervised; states are labelled post-hoc by mean (``bull``
        highest, ``bear`` lowest) and volatility (``sideways`` lower,
        ``high_vol`` higher).
        """
        if not market_view.is_train:
            raise ValueError(
                "CalibratedPriceProcess.fit requires a train MarketView "
                "(is_train=True); fitting on eval data is look-ahead"
            )

        bars = market_view.reference_bars(
            market_view.active_start_utc, market_view.active_end_utc
        )
        closes = np.asarray(bars["close"].to_numpy(), dtype=float)
        if closes.size < 2:
            raise ValueError("need at least 2 reference bars to fit the MRSJD")

        stride = max(_DOWNSAMPLE_STRIDE, 1)
        # Annualise against the feed's actual cadence rather than assuming a
        # 1-minute bar. On the pinned 1-minute feed this equals DEFAULT_DT.
        dt = _decision_dt_years(bars["close_time"].to_list(), stride)

        sampled = closes[::stride]
        if sampled.size < 2:
            sampled = closes
        returns = np.diff(np.log(np.clip(sampled, 1e-12, None)))
        if returns.size < 2:
            raise ValueError("need at least 2 reference log-returns to fit the MRSJD")

        _, transition, means, sigmas, gamma = _fit_gaussian_hmm(returns, n_states=4)

        order = np.argsort(means)[::-1]
        bull = int(order[0])
        bear = int(order[-1])
        middle = [i for i in range(4) if i not in (bull, bear)]
        if sigmas[middle[0]] <= sigmas[middle[1]]:
            sideways, high_vol = int(middle[0]), int(middle[1])
        else:
            sideways, high_vol = int(middle[1]), int(middle[0])
        perm = [bull, bear, sideways, high_vol]

        # Per-state Student-t jump parameters from residuals beyond 3 sigma.
        assignments = np.argmax(gamma, axis=1)
        residuals = returns - means[assignments]
        jump_loc: list[float] = []
        jump_scale: list[float] = []
        jump_dof: list[float] = []
        jump_intensity: list[float] = []
        for state in range(4):
            mask = assignments == state
            count = int(mask.sum())
            state_residuals = residuals[mask]
            threshold = 3.0 * float(sigmas[state])
            tail = state_residuals[np.abs(state_residuals) > threshold]
            sample = tail if tail.size >= 4 else state_residuals
            loc, scale, dof = _fit_student_t(sample)
            jump_loc.append(loc)
            jump_scale.append(scale)
            jump_dof.append(dof)
            years = max(count * dt, dt)
            jump_intensity.append(float(tail.size) / years if tail.size >= 4 else 0.0)

        drift = tuple(float(means[i] / dt) for i in perm)
        diffusion = tuple(float(sigmas[i] / math.sqrt(dt)) for i in perm)
        reordered = np.asarray(transition, dtype=float)[np.ix_(perm, perm)]
        row_totals = reordered.sum(axis=1, keepdims=True)
        reordered = reordered / np.maximum(row_totals, 1e-300)

        return MRSJDParams(
            state_names=("bull", "bear", "sideways", "high_vol"),
            transition_matrix=reordered,
            drift=drift,
            diffusion=diffusion,
            jump_intensity=tuple(jump_intensity[i] for i in perm),
            jump_loc=tuple(jump_loc[i] for i in perm),
            jump_scale=tuple(jump_scale[i] for i in perm),
            jump_dof=tuple(jump_dof[i] for i in perm),
        )
