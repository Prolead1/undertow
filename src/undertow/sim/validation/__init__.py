"""Parity & look-ahead validation for ``undertow.sim`` (S14, CONTRACTS.md §16).

Public surface:

* :class:`~undertow.sim.validation.parity.ParityReport` — sim-vs-backtester
  agreement on identical inputs.
* :func:`~undertow.sim.validation.parity.run_parity_check` — run replay-mode
  simulator episodes and the backtester over the identical window, quantify the
  per-step fee/IL/PnL drift against a stated, justified tolerance.
* :class:`~undertow.sim.validation.lookahead.LookAheadReport` — the composed
  look-ahead probe results.
* :func:`~undertow.sim.validation.lookahead.run_lookahead_probes` — attack the
  composed simulator + backtester for future-data leaks.

This sub-package is a *gate* (PLAN.md §3): S15 may not run until
:func:`run_parity_check` reports ``passed=True``.
"""

from __future__ import annotations

from undertow.sim.validation.lookahead import LookAheadReport, run_lookahead_probes
from undertow.sim.validation.parity import (
    DEFAULT_PARITY_TOLERANCE,
    ParityReport,
    run_parity_check,
)

__all__ = [
    "DEFAULT_PARITY_TOLERANCE",
    "LookAheadReport",
    "ParityReport",
    "run_lookahead_probes",
    "run_parity_check",
]
