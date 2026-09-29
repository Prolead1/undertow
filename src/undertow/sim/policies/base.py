"""The frozen ``Policy`` protocol for ``undertow.sim`` (CONTRACTS.md §10).

A ``Policy`` is a frozen decision rule: it maps an ``Observation`` to an
``Action``.  The protocol deliberately exposes **no** learning/update method, so
"the policy learned during the backtest" is unrepresentable (PLAN.md §4, "Frozen
means frozen").
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from undertow.sim.types import Action

if TYPE_CHECKING:
    import numpy as np

    # S12 owns the ``Observation`` dataclass (CONTRACTS.md §11).  Importing it only
    # under ``TYPE_CHECKING`` keeps this module importable before S12 lands while
    # still typing ``act()`` against the frozen contract shape.
    from undertow.sim.env.observations import Observation

#: The canonical no-op action.  Policies that merely hold can return this directly.
HOLDAction = Action(action_type="hold", center_offset=0, width=0)


@runtime_checkable
class Policy(Protocol):
    """A frozen decision rule (CONTRACTS.md §10).

    The backtester (S11) calls only :meth:`act`; there is no ``update`` method on
    the protocol, so a policy cannot be trained from the backtester's call path.
    """

    def act(
        self,
        observation: Observation,
        rng: np.random.Generator | None = None,
    ) -> Action:
        """Return the action for ``observation``.

        ``rng`` is ``None`` for deterministic policies; stochastic policies use the
        injected generator (never a module-level RNG).
        """
        ...

    @property
    def name(self) -> str:
        """Stable identifier used in run manifests and result tables."""
        ...

    def reset(self) -> None:
        """Reset episode-local state at episode start. Default: no-op."""
        ...
