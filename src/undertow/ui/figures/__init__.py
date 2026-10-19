"""``undertow.ui.figures`` — pure Plotly figure builders + dark theme (U04).

Every builder takes plain data (dicts/lists per ``CONTRACTS.md`` §1–§4, never a
reader object) and returns a JSON-serialisable :class:`FigureSpec`
(``{"data", "layout", "config"}``).  ``undertow.sim``/``undertow.data`` are never
imported here; the package is a pure artifact consumer.

:data:`FIGURE_REGISTRY` is the API-name -> builder map U05 serves at
``GET /api/runs/{run_id}/figures/{name}`` (``CONTRACTS.md`` §7).
"""

from __future__ import annotations

from collections.abc import Callable

from undertow.ui.figures.actions import action_heatmap_figure
from undertow.ui.figures.compare import compare_equity_figure
from undertow.ui.figures.diagnostics import diagnostics_figure
from undertow.ui.figures.leaderboard import leaderboard_figure
from undertow.ui.figures.learning import learning_curve_figure
from undertow.ui.figures.ribbon import ribbon_figure
from undertow.ui.figures.theme import (
    DARK_TEMPLATE,
    TEMPLATE_NAME,
    TOKENS,
    FigureSpec,
    apply_theme,
)

#: API figure name -> builder (``CONTRACTS.md`` §7).
FIGURE_REGISTRY: dict[str, Callable[..., FigureSpec]] = {
    "learning": learning_curve_figure,
    "diagnostics": diagnostics_figure,
    "actions": action_heatmap_figure,
    "ribbon": ribbon_figure,
    "leaderboard": leaderboard_figure,
}

__all__ = [
    "DARK_TEMPLATE",
    "FIGURE_REGISTRY",
    "TEMPLATE_NAME",
    "TOKENS",
    "FigureSpec",
    "action_heatmap_figure",
    "apply_theme",
    "compare_equity_figure",
    "diagnostics_figure",
    "leaderboard_figure",
    "learning_curve_figure",
    "ribbon_figure",
]
