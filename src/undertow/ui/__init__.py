"""``undertow.ui`` — the Undertow live policy cockpit.

The third top-level package: a read-only FastAPI/WebSocket backend plus a React
frontend that make training runs legible (`docs/plans/ui/PLAN.md`).  It consumes
only the public ``undertow.sim`` / ``undertow.data`` surfaces; neither of those
packages may ever import this one.

U01 ships the config surface; U05 appends ``create_app`` and ``serve``.
"""

from __future__ import annotations

from undertow.ui.app import create_app
from undertow.ui.cli import serve
from undertow.ui.config import UiConfig

__all__ = ["UiConfig", "create_app", "serve"]
