"""Static SPA mounting for ``undertow.ui`` (U05, CONTRACTS §7).

The built frontend is a catch-all mount registered **after** every ``/api``
route, with an SPA fallback to ``index.html`` for client-side routes.  Keeping
this in its own module leaves the application factory focused on route wiring.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

__all__ = ["mount_spa"]


def mount_spa(app: FastAPI, web_dist: Path) -> None:
    """Serve a built frontend at ``/`` with an SPA fallback to ``index.html``.

    API routes are registered before this catch-all, so ``/api/*`` and the
    FastAPI docs keep precedence.  Unknown ``/api/*`` paths are still a ``404``.
    """
    dist = Path(web_dist).resolve()
    index = dist / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail=f"unknown API path /{full_path}")
        candidate = (dist / full_path).resolve()
        if dist in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail=f"no index.html under {dist}")
