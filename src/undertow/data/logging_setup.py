"""Logging configuration for ``undertow.data`` — one call in ``cli.main()`` wires the
whole package tree (T14 progress-reporting AC, CONTRACTS.md §9).

Idempotent: a second call replaces the handler rather than duplicating it. The handler
writes to **stderr** so progress and diagnostics never mix with stdout (which may carry
``--json`` machine output). A sentinel attribute on the root ``undertow.data`` logger
prevents ``basicConfig`` from being a guest at the wrong house (it configures the root
logger, which is too wide — sub-packages that the pipeline imports should not suddenly
emit INFO messages because the pipeline turned on its own logger).

Never imported from ``undertow.sim`` or any non-data package — each package owns its
logging config, per PLAN.md §4.
"""

from __future__ import annotations

import logging
import sys

PACKAGE_LOGGER_NAME: str = "undertow.data"
"""Logger name for the root of the data-package hierarchy. All modules that do
``logging.getLogger("undertow.data.…")`` inherit the level and handler set here."""

_SENTINEL: str = "_undertow_data_logging_configured"

_STREAM: logging.StreamHandler | None = None


def _stderr_handler() -> logging.Handler:
    """Return a stderr ``StreamHandler`` with a concise, machine-parseable format.

    The handler is shared across calls — the same instance is reused so
    ``configure_logging(force=True)`` does not lose the custom formatter.
    """
    global _STREAM  # noqa: PLW0603 — deliberately shared across calls
    if _STREAM is None:
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _STREAM = logging.StreamHandler(sys.stderr)
        _STREAM.setFormatter(fmt)
    return _STREAM


def configure_logging(
    *,
    level: str = "INFO",
    force: bool = False,
) -> None:
    """Configure logging for the ``undertow.data`` package tree.

    Called once from ``cli.main()``. The data pipeline owns its own logging; other
    packages run with their own level. Idempotent by default (second call is a no-op);
    pass ``force=True`` to replace.

    ``level`` is the Python logging level name: ``"DEBUG"`` / ``"INFO"`` / ``"WARNING"``.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if getattr(logger, _SENTINEL, False) and not force:
        return

    # Remove any handler we attached so a force call never stacks handlers.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.setLevel(getattr(logging, level.upper()))
    logger.addHandler(_stderr_handler())
    # Never propagate to root — other packages' handlers are their own business.
    logger.propagate = False
    setattr(logger, _SENTINEL, True)