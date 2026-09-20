"""Logging configuration for ``undertow.data`` — one call in ``cli.main()`` wires the
whole package tree (T14 progress-reporting AC, CONTRACTS.md §9).

Console output goes to **stderr** so progress and diagnostics never mix with stdout
(which may carry ``--json`` machine output). A sentinel attribute on the root
``undertow.data`` logger prevents ``basicConfig`` from being a guest at the wrong house
(it configures the root logger, which is too wide — sub-packages that the pipeline
imports should not suddenly emit INFO messages because the pipeline turned on its own
logger).

:func:`add_file_handler` additionally captures the same stream to a file at DEBUG
level. The pull/snapshot commands call it automatically so a failed download leaves a
full diagnostic log behind; the console level stays as requested because each handler
carries its own level and only the logger's floor is raised to DEBUG. ``configure_logging``
is idempotent and, with ``force=True``, closes and removes any file handlers it added.

Never imported from ``undertow.sim`` or any non-data package — each package owns its
logging config, per PLAN.md §4.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PACKAGE_LOGGER_NAME: str = "undertow.data"
"""Logger name for the root of the data-package hierarchy. All modules that do
``logging.getLogger("undertow.data.…")`` inherit the level and handler set here."""

DEFAULT_FILE_LEVEL: str = "DEBUG"
"""Default level for file handlers: the full stream, so errors are diagnosable."""

_SENTINEL: str = "_undertow_data_logging_configured"
_FILE_SENTINEL: str = "_undertow_data_file_handler_paths"

_STREAM: logging.StreamHandler | None = None


def _formatter() -> logging.Formatter:
    """Concise, machine-parseable format shared by the console and file handlers."""
    return logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _stderr_handler() -> logging.StreamHandler:
    """Return a stderr ``StreamHandler``, shared across calls.

    The same instance is reused so ``configure_logging(force=True)`` does not lose
    the custom formatter. It is never closed on ``force`` (unlike file handlers) —
    it wraps the process's real stderr.
    """
    global _STREAM  # noqa: PLW0603 — deliberately shared across calls
    if _STREAM is None:
        _STREAM = logging.StreamHandler(sys.stderr)
        _STREAM.setFormatter(_formatter())
    return _STREAM


def configure_logging(
    *,
    level: str = "INFO",
    force: bool = False,
) -> None:
    """Configure console logging for the ``undertow.data`` package tree.

    Called once from ``cli.main()``. The data pipeline owns its own logging; other
    packages run with their own level. Idempotent by default (second call is a no-op);
    pass ``force=True`` to replace — this closes and removes any file handlers added
    by :func:`add_file_handler`, so a subsequent ``add_file_handler`` re-attaches.

    ``level`` is the Python logging level name: ``"DEBUG"`` / ``"INFO"`` / ``"WARNING"``.
    It is applied to the console handler, so raising the logger to DEBUG in
    :func:`add_file_handler` does not leak debug output to stderr.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if getattr(logger, _SENTINEL, False) and not force:
        return

    # Remove any handler we attached so a force call never stacks handlers. File
    # handlers are closed; the shared stderr handler is kept alive.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        if isinstance(h, logging.FileHandler):
            h.close()

    console_level = getattr(logging, level.upper())
    stream = _stderr_handler()
    stream.setLevel(console_level)
    logger.setLevel(console_level)
    logger.addHandler(stream)
    # Never propagate to root — other packages' handlers are their own business.
    logger.propagate = False
    setattr(logger, _SENTINEL, True)
    setattr(logger, _FILE_SENTINEL, set())


def add_file_handler(
    path: str | Path,
    *,
    level: str = DEFAULT_FILE_LEVEL,
) -> Path:
    """Attach a DEBUG-capable file handler to the ``undertow.data`` logger.

    Creates ``path``'s parent directory as needed. The console handler keeps its
    own (quieter) level; the logger floor is raised to DEBUG only if it is higher,
    so ``debug()`` records reach the file without leaking to stderr. Idempotent per
    resolved path within a process — a repeated call for the same file is a no-op.
    Returns the path (the caller may log it or let it fail upstream).
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if not getattr(logger, _SENTINEL, False):
        configure_logging()

    target = Path(path)
    resolved = str(target.resolve())
    attached: set[str] = getattr(logger, _FILE_SENTINEL, set())
    if resolved in attached:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(_formatter())
    handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(handler)

    # Raise only the logger's floor; each handler still filters at its own level, so
    # stderr stays at the console level while the file captures the full DEBUG stream.
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)

    attached.add(resolved)
    setattr(logger, _FILE_SENTINEL, attached)
    return target
