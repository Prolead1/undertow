"""Lightweight progress reporter for long-running fetches (T14 progress-reporting AC).

Writes a single updating line to stderr (carriage-return in-place update on a TTY;
periodic milestone lines on a non-TTY). Thread-safe: all state is local to each call;
concurrent fetchers share stderr but each reporter instance owns its label.

CONTRACTS.md §9: progress writes to stderr, never stdout (so ``--json`` output is
clean). No imports from other ``undertow`` modules — this is pure I/O.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

LOGGER = logging.getLogger("undertow.data.progress")

# ---------------------------------------------------------------------------
# Rate limiter — shared across reporters so concurrent fetchers don't
# collectively flood stderr with updates faster than a human can read.
# ---------------------------------------------------------------------------

_last_flush: float = 0.0
_flush_lock: threading.Lock = threading.Lock()
_MIN_INTERVAL_S: float = 0.15  # never flush more than ~6 times/sec


def _should_flush() -> bool:
    """Return True when enough time has passed since the last flush."""
    global _last_flush  # noqa: PLW0603 — deliberate module-level rate limiter
    now = time.monotonic()
    with _flush_lock:
        if now - _last_flush >= _MIN_INTERVAL_S:
            _last_flush = now
            return True
    return False


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Single-line progress for one fetch operation.

    Call ``chunk_done(completed, total, cached)`` for each completed chunk;
    call ``finish()`` when the last chunk is done to emit the final summary
    line.

    On a TTY: ``\\r label: ████████░░  C/T  (C  cached, H  fetched)`` — updated
    in-place. On a non-TTY (CI, pipe): logs a line at 10% milestones, and a
    final summary.

    ``flush()`` pushes the current bar to stderr; called internally by
    ``chunk_done`` and ``finish`` at the rate-limit interval.
    """

    _BAR_WIDTH = 40

    def __init__(self, label: str, total: int, *, enabled: bool = True) -> None:
        self._label = label
        self._total = max(1, total)
        self._enabled = enabled
        self._completed = 0
        self._cached = 0
        self._fetched = 0
        self._tty = _stderr_is_tty()
        self._last_logged_pct = -1  # for non-TTY milestone logging
        if self._total > 1 and self._enabled:
            # Initial tick so the user sees the label immediately.
            self.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_done(self, *, completed: int, cached: bool) -> None:
        """Record one completed chunk."""
        self._completed = completed
        if cached:
            self._cached += 1
        else:
            self._fetched += 1
        if self._total > 1 and self._enabled:
            self.flush()

    def finish(self) -> None:
        """Finalise and write the summary line."""
        if self._total <= 1 or not self._enabled:
            return
        if self._tty:
            # Clear the in-place bar and write the final line.
            sys.stderr.write("\r" + " " * 120 + "\r")
        summary = f"{self._label}: done — {self._cached} cached, {self._fetched} fetched"
        LOGGER.info(summary)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Render and write the current progress bar, rate-limited."""
        if not _should_flush():
            return
        line = self._render()
        if self._tty:
            sys.stderr.write("\r" + line + "\r")
            sys.stderr.flush()
        else:
            pct = (self._completed * 100) // self._total
            milestone = (pct // 10) * 10
            if milestone > self._last_logged_pct:
                self._last_logged_pct = milestone
                LOGGER.info(line)

    def _render(self) -> str:
        filled = min(
            self._BAR_WIDTH,
            (self._completed * self._BAR_WIDTH) // self._total,
        )
        bar = "█" * filled + "░" * (self._BAR_WIDTH - filled)
        return (
            f"{self._label}: {bar} "
            f"{self._completed}/{self._total} "
            f"({self._cached} cached, {self._fetched} fetched)"
        )


def _stderr_is_tty() -> bool:
    try:
        return sys.stderr.isatty()
    except (AttributeError, OSError):
        return False