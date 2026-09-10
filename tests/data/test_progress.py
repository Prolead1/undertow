"""Tests for ``undertow.data.progress`` (T14 progress-reporting AC)."""

from __future__ import annotations

import logging

import pytest

from undertow.data import progress


@pytest.fixture(autouse=True)
def _reset_rate_limiter(monkeypatch):
    """Each test starts with a fresh rate-limiter so flushes are deterministic."""
    monkeypatch.setattr(progress, "_last_flush", 0.0)


@pytest.fixture
def forced_flush(monkeypatch):
    """Make every ``flush()`` call pass the rate limiter."""
    monkeypatch.setattr(progress, "_should_flush", lambda: True)


def _make(
    label: str = "swaps",
    total: int = 10,
    *,
    enabled: bool = True,
    tty: bool = False,
) -> progress.ProgressReporter:
    """Construct a reporter with a pinned tty flag."""
    reporter = progress.ProgressReporter(label, total, enabled=enabled)
    reporter._tty = tty  # noqa: SLF001 — test pins the terminal mode
    return reporter


# ---------------------------------------------------------------------------
# _stderr_is_tty
# ---------------------------------------------------------------------------


def test_stderr_is_tty_true(monkeypatch):
    class FakeStderr:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(progress.sys, "stderr", FakeStderr())
    assert progress._stderr_is_tty() is True


def test_stderr_is_tty_false_when_no_isatty(monkeypatch):
    monkeypatch.setattr(progress.sys, "stderr", object())
    assert progress._stderr_is_tty() is False


def test_stderr_is_tty_false_when_isatty_raises(monkeypatch):
    class BrokenStderr:
        def isatty(self) -> bool:  # noqa: PLR6301
            raise OSError("no tty here")

    monkeypatch.setattr(progress.sys, "stderr", BrokenStderr())
    assert progress._stderr_is_tty() is False


# ---------------------------------------------------------------------------
# _should_flush — rate limiting
# ---------------------------------------------------------------------------


def test_should_flush_first_call_is_true(monkeypatch):
    monkeypatch.setattr(progress, "_last_flush", 0.0)
    monkeypatch.setattr(progress.time, "monotonic", lambda: 100.0)
    assert progress._should_flush() is True


def test_should_flush_suppresses_within_interval(monkeypatch):
    monkeypatch.setattr(progress.time, "monotonic", lambda: 100.0)
    assert progress._should_flush() is True
    # Same clock still within the 0.15 s interval → suppressed.
    assert progress._should_flush() is False


def test_should_flush_after_interval(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock["t"])
    assert progress._should_flush() is True
    clock["t"] += progress._MIN_INTERVAL_S
    assert progress._should_flush() is True


# ---------------------------------------------------------------------------
# _render
# ---------------------------------------------------------------------------


def test_render_draws_partial_bar():
    reporter = _make(total=10, enabled=False)
    reporter._completed = 3  # noqa: SLF001
    reporter._cached = 2  # noqa: SLF001
    reporter._fetched = 1  # noqa: SLF001
    line = reporter._render()  # noqa: SLF001
    assert "████████████" in line  # 3/10 * 40 = 12 filled)
    assert line.startswith("swaps: ")
    assert "3/10" in line
    assert "(2 cached, 1 fetched)" in line


def test_render_full_bar():
    reporter = _make(total=4, enabled=False)
    reporter._completed = 4  # noqa: SLF001
    line = reporter._render()  # noqa: SLF001
    assert "░" not in line.split("(")[0]  # no unfilled cells when complete


def test_render_empty_bar():
    reporter = _make(total=10, enabled=False)
    line = reporter._render()  # noqa: SLF001
    assert "█" * 0 + "░" * 40 in line  # all unfilled
    assert "0/10" in line


def test_filled_cells_never_exceed_width():
    reporter = _make(total=3, enabled=False)
    # completed beyond total (defensive) — the bar must clamp, not overflow.
    reporter._completed = 4  # noqa: SLF001
    line = reporter._render()  # noqa: SLF001
    # The bar is the first _BAR_WIDTH characters after the "label: " prefix.
    bar = line.split(": ", 1)[1][: progress.ProgressReporter._BAR_WIDTH]
    assert len(bar) == progress.ProgressReporter._BAR_WIDTH
    assert "░" not in bar  # fully clamped → all filled cells


# ---------------------------------------------------------------------------
# chunk_done — state tracking
# ---------------------------------------------------------------------------


def test_chunk_done_tracks_cached_and_fetched():
    reporter = _make(total=10, enabled=False)
    reporter.chunk_done(completed=1, cached=True)
    reporter.chunk_done(completed=2, cached=False)
    reporter.chunk_done(completed=3, cached=False)
    assert reporter._completed == 3  # noqa: SLF001
    assert reporter._cached == 1  # noqa: SLF001
    assert reporter._fetched == 2  # noqa: SLF001


def test_chunk_done_disabled_does_not_track():
    reporter = _make(total=10, enabled=False)
    reporter.chunk_done(completed=5, cached=True)
    # Disabled reporters still record state; they just produce no output.
    assert reporter._completed == 5  # noqa: SLF001
    assert reporter._cached == 1  # noqa: SLF001


# ---------------------------------------------------------------------------
# constructor — initial tick + total clamping
# ---------------------------------------------------------------------------


def test_total_is_clamped_to_one():
    reporter = progress.ProgressReporter("x", 0)
    assert reporter._total == 1  # noqa: SLF001


def test_no_initial_flush_when_total_is_one(forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: True)
    progress.ProgressReporter("x", 1)
    err = capsys.readouterr().err
    assert err == ""


def test_initial_flush_when_total_gt_one(forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: True)
    progress.ProgressReporter("swaps", 10)
    err = capsys.readouterr().err
    assert "swaps:" in err
    assert "0/10" in err


def test_no_initial_flush_when_disabled(forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: True)
    progress.ProgressReporter("swaps", 10, enabled=False)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# tty in-place rendering
# ---------------------------------------------------------------------------


def test_tty_flush_writes_in_place(forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: True)
    reporter = progress.ProgressReporter("swaps", 10)
    reporter._last_logged_pct = -1  # noqa: SLF001
    capsys.readouterr()  # discard the constructor's initial flush
    reporter.chunk_done(completed=5, cached=True)
    err = capsys.readouterr().err
    assert "\r" in err  # carriage-return, in-place update


def test_tty_finish_clears_and_logs_summary(caplog, forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: True)
    caplog.set_level(logging.INFO, logger="undertow.data.progress")
    reporter = progress.ProgressReporter("swaps", 10)
    capsys.readouterr()  # discard initial flush
    reporter.chunk_done(completed=10, cached=False)
    reporter.finish()
    err = capsys.readouterr().err
    # The clear sequence writes the 120-space overwrite line.
    assert "\r" + " " * 120 + "\r" in err
    # The summary is a single LOGGER.info call.
    records = [r for r in caplog.records if r.name == "undertow.data.progress"]
    summaries = [r.message for r in records if "done —" in r.message]
    assert summaries == ["swaps: done — 0 cached, 1 fetched"]


# ---------------------------------------------------------------------------
# non-tty milestone logging
# ---------------------------------------------------------------------------


def test_nontty_logs_milestones(caplog, forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: False)
    caplog.set_level(logging.INFO, logger="undertow.data.progress")
    reporter = progress.ProgressReporter("swaps", 10)
    capsys.readouterr()  # discard initial flush
    for i in range(1, 11):
        reporter.chunk_done(completed=i, cached=(i % 2 == 0))
    records = [r.message for r in caplog.records if r.name == "undertow.data.progress"]
    # Milestones fire at 0% (initial), then 10, 20, ... 100 — never twice per band.
    assert records[0].startswith("swaps:")
    assert len(records) == 11  # initial + 10 milestone crossings


def test_nontty_does_not_write_stderr(forced_flush, monkeypatch, capsys):
    monkeypatch.setattr(progress, "_stderr_is_tty", lambda: False)
    reporter = progress.ProgressReporter("swaps", 10)
    capsys.readouterr()
    reporter.chunk_done(completed=5, cached=True)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# finish edge cases
# ---------------------------------------------------------------------------


def test_finish_is_noop_when_total_is_one(caplog):
    reporter = progress.ProgressReporter("x", 1)
    reporter.finish()
    records = [r for r in caplog.records if r.name == "undertow.data.progress"]
    assert records == []


def test_finish_is_noop_when_disabled(caplog):
    reporter = progress.ProgressReporter("x", 10, enabled=False)
    reporter.finish()
    records = [r for r in caplog.records if r.name == "undertow.data.progress"]
    assert records == []