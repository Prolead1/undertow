"""Tests for ``undertow.data.logging_setup``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from undertow.data import logging_setup


def test_package_logger_name_is_correct():
    assert logging_setup.PACKAGE_LOGGER_NAME == "undertow.data"


def test_configure_logging_sets_level_info_by_default():
    logging_setup.configure_logging(force=True)
    logger = logging.getLogger("undertow.data")
    assert logger.level == logging.INFO


def test_configure_logging_debug_level():
    logging_setup.configure_logging(level="DEBUG", force=True)
    logger = logging.getLogger("undertow.data")
    assert logger.level == logging.DEBUG


def test_configure_logging_warning_level():
    logging_setup.configure_logging(level="WARNING", force=True)
    logger = logging.getLogger("undertow.data")
    assert logger.level == logging.WARNING


def test_configure_logging_is_idempotent():
    """Second call without force is a no-op — level stays as set by first call."""
    logging_setup.configure_logging(level="WARNING", force=True)
    logging_setup.configure_logging(level="DEBUG")  # no force → should be ignored
    logger = logging.getLogger("undertow.data")
    assert logger.level == logging.WARNING


def test_configure_logging_force_replaces():
    logging_setup.configure_logging(level="WARNING", force=True)
    logging_setup.configure_logging(level="DEBUG", force=True)
    logger = logging.getLogger("undertow.data")
    assert logger.level == logging.DEBUG


def test_logger_has_stderr_handler(monkeypatch):
    # Drop pytest's captured stderr so the handler wraps the real sys.stderr.
    import sys
    real_stderr = sys.__stderr__
    monkeypatch.setattr(sys, "stderr", real_stderr)
    # Reset the shared handler so it is re-created against the real stderr.
    monkeypatch.setattr(logging_setup, "_STREAM", None)

    logging_setup.configure_logging(force=True)
    logger = logging.getLogger("undertow.data")
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    # The stream must be stderr, never stdout.
    assert handler.stream is sys.stderr


def test_logger_does_not_propagate_to_root():
    logging_setup.configure_logging(force=True)
    logger = logging.getLogger("undertow.data")
    assert logger.propagate is False


def test_sub_logger_inherits_configuration():
    """A child logger (e.g. undertow.data.fetchers) inherits the package level."""
    logging_setup.configure_logging(level="DEBUG", force=True)
    child = logging.getLogger("undertow.data.fetchers.gas")
    assert child.level == logging.NOTSET  # child inherits; effective level is DEBUG
    assert child.isEnabledFor(logging.DEBUG)


def test_configure_logging_force_does_not_stack_handlers():
    """Multiple force calls should leave exactly one handler, not stack them."""
    logging_setup.configure_logging(force=True)
    logging_setup.configure_logging(force=True)
    logging_setup.configure_logging(force=True)
    logger = logging.getLogger("undertow.data")
    assert len(logger.handlers) == 1


@pytest.mark.parametrize("level", ["INFO", "DEBUG", "WARNING"])
def test_configure_logging_accepts_standard_levels(level):
    logging_setup.configure_logging(level=level, force=True)
    logger = logging.getLogger("undertow.data")
    assert logger.level == getattr(logging, level)


# ---------------------------------------------------------------------------
# add_file_handler — automatic DEBUG capture to a file
# ---------------------------------------------------------------------------


def _file_handlers() -> list[logging.Handler]:
    logger = logging.getLogger("undertow.data")
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_add_file_handler_creates_parent_dirs_and_captures_debug(tmp_path: Path) -> None:
    logging_setup.configure_logging(level="INFO", force=True)
    log_path = tmp_path / "nested" / "run.log"
    returned = logging_setup.add_file_handler(log_path)
    assert returned == log_path
    assert log_path.parent.is_dir()

    # A child logger's debug record must land in the file...
    logging.getLogger("undertow.data.fetchers.reference").debug("diagnostic %d", 7)
    for handler in _file_handlers():
        handler.flush()
    assert "diagnostic 7" in log_path.read_text(encoding="utf-8")


def test_add_file_handler_raises_logger_floor_but_not_console_level(tmp_path: Path) -> None:
    logging_setup.configure_logging(level="INFO", force=True)
    logging_setup.add_file_handler(tmp_path / "run.log")
    logger = logging.getLogger("undertow.data")
    assert logger.level == logging.DEBUG  # so debug records reach the file
    assert logging_setup._stderr_handler().level == logging.INFO  # console unchanged


def test_add_file_handler_file_level_defaults_to_debug(tmp_path: Path) -> None:
    logging_setup.configure_logging(force=True)
    logging_setup.add_file_handler(tmp_path / "run.log")
    assert _file_handlers()[0].level == logging.DEBUG


def test_add_file_handler_is_idempotent_per_path(tmp_path: Path) -> None:
    logging_setup.configure_logging(force=True)
    log_path = tmp_path / "run.log"
    logging_setup.add_file_handler(log_path)
    logging_setup.add_file_handler(log_path)
    assert len(_file_handlers()) == 1
    # A distinct path still attaches a second handler.
    logging_setup.add_file_handler(tmp_path / "run2.log")
    assert len(_file_handlers()) == 2


def test_configure_logging_force_removes_file_handlers(tmp_path: Path) -> None:
    logging_setup.configure_logging(force=True)
    logging_setup.add_file_handler(tmp_path / "run.log")
    assert len(_file_handlers()) == 1
    logging_setup.configure_logging(force=True)
    assert _file_handlers() == []
    # The sentinel is reset, so the same path can be attached again after force.
    logging_setup.add_file_handler(tmp_path / "run.log")
    assert len(_file_handlers()) == 1