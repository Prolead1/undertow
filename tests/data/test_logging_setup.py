"""Tests for ``undertow.data.logging_setup``."""

from __future__ import annotations

import logging

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