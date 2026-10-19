"""Shared pytest fixtures for the ``undertow.ui`` test suite.

U01 seeds the config fixture; U02 appends the shared ``run_dir`` factory here.
Keep this module tiny — later tasks add fixtures, they do not edit existing ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from undertow.ui.config import UiConfig


@pytest.fixture
def ui_config(tmp_path: Path) -> UiConfig:
    """A :class:`UiConfig` rooted entirely inside an isolated ``tmp_path``."""
    return UiConfig(runs_dir=tmp_path, results_dir=tmp_path)
