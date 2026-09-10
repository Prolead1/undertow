"""Shared test fixtures for undertow.sim."""

from __future__ import annotations

import pytest
import numpy as np


def pytest_configure(config: pytest.Config) -> None:
    """Register sim-specific markers."""
    config.addinivalue_line("markers", "slow: > 5s (default-on, budgeted)")


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded Generator for deterministic tests."""
    return np.random.default_rng(42)