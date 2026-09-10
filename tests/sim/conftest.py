"""Shared test fixtures for undertow.sim."""

from __future__ import annotations

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register sim-specific markers."""
    config.addinivalue_line("markers", "slow: > 5s (default-on, budgeted)")


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded Generator for deterministic tests."""
    return np.random.default_rng(42)