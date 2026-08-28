"""Shared pytest configuration for the Undertow test suite.

Defines the ``--run-network`` flag and a collection hook that skips
Network-marked tests unless the flag is passed. Network tests hit live
endpoints and are therefore deselected by default (see README).
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.network (hit live endpoints)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``network``-marked tests unless ``--run-network`` is passed.

    Uses explicit ``skip`` (not deselection) so that a ``--collect-only``/
    ``--run-network`` invocation still exposes the tests, and so that the
    ``network`` marker can be combined with ``-m`` fixtures without colliding
    with an ``addopts``-level ``-m "not network"``.
    """
    if config.getoption("--run-network"):
        return  # explicitly requested: run them
    skip_network = pytest.mark.skip(reason="network test; pass --run-network to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)
