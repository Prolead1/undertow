"""Shared pytest configuration for the Undertow test suite.

Defines the ``--run-network`` flag and a collection hook that skips
Network-marked tests unless the flag is passed. Network tests hit live
endpoints and are therefore deselected by default (see README).
"""

from __future__ import annotations

import os

import pytest

# Pin BLAS/OpenMP/NumExpr to a single thread per test process. The suite's
# training and backtest tests are small enough that thread-pool synchronisation
# costs more than the work; left at the host default, numpy/torch spawn one
# thread per core and oversubscribe the machine -- badly so under ``-n auto``.
# Set at import time, before numpy/torch are first imported by any test module,
# and inherited by the subprocess-based CLI tests. An explicit non-empty value
# (e.g. ``OMP_NUM_THREADS=8``) is honoured; empty values are treated as unset so
# a stray ``ENV OMP_NUM_THREADS=`` in a container cannot silently defeat it.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    if not os.environ.get(_thread_var):
        os.environ[_thread_var] = "1"


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
