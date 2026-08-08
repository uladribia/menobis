"""pytest configuration: fast/heavy test splitting.

Usage:
    pytest                     # fast tests only (skip heavy)
    pytest --run-heavy         # all tests including heavy
    pytest -m heavy            # heavy tests only
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: slow E2E / benchmark-level tests (>2s). Skip unless --run-heavy or -m heavy.",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="include tests marked @pytest.mark.heavy (slow E2E / benchmarking)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-heavy"):
        return  # explicit run-heavy → include all
    # If user passed -m and it mentions "heavy", honour their filter.
    markexpr = config.getoption("markexpr") or ""
    if "heavy" in markexpr:
        return
    skip_heavy = pytest.mark.skip(
        reason="heavy test; use --run-heavy or -m heavy to include"
    )
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)