"""Tests for Python package layout boundaries."""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SRC_ODME = PROJECT_ROOT / "src" / "odme"


def test_benchmark_cli_is_not_packaged() -> None:
    """Repository benchmarks should not add package entry points under odme."""
    assert not (SRC_ODME / "benchmark_cli.py").exists()

    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = pyproject["project"].get("scripts", {})
    assert "odme-bench" not in scripts


def test_filtering_is_a_domain_package() -> None:
    """Filtering should be grouped as a package behind the filter CLI domain."""
    assert not (SRC_ODME / "filtering.py").exists()
    assert not (SRC_ODME / "filtering_types.py").exists()

    filtering = SRC_ODME / "filtering"
    assert (filtering / "__init__.py").exists()
    assert (filtering / "types.py").exists()
    assert (filtering / "classify.py").exists()
    assert (filtering / "models.py").exists()


def test_utilities_package_exists() -> None:
    """Shared utilities live under odme.utilities, not at package root."""
    utilities = SRC_ODME / "utilities"
    assert (utilities / "__init__.py").exists()
    assert (utilities / "synthetic.py").exists()
    assert (utilities / "logging.py").exists()
    # No shims at root
    assert not (SRC_ODME / "synthetic.py").exists()
    assert not (SRC_ODME / "logging.py").exists()
