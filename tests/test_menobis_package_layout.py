"""Tests for Python package layout boundaries."""

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
SRC_MENOBIS = PROJECT_ROOT / "src" / "menobis"


def test_benchmark_cli_is_not_packaged() -> None:
    """Repository benchmarks should not add package entry points under menobis."""
    assert not (SRC_MENOBIS / "benchmark_cli.py").exists()

    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = pyproject["project"].get("scripts", {})
    assert "menobis-bench" not in scripts


def test_filtering_is_a_domain_package() -> None:
    """Filtering should be grouped as a package behind the filter CLI domain."""
    assert not (SRC_MENOBIS / "filtering.py").exists()
    assert not (SRC_MENOBIS / "filtering_types.py").exists()

    filtering = SRC_MENOBIS / "filtering"
    assert (filtering / "__init__.py").exists()
    assert (filtering / "types.py").exists()
    assert (filtering / "classify.py").exists()
    assert (filtering / "models.py").exists()


def test_utilities_package_exists() -> None:
    """Shared utilities live under menobis.utilities, not at package root."""
    utilities = SRC_MENOBIS / "utilities"
    assert (utilities / "__init__.py").exists()
    assert (utilities / "synthetic.py").exists()
    assert (utilities / "logging.py").exists()
    # No shims at root
    assert not (SRC_MENOBIS / "synthetic.py").exists()
    assert not (SRC_MENOBIS / "logging.py").exists()


def test_unified_router_is_not_inside_models_domain() -> None:
    """Verb routing belongs above model/filter domains to avoid cycles."""
    routing = SRC_MENOBIS / "routing.py"
    assert routing.exists()
    assert not (SRC_MENOBIS / "models" / "routing.py").exists()

    for path in (SRC_MENOBIS / "models").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "menobis.filtering" not in source, path
