"""Tests for graph-library dependency boundaries."""

import tomllib
from pathlib import Path


def test_no_interop_package_is_shipped() -> None:
    """ODME should not ship graph-library adapter packages."""
    interop_dir = Path(__file__).parents[1] / "src" / "odme" / "interop"
    assert not interop_dir.exists()


def test_project_has_no_graph_library_runtime_dependency() -> None:
    """ODME runtime dependencies must stay decoupled from graph libraries."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    dependencies = "\n".join(project["dependencies"])
    assert "networkx" not in dependencies.lower()
    assert "rustworkx" not in dependencies.lower()
