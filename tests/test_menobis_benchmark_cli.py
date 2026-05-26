"""Tests for the canonical benchmark CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli import app


def test_benchmark_cli_generates_pa_geographic_network(tmp_path) -> None:
    """Generate stage writes canonical network artifacts and JSON rows."""
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--nodes",
            "20",
            "--families",
            "me",
            "--constraints",
            "strength",
            "--output",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert rows[0]["case"] == "pa-geographic"
    assert "edges=" in rows[0]["message"]
    assert (tmp_path / "networks" / "n20.npz").exists()
