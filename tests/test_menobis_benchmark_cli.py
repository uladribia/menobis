"""Tests for the modern benchmark CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli import app


def test_benchmark_cli_runs_modern_e2e_smoke(tmp_path: Path) -> None:
    """The benchmark CLI runs PA-geographic fit/sample/filter rows."""
    output = tmp_path / "summary.json"
    result = CliRunner().invoke(
        app,
        [
            "all",
            "--nodes",
            "12",
            "--families",
            "me",
            "--constraints",
            "strength",
            "--regime",
            "sparse",
            "--known-pairs",
            "0.0",
            "--filter-samples",
            "1",
            "--no-memory",
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert output.exists()
    stages = {row["stage"] for row in rows}
    assert "generate" in stages
    assert "fit" in stages


def test_benchmark_cli_partial_fitting(tmp_path: Path) -> None:
    """Partial fitting with known pairs runs without error."""
    output = tmp_path / "partial.json"
    result = CliRunner().invoke(
        app,
        [
            "fit",
            "--nodes",
            "12",
            "--families",
            "me",
            "--constraints",
            "strength",
            "--regime",
            "sparse",
            "--known-pairs",
            "0.0,0.05,0.20",
            "--no-memory",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output


def test_benchmark_cli_compare_subcommand() -> None:
    """Compare subcommand runs without error."""
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--nodes",
            "12",
            "--families",
            "me",
            "--constraints",
            "strength",
            "--no-memory",
        ],
    )
    assert result.exit_code == 0, result.output
