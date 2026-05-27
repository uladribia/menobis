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
            "--filter-samples",
            "1",
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    rows = json.loads(result.stdout)
    assert output.exists()
    assert {row["stage"] for row in rows} == {
        "generate",
        "fit",
        "sample-check",
        "filter-fpr",
    }
    assert rows[0]["constraint"] == "pa-geographic"
    assert rows[0]["sampled_edges"] > 0
