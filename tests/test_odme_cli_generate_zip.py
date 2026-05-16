"""Tests for strength-degree ZIP generation CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from odme.cli.main import app

runner = CliRunner()


def test_generate_strength_degree_zip_json(tmp_path: Path) -> None:
    """The generate strength-degree-zip command emits sampled edges."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n0,2,3\n")

    result = runner.invoke(
        app,
        ["generate", "strength-degree-zip", str(input_path), "--json", "--seed", "42"],
    )

    assert result.exit_code == 0, result.output
    assert '"weight"' in result.output
