"""Tests for fixed-degree fitting CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from odme.cli.main import app

runner = CliRunner()


def test_fit_degrees_json(tmp_path: Path) -> None:
    """The fit degrees command emits degree multipliers as JSON."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n0,2,3\n")

    result = runner.invoke(app, ["fit", "degrees", str(input_path), "--json"])

    assert result.exit_code == 0, result.output
    assert '"x"' in result.output
    assert '"y"' in result.output


def test_fit_strength_degree_zip_to_file(tmp_path: Path) -> None:
    """The fit strength-degree-zip command writes all multipliers."""
    input_path = tmp_path / "edges.csv"
    output_path = tmp_path / "zip.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n0,2,3\n")

    result = runner.invoke(
        app,
        ["fit", "strength-degree-zip", str(input_path), "-o", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    content = output_path.read_text()
    assert "degree_x" in content
    assert "excess_x" in content
