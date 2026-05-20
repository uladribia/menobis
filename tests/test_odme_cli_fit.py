"""Tests for the `odme fit` CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from odme.cli.main import app

runner = CliRunner()


def test_fit_strengths_to_file(tmp_path: Path) -> None:
    """The fit strengths command writes multipliers to --output."""
    input_path = tmp_path / "edges.csv"
    output_path = tmp_path / "multipliers.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n0,2,3\n")

    result = runner.invoke(
        app, ["fit", "strength-poisson", str(input_path), "-o", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    content = output_path.read_text()
    assert "node" in content
    assert "x" in content


def test_fit_strength_geometric_json(tmp_path: Path) -> None:
    """The W geometric strength fit command emits diagnostics as JSON."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,1\n1,2,2\n")

    result = runner.invoke(
        app, ["fit", "strength-geometric", str(input_path), "--json"]
    )

    assert result.exit_code == 0, result.output
    assert '"status"' in result.output
    assert '"max_strength_residual"' in result.output
    assert '"nodes"' in result.output


def test_fit_strengths_json(tmp_path: Path) -> None:
    """--json outputs JSON."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n0,2,3\n")

    result = runner.invoke(app, ["fit", "strength-poisson", str(input_path), "--json"])

    assert result.exit_code == 0, result.output
    assert '"x"' in result.output
