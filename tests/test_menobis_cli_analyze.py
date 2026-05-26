"""Tests for the `menobis analyze` command."""

from pathlib import Path

from typer.testing import CliRunner

from menobis.cli.main import app

runner = CliRunner()


def test_analyze_strengths_to_file(tmp_path: Path) -> None:
    """The analyze strengths command writes CSV to --output."""
    input_path = tmp_path / "edges.csv"
    output_path = tmp_path / "strengths.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n")

    result = runner.invoke(
        app, ["analyze", "strengths", str(input_path), "-o", str(output_path)]
    )

    assert result.exit_code == 0
    assert (
        output_path.read_text()
        == "node,strength_out,strength_in\n0,3,0\n1,4,3\n2,0,4\n"
    )


def test_analyze_strengths_to_stdout(tmp_path: Path) -> None:
    """Without --output, CSV goes to stdout."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n")

    result = runner.invoke(app, ["analyze", "strengths", str(input_path), "--quiet"])

    assert result.exit_code == 0
    assert "strength_out" in result.output


def test_analyze_strengths_json(tmp_path: Path) -> None:
    """--json outputs JSON to stdout."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,3\n1,2,4\n")

    result = runner.invoke(app, ["analyze", "strengths", str(input_path), "--json"])

    assert result.exit_code == 0
    assert '"strength_out"' in result.output
