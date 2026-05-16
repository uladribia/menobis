"""Tests for the `odme generate` CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from odme.cli.main import app

runner = CliRunner()


def test_generate_poisson_to_file(tmp_path: Path) -> None:
    """The generate poisson command writes a sampled edge table."""
    input_path = tmp_path / "edges.csv"
    output_path = tmp_path / "sample.csv"
    input_path.write_text("source,target,weight\n0,1,10\n1,2,20\n0,2,30\n")

    result = runner.invoke(
        app,
        [
            "generate",
            "poisson",
            str(input_path),
            "-o",
            str(output_path),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    content = output_path.read_text()
    assert "source" in content
    assert "weight" in content


def test_generate_poisson_json(tmp_path: Path) -> None:
    """--json outputs JSON."""
    input_path = tmp_path / "edges.csv"
    input_path.write_text("source,target,weight\n0,1,10\n1,2,20\n0,2,30\n")

    result = runner.invoke(
        app, ["generate", "poisson", str(input_path), "--json", "--seed", "42"]
    )

    assert result.exit_code == 0, result.output
    assert '"weight"' in result.output
