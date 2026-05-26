"""Tests for ME model generation CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from menobis.cli.main import app

runner = CliRunner()


def _edges(path: Path) -> None:
    path.write_text("source,target,weight\n0,1,10\n1,2,20\n0,2,30\n")


def test_generate_multinomial_json(tmp_path: Path) -> None:
    input_path = tmp_path / "edges.csv"
    _edges(input_path)
    result = runner.invoke(
        app,
        [
            "generate",
            "strength-multinomial",
            str(input_path),
            "--total-events",
            "60",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"weight"' in result.output


def test_generate_degree_events_me_json(tmp_path: Path) -> None:
    input_path = tmp_path / "edges.csv"
    _edges(input_path)
    result = runner.invoke(
        app,
        [
            "generate",
            "degree-events-poisson",
            str(input_path),
            "--total-events",
            "60",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"weight"' in result.output


def test_generate_strength_edges_me_json(tmp_path: Path) -> None:
    input_path = tmp_path / "edges.csv"
    _edges(input_path)
    result = runner.invoke(
        app, ["generate", "strength-edges-poisson", str(input_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert '"weight"' in result.output


def test_generate_custom_pij_json(tmp_path: Path) -> None:
    probabilities_path = tmp_path / "probabilities.csv"
    probabilities_path.write_text("source,target,probability\n0,1,0.25\n1,2,0.75\n")
    result = runner.invoke(
        app,
        [
            "generate",
            "custom-poisson",
            str(probabilities_path),
            "--total-events",
            "20",
            "--ensemble",
            "multinomial",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"weight"' in result.output
