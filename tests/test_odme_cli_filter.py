"""CLI tests for statistical filtering."""

from pathlib import Path

from typer.testing import CliRunner

from odme.cli.main import app


def test_filter_custom_rates_cli(tmp_path: Path) -> None:
    edges = tmp_path / "edges.csv"
    rates = tmp_path / "rates.csv"
    out = tmp_path / "filtered"
    edges.write_text("source,target,weight\n0,1,10\n", encoding="utf-8")
    rates.write_text("source,target,rate\n0,1,1.0\n1,0,5.0\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "filter",
            "custom-poisson",
            str(edges),
            "--rates",
            str(rates),
            "--output-prefix",
            str(out),
            "--detect-absent",
        ],
    )

    assert result.exit_code == 0
    assert (out / "upper.csv").exists()
    assert (out / "absent_lower.csv").exists()
