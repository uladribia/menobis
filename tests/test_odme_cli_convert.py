"""Test odme convert CLI command."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from odme.cli.main import app
from odme.data.frames import EdgeTable
from odme.data.io import read_edges, write_edges

runner = CliRunner()


def test_csv_to_parquet(tmp_path: Path) -> None:
    """Convert CSV to Parquet and read back."""
    csv_path = tmp_path / "edges.csv"
    parquet_path = tmp_path / "edges.parquet"
    edges = EdgeTable(
        source=np.array([0, 1], dtype=np.uint64),
        target=np.array([1, 0], dtype=np.uint64),
        weight=np.array([3, 2], dtype=np.uint64),
    )
    write_edges(edges, csv_path)
    result = runner.invoke(app, ["convert", str(csv_path), "-o", str(parquet_path)])
    assert result.exit_code == 0, result.output
    recovered = read_edges(parquet_path)
    assert len(recovered) == 2
    np.testing.assert_array_equal(recovered.source, edges.source)
    np.testing.assert_array_equal(recovered.weight, edges.weight)


def test_parquet_to_csv(tmp_path: Path) -> None:
    """Convert Parquet to CSV and read back."""
    parquet_path = tmp_path / "edges.parquet"
    csv_path = tmp_path / "edges.csv"
    edges = EdgeTable(
        source=np.array([0, 1, 2], dtype=np.uint64),
        target=np.array([1, 2, 0], dtype=np.uint64),
        weight=np.array([1, 4, 2], dtype=np.uint64),
    )
    write_edges(edges, parquet_path)
    result = runner.invoke(app, ["convert", str(parquet_path), "-o", str(csv_path)])
    assert result.exit_code == 0, result.output
    recovered = read_edges(csv_path)
    assert len(recovered) == 3
    np.testing.assert_array_equal(recovered.target, edges.target)
