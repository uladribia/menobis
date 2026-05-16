"""Tests for I/O."""

from pathlib import Path

import numpy as np

from odme.data.io import read_edges, write_edges


def test_read_csv(tmp_path: Path) -> None:
    path = tmp_path / "edges.csv"
    path.write_text("source,target,weight\n0,1,3\n1,2,4\n")
    edges = read_edges(path)
    assert edges.total_events == 7
    assert edges.num_edges == 2


def test_read_tsv(tmp_path: Path) -> None:
    path = tmp_path / "edges.tsv"
    path.write_text("source\ttarget\tweight\n0\t1\t3\n")
    edges = read_edges(path)
    assert edges.total_events == 3


def test_roundtrip_parquet(tmp_path: Path) -> None:
    from odme.data.frames import EdgeTable

    original = EdgeTable(
        source=np.array([0, 1], dtype=np.uint64),
        target=np.array([1, 2], dtype=np.uint64),
        weight=np.array([3, 4], dtype=np.uint64),
    )
    path = tmp_path / "edges.parquet"
    write_edges(original, path)
    roundtrip = read_edges(path)
    assert roundtrip.total_events == 7
