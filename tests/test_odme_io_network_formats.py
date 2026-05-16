"""Tests for network format readers."""

from pathlib import Path

import pytest

from odme.data.io import read_edges


def test_graphml(tmp_path: Path) -> None:
    path = tmp_path / "n.graphml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        '  <key id="w" for="edge" attr.name="weight" attr.type="int"/>\n'
        '  <graph edgedefault="directed">\n'
        '    <node id="0"/><node id="1"/><node id="2"/>\n'
        '    <edge source="0" target="1"><data key="w">3</data></edge>\n'
        '    <edge source="1" target="2"><data key="w">4</data></edge>\n'
        "  </graph>\n"
        "</graphml>\n"
    )
    edges = read_edges(path)
    assert edges.total_events == 7


def test_matrix_market(tmp_path: Path) -> None:
    path = tmp_path / "n.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate integer general\n3 3 2\n1 2 3\n2 3 4\n"
    )
    edges = read_edges(path)
    assert edges.total_events == 7


def test_pajek(tmp_path: Path) -> None:
    path = tmp_path / "n.net"
    path.write_text('*Vertices 3\n1 "A"\n2 "B"\n3 "C"\n*Arcs\n1 2 3\n2 3 4\n')
    edges = read_edges(path)
    assert edges.total_events == 7


def test_graphml_rejects_non_integer(tmp_path: Path) -> None:
    path = tmp_path / "n.graphml"
    path.write_text(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        '  <key id="w" for="edge" attr.name="weight" attr.type="double"/>\n'
        '  <graph edgedefault="directed">\n'
        '    <node id="0"/><node id="1"/>\n'
        '    <edge source="0" target="1"><data key="w">1.5</data></edge>\n'
        "  </graph>\n"
        "</graphml>\n"
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        read_edges(path)


def test_graphml_ignores_zero_weights(tmp_path: Path) -> None:
    path = tmp_path / "n.graphml"
    path.write_text(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        '  <key id="w" for="edge" attr.name="weight" attr.type="int"/>\n'
        '  <graph edgedefault="directed">\n'
        '    <node id="0"/><node id="1"/><node id="2"/>\n'
        '    <edge source="0" target="1"><data key="w">0</data></edge>\n'
        '    <edge source="1" target="2"><data key="w">4</data></edge>\n'
        "  </graph>\n"
        "</graphml>\n"
    )
    edges = read_edges(path)
    assert edges.num_edges == 1
    assert edges.total_events == 4
