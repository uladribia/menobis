"""Input/output helpers for ODME data files.

Uses pyarrow for Parquet/IPC and Python stdlib for CSV/text formats.
"""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

from odme.data.frames import (
    EdgeTable,
    ProbabilityTable,
    normalize_edges,
    normalize_probabilities,
)

CSV_SUFFIXES = {".csv"}
TSV_SUFFIXES = {".tsv", ".tab"}
PARQUET_SUFFIXES = {".parquet", ".pq"}
IPC_SUFFIXES = {".arrow", ".ipc", ".feather"}
GRAPHML_SUFFIXES = {".graphml"}
MATRIX_MARKET_SUFFIXES = {".mtx", ".mm"}
PAJEK_SUFFIXES = {".net", ".paj"}


def read_edges(path: Path | str) -> EdgeTable:
    """Read an ODME weighted edge table from a supported file format.

    Args:
        path: Input path.

    Returns:
        Normalized EdgeTable.
    """
    edge_path = Path(path)
    suffix = edge_path.suffix.lower()

    if suffix in CSV_SUFFIXES:
        return _read_csv(edge_path, ",")
    if suffix in TSV_SUFFIXES:
        return _read_csv(edge_path, "\t")
    if suffix in PARQUET_SUFFIXES:
        return _read_parquet(edge_path)
    if suffix in IPC_SUFFIXES:
        return _read_ipc(edge_path)
    if suffix in GRAPHML_SUFFIXES:
        return _read_graphml(edge_path)
    if suffix in MATRIX_MARKET_SUFFIXES:
        return _read_matrix_market(edge_path)
    if suffix in PAJEK_SUFFIXES:
        return _read_pajek(edge_path)
    msg = f"unsupported edge file format: {suffix or '<none>'}"
    raise ValueError(msg)


def read_probabilities(path: Path | str) -> ProbabilityTable:
    """Read a sparse custom p_ij table with source, target, probability columns."""
    probability_path = Path(path)
    suffix = probability_path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        table = pa_csv.read_csv(probability_path)
    elif suffix in TSV_SUFFIXES:
        table = pa_csv.read_csv(
            probability_path, parse_options=pa_csv.ParseOptions(delimiter="\t")
        )
    elif suffix in PARQUET_SUFFIXES:
        table = pq.read_table(probability_path)
    elif suffix in IPC_SUFFIXES:
        with pa.OSFile(str(probability_path), "rb") as f:
            table = pa.ipc.open_file(f).read_all()
    else:
        msg = f"unsupported probability file format: {suffix or '<none>'}"
        raise ValueError(msg)
    return normalize_probabilities(
        source=table.column("source").to_numpy(),
        target=table.column("target").to_numpy(),
        probability=table.column("probability").to_numpy(),
    )


def write_edges(edges: EdgeTable, path: Path | str) -> None:
    """Write an ODME edge table to a supported file format."""
    edge_path = Path(path)
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = edge_path.suffix.lower()

    table = pa.table(
        {
            "source": pa.array(edges.source, type=pa.uint64()),
            "target": pa.array(edges.target, type=pa.uint64()),
            "weight": pa.array(edges.weight, type=pa.uint64()),
        }
    )

    if suffix in CSV_SUFFIXES:
        pa_csv.write_csv(table, edge_path)
    elif suffix in TSV_SUFFIXES:
        pa_csv.write_csv(table, edge_path, pa_csv.WriteOptions(delimiter="\t"))
    elif suffix in PARQUET_SUFFIXES:
        pq.write_table(table, edge_path)
    elif suffix in IPC_SUFFIXES:
        with pa.OSFile(str(edge_path), "wb") as f:
            writer = pa.ipc.new_file(f, table.schema)
            writer.write_table(table)
            writer.close()
    else:
        msg = f"unsupported edge file format: {suffix or '<none>'}"
        raise ValueError(msg)


def _read_csv(path: Path, delimiter: str) -> EdgeTable:
    table = pa_csv.read_csv(
        path, parse_options=pa_csv.ParseOptions(delimiter=delimiter)
    )
    return _arrow_to_edges(table)


def _read_parquet(path: Path) -> EdgeTable:
    table = pq.read_table(path)
    return _arrow_to_edges(table)


def _read_ipc(path: Path) -> EdgeTable:
    with pa.OSFile(str(path), "rb") as f:
        reader = pa.ipc.open_file(f)
        table = reader.read_all()
    return _arrow_to_edges(table)


def _arrow_to_edges(table: pa.Table) -> EdgeTable:
    return normalize_edges(
        source=table.column("source").to_numpy(),
        target=table.column("target").to_numpy(),
        weight=table.column("weight").to_numpy(),
    )


def _read_graphml(path: Path) -> EdgeTable:
    """Read GraphML edges with integer weights."""
    root = ElementTree.parse(path).getroot()
    namespace = _xml_namespace(root.tag)
    key_to_name = {
        key.attrib["id"]: key.attrib.get("attr.name", key.attrib["id"])
        for key in root.findall(f"{namespace}key")
        if key.attrib.get("for") in {"edge", "all"}
    }
    sources, targets, weights = [], [], []
    for edge in root.findall(f".//{namespace}edge"):
        s = _parse_non_negative_integer(edge.attrib["source"])
        t = _parse_non_negative_integer(edge.attrib["target"])
        w = 1
        for data in edge.findall(f"{namespace}data"):
            if key_to_name.get(data.attrib.get("key", "")) == "weight":
                w = _parse_non_negative_integer(data.text or "")
                break
        sources.append(s)
        targets.append(t)
        weights.append(w)
    return normalize_edges(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.array(weights, dtype=np.uint64),
    )


def _read_matrix_market(path: Path) -> EdgeTable:
    sources, targets, weights = [], [], []
    dimensions_seen = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if not dimensions_seen:
            dimensions_seen = True
            continue
        if len(parts) < 3:
            msg = "Matrix Market coordinate entries must contain source target weight"
            raise ValueError(msg)
        sources.append(_parse_positive_integer(parts[0]) - 1)
        targets.append(_parse_positive_integer(parts[1]) - 1)
        weights.append(_parse_non_negative_integer(parts[2]))
    return normalize_edges(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.array(weights, dtype=np.uint64),
    )


def _read_pajek(path: Path) -> EdgeTable:
    sources, targets, weights = [], [], []
    in_edges = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("*"):
            in_edges = line.lower().startswith(("*arcs", "*edges"))
            continue
        if not in_edges:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sources.append(_parse_positive_integer(parts[0]) - 1)
        targets.append(_parse_positive_integer(parts[1]) - 1)
        weights.append(_parse_non_negative_integer(parts[2]) if len(parts) >= 3 else 1)
    return normalize_edges(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.array(weights, dtype=np.uint64),
    )


def _xml_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", maxsplit=1)[0] + "}"
    return ""


def _parse_non_negative_integer(value: str) -> int:
    stripped = value.strip()
    if not stripped.isdecimal():
        msg = f"expected non-negative integer value, got {value!r}"
        raise ValueError(msg)
    return int(stripped)


def _parse_positive_integer(value: str) -> int:
    stripped = value.strip()
    if not stripped.isdecimal() or int(stripped) <= 0:
        msg = f"expected positive integer value, got {value!r}"
        raise ValueError(msg)
    return int(stripped)


__all__ = ["read_edges", "read_probabilities", "write_edges"]
