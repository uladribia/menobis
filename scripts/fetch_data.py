#!/usr/bin/env python
"""Fetch real-world Origin-Destination (OD) flow networks for MENoBiS.

Downloads directed weighted networks from public sources, converts them
to MENoBiS EdgeTable format (source, target, weight), and stores them
in a gitignored data directory.

Built-in OD datasets (use ``list`` to see all)::

    openflights    Global airport route network (directed, weighted by
                   number of airlines serving each route). Perfect OD
                   test case for MENoBiS.

    email-eu       Email-Eu-core directed communication network from
                   SNAP. All edges have weight 1. Simple test case.

Usage::

    # List available datasets
    python fetch_data.py list

    # Download and prepare the OpenFlights OD network (default: data/)
    python fetch_data.py openflights

    # Custom output directory
    python fetch_data.py email-eu --output-dir /tmp/my-data

    # Use a custom URL (Matrix Market or CSV with source,target[,weight])
    python fetch_data.py url https://example.com/graph.mtx

    # Force re-download even if cached
    python fetch_data.py openflights --force
"""

from __future__ import annotations

import csv
import gzip
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from typer import Option, Typer

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

CACHE_DIR_DEFAULT = Path("~/.cache/menobis-datasets").expanduser()


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a built-in downloadable dataset."""

    name: str
    description: str
    urls: dict[str, str]  # logical name -> download URL
    format: str
    node_count_hint: int | None = None
    edge_count_hint: int | None = None


BUILTIN_DATASETS: list[DatasetInfo] = [
    DatasetInfo(
        name="openflights",
        description=(
            "Global airport route network (directed, weighted by number of "
            "airlines per route). Origin-destination pairs from the "
            "OpenFlights dataset with IATA codes mapped to integer node IDs."
        ),
        urls={
            "airports": (
                "https://raw.githubusercontent.com/jpatokal/openflights/"
                "master/data/airports.dat"
            ),
            "routes": (
                "https://raw.githubusercontent.com/jpatokal/openflights/"
                "master/data/routes.dat"
            ),
        },
        format="openflights",
        node_count_hint=17612,
        edge_count_hint=37595,
    ),
    DatasetInfo(
        name="email-eu",
        description=(
            "Email-Eu-core directed communication network from SNAP. "
            "All edges have weight 1. Simple test case for MENoBiS."
        ),
        urls={
            "edges": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
        },
        format="snap-edge",
        node_count_hint=1005,
        edge_count_hint=25571,
    ),
]

_DATASET_MAP: dict[str, DatasetInfo] = {d.name: d for d in BUILTIN_DATASETS}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Download a (possibly gzipped) file to *dest*.

    Returns the path to the decompressed file (or the original if not gzipped).
    """
    if dest.exists() and not force:
        print(f"  Already cached: {dest}", file=sys.stderr)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "MENoBiS/0.1"})
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    # Auto-decompress gzip
    if url.endswith(".gz"):
        decompressed = dest.with_suffix("")
        try:
            data = gzip.decompress(raw)
        except gzip.BadGzipFile:
            data = raw  # not actually gzipped despite .gz extension
        decompressed.write_bytes(data)
        dest.write_bytes(raw)  # keep gz for re-cache
        print(f"  Decompressed to: {decompressed}", file=sys.stderr)
        return decompressed

    dest.write_bytes(raw)
    return dest


def _download_to_text(url: str, dest: Path, force: bool = False) -> list[str]:
    """Download and return lines, optionally caching."""
    raw_path = _download(url, dest, force=force)
    return raw_path.read_text().splitlines()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_openflights(
    cache_dir: Path, force: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse OpenFlights airports + routes into an OD matrix.

    Airport IATA codes are mapped to sequential integer node IDs.
    Weight = number of airlines serving each directed route.
    """
    ds = _DATASET_MAP["openflights"]
    airport_url = ds.urls["airports"]
    routes_url = ds.urls["routes"]

    a_dest = cache_dir / "openflights" / "airports.dat"
    r_dest = cache_dir / "openflights" / "routes.dat"

    airport_lines = _download_to_text(airport_url, a_dest, force=force)
    route_lines = _download_to_text(routes_url, r_dest, force=force)

    # Build IATA -> integer ID mapping
    iata_to_id: dict[str, int] = {}
    for line in airport_lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            # Airport ID, name, city, country, IATA, ICAO, ...
            iata = parts[4].strip('"')
            icao = parts[5].strip('"')
        except IndexError:
            continue
        code = iata or icao
        if code and code not in iata_to_id:
            iata_to_id[code] = len(iata_to_id)

    # Aggregate routes: (src_code, dst_code) -> airline_count
    route_agg: dict[tuple[str, str], int] = {}
    for line in route_lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        src = parts[2]
        dst = parts[4]
        if not src or not dst:
            continue
        key = (src, dst)
        route_agg[key] = route_agg.get(key, 0) + 1

    # Build arrays
    sources, targets, weights = [], [], []
    for (src, dst), cnt in route_agg.items():
        if src in iata_to_id and dst in iata_to_id:
            sources.append(iata_to_id[src])
            targets.append(iata_to_id[dst])
            weights.append(cnt)

    return _normalize(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.array(weights, dtype=np.uint64),
    )


def _parse_snap_edge_list(
    url: str, cache_dir: Path, force: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a SNAP edge-list file (directed, unweighted).

    Each line: ``from_id  to_id``
    Lines starting with ``#`` are comments.
    """
    filename = url.rstrip("/").split("/")[-1]
    dest = cache_dir / "snap" / filename
    lines = _download_to_text(url, dest, force=force)

    sources, targets = [], []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            sources.append(int(parts[0]))
            targets.append(int(parts[1]))

    return _normalize(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.ones(len(sources), dtype=np.uint64),
    )


def _parse_mtx(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Matrix Market coordinate file."""
    sources, targets, weights = [], [], []
    in_body = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if not in_body:
            in_body = True
            continue
        s, t = int(parts[0]), int(parts[1])
        w = int(parts[2]) if len(parts) >= 3 else 1
        sources.append(s - 1)
        targets.append(t - 1)
        weights.append(w)
    return _normalize(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.array(weights, dtype=np.uint64),
    )


def _parse_generic_csv(
    path: Path, delimiter: str = ",", has_header: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a CSV/TSV file with source,target[,weight] columns."""
    sources, targets, weights = [], [], []
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if has_header and i == 0:
                continue
            if not row or len(row) < 2:
                continue
            sources.append(int(row[0]))
            targets.append(int(row[1]))
            w = int(row[2]) if len(row) >= 3 and row[2].strip() else 1
            weights.append(w)
    return _normalize(
        np.array(sources, dtype=np.uint64),
        np.array(targets, dtype=np.uint64),
        np.array(weights, dtype=np.uint64),
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply MENoBiS normalization: drop zero-weight edges, sort.

    Returns (source, target, weight) as uint64 arrays.
    """
    w = weights.astype(np.int64, copy=False)
    mask = w > 0
    s = sources[mask].astype(np.uint64, copy=False)
    t = targets[mask].astype(np.uint64, copy=False)
    wgt = w[mask].astype(np.uint64, copy=False)
    order = np.lexsort((t, s))
    return s[order], t[order], wgt[order]


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    dataset: DatasetInfo,
    cache_dir: Path,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download and parse a built-in dataset.

    Returns (sources, targets, weights) as uint64 arrays.
    """
    fmt = dataset.format
    if fmt == "openflights":
        return _parse_openflights(cache_dir, force)

    if fmt == "snap-edge":
        url = dataset.urls["edges"]
        return _parse_snap_edge_list(url, cache_dir, force)

    msg = f"Unknown format: {fmt}"
    raise ValueError(msg)


def prepare_from_url(
    url: str,
    cache_dir: Path,
    format_hint: str | None = None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download from a custom URL and parse.

    Args:
        url: Download URL.
        cache_dir: Cache directory.
        format_hint: Override format detection (mtx, csv, tsv).
        force: Re-download even if cached.

    Returns:
        (sources, targets, weights) as uint64 arrays.
    """
    filename = url.rstrip("/").split("/")[-1]
    dest = cache_dir / "custom" / filename
    path = _download(url, dest, force=force)

    fmt = format_hint.lower() if format_hint else _infer_format(path.suffix.lower())

    if fmt == "mtx":
        return _parse_mtx(path)
    if fmt == "csv":
        return _parse_generic_csv(path, delimiter=",")
    if fmt == "tsv":
        return _parse_generic_csv(path, delimiter="\t")
    raise ValueError(f"Cannot detect format for {filename}. Use --format mtx|csv|tsv")


def _infer_format(suffix: str) -> str:
    mapping = {
        ".mtx": "mtx",
        ".mm": "mtx",
        ".csv": "csv",
        ".tsv": "tsv",
        ".tab": "tsv",
    }
    return mapping.get(suffix, "")


def _guess_dataset_name(dataset: DatasetInfo | None, url: str | None) -> str:
    if dataset:
        return dataset.name
    if url:
        return url.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
    return "dataset"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_menobis_edges(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    output_dir: Path,
    dataset_name: str,
) -> Path:
    """Write a MENoBiS-compatible CSV edge table.

    Format::

        source,target,weight
        0,12,5
        0,34,2

    Returns the path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_name}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])
        for s, t, w in zip(sources, targets, weights, strict=True):
            writer.writerow([int(s), int(t), int(w)])
    return out_path


def write_summary(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    output_dir: Path,
    dataset_name: str,
    dataset_info: DatasetInfo | None = None,
) -> None:
    """Write a YAML summary file next to the edge table."""
    n_nodes = int(max(sources.max(), targets.max())) + 1
    n_edges = len(sources)
    total_weight = int(weights.sum())
    max_weight = int(weights.max())
    mean_weight = float(weights.mean())
    has_self_loops = bool((sources == targets).sum() > 0)
    lines = [
        f"dataset: {dataset_name}",
        f"nodes: {n_nodes}",
        f"edges: {n_edges}",
        f"total_weight: {total_weight}",
        f"max_weight: {max_weight}",
        f"mean_weight: {mean_weight:.2f}",
        "directed: true",
        f"weighted: {bool((weights > 1).any())}",
        f"self_loops: {has_self_loops}",
    ]
    if dataset_info:
        urls = list(dataset_info.urls.values())
        lines.append(f"source: {urls[0]}" if len(urls) == 1 else f"sources: {urls}")
        lines.append(f"description: {dataset_info.description}")

    summary_path = output_dir / f"{dataset_name}.yml"
    summary_path.write_text("\n".join(lines) + "\n")


def ensure_gitignore(output_dir: Path) -> None:
    """Ensure the parent repo's .gitignore excludes the output directory."""
    resolved = output_dir.resolve()
    for parent in [resolved, *list(resolved.parents)]:
        gitignore = parent / ".gitignore"
        if gitignore.exists():
            # Compute relative path from repo root to output dir
            try:
                rel = resolved.relative_to(parent)
            except ValueError:
                continue
            entry = str(rel)
            if entry not in gitignore.read_text():
                with gitignore.open("a") as f:
                    f.write(f"\n{entry}/\n")
                print(f"  Added '{entry}/' to {gitignore}", file=sys.stderr)
            return
    print(
        "  Warning: no .gitignore found above output directory. "
        "Consider excluding this directory from version control.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = Typer(
    no_args_is_help=True,
    help="Fetch real-world OD flow networks for MENoBiS.",
)


@app.command(name="list")
def list_datasets() -> None:
    """List available built-in datasets."""
    print(f"{'Name':<20} {'Nodes':>8} {'Edges':>8}  Description")
    print("-" * 80)
    for ds in BUILTIN_DATASETS:
        nc = str(ds.node_count_hint or "?")
        ec = str(ds.edge_count_hint or "?")
        print(f"{ds.name:<20} {nc:>8} {ec:>8}  {ds.description}")


@app.command()
def download(
    dataset: Annotated[
        str,
        typer.Argument(help="Dataset name (see ``list``) or 'url' for custom URL."),
    ],
    url: Annotated[
        str | None,
        typer.Argument(
            help="Custom URL (required when dataset='url').",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        Option(
            "--output-dir",
            "-o",
            help="Output directory.  Defaults to ``data/`` at the repo root.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    cache_dir: Annotated[
        Path | None,
        Option(
            "--cache-dir",
            "-c",
            help="Cache directory for raw downloads.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    format: Annotated[
        str | None,
        Option(
            "--format",
            "-f",
            help="Force format (mtx, csv, tsv) for custom URLs.",
        ),
    ] = None,
    force: Annotated[
        bool,
        Option("--force", help="Re-download even if cached."),
    ] = False,
) -> None:
    """Download and prepare a dataset for MENoBiS.

    Use ``list`` to see available built-in datasets.  Use ``url <URL>``
    to download from a custom URL (Matrix Market, CSV, or TSV).
    """
    effective_cache = cache_dir or CACHE_DIR_DEFAULT
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data"

    if dataset == "url":
        if not url:
            print("Error: --url <URL> is required when dataset='url'.", file=sys.stderr)
            raise typer.Exit(code=2)
        print(f"Preparing custom dataset from: {url}", file=sys.stderr)
        sources, targets, weights = prepare_from_url(
            url, effective_cache, format_hint=format, force=force
        )
        ds_info = None
        ds_name = _guess_dataset_name(None, url)
    elif dataset in _DATASET_MAP:
        info = _DATASET_MAP[dataset]
        print(f"Preparing dataset: {dataset}", file=sys.stderr)
        sources, targets, weights = prepare_dataset(info, effective_cache, force=force)
        ds_info = info
        ds_name = dataset
    else:
        available = ", ".join(_DATASET_MAP.keys())
        print(f"Unknown dataset: {dataset!r}. Available: {available}", file=sys.stderr)
        raise typer.Exit(code=2)

    # Write MENoBiS edge table
    out_path = write_menobis_edges(sources, targets, weights, output_dir, ds_name)
    print(f"  Wrote {len(sources)} edges to: {out_path}", file=sys.stderr)

    # Write summary
    write_summary(sources, targets, weights, output_dir, ds_name, dataset_info=ds_info)

    # Ensure gitignore
    ensure_gitignore(output_dir)

    n_nodes = int(max(sources.max(), targets.max())) + 1
    n_edges = len(sources)
    total_weight = int(weights.sum())
    max_weight = int(weights.max())
    print("\nSummary:", file=sys.stderr)
    print(f"  Nodes:        {n_nodes}", file=sys.stderr)
    print(f"  Edges:        {n_edges}", file=sys.stderr)
    print(f"  Total weight: {total_weight}", file=sys.stderr)
    print(f"  Max weight:   {max_weight}", file=sys.stderr)
    print("  Format:       MENoBiS CSV (source, target, weight)", file=sys.stderr)
    print("\nUsage:", file=sys.stderr)
    print(f"  menobis analyze strengths {out_path}", file=sys.stderr)
    print(f"  menobis fit strength-poisson {out_path}", file=sys.stderr)
    print(f"  menobis generate strength-poisson {out_path} --seed 42", file=sys.stderr)
    print(f"  menobis filter strength-poisson {out_path} --alpha 0.05", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
