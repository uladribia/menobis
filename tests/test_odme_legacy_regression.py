"""Regression checks against archived thesis-era behavior."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from odme.analysis import compute_all_stats, weight_distribution
from odme.data import normalize_edges

LEGACY_PARENT = "HEAD^"


def _legacy_fixture_edges():
    return normalize_edges(
        np.array([0, 0, 1, 2], dtype=np.uint64),
        np.array([1, 2, 2, 0], dtype=np.uint64),
        np.array([2, 3, 4, 5], dtype=np.uint64),
    )


def test_modern_stats_match_archived_legacy_analyzer_fixture() -> None:
    """Compare modern Rust stats with a hand-audited legacy analyzer fixture."""
    stats = compute_all_stats(_legacy_fixture_edges())

    np.testing.assert_array_equal(stats.strength_out, np.array([5, 4, 5]))
    np.testing.assert_array_equal(stats.strength_in, np.array([5, 2, 7]))
    np.testing.assert_array_equal(stats.degree_out, np.array([2, 1, 1]))
    np.testing.assert_array_equal(stats.degree_in, np.array([1, 1, 2]))
    np.testing.assert_allclose(stats.y2_out, np.array([13 / 25, 1, 1]))
    np.testing.assert_allclose(stats.y2_in, np.array([1, 1, 25 / 49]))
    np.testing.assert_allclose(stats.k_nn_out, np.array([1.5, 2.0, 1.0]))
    np.testing.assert_allclose(stats.k_nn_in, np.array([1.0, 2.0, 1.5]))
    np.testing.assert_allclose(stats.s_nn_out, np.array([5.0, 7.0, 5.0]))
    np.testing.assert_allclose(stats.s_nn_in, np.array([5.0, 5.0, 31 / 7]))

    weights = weight_distribution(_legacy_fixture_edges())
    np.testing.assert_array_equal(weights.weight, np.array([2, 3, 4, 5]))
    np.testing.assert_array_equal(weights.count, np.ones(4, dtype=np.uint64))


@pytest.mark.skipif(
    os.environ.get("ODME_RUN_LEGACY_COMPARISON") != "1",
    reason="set ODME_RUN_LEGACY_COMPARISON=1 to compile and run archived C code",
)
def test_optional_live_legacy_analyzer_comparison(tmp_path: Path) -> None:
    """Run archived C analyzer from git history and compare key node outputs."""
    if shutil.which("gcc") is None or shutil.which("make") is None:
        pytest.skip("gcc and make are required for live legacy comparison")

    archive = tmp_path / "legacy.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), LEGACY_PARENT],
        check=True,
    )
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    subprocess.run(["tar", "xf", str(archive), "-C", str(legacy_root)], check=True)

    analyzer_dir = legacy_root / "1. Network analysis"
    if not analyzer_dir.exists():
        pytest.skip("legacy analyzer is unavailable in git parent")

    make = subprocess.run(
        ["make", "CC=gcc"],
        cwd=analyzer_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if make.returncode != 0:
        pytest.skip(f"legacy analyzer did not compile: {make.stderr[-500:]}")

    edge_file = tmp_path / "edges.tr"
    edge_file.write_text("0 1 2\n0 2 3\n1 2 4\n2 0 5\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    subprocess.run(
        [
            str(analyzer_dir / "MultiEdgeAnalyzer"),
            "-N",
            "3",
            "-d",
            "1",
            "-f",
            str(edge_file),
            "-h",
            "0",
            "-l",
            "1",
        ],
        cwd=run_dir,
        text=True,
        capture_output=True,
        check=True,
    )

    node_files = list(run_dir.glob("*node_list.list"))
    assert node_files, "legacy analyzer did not produce node_list.list"
    legacy_nodes = np.loadtxt(node_files[0], comments="#")
    stats = compute_all_stats(_legacy_fixture_edges())

    np.testing.assert_array_equal(legacy_nodes[:, 0].astype(np.uint64), stats.node)
    np.testing.assert_array_equal(
        legacy_nodes[:, 1].astype(np.uint64), stats.degree_out
    )
    np.testing.assert_array_equal(
        legacy_nodes[:, 3].astype(np.uint64), stats.strength_out
    )
    np.testing.assert_allclose(legacy_nodes[:, 4], stats.y2_out, rtol=1e-6)
    np.testing.assert_array_equal(legacy_nodes[:, 8].astype(np.uint64), stats.degree_in)
    np.testing.assert_array_equal(
        legacy_nodes[:, 10].astype(np.uint64), stats.strength_in
    )
    np.testing.assert_allclose(legacy_nodes[:, 11], stats.y2_in, rtol=1e-6)
