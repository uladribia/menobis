"""Tests for ME model generation CLI commands."""

import json
from pathlib import Path

import numpy as np
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
    assert '"occ_num"' in result.output


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
    assert '"occ_num"' in result.output


def test_generate_strength_edges_me_json(tmp_path: Path) -> None:
    input_path = tmp_path / "edges.csv"
    _edges(input_path)
    result = runner.invoke(
        app, ["generate", "strength-edges-poisson", str(input_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert '"occ_num"' in result.output


def test_generate_strength_degree_mcmc_json(tmp_path: Path) -> None:
    """Microcanonical fixed-(s,k) CLI route: exact s and k from the input."""
    input_path = tmp_path / "edges.csv"
    _edges(input_path)
    result = runner.invoke(
        app,
        ["generate", "strength-degree-mcmc", str(input_path), "--json"],
    )
    assert result.exit_code == 0, result.output
    assert '"occ_num"' in result.output
    # The output must reproduce the input strengths and degrees exactly.
    data = json.loads(result.output)
    src = np.asarray([row["source"] for row in data], dtype=np.uint64)
    tgt = np.asarray([row["target"] for row in data], dtype=np.uint64)
    occ = np.asarray([row["occ_num"] for row in data], dtype=np.uint64)
    s_out = np.zeros(3, dtype=np.uint64)
    s_in = np.zeros(3, dtype=np.uint64)
    k_out = np.zeros(3, dtype=np.uint32)
    k_in = np.zeros(3, dtype=np.uint32)
    for s, t, w in zip(src, tgt, occ, strict=True):
        s_out[int(s)] += int(w)
        s_in[int(t)] += int(w)
        k_out[int(s)] += 1
        k_in[int(t)] += 1
    np.testing.assert_array_equal(s_out, np.array([40, 20, 0], dtype=np.uint64))
    np.testing.assert_array_equal(s_in, np.array([0, 10, 50], dtype=np.uint64))
    np.testing.assert_array_equal(k_out, np.array([2, 1, 0], dtype=np.uint32))
    np.testing.assert_array_equal(k_in, np.array([0, 1, 2], dtype=np.uint32))


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
    assert '"occ_num"' in result.output
