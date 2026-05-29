#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.19",
#     "numpy>=2.0",
# ]
# ///
r"""Evaluate MENoBiS fitting/sampling/filtering on real-world OD data.

Loads a real dataset prepared by ``fetch_data.py``, derives constraints,
and runs the full MENoBiS pipeline (fit -> sample -> filter) across
model families and constraint types.

Usage::

    # First, download a dataset
    python scripts/fetch_data.py download openflights

    # Run basic evaluation (strength constraint, all families)
    python scripts/evaluate_real_data.py

    # Specific dataset and constraints
    python scripts/evaluate_real_data.py openflights \\
        --families me,b --constraints strength,strength-edges

    # Include sampling and filter FPR estimation
    python scripts/evaluate_real_data.py openflights \\
        --sample --filter-samples 3

    # With cost constraint (requires Mercator coordinates)
    python scripts/evaluate_real_data.py openflights \\
        --constraints strength-cost

    # Use the smaller email-eu dataset for quick testing
    python scripts/evaluate_real_data.py email-eu \\
        --families me,b --constraints strength

    # Output JSON
    python scripts/evaluate_real_data.py openflights \\
        --output results/openflights-eval.json --json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from typer import Option

from menobis.analysis.summary import directed_degrees, directed_strengths
from menobis.data.io import read_edges
from menobis.models.spec import Constraint, ModelFamily
from menobis.routing import filter_model, fit_model, sample_model

app = typer.Typer(help="Evaluate MENoBiS on real-world OD data.")

FAMILIES: tuple[str, ...] = ("me", "b", "w")
CONSTRAINTS: tuple[str, ...] = (
    "strength",
    "strength-edges",
    "strength-degree",
    "strength-cost",
)

# When N > this threshold, skip expensive per-pair precision computation
PRECISION_MAX_N = 300


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    """One evaluation measurement row."""

    dataset: str
    family: str
    constraint: str
    nodes: int
    edges: int
    total_weight: int
    fit_wall_seconds: float = 0.0
    fit_converged: bool | None = None
    fit_iterations: int | None = None
    fit_max_s_out_err: float | None = None
    fit_max_s_in_err: float | None = None
    fit_rel_s_err: float | None = None
    fit_max_k_out_err: float | None = None
    fit_max_k_in_err: float | None = None
    fit_edge_count_err: float | None = None
    fit_status: str = "ok"
    fit_message: str = ""
    sample_wall_seconds: float | None = None
    sample_edges: int | None = None
    sample_status: str | None = None
    filter_fpr: float | None = None
    filter_status: str | None = None

    @property
    def fit_precision(self) -> str:
        """Return a short precision summary string."""
        parts = []
        if self.fit_max_s_out_err is not None:
            parts.append(f"s={self.fit_max_s_out_err:.2e}")
        if self.fit_max_k_out_err is not None:
            parts.append(f"k={self.fit_max_k_out_err:.2e}")
        if self.fit_edge_count_err is not None:
            parts.append(f"E={self.fit_edge_count_err:.2e}")
        return ", ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _find_dataset(name_or_path: str) -> Path:
    """Resolve a dataset name to a path under the repo ``data/`` directory."""
    p = Path(name_or_path)
    if p.suffix in (".csv", ".tsv", ".parquet", ".mtx", ".net", ".paj"):
        return p
    script_dir = Path(__file__).resolve().parent
    for candidate in [
        script_dir.parent / "data" / f"{name_or_path}.csv",
    ]:
        if candidate.exists():
            return candidate
    return p


def _find_coords(name_or_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load Mercator coordinates from ``{name}_coords.npz`` if present."""
    p = Path(name_or_path)
    ds_name = p.stem
    coords_path = p.parent / f"{ds_name}_coords.npz"
    if not coords_path.exists():
        return None, None
    try:
        d = np.load(coords_path)
        return d["x"].astype(np.float64), d["y"].astype(np.float64)
    except (OSError, KeyError):
        return None, None


# ---------------------------------------------------------------------------
# Constraint derivation
# ---------------------------------------------------------------------------


def derive_constraints(
    edge_path: Path,
) -> dict[str, Any]:
    """Derive constraints from a real dataset edge table.

    Computes strength sequences, degree sequences, total edges/events,
    and binomial layers estimate.
    """
    edges = read_edges(edge_path)
    n_nodes = int(max(edges.source.max(), edges.target.max())) + 1

    s = directed_strengths(edges)
    k = directed_degrees(edges)

    s_out = s.out.astype(np.float64)
    s_in = s.incoming.astype(np.float64)
    k_out = k.out.astype(np.float64)
    k_in = k.incoming.astype(np.float64)

    total_events = edges.total_events
    total_edges = float(edges.num_edges)

    # Estimate binomial layers
    pairs_per_node = n_nodes
    max_strength = float(max(s_out.max(), s_in.max()))
    binomial_layers = max(10, 4 * int(np.ceil(max_strength / max(pairs_per_node, 1))))

    return {
        "edges": edges,
        "n_nodes": n_nodes,
        "strength_out": s_out,
        "strength_in": s_in,
        "degree_out": k_out,
        "degree_in": k_in,
        "total_edges": total_edges,
        "total_events": total_events,
        "binomial_layers": binomial_layers,
    }


# ---------------------------------------------------------------------------
# Precision computation (vectorized)
# ---------------------------------------------------------------------------


def _compute_precision_vectorized(
    fit_result: Any,  # noqa: ANN401  # any fit result type
    c: dict[str, Any],
    constraint: str,
    *,
    coord_x: np.ndarray | None = None,
    coord_y: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute constraint recovery precision from fitted multipliers.

    Uses vectorized operations over all N x N node pairs.
    Only called when N <= PRECISION_MAX_N.
    """
    result: dict[str, float | None] = {
        "max_s_out_err": None,
        "max_s_in_err": None,
        "rel_s_err": None,
        "max_k_out_err": None,
        "max_k_in_err": None,
        "edge_count_err": None,
    }

    if not hasattr(fit_result, "x") or fit_result.x is None:
        return result

    x = np.asarray(fit_result.x, dtype=np.float64)
    y = np.asarray(fit_result.y, dtype=np.float64)
    n = len(x)
    layers = getattr(fit_result, "layers", None) or 1
    family = getattr(fit_result, "family", "poisson")
    self_loops = getattr(fit_result, "self_loops", True)
    lam = getattr(fit_result, "lam", None)
    gamma = getattr(fit_result, "gamma", None)

    z_val = (
        np.asarray(fit_result.z, dtype=np.float64)
        if hasattr(fit_result, "z") and fit_result.z is not None
        else None
    )
    w_val = (
        np.asarray(fit_result.w, dtype=np.float64)
        if hasattr(fit_result, "w") and fit_result.w is not None
        else None
    )

    # Build mesh of x_i * y_j
    q_mat = np.outer(x, y)  # (N, N)
    if constraint == "strength-cost" and gamma is not None and coord_x is not None:
        # Build distance matrix
        dx = coord_x[:, np.newaxis] - coord_x[np.newaxis, :]
        dy = coord_y[:, np.newaxis] - coord_y[np.newaxis, :]
        d_mat = np.hypot(dx, dy)
        q_mat *= np.exp(-float(gamma) * d_mat)

    if not self_loops:
        np.fill_diagonal(q_mat, 0.0)

    mask = q_mat > 0.0
    q_pos = q_mat[mask]

    if constraint == "strength" or constraint == "strength-cost":
        if "binomial" in family:
            m = float(layers)
            w_ij = m * q_pos / (1.0 + q_pos)
            occ = np.ones_like(w_ij)
        elif "geometric" in family:
            valid = q_pos < 1.0
            m = float(layers)
            w_ij = np.zeros_like(q_pos)
            w_ij[valid] = m * q_pos[valid] / (1.0 - q_pos[valid])
            occ = np.ones_like(w_ij)
        else:
            w_ij = q_pos
            occ = np.ones_like(w_ij)

        # Only compute strength errors
        exp_s_out = np.zeros(n, dtype=np.float64)
        exp_s_in = np.zeros(n, dtype=np.float64)
        src_idx, tgt_idx = np.where(mask)
        np.add.at(exp_s_out, src_idx, w_ij)
        np.add.at(exp_s_in, tgt_idx, w_ij)

        s_out_true = c["strength_out"]
        s_in_true = c["strength_in"]
        result["rel_s_err"] = float(
            np.abs(exp_s_out.sum() - s_out_true.sum()) / max(s_out_true.sum(), 1.0)
        )
        result["max_s_out_err"] = float(np.max(np.abs(exp_s_out - s_out_true)))
        result["max_s_in_err"] = float(np.max(np.abs(exp_s_in - s_in_true)))
        return result

    # ZI constraints (strength-edges, strength-degree)
    is_b = "binomial" in family
    is_w = "geometric" in family

    if constraint == "strength-edges":
        if lam is None:
            return result
        l_val = float(lam)
        if is_b:
            g_pos = (1.0 + q_pos) ** layers - 1.0
            den = 1.0 + l_val * g_pos
            occ = l_val * g_pos / den
            w_ij = l_val * float(layers) * q_pos * (1.0 + q_pos) ** (layers - 1) / den
        elif is_w:
            valid = q_pos < 1.0
            g_pos = np.zeros_like(q_pos)
            g_pos[valid] = (1.0 - q_pos[valid]) ** (-layers) - 1.0
            den = 1.0 + l_val * g_pos
            occ = l_val * g_pos / den
            w_ij = l_val * float(layers) * q_pos * (1.0 - q_pos) ** (-layers - 1) / den
        else:
            # ME
            exp_q = np.exp(q_pos)
            g_pos = exp_q - 1.0
            den = 1.0 + l_val * g_pos
            occ = l_val * g_pos / den
            w_ij = l_val * q_pos * exp_q / den

        valid_mask = ~(np.isnan(occ) | np.isinf(occ))
    elif constraint == "strength-degree":
        if z_val is None or w_val is None:
            return result
        v_mat = np.outer(z_val, w_val)  # (N, N)
        v_pos = v_mat[mask]
        if is_b:
            g_pos = (1.0 + q_pos) ** layers - 1.0
            den = 1.0 + v_pos * g_pos
            occ = v_pos * g_pos / den
            w_ij = v_pos * float(layers) * q_pos * (1.0 + q_pos) ** (layers - 1) / den
        elif is_w:
            valid = q_pos < 1.0
            g_pos = np.zeros_like(q_pos)
            g_pos[valid] = (1.0 - q_pos[valid]) ** (-layers) - 1.0
            den = 1.0 + v_pos * g_pos
            occ = v_pos * g_pos / den
            w_ij = v_pos * float(layers) * q_pos * (1.0 - q_pos) ** (-layers - 1) / den
        else:
            exp_q = np.exp(q_pos)
            g_pos = exp_q - 1.0
            den = 1.0 + v_pos * g_pos
            occ = v_pos * g_pos / den
            w_ij = v_pos * q_pos * exp_q / den

        valid_mask = ~(np.isnan(occ) | np.isinf(occ))
    else:
        return result

    occ = occ[valid_mask]
    w_ij = w_ij[valid_mask]
    src_idx, tgt_idx = np.where(mask)
    src_idx = src_idx[valid_mask]
    tgt_idx = tgt_idx[valid_mask]

    exp_s_out = np.zeros(n, dtype=np.float64)
    exp_s_in = np.zeros(n, dtype=np.float64)
    exp_k_out = np.zeros(n, dtype=np.float64)
    exp_k_in = np.zeros(n, dtype=np.float64)
    np.add.at(exp_s_out, src_idx, w_ij)
    np.add.at(exp_s_in, tgt_idx, w_ij)
    np.add.at(exp_k_out, src_idx, occ)
    np.add.at(exp_k_in, tgt_idx, occ)

    s_out_true = c["strength_out"]
    s_in_true = c["strength_in"]
    k_out_true = c["degree_out"]
    k_in_true = c["degree_in"]

    result["rel_s_err"] = float(
        np.abs(exp_s_out.sum() - s_out_true.sum()) / max(s_out_true.sum(), 1.0)
    )
    result["max_s_out_err"] = float(np.max(np.abs(exp_s_out - s_out_true)))
    result["max_s_in_err"] = float(np.max(np.abs(exp_s_in - s_in_true)))
    result["max_k_out_err"] = float(np.max(np.abs(exp_k_out - k_out_true)))
    result["max_k_in_err"] = float(np.max(np.abs(exp_k_in - k_in_true)))
    if constraint == "strength-edges":
        result["edge_count_err"] = float(np.abs(occ.sum() - c["total_edges"]))
    return result


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_evaluation(
    edge_path: Path,
    *,
    families: tuple[str, ...],
    constraints: tuple[str, ...],
    coord_x: np.ndarray | None = None,
    coord_y: np.ndarray | None = None,
    sample: bool,
    filter_samples: int,
    alpha: float,
    tolerance: float,
    max_iterations: int,
    verbose: bool,
) -> list[EvalRow]:
    """Run the evaluation matrix on a real dataset."""
    if verbose:
        print(f"Loading dataset: {edge_path}", file=sys.stderr)

    c = derive_constraints(edge_path)
    n_nodes = c["n_nodes"]
    n_edges = c["edges"].num_edges
    total_weight = c["edges"].total_events

    ds_name = edge_path.stem

    if verbose:
        print(
            f"  Nodes: {n_nodes}, Edges: {n_edges}, Total weight: {total_weight}",
            file=sys.stderr,
        )
        print(
            f"  Str out: [{c['strength_out'].min():.0f} .. "
            f"{c['strength_out'].max():.0f}], "
            f"Str in: [{c['strength_in'].min():.0f} .. "
            f"{c['strength_in'].max():.0f}]",
            file=sys.stderr,
        )
        has_coords = coord_x is not None
        print(f"  Coordinates: {'Yes' if has_coords else 'No'}", file=sys.stderr)
        print(file=sys.stderr)

    family_map = {"me": ModelFamily.ME, "b": ModelFamily.B, "w": ModelFamily.W}
    constraint_map = {
        "strength": Constraint.STRENGTH,
        "strength-edges": Constraint.STRENGTH_EDGES,
        "strength-degree": Constraint.STRENGTH_DEGREE,
        "strength-cost": Constraint.STRENGTH_COST,
    }

    rows: list[EvalRow] = []

    for family in families:
        for constraint in constraints:
            row = EvalRow(
                dataset=ds_name,
                family=family,
                constraint=constraint,
                nodes=n_nodes,
                edges=n_edges,
                total_weight=total_weight,
            )

            fit_kwargs: dict[str, Any] = {
                "family": family_map[family],
                "constraint": constraint_map[constraint],
                "strength_out": c["strength_out"],
                "strength_in": c["strength_in"],
                "tolerance": tolerance,
                "max_iterations": max_iterations,
            }
            if constraint == "strength-edges":
                fit_kwargs["target_edges"] = c["total_edges"]
            if constraint == "strength-degree":
                fit_kwargs["degree_out"] = c["degree_out"]
                fit_kwargs["degree_in"] = c["degree_in"]
            if constraint == "strength-cost":
                if coord_x is None or coord_y is None:
                    if verbose:
                        print(
                            f"  [{family:<3} {constraint:<16}] SKIP (no coordinates)",
                            file=sys.stderr,
                        )
                    row.fit_status = "skip"
                    row.fit_message = "no coordinates available"
                    rows.append(row)
                    continue
                fit_kwargs["coord_x"] = coord_x
                fit_kwargs["coord_y"] = coord_y
                fit_kwargs["target_cost"] = c["total_events"]
            if family == "b":
                n_layers = c["binomial_layers"]
                fit_kwargs["layers"] = n_layers

            if verbose:
                print(
                    f"  [{family:<3} {constraint:<16}] Fitting...",
                    file=sys.stderr,
                    end=" " if constraint == "strength-cost" else "",
                )
                if constraint == "strength-cost":
                    print(
                        f"(cost={c['total_events']}, gamma init=auto)",
                        file=sys.stderr,
                    )
                else:
                    print(file=sys.stderr)

            try:
                wall_start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_result = fit_model(**fit_kwargs)
                fit_wall = time.perf_counter() - wall_start

                row.fit_wall_seconds = fit_wall
                row.fit_converged = getattr(fit_result, "converged", None)
                row.fit_iterations = getattr(fit_result, "iterations", None)

                # Precision (only for datasets small enough for vectorized computation)
                if n_nodes <= PRECISION_MAX_N:
                    precision = _compute_precision_vectorized(
                        fit_result,
                        c,
                        constraint,
                        coord_x=coord_x,
                        coord_y=coord_y,
                    )
                else:
                    precision = {
                        "max_s_out_err": None,
                        "max_s_in_err": None,
                        "rel_s_err": None,
                        "max_k_out_err": None,
                        "max_k_in_err": None,
                        "edge_count_err": None,
                    }

                row.fit_max_s_out_err = precision["max_s_out_err"]
                row.fit_max_s_in_err = precision["max_s_in_err"]
                row.fit_rel_s_err = precision["rel_s_err"]
                row.fit_max_k_out_err = precision["max_k_out_err"]
                row.fit_max_k_in_err = precision["max_k_in_err"]
                row.fit_edge_count_err = precision["edge_count_err"]

                conv = "✓" if row.fit_converged else "✗"
                if verbose:
                    precision_str = row.fit_precision
                    print(
                        f"    {fit_wall:.2f}s  {conv}  "
                        f"{row.fit_iterations or '?'} iters  "
                        f"{precision_str}",
                        file=sys.stderr,
                    )

            except Exception as exc:
                row.fit_status = "error"
                row.fit_message = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"    ERROR: {exc}", file=sys.stderr)
                rows.append(row)
                continue

            # Sample
            if sample and row.fit_status == "ok":
                sample_kwargs: dict[str, Any] = {
                    "family": family_map[family],
                    "constraint": constraint_map[constraint],
                    "fit": fit_result,
                    "seed": 42,
                }
                if constraint == "strength-cost":
                    sample_kwargs["coord_x"] = coord_x
                    sample_kwargs["coord_y"] = coord_y
                if family == "b":
                    sample_kwargs["layers"] = c["binomial_layers"]

                if verbose:
                    print("    Sampling...", file=sys.stderr, end=" ")

                try:
                    wall_start = time.perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sample_edge_table = sample_model(**sample_kwargs)
                    sample_wall = time.perf_counter() - wall_start
                    row.sample_wall_seconds = sample_wall
                    row.sample_edges = (
                        sample_edge_table.num_edges if sample_edge_table else 0
                    )
                    if verbose:
                        print(
                            f"{sample_wall:.2f}s  {row.sample_edges} edges",
                            file=sys.stderr,
                        )

                    # Filter FPR
                    if filter_samples > 0 and sample_edge_table is not None:
                        if verbose:
                            print(
                                f"    Filter FPR ({filter_samples} null samples)...",
                                file=sys.stderr,
                            )
                        fp_total = 0
                        edge_total = 0
                        for offset in range(filter_samples):
                            null_sample = sample_model(
                                **{**sample_kwargs, "seed": 10000 + offset}
                            )
                            if null_sample is None:
                                continue
                            filter_kwargs: dict[str, Any] = {
                                "family": family_map[family],
                                "constraint": constraint_map[constraint],
                                "fit": fit_result,
                                "alpha": alpha,
                                "tail": "upper",
                            }
                            if constraint == "strength-cost":
                                filter_kwargs["coord_x"] = coord_x
                                filter_kwargs["coord_y"] = coord_y
                            if family == "b":
                                filter_kwargs["layers"] = c["binomial_layers"]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                filter_result = filter_model(
                                    null_sample, **filter_kwargs
                                )
                            fp_total += filter_result.upper.edges.num_edges
                            edge_total += null_sample.num_edges

                        row.filter_fpr = fp_total / max(edge_total, 1)
                        if verbose:
                            print(f"    FPR = {row.filter_fpr:.4f}", file=sys.stderr)
                except Exception as exc:
                    if row.sample_wall_seconds is None:
                        row.sample_status = f"error: {type(exc).__name__}: {exc}"
                    else:
                        row.filter_status = f"error: {type(exc).__name__}: {exc}"
                    if verbose:
                        print(f"    ERROR: {exc}", file=sys.stderr)

            rows.append(row)
            if verbose:
                print(file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    dataset: Annotated[
        str,
        typer.Argument(
            help="Dataset name (see ``fetch_data.py list``) or path to edge CSV.",
        ),
    ] = "openflights",
    families: Annotated[
        str,
        Option(
            "--families",
            "-f",
            help="Comma-separated model families: me,b,w.",
        ),
    ] = "me,b,w",
    constraints: Annotated[
        str,
        Option(
            "--constraints",
            "-c",
            help="Comma-separated constraint types.",
        ),
    ] = "strength,strength-edges,strength-degree",
    sample: Annotated[
        bool,
        Option("--sample/--no-sample", help="Generate a sample from the fit."),
    ] = False,
    filter_samples: Annotated[
        int,
        Option(
            "--filter-samples",
            help="Number of null samples for FPR (0 to skip).",
        ),
    ] = 0,
    alpha: Annotated[
        float,
        Option("--alpha", help="Upper-tail alpha for filtering."),
    ] = 0.05,
    tolerance: Annotated[
        float,
        Option("--tolerance", "-t", help="Solver convergence tolerance."),
    ] = 1e-6,
    max_iterations: Annotated[
        int,
        Option("--max-iterations", help="Maximum solver iterations."),
    ] = 10000,
    output: Annotated[
        Path | None,
        Option("--output", "-o", help="Output JSON path."),
    ] = None,
    output_json: Annotated[
        bool,
        Option("--json", help="Print results as JSON to stdout."),
    ] = False,
    no_header: Annotated[
        bool,
        Option("--no-header", help="Suppress column header."),
    ] = False,
) -> None:
    """Evaluate MENoBiS fitting on a real-world dataset."""
    edge_path = _find_dataset(dataset)
    if not edge_path.exists():
        print(
            f"Error: dataset not found: {edge_path}\n"
            f"Run 'python scripts/fetch_data.py list' to see available datasets.\n"
            f"Then download: python scripts/fetch_data.py download {dataset}",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)

    coord_x, coord_y = _find_coords(str(edge_path))

    parsed_families = tuple(p.strip().lower() for p in families.split(",") if p.strip())
    parsed_constraints = tuple(
        p.strip().lower() for p in constraints.split(",") if p.strip()
    )

    _check_tokens(parsed_families, FAMILIES, "family")
    _check_tokens(parsed_constraints, CONSTRAINTS, "constraint")

    rows = run_evaluation(
        edge_path=edge_path,
        families=parsed_families,
        constraints=parsed_constraints,
        coord_x=coord_x,
        coord_y=coord_y,
        sample=sample,
        filter_samples=filter_samples,
        alpha=alpha,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=True,
    )

    if not no_header:
        _print_header()

    for row in rows:
        print(_format_row(row), file=sys.stderr)

    _print_summary(rows)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
        )

    if output_json:
        print(json.dumps([asdict(r) for r in rows], indent=2))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _format_row(row: EvalRow) -> str:
    conv = "✓" if row.fit_converged else ("✗" if row.fit_converged is False else "?")
    iters = str(row.fit_iterations) if row.fit_iterations else ""
    prec = row.fit_precision
    sample_info = ""
    if row.sample_wall_seconds is not None:
        sample_info = f" sample={row.sample_edges}e {row.sample_wall_seconds:.2f}s"
    fpr_info = ""
    if row.filter_fpr is not None:
        fpr_info = f" fpr={row.filter_fpr:.4f}"
    status = row.fit_status if row.fit_status != "ok" else ""
    return (
        f"{row.dataset:<16} {row.family:<3} {row.constraint:<16} "
        f"{row.nodes:>5} {row.edges:>5} {row.total_weight:>8} "
        f"{row.fit_wall_seconds:>8.3f} {conv:>2} {iters:>5} "
        f"{prec:<30} {status}{sample_info}{fpr_info}"
    )


def _print_header() -> None:
    header = (
        f"{'dataset':<16} {'fam':<3} {'constraint':<16} "
        f"{'N':>5} {'E':>5} {'W':>8} "
        f"{'wall(s)':>8} {'':2} {'iters':>5} precision"
    )
    print(header, file=sys.stderr)
    print("─" * 130, file=sys.stderr)


def _print_summary(rows: list[EvalRow]) -> None:
    n_converged = sum(1 for r in rows if r.fit_converged is True)
    n_err = sum(1 for r in rows if r.fit_status not in ("ok", "skip"))
    total_wall = sum(r.fit_wall_seconds for r in rows)
    print(
        f"\nDone: {len(rows)} rows, {n_converged} converged, "
        f"{n_err} errors, wall={total_wall:.1f}s",
        file=sys.stderr,
    )


def _check_tokens(
    values: tuple[str, ...],
    allowed: tuple[str, ...],
    name: str,
) -> None:
    bad = sorted(set(values) - set(allowed))
    if bad:
        print(
            f"Error: unknown {name}: {','.join(bad)}. Allowed: {','.join(allowed)}",
            file=sys.stderr,
        )
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
