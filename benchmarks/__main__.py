"""ODME benchmark CLI.

Usage:
    uv run python -m benchmarks fit --max-n 1000 --tolerance 1e-4 --plot
    uv run python -m benchmarks sample --max-n 10000 --plot
    uv run python -m benchmarks e2e --max-n 100 --ensemble 300 --plot
    uv run python -m benchmarks filter --max-n 100 --ensemble 100 --plot
"""

import typer

app = typer.Typer(help="ODME benchmark suite.", no_args_is_help=True)


@app.command()
def fit(
    max_n: int = typer.Option(1000, help="Maximum network size"),
    nodes: str | None = typer.Option(
        None, help="Comma-separated exact sizes, e.g. 50,100"
    ),
    tolerance: float = typer.Option(1e-4, help="Convergence tolerance"),
    known_fractions: str = typer.Option(
        "0.05,0.40", help="Comma-separated known-weight fractions for partial cases"
    ),
    verbose: int = typer.Option(0, help="Verbosity (0=quiet, 2=convergence)"),
    plot: bool = typer.Option(False, help="Generate scaling/residual plots"),
    output: str = typer.Option("benchmarks/results", help="Output directory"),
):
    """Benchmark all fitting solvers across ME/W/B families."""
    from benchmarks.bench_fitting import (
        bench_all,
        bench_partial,
        parse_floats,
        parse_nodes,
        plot_results,
        save_results,
    )

    requested_nodes = parse_nodes(nodes)
    results = bench_all(
        max_n=max_n, tolerance=tolerance, verbose=verbose, nodes=requested_nodes
    )
    partial = bench_partial(
        max_n=max_n,
        tolerance=tolerance,
        verbose=verbose,
        nodes=requested_nodes,
        known_fractions=parse_floats(known_fractions),
    )
    results.extend(partial)
    save_results(results, output)
    if plot:
        plot_results(results, output)


@app.command()
def sample(
    max_n: int = typer.Option(10000, help="Maximum network size"),
    nodes: str | None = typer.Option(
        None, help="Comma-separated exact sizes, e.g. 50,100"
    ),
    repeats: int = typer.Option(3, help="Repeats per case"),
    plot: bool = typer.Option(False, help="Generate plots"),
):
    """Benchmark streaming generation at scale."""
    from benchmarks.bench_fitting import parse_nodes
    from benchmarks.bench_generation import bench_generation, plot_generation

    results = bench_generation(max_n=max_n, repeats=repeats, nodes=parse_nodes(nodes))
    if plot:
        plot_generation(results)


@app.command()
def e2e(
    max_n: int = typer.Option(100, help="Maximum network size"),
    nodes: str | None = typer.Option(
        None, help="Comma-separated exact sizes, e.g. 50,100"
    ),
    ensemble: int = typer.Option(300, help="Ensemble size for z-score checks"),
    tolerance: float = typer.Option(1e-4, help="Fitting tolerance"),
    plot: bool = typer.Option(False, help="Generate plots"),
):
    """End-to-end: fit → sample null → compare to input constraints."""
    from benchmarks.bench_e2e import bench_e2e, plot_e2e
    from benchmarks.bench_fitting import parse_nodes

    results = bench_e2e(
        max_n=max_n, ensemble=ensemble, tolerance=tolerance, nodes=parse_nodes(nodes)
    )
    if plot:
        plot_e2e(results)


@app.command()
def filter(
    max_n: int = typer.Option(100, help="Maximum network size"),
    nodes: str | None = typer.Option(
        None, help="Comma-separated exact sizes, e.g. 50,100"
    ),
    ensemble: int = typer.Option(100, help="Samples per alpha level"),
    plot: bool = typer.Option(False, help="Generate plots"),
):
    """Filter benchmark: fit → sample null → filter → verify FPR."""
    from benchmarks.bench_filter import bench_filter, plot_filter
    from benchmarks.bench_fitting import parse_nodes

    results = bench_filter(max_n=max_n, ensemble=ensemble, nodes=parse_nodes(nodes))
    if plot:
        plot_filter(results)


if __name__ == "__main__":
    app()
