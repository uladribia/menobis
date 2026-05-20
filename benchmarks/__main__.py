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
    tolerance: float = typer.Option(1e-4, help="Convergence tolerance"),
    verbose: int = typer.Option(0, help="Verbosity (0=quiet, 2=convergence)"),
    plot: bool = typer.Option(False, help="Generate scaling/residual plots"),
    output: str = typer.Option("benchmarks/results", help="Output directory"),
):
    """Benchmark all fitting solvers across ME/W/B families."""
    from benchmarks.bench_fitting import bench_all, bench_partial, plot_results, save_results

    results = bench_all(max_n=max_n, tolerance=tolerance, verbose=verbose)
    partial = bench_partial(max_n=max_n, tolerance=tolerance, verbose=verbose)
    results.extend(partial)
    save_results(results, output)
    if plot:
        plot_results(results, output)


@app.command()
def sample(
    max_n: int = typer.Option(10000, help="Maximum network size"),
    repeats: int = typer.Option(3, help="Repeats per case"),
    plot: bool = typer.Option(False, help="Generate plots"),
):
    """Benchmark streaming generation at scale."""
    from benchmarks.bench_generation import bench_generation, plot_generation

    results = bench_generation(max_n=max_n, repeats=repeats)
    if plot:
        plot_generation(results)


@app.command()
def e2e(
    max_n: int = typer.Option(100, help="Maximum network size"),
    ensemble: int = typer.Option(300, help="Ensemble size for z-score checks"),
    tolerance: float = typer.Option(1e-4, help="Fitting tolerance"),
    plot: bool = typer.Option(False, help="Generate plots"),
):
    """End-to-end: fit → sample null → compare to input constraints."""
    from benchmarks.bench_e2e import bench_e2e, plot_e2e

    results = bench_e2e(max_n=max_n, ensemble=ensemble, tolerance=tolerance)
    if plot:
        plot_e2e(results)


@app.command()
def filter(
    max_n: int = typer.Option(100, help="Maximum network size"),
    ensemble: int = typer.Option(100, help="Samples per alpha level"),
    plot: bool = typer.Option(False, help="Generate plots"),
):
    """Filter benchmark: fit → sample null → filter → verify FPR."""
    from benchmarks.bench_filter import bench_filter, plot_filter

    results = bench_filter(max_n=max_n, ensemble=ensemble)
    if plot:
        plot_filter(results)


if __name__ == "__main__":
    app()
