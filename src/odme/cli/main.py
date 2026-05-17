"""Main Typer application for ODME."""

import sys

import typer
from typer import Option, Typer

from odme import __version__
from odme.cli.analyze import app as analyze_app
from odme.cli.filter import app as filter_app
from odme.cli.fit import app as fit_app
from odme.cli.generate import app as generate_app

app = Typer(no_args_is_help=True)
app.add_typer(analyze_app, name="analyze", help="Analyze ODME edge tables.")
app.add_typer(fit_app, name="fit", help="Fit model parameters.")
app.add_typer(generate_app, name="generate", help="Generate network samples.")
app.add_typer(filter_app, name="filter", help="Filter edges against null models.")


@app.callback(invoke_without_command=True)
def main_callback(
    version: bool = Option(False, "--version", help="Show the package version."),
) -> None:
    """ODME - Origin-Destination Multi-Edge network models."""
    if version:
        typer.echo(f"ODME, version {__version__} on {sys.platform.capitalize()}")
        raise typer.Exit


typer_click_object = typer.main.get_command(app)
"""Main Click object derived from the Typer app."""
