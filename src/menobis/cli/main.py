"""Main Typer application for MENoBiS."""

import sys

import typer
from typer import Option, Typer

from menobis import __version__
from menobis.cli.analyze import app as analyze_app
from menobis.cli.convert import convert as convert_command
from menobis.cli.filter import app as filter_app
from menobis.cli.fit import app as fit_app
from menobis.cli.generate import app as generate_app

app = Typer(no_args_is_help=True)
app.add_typer(analyze_app, name="analyze", help="Analyze MENoBiS edge tables.")
app.add_typer(fit_app, name="fit", help="Fit model parameters.")
app.add_typer(generate_app, name="generate", help="Generate network samples.")
app.add_typer(filter_app, name="filter", help="Filter edges against null models.")
app.command(name="convert")(convert_command)


@app.callback(invoke_without_command=True)
def main_callback(
    version: bool = Option(False, "--version", help="Show the package version."),
) -> None:
    """MENoBiS - Max Entropy NOn Binary Suite for null modeling."""
    if version:
        typer.echo(f"MENoBiS, version {__version__} on {sys.platform.capitalize()}")
        raise typer.Exit


typer_click_object = typer.main.get_command(app)
"""Main Click object derived from the Typer app."""
