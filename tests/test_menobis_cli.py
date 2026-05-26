"""Tests for the MENoBiS command line interface."""

import sys

from typer.testing import CliRunner

from menobis import __version__
from menobis.cli.main import app

runner = CliRunner()


def test_version() -> None:
    """The CLI exposes the package version and current platform."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "MENoBiS" in result.output
    assert __version__ in result.output
    assert sys.platform.capitalize() in result.output
