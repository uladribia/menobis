"""Import tests for the ODME Python package."""

import odme


def test_package_exposes_version() -> None:
    """The package exposes a non-empty version string."""
    assert isinstance(odme.__version__, str)
    assert odme.__version__
