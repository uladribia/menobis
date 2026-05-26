"""Import tests for the MENoBiS Python package."""

import menobis


def test_package_exposes_version() -> None:
    """The package exposes a non-empty version string."""
    assert isinstance(menobis.__version__, str)
    assert menobis.__version__
