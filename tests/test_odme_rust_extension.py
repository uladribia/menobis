"""Tests for Rust extension."""

from odme import _odme


def test_rust_core_version() -> None:
    assert _odme.rust_core_version() == "0.1.0"
