"""Tests for Rust extension."""

from menobis import _menobis


def test_rust_core_version() -> None:
    assert _menobis.rust_core_version() == "1.0.0"
