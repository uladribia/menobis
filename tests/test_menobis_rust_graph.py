"""Tests for Rust-backed graph kernels."""

from menobis import _menobis


def test_directed_strengths_ignores_zero_weights() -> None:
    s_out, s_in = _menobis.directed_strengths(3, [0, 1], [1, 2], [0, 4])
    assert s_out == [0, 4, 0]
    assert s_in == [0, 0, 4]


def test_directed_strengths() -> None:
    s_out, s_in = _menobis.directed_strengths(3, [0, 1], [1, 2], [3, 4])
    assert s_out == [3, 4, 0]
    assert s_in == [0, 3, 4]
