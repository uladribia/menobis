"""Tests for Rust-backed graph kernels."""

from odme import _odme


def test_directed_strengths_ignores_zero_weights() -> None:
    s_out, s_in = _odme.directed_strengths(3, [0, 1], [1, 2], [0, 4])
    assert s_out == [0, 4, 0]
    assert s_in == [0, 0, 4]


def test_directed_strengths() -> None:
    s_out, s_in = _odme.directed_strengths(3, [0, 1], [1, 2], [3, 4])
    assert s_out == [3, 4, 0]
    assert s_in == [0, 3, 4]
