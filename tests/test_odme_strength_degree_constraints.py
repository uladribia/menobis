"""Tests for strength-degree model constraints."""

import numpy as np
import pytest

from odme.models import validate_strength_degree_constraints


def test_validate_strength_degree_constraints_accepts_valid_sequences() -> None:
    validate_strength_degree_constraints(
        np.array([2.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
    )


def test_validate_strength_degree_constraints_rejects_strength_below_degree() -> None:
    with pytest.raises(ValueError, match=r"strength.*degree"):
        validate_strength_degree_constraints(
            np.array([0.5, 1.0]),
            np.array([1.0, 0.5]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )
