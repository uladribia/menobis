"""Documentation contract test: the 72-cell benchmark matrix preset (§35).

Guards that the ``benchmarks matrix`` default preset still expands to
3 N values x 2 regimes x 3 families x 2 constraints x 2 self-loop
policies = 72 cells.
"""

from __future__ import annotations

import inspect

from benchmarks.cli import matrix_command

# (parameter name, default value, allowed tokens) — mirrors the CLI defaults.
_DIMENSIONS: dict[str, tuple[object, tuple[str, ...]]] = {
    "nodes": ("100,500,1000", ("100", "500", "1000")),
    "regimes": ("sparse,dense", ("sparse", "dense")),
    "families": ("me,b,w", ("me", "b", "w")),
    "constraints": ("strength,strength-cost", ("strength", "strength-cost")),
}


def _default_value(param_name: str) -> str:
    """Read the declared default value from the matrix command signature."""
    for name, param in inspect.signature(matrix_command).parameters.items():
        if name == param_name:
            default = param.default
            if not isinstance(default, str) or not default:
                msg = f"{param_name} default {default!r} is not a token list"
                raise AssertionError(msg)
            return default
    msg = f"parameter {param_name!r} not found in matrix_command"
    raise AssertionError(msg)


def test_matrix_preset_dimensions() -> None:
    """The documented 3x2x3x2x2 preset matches the CLI defaults."""
    for param_name, (_, tokens) in _DIMENSIONS.items():
        default = _default_value(param_name)
        values = tuple(v.strip().lower() for v in default.split(","))
        assert values == tokens, f"{param_name}: {values} != {tokens}"


def test_matrix_preset_is_72_cells() -> None:
    """3 N x 2 regimes x 3 families x 2 constraints x 2 self-loop policies = 72."""
    n = len(_default_value("nodes").split(","))
    regimes = len(_default_value("regimes").split(","))
    families = len(_default_value("families").split(","))
    constraints = len(_default_value("constraints").split(","))
    self_loop_policies = 2  # the CLI exposes both --self-loops and --no-self-loops
    assert n * regimes * families * constraints * self_loop_policies == 72, (
        f"matrix expands to {n * regimes * families * constraints * self_loop_policies}"
        " cells, expected 72"
    )
