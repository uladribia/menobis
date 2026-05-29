"""Pure formula verification tests for ME, B, and W distribution families.

Tests verify E[t_ij] and E[Θ(t_ij>0)] formulas from the thesis against
hand-computed values. No solver calls — only direct formula evaluation.

Reference: https://hdl.handle.net/10803/400560
"""

import math

import numpy as np
import pytest

# --- ME (Poisson) formulas ---


class TestMEFormulas:
    """Verify ME/Poisson expected weight and occupation formulas."""

    def test_non_zero_inflated_expected_weight(self) -> None:
        """ME non-ZI: E[t_ij] = q_ij = x_i * y_j."""
        x = np.array([0.5, 1.0, 2.0, 0.3, 1.5])
        y = np.array([0.4, 0.8, 1.2, 0.6, 1.0])
        expected = np.outer(x, y)
        assert expected[0, 0] == pytest.approx(0.2)
        assert expected[2, 2] == pytest.approx(2.4)
        assert expected[4, 1] == pytest.approx(1.2)

    def test_zero_inflated_partition_factor(self) -> None:
        """G_ME(q) = exp(q) - 1."""
        q_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        for q in q_values:
            g = math.exp(q) - 1.0
            assert g == pytest.approx(math.expm1(q), rel=1e-14)

    def test_zero_inflated_occupation(self) -> None:
        """E[Θ] = v*G(q) / (1 + v*G(q)) where G(q)=exp(q)-1."""
        q, v = 0.4, 0.7
        g = math.exp(q) - 1.0
        expected_occ = v * g / (1.0 + v * g)
        # Manual: exp(0.4)=1.4918..., g=0.4918..., v*g=0.3443...
        # occ = 0.3443 / 1.3443 = 0.2562...
        assert expected_occ == pytest.approx(0.25617, rel=1e-3)

    def test_zero_inflated_expected_weight(self) -> None:
        """E[t_ij] = v * q * exp(q) / (1 + v*G(q))."""
        q, v = 0.4, 0.7
        exp_q = math.exp(q)
        g = exp_q - 1.0
        expected_weight = v * q * exp_q / (1.0 + v * g)
        # v*q*exp(q) = 0.7*0.4*1.4918 = 0.4177
        # denom = 1 + 0.7*0.4918 = 1.3443
        # weight = 0.4177/1.3443 = 0.3107...
        assert expected_weight == pytest.approx(0.3107, rel=1e-3)

    def test_conditional_weight_given_positive(self) -> None:
        """E[t|t>0] = q*exp(q)/(exp(q)-1) = q/(1-exp(-q))."""
        q = 0.4
        conditional = q / (1.0 - math.exp(-q))
        alt = q * math.exp(q) / (math.exp(q) - 1.0)
        assert conditional == pytest.approx(alt, rel=1e-14)
        # For q=0.4: 0.4/(1-0.6703) = 0.4/0.3297 = 1.2132
        assert conditional == pytest.approx(1.2132, rel=1e-3)


# --- B (Binomial) formulas ---


class TestBFormulas:
    """Verify B/Binomial expected weight and occupation formulas."""

    def test_non_zero_inflated_expected_weight(self) -> None:
        """B non-ZI: E[t_ij] = M*q/(1+q)."""
        q, m = 0.5, 10
        expected = m * q / (1.0 + q)
        # 10*0.5/1.5 = 3.333...
        assert expected == pytest.approx(10.0 / 3.0, rel=1e-10)

    def test_zero_inflated_partition_factor(self) -> None:
        """G_B(q) = (1+q)^M - 1."""
        q, m = 0.3, 5
        g = (1.0 + q) ** m - 1.0
        # (1.3)^5 - 1 = 3.71293 - 1 = 2.71293
        assert g == pytest.approx(2.71293, rel=1e-4)

    def test_zero_inflated_occupation(self) -> None:
        """E[Θ] = v*G_B(q) / (1 + v*G_B(q))."""
        q, v, m = 0.3, 0.5, 5
        g = (1.0 + q) ** m - 1.0
        occ = v * g / (1.0 + v * g)
        # v*g = 0.5*2.71293 = 1.35647
        # occ = 1.35647/2.35647 = 0.57564
        assert occ == pytest.approx(0.57564, rel=1e-3)

    def test_zero_inflated_expected_weight(self) -> None:
        """E[t_ij] = v * M * q * (1+q)^(M-1) / (1 + v*G_B(q))."""
        q, v, m = 0.3, 0.5, 5
        g = (1.0 + q) ** m - 1.0
        weight = v * m * q * (1.0 + q) ** (m - 1) / (1.0 + v * g)
        # numerator: 0.5*5*0.3*(1.3)^4 = 0.75*2.8561 = 2.1421
        # denom: 1 + 0.5*2.71293 = 2.35647
        # weight = 2.1421/2.35647 = 0.90903
        assert weight == pytest.approx(0.90903, rel=1e-3)

    def test_conditional_weight_given_positive(self) -> None:
        """E[t|t>0] = M*q*(1+q)^(M-1) / ((1+q)^M - 1)."""
        q, m = 0.3, 5
        g = (1.0 + q) ** m - 1.0
        conditional = m * q * (1.0 + q) ** (m - 1) / g
        # = 5*0.3*(1.3)^4 / 2.71293 = 4.2842/2.71293 = 1.5792
        assert conditional == pytest.approx(1.5792, rel=1e-3)


# --- W (Geometric / Negative Binomial) formulas ---


class TestWFormulas:
    """Verify W/Geometric/NegBin expected weight and occupation formulas."""

    def test_non_zero_inflated_expected_weight(self) -> None:
        """W non-ZI: E[t_ij] = M*q/(1-q) with q in (0,1)."""
        q, m = 0.3, 1
        expected = m * q / (1.0 - q)
        # 1*0.3/0.7 = 0.4286
        assert expected == pytest.approx(3.0 / 7.0, rel=1e-10)

    def test_non_zero_inflated_negbin(self) -> None:
        """W non-ZI with M layers: E[t_ij] = M*q/(1-q)."""
        q, m = 0.2, 5
        expected = m * q / (1.0 - q)
        # 5*0.2/0.8 = 1.25
        assert expected == pytest.approx(1.25, rel=1e-10)

    def test_zero_inflated_partition_factor(self) -> None:
        """G_W(q) = (1-q)^(-M) - 1."""
        q, m = 0.3, 3
        g = (1.0 - q) ** (-m) - 1.0
        # (0.7)^(-3) - 1 = 2.9155 - 1 = 1.9155
        assert g == pytest.approx(1.9155, rel=1e-3)

    def test_zero_inflated_occupation(self) -> None:
        """E[Θ] = v*G_W(q) / (1 + v*G_W(q))."""
        q, v, m = 0.3, 0.6, 3
        g = (1.0 - q) ** (-m) - 1.0
        occ = v * g / (1.0 + v * g)
        # v*g = 0.6*1.9155 = 1.1493
        # occ = 1.1493/2.1493 = 0.5347
        assert occ == pytest.approx(0.5347, rel=1e-3)

    def test_zero_inflated_expected_weight(self) -> None:
        """E[t_ij] = v * M * q * (1-q)^(-M-1) / (1 + v*G_W(q))."""
        q, v, m = 0.3, 0.6, 3
        g = (1.0 - q) ** (-m) - 1.0
        weight = v * m * q * (1.0 - q) ** (-m - 1) / (1.0 + v * g)
        # numerator: 0.6*3*0.3*(0.7)^(-4) = 0.54*4.1650 = 2.2491
        # denom: 1 + 0.6*1.9155 = 2.1493
        # weight = 2.2491/2.1493 = 1.0465
        assert weight == pytest.approx(1.0465, rel=1e-3)

    def test_conditional_weight_given_positive(self) -> None:
        """E[t|t>0] = M*q*(1-q)^(-M-1) / ((1-q)^(-M) - 1)."""
        q, m = 0.3, 3
        g = (1.0 - q) ** (-m) - 1.0
        conditional = m * q * (1.0 - q) ** (-m - 1) / g
        # = 3*0.3*(0.7)^(-4) / 1.9155 = 3.7485/1.9155 = 1.9570
        assert conditional == pytest.approx(1.9570, rel=1e-3)

    def test_q_bound_below_one(self) -> None:
        """W requires q_ij < 1 for convergence."""
        q = 0.999
        m = 1
        g = (1.0 - q) ** (-m) - 1.0
        assert g > 0
        assert math.isfinite(g)
        # At q=1.0, G_W diverges
        assert (1.0 - 0.9999) ** (-m) - 1.0 > 9999


# --- Cross-family comparison ---


class TestCrossFamilyDifference:
    """Verify ME ≠ B ≠ W on same inputs."""

    def test_same_q_different_expectations(self) -> None:
        """Given same q=0.4 and layers=5, all families produce different E[t|t>0]."""
        q = 0.4
        m = 5

        # ME: q/(1-exp(-q))
        me_cond = q / (1.0 - math.exp(-q))

        # B: M*q*(1+q)^(M-1) / ((1+q)^M - 1)
        b_cond = m * q * (1.0 + q) ** (m - 1) / ((1.0 + q) ** m - 1.0)

        # W: M*q*(1-q)^(-M-1) / ((1-q)^(-M) - 1)
        w_cond = m * q * (1.0 - q) ** (-m - 1) / ((1.0 - q) ** (-m) - 1.0)

        # All must be different
        assert me_cond != pytest.approx(b_cond, rel=1e-3)
        assert me_cond != pytest.approx(w_cond, rel=1e-3)
        assert b_cond != pytest.approx(w_cond, rel=1e-3)

        # ME < B < W for moderate q (ordering property)
        assert me_cond < b_cond < w_cond
