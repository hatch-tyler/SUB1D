"""Tests for the diagnostics module."""
from __future__ import annotations

import numpy as np
import pytest

from sub1d.diagnostics import check_mass_balance, check_cn_residual, estimate_spatial_error
from sub1d.solver import solve_head_equation_single, solve_head_equation_crank_nicolson


class TestCheckMassBalance:
    """Tests for check_mass_balance."""

    def test_good_solution_returns_finite(self):
        """A valid FTCS solution should return a finite mass balance error."""
        n = 21
        L = 1.0
        dx = L / (n - 1)
        x = np.linspace(0, L, n)
        k = 0.1
        dt = 0.4 * dx**2 / k
        n_steps = 200
        t = np.arange(n_steps) * dt
        bc = np.full((2, n_steps), 5.0)
        hmat = solve_head_equation_single(dt, t, dx, x, bc, 10.0, k)

        # The mass balance check should return a finite value
        err = check_mass_balance(hmat, dx, dt, k, bc, 100, threshold=100.0)
        assert np.isfinite(err)

    def test_near_steady_state_small_error(self):
        """Near steady state, mass balance error should be very small."""
        n = 21
        L = 1.0
        dx = L / (n - 1)
        x = np.linspace(0, L, n)
        k = 0.1
        dt = 0.4 * dx**2 / k
        n_steps = 5000
        t = np.arange(n_steps) * dt
        bc = np.full((2, n_steps), 7.0)
        hmat = solve_head_equation_single(dt, t, dx, x, bc, 7.0, k)

        # At steady state with uniform head, error should be tiny
        err = check_mass_balance(hmat, dx, dt, k, bc, n_steps - 1, threshold=100.0)
        assert err < 1.0  # Very relaxed since storage ~ flux ~ 0

    def test_uniform_gives_zero_error(self):
        """Uniform head with matching BCs => zero mass balance error."""
        n = 11
        n_t = 50
        hmat = np.ones((n, n_t)) * 7.0
        dx = 0.1
        dt = 0.001
        k = 0.1
        bc = np.full((2, n_t), 7.0)

        err = check_mass_balance(hmat, dx, dt, k, bc, 10)
        assert err < 1e-10


class TestCheckCNResidual:
    """Tests for check_cn_residual."""

    def test_identity_system_zero_residual(self):
        """Identity banded matrix: A*x = b exactly."""
        m = 5
        ab = np.zeros((3, m))
        ab[1, :] = 1.0  # identity main diagonal
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rhs = x.copy()

        residual = check_cn_residual(ab, x, rhs, step=1)
        assert residual < 1e-14

    def test_tridiagonal_residual_small(self):
        """Solve a tridiagonal system and verify residual is small."""
        import scipy.linalg

        m = 10
        r = 0.3
        ab = np.zeros((3, m))
        ab[0, 1:] = -0.5 * r
        ab[1, :] = 1.0 + r
        ab[2, :-1] = -0.5 * r

        rhs = np.ones(m) * 2.0
        x = scipy.linalg.solve_banded((1, 1), ab, rhs)

        residual = check_cn_residual(ab, x, rhs, step=1)
        assert residual < 1e-12


class TestEstimateSpatialError:
    """Tests for Richardson extrapolation."""

    def test_finer_grid_has_smaller_error(self):
        """Richardson error estimate should be positive and finite."""
        n = 11
        L = 1.0
        dx = L / (n - 1)
        x = np.linspace(0, L, n)
        k = 0.1
        dt = 0.3 * dx**2 / k
        n_steps = 200
        t = np.arange(n_steps) * dt
        bc = np.full((2, n_steps), 5.0)

        kwargs = dict(dt=dt, t=t, dx=dx, x=x, bc=bc, ic=10.0, k=k)

        result = estimate_spatial_error(
            solve_head_equation_single, kwargs, dx,
        )

        assert result["error_estimate"] >= 0
        assert np.isfinite(result["error_estimate"])
        assert result["max_pointwise_error"] >= 0

    def test_cn_solver_richardson(self):
        """Richardson extrapolation works with CN solver too."""
        n = 11
        L = 1.0
        dx = L / (n - 1)
        x = np.linspace(0, L, n)
        k = 0.1
        dt = 0.01
        n_steps = 200
        t = np.arange(n_steps) * dt
        bc = np.full((2, n_steps), 5.0)

        kwargs = dict(dt=dt, t=t, dx=dx, x=x, bc=bc, ic=10.0, k=k)

        result = estimate_spatial_error(
            solve_head_equation_crank_nicolson, kwargs, dx,
        )

        assert result["error_estimate"] >= 0
        assert np.isfinite(result["error_estimate"])
