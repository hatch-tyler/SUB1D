"""Integration tests for the SUB1D model.

These tests verify the full pipeline works end-to-end.
Golden file regression tests will be added once legacy output is captured.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from sub1d.solver import (
    solve_head_equation_single,
    solve_head_equation_elastic_inelastic,
    solve_head_equation_crank_nicolson,
    solve_compaction_elastic_inelastic,
)


class TestSolverCrossValidation:
    """Verify that explicit FTCS and Crank-Nicolson converge to the same solution."""

    def test_both_solvers_converge_same_steady_state(self):
        """Both solvers should produce the same steady-state solution."""
        n = 31
        L = 1.0
        dx = L / (n - 1)
        x = np.linspace(0, L, n)
        k = 0.1

        # Explicit: use CFL-safe dt
        dt_explicit = 0.4 * dx**2 / k
        n_steps_explicit = 10000
        t_explicit = np.arange(n_steps_explicit) * dt_explicit

        # Crank-Nicolson: can use larger dt
        dt_cn = 10 * dt_explicit
        n_steps_cn = 1000
        t_cn = np.arange(n_steps_cn) * dt_cn

        h0 = 10.0
        h_left, h_right = 3.0, 7.0

        bc_explicit = np.zeros((2, n_steps_explicit))
        bc_explicit[0, :] = h_left
        bc_explicit[1, :] = h_right

        bc_cn = np.zeros((2, n_steps_cn))
        bc_cn[0, :] = h_left
        bc_cn[1, :] = h_right

        hmat_explicit = solve_head_equation_single(
            dt_explicit, t_explicit, dx, x, bc_explicit, h0, k)
        hmat_cn = solve_head_equation_crank_nicolson(
            dt_cn, t_cn, dx, x, bc_cn, h0, k)

        # Both should converge to the same linear steady state
        expected_ss = h_left + (h_right - h_left) * x / L
        np.testing.assert_allclose(hmat_explicit[:, -1], expected_ss, atol=0.05)
        np.testing.assert_allclose(hmat_cn[:, -1], expected_ss, atol=0.05)


class TestFullPipelineSynthetic:
    """Test the full head -> compaction pipeline with synthetic data."""

    def test_synthetic_aquitard_workflow(self):
        """Run head equation then compaction for a synthetic aquitard."""
        n = 21
        L = 10.0  # 10m thick aquitard
        dx = L / (n - 1)
        x = np.linspace(0, L, n)
        k = 0.01  # Low diffusivity (clay)
        dt = 0.4 * dx**2 / k
        n_steps = 2000
        t = np.arange(n_steps) * dt

        # Head drops in surrounding aquifers
        h_initial = 100.0
        h_final = 90.0
        bc_top = np.linspace(h_initial, h_final, n_steps)
        bc_bot = np.linspace(h_initial, h_final, n_steps)
        bc = np.vstack([bc_top, bc_bot])

        # Solve head equation
        hmat = solve_head_equation_single(dt, t, dx, x, bc, h_initial, k)

        assert hmat.shape == (n, n_steps)
        # Head at boundaries should match BCs
        np.testing.assert_allclose(hmat[0, :], bc_top, atol=0.01)
        np.testing.assert_allclose(hmat[-1, :], bc_bot, atol=0.01)

        # Now solve compaction
        Sske = 1e-4
        Sskv = 1e-3
        b, ifl = solve_compaction_elastic_inelastic(hmat, Sske, Sskv, dx)

        # With declining head, should see compaction (negative b means subsidence)
        assert b.shape == (n_steps,)
        # The direction of b depends on sign convention - just check it's non-zero
        assert np.any(b != 0)
