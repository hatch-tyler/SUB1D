"""Golden file regression tests: legacy vs new solvers.

These tests load the original (unmodified) legacy solver functions and run
them side-by-side with the new vectorised implementations on identical
synthetic inputs.  The goal is to verify numerical equivalence.

Known behavioural improvements in the new code
-----------------------------------------------
1. **Elastic-inelastic head solver, last 2 timesteps of ``inelastic_flag``**:
   The legacy code has an off-by-one where ``inelastic_flag[:, -2:]`` is never
   set (always 0), because the ``if j <= len(t) - 3`` guard stops updating one
   step too early.  The new code fixes this.  As a consequence, the head at the
   very last timestep *may* differ slightly because the legacy solver always
   uses elastic CFL for the final step regardless of actual stress state.

2. **CFL check**: The legacy code only checks elastic CFL; the new code checks
   both elastic and inelastic CFL.

3. **Mutable default arguments**: The legacy code has ``overburden_data=[]``
   and ``ic_precons=[]`` which are classic Python bugs.  The new code uses
   ``None`` with an explicit check.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import legacy solver via importlib (it expects ``from utils import ...``)
# ---------------------------------------------------------------------------
_LEGACY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "legacy")
)


def _load_legacy_solver():
    """Load ``solver_legacy.py`` with a shimmed ``utils`` module."""
    # Provide a fake 'utils' module so ``from utils import printProgressBar``
    # inside the legacy solver resolves correctly.
    utils_shim = types.ModuleType("utils")
    utils_shim.printProgressBar = lambda *args, **kwargs: None  # no-op
    saved = sys.modules.get("utils")
    sys.modules["utils"] = utils_shim

    spec = importlib.util.spec_from_file_location(
        "solver_legacy",
        os.path.join(_LEGACY_DIR, "solver_legacy.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Restore
    if saved is None:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = saved

    return module


legacy = _load_legacy_solver()

# Import new solvers
from sub1d.solver import (
    solve_head_equation_single,
    solve_head_equation_elastic_inelastic,
    solve_compaction_elastic_inelastic,
)


# =========================================================================
# Helpers
# =========================================================================
def _make_step_bc(n_x, n_t, ic, h_left, h_right):
    """Constant Dirichlet BCs at both boundaries."""
    bc = np.zeros((2, n_t))
    bc[0, :] = h_left
    bc[1, :] = h_right
    return bc


def _make_declining_bc(n_x, n_t, h_start, h_end):
    """Linearly declining BCs at both boundaries."""
    bc = np.zeros((2, n_t))
    bc[0, :] = np.linspace(h_start, h_end, n_t)
    bc[1, :] = np.linspace(h_start, h_end, n_t)
    return bc


# =========================================================================
# 1. Single-value head solver
# =========================================================================
class TestGoldenSingleValueHead:
    """Legacy ``solve_head_equation_singlevalue`` vs new ``solve_head_equation_single``."""

    @pytest.fixture()
    def setup(self):
        n_x = 21
        L = 10.0
        dx = L / (n_x - 1)
        x = np.linspace(0, L, n_x)
        k = 0.05
        dt = 0.4 * dx**2 / k  # CFL = 0.4
        n_t = 500
        t = np.arange(n_t) * dt
        ic = 50.0
        bc = _make_step_bc(n_x, n_t, ic, h_left=40.0, h_right=45.0)
        return dt, t, dx, x, bc, ic, k, n_t

    def test_hmat_identical(self, setup):
        """Head matrices must be identical to machine precision."""
        dt, t, dx, x, bc, ic, k, n_t = setup

        hmat_legacy = legacy.solve_head_equation_singlevalue(dt, t, dx, x, bc, ic, k)
        hmat_new = solve_head_equation_single(dt, t, dx, x, bc, ic, k)

        assert hmat_legacy.shape == hmat_new.shape
        np.testing.assert_allclose(
            hmat_new, hmat_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Single-value head solver: new vs legacy mismatch",
        )

    def test_varied_ic(self, setup):
        """Test with a different initial condition."""
        dt, t, dx, x, bc, ic, k, n_t = setup
        ic2 = 100.0

        hmat_legacy = legacy.solve_head_equation_singlevalue(dt, t, dx, x, bc, ic2, k)
        hmat_new = solve_head_equation_single(dt, t, dx, x, bc, ic2, k)

        np.testing.assert_allclose(hmat_new, hmat_legacy, atol=1e-12, rtol=1e-12)

    def test_time_varying_bc(self):
        """Test with time-varying boundary conditions."""
        n_x = 15
        L = 5.0
        dx = L / (n_x - 1)
        x = np.linspace(0, L, n_x)
        k = 0.1
        dt = 0.4 * dx**2 / k
        n_t = 300
        t = np.arange(n_t) * dt
        ic = 20.0

        # Sinusoidal top BC, constant bottom
        bc = np.zeros((2, n_t))
        bc[0, :] = 20.0 + 5.0 * np.sin(2 * np.pi * t / t[-1])
        bc[1, :] = 20.0

        hmat_legacy = legacy.solve_head_equation_singlevalue(dt, t, dx, x, bc, ic, k)
        hmat_new = solve_head_equation_single(dt, t, dx, x, bc, ic, k)

        np.testing.assert_allclose(hmat_new, hmat_legacy, atol=1e-12, rtol=1e-12)


# =========================================================================
# 2. Elastic-inelastic head solver
# =========================================================================
class TestGoldenElasticInelasticHead:
    """Legacy ``solve_head_equation_elasticinelastic`` vs new ``solve_head_equation_elastic_inelastic``."""

    @pytest.fixture()
    def setup(self):
        n_x = 21
        L = 10.0
        dx = L / (n_x - 1)
        x = np.linspace(0, L, n_x)
        k_e = 0.05
        k_i = 0.02
        dt = 0.4 * dx**2 / max(k_e, k_i)  # CFL safe for both
        n_t = 500
        t = np.arange(n_t) * dt
        ic = 50.0
        bc = _make_declining_bc(n_x, n_t, h_start=50.0, h_end=40.0)
        return dt, t, dx, x, bc, ic, k_e, k_i, n_t

    def test_hmat_matches_except_last_step(self, setup):
        """Head matrices must match for all but possibly the last timestep.

        The legacy code has an off-by-one in inelastic_flag[:, -2:] = 0,
        which may cause the last step to use elastic CFL incorrectly.
        """
        dt, t, dx, x, bc, ic, k_e, k_i, n_t = setup

        hmat_legacy, ifl_legacy = legacy.solve_head_equation_elasticinelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )
        hmat_new, ifl_new = solve_head_equation_elastic_inelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )

        assert hmat_legacy.shape == hmat_new.shape

        # All timesteps except the last must match exactly
        np.testing.assert_allclose(
            hmat_new[:, :-1], hmat_legacy[:, :-1],
            atol=1e-12, rtol=1e-12,
            err_msg="Elastic-inelastic head: mismatch before last timestep",
        )

        # The last timestep differs because the legacy code always uses elastic
        # CFL for the final step (off-by-one bug).  With k_elastic=0.05 vs
        # k_inelastic=0.02, the difference can be up to ~0.03 in head.
        np.testing.assert_allclose(
            hmat_new[:, -1], hmat_legacy[:, -1],
            atol=0.05,
            err_msg="Elastic-inelastic head: last timestep differs beyond tolerance",
        )

    def test_inelastic_flag_matches_except_last_two(self, setup):
        """Inelastic flags must match except for the last 2 columns (legacy bug)."""
        dt, t, dx, x, bc, ic, k_e, k_i, n_t = setup

        _, ifl_legacy = legacy.solve_head_equation_elasticinelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )
        _, ifl_new = solve_head_equation_elastic_inelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )

        # Legacy stores floats; new stores bools
        ifl_legacy_bool = ifl_legacy.astype(bool)

        # All but last 2 columns must match
        np.testing.assert_array_equal(
            ifl_new[:, :-2], ifl_legacy_bool[:, :-2],
            err_msg="Inelastic flag mismatch before last 2 timesteps",
        )

        # Last 2 columns in legacy are always 0 (bug); new code sets them properly
        np.testing.assert_array_equal(
            ifl_legacy[:, -2:], 0,
            err_msg="Legacy inelastic_flag last 2 columns should be 0 (known bug)",
        )

    def test_with_overburden_stress(self, setup):
        """Test with overburden stress enabled."""
        dt, t, dx, x, bc, ic, k_e, k_i, n_t = setup

        # Small linearly increasing overburden
        ob = np.linspace(0, 0.01, n_t)

        hmat_legacy, _ = legacy.solve_head_equation_elasticinelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
            overburdenstress=True, overburden_data=ob,
        )
        hmat_new, _ = solve_head_equation_elastic_inelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
            overburden_stress=True, overburden_data=ob,
        )

        # All but last timestep
        np.testing.assert_allclose(
            hmat_new[:, :-1], hmat_legacy[:, :-1],
            atol=1e-12, rtol=1e-12,
        )

    def test_with_preset_preconsolidation(self, setup):
        """Test with initial preconsolidation stress provided."""
        dt, t, dx, x, bc, ic, k_e, k_i, n_t = setup

        # Preset preconsolidation below initial head
        precons = np.ones(len(x)) * 48.0

        hmat_legacy, _ = legacy.solve_head_equation_elasticinelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
            initial_precons=True, initial_condition_precons=precons,
        )
        hmat_new, _ = solve_head_equation_elastic_inelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
            initial_precons=True, initial_condition_precons=precons,
        )

        np.testing.assert_allclose(
            hmat_new[:, :-1], hmat_legacy[:, :-1],
            atol=1e-12, rtol=1e-12,
        )

    def test_purely_elastic_regime(self):
        """When head never drops below preconsolidation, results must be exact."""
        n_x = 15
        L = 5.0
        dx = L / (n_x - 1)
        x = np.linspace(0, L, n_x)
        k_e = 0.05
        k_i = 0.02
        dt = 0.4 * dx**2 / max(k_e, k_i)
        n_t = 200
        t = np.arange(n_t) * dt
        ic = 50.0

        # Head RISES (never triggers inelastic) => purely elastic
        bc = np.zeros((2, n_t))
        bc[0, :] = np.linspace(50.0, 55.0, n_t)
        bc[1, :] = np.linspace(50.0, 55.0, n_t)

        hmat_legacy, ifl_legacy = legacy.solve_head_equation_elasticinelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )
        hmat_new, ifl_new = solve_head_equation_elastic_inelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )

        # Purely elastic: no inelastic flags at all
        assert not np.any(ifl_new), "Expected no inelastic flags for rising head"
        assert not np.any(ifl_legacy), "Legacy should also have no inelastic flags"

        # All timesteps must match exactly (no off-by-one issue in elastic regime)
        np.testing.assert_allclose(
            hmat_new, hmat_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Purely elastic regime should be identical everywhere",
        )


# =========================================================================
# 3. Compaction solver
# =========================================================================
class TestGoldenCompactionSolver:
    """Legacy ``subsidence_solver_aquitard_elasticinelastic`` vs new ``solve_compaction_elastic_inelastic``."""

    def _make_hmat(self, n_z, n_t, h_start, h_end):
        """Create a synthetic head matrix with linearly declining heads."""
        hmat = np.zeros((n_z, n_t))
        for i in range(n_z):
            # Interior nodes decline; boundaries decline at same rate
            hmat[i, :] = np.linspace(h_start, h_end, n_t)
        return hmat

    def _make_hmat_with_diffusion(self, n_z, n_t, h_start, h_end, dx, k):
        """Create a more realistic hmat by actually running the solver."""
        dt = 0.4 * dx**2 / k
        x = np.linspace(0, (n_z - 1) * dx, n_z)
        t = np.arange(n_t) * dt

        bc = np.zeros((2, n_t))
        bc[0, :] = np.linspace(h_start, h_end, n_t)
        bc[1, :] = np.linspace(h_start, h_end, n_t)

        hmat = solve_head_equation_single(dt, t, dx, x, bc, h_start, k)
        return hmat

    def test_compaction_identical_uniform_decline(self):
        """Compaction from uniform head decline must match legacy."""
        n_z = 11
        n_t = 200
        dz = 0.5
        Sske = 1e-4
        Sskv = 1e-3

        hmat = self._make_hmat(n_z, n_t, h_start=50.0, h_end=40.0)

        b_legacy, ifl_legacy = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz,
        )
        b_new, ifl_new = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz,
        )

        assert b_legacy.shape == b_new.shape
        np.testing.assert_allclose(
            b_new, b_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Compaction (uniform decline): new vs legacy mismatch",
        )

    def test_compaction_identical_diffused_head(self):
        """Compaction from a diffusion-derived head field must match legacy."""
        n_z = 21
        n_t = 300
        dz = 0.5
        k = 0.05
        Sske = 1e-4
        Sskv = 1e-3

        hmat = self._make_hmat_with_diffusion(n_z, n_t, 50.0, 40.0, dz, k)

        b_legacy, ifl_legacy = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz,
        )
        b_new, ifl_new = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz,
        )

        np.testing.assert_allclose(
            b_new, b_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Compaction (diffused head): new vs legacy mismatch",
        )

    def test_inelastic_flags_match(self):
        """Inelastic flag arrays from compaction solver must match."""
        n_z = 11
        n_t = 200
        dz = 0.5
        Sske = 1e-4
        Sskv = 1e-3

        hmat = self._make_hmat(n_z, n_t, h_start=50.0, h_end=40.0)

        _, ifl_legacy = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz,
        )
        _, ifl_new = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz,
        )

        np.testing.assert_array_equal(
            ifl_new, ifl_legacy.astype(bool),
            err_msg="Compaction inelastic flags: new vs legacy mismatch",
        )

    def test_compaction_with_overburden(self):
        """Compaction with overburden stress must match legacy."""
        n_z = 11
        n_t = 200
        dz = 0.5
        Sske = 1e-4
        Sskv = 1e-3

        hmat = self._make_hmat(n_z, n_t, h_start=50.0, h_end=40.0)
        ob = np.linspace(0, 0.05, n_t)

        b_legacy, _ = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz,
            overburden=True, overburden_data=ob,
        )
        b_new, _ = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz,
            overburden=True, overburden_data=ob,
        )

        np.testing.assert_allclose(
            b_new, b_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Compaction with overburden: new vs legacy mismatch",
        )

    def test_compaction_with_preset_preconsolidation(self):
        """Compaction with preset preconsolidation must match legacy."""
        n_z = 11
        n_t = 200
        dz = 0.5
        Sske = 1e-4
        Sskv = 1e-3

        hmat = self._make_hmat(n_z, n_t, h_start=50.0, h_end=40.0)
        ic_precons = np.ones(n_z) * 48.0  # some nodes start already inelastic

        b_legacy, _ = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz,
            preset_precons=True, ic_precons=ic_precons,
        )
        b_new, _ = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz,
            preset_precons=True, ic_precons=ic_precons,
        )

        np.testing.assert_allclose(
            b_new, b_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Compaction with preset precons: new vs legacy mismatch",
        )

    def test_compaction_endnodes_mode(self):
        """Compaction with endnodes=True (skip midpoint interpolation)."""
        n_z = 11
        n_t = 200
        dz = 0.5
        Sske = 1e-4
        Sskv = 1e-3

        hmat = self._make_hmat(n_z, n_t, h_start=50.0, h_end=40.0)

        b_legacy, _ = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz, endnodes=True,
        )
        b_new, _ = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz, endnodes=True,
        )

        np.testing.assert_allclose(
            b_new, b_legacy,
            atol=1e-12, rtol=1e-12,
            err_msg="Compaction endnodes mode: new vs legacy mismatch",
        )

    def test_compaction_rising_head_purely_elastic(self):
        """Rising head => purely elastic compaction; must match exactly."""
        n_z = 11
        n_t = 200
        dz = 0.5
        Sske = 1e-4
        Sskv = 1e-3

        # Head rises: stress decreases, stays below initial preconsolidation
        hmat = self._make_hmat(n_z, n_t, h_start=50.0, h_end=55.0)

        b_legacy, ifl_legacy = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat, Sske, Sskv, dz,
        )
        b_new, ifl_new = solve_compaction_elastic_inelastic(
            hmat, Sske, Sskv, dz,
        )

        # No inelastic deformation expected
        assert not np.any(ifl_new), "Expected purely elastic for rising head"
        assert not np.any(ifl_legacy), "Legacy should also be purely elastic"

        np.testing.assert_allclose(
            b_new, b_legacy,
            atol=1e-12, rtol=1e-12,
        )


# =========================================================================
# 4. Combined head + compaction pipeline
# =========================================================================
class TestGoldenFullPipeline:
    """Run the complete head -> compaction pipeline through both codebases."""

    def test_full_pipeline_declining_head(self):
        """Full pipeline: solve head, then compaction, compare all outputs."""
        # Setup
        n_x = 21
        L = 10.0
        dx = L / (n_x - 1)
        x = np.linspace(0, L, n_x)
        k = 0.05
        dt = 0.4 * dx**2 / k
        n_t = 400
        t = np.arange(n_t) * dt
        ic = 50.0

        bc = np.zeros((2, n_t))
        bc[0, :] = np.linspace(50.0, 42.0, n_t)
        bc[1, :] = np.linspace(50.0, 42.0, n_t)

        Sske = 1e-4
        Sskv = 1e-3

        # Legacy pipeline
        hmat_legacy = legacy.solve_head_equation_singlevalue(
            dt, t, dx, x, bc, ic, k,
        )
        b_legacy, ifl_legacy = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat_legacy, Sske, Sskv, dx,
        )

        # New pipeline
        hmat_new = solve_head_equation_single(dt, t, dx, x, bc, ic, k)
        b_new, ifl_new = solve_compaction_elastic_inelastic(
            hmat_new, Sske, Sskv, dx,
        )

        # Head must be identical
        np.testing.assert_allclose(
            hmat_new, hmat_legacy, atol=1e-12, rtol=1e-12,
            err_msg="Pipeline: head mismatch",
        )

        # Compaction must be identical (same hmat input)
        np.testing.assert_allclose(
            b_new, b_legacy, atol=1e-12, rtol=1e-12,
            err_msg="Pipeline: compaction mismatch",
        )

    def test_full_pipeline_elastic_inelastic_head(self):
        """Full pipeline with elastic-inelastic head solver."""
        n_x = 21
        L = 10.0
        dx = L / (n_x - 1)
        x = np.linspace(0, L, n_x)
        k_e = 0.05
        k_i = 0.02
        dt = 0.4 * dx**2 / max(k_e, k_i)
        n_t = 400
        t = np.arange(n_t) * dt
        ic = 50.0

        bc = np.zeros((2, n_t))
        bc[0, :] = np.linspace(50.0, 42.0, n_t)
        bc[1, :] = np.linspace(50.0, 42.0, n_t)

        Sske = 1e-4
        Sskv = 1e-3

        # Legacy pipeline
        hmat_legacy, _ = legacy.solve_head_equation_elasticinelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )
        b_legacy, _ = legacy.subsidence_solver_aquitard_elasticinelastic(
            hmat_legacy, Sske, Sskv, dx,
        )

        # New pipeline (use legacy hmat to isolate compaction comparison)
        b_new_from_legacy_hmat, _ = solve_compaction_elastic_inelastic(
            hmat_legacy, Sske, Sskv, dx,
        )

        # Compaction from same hmat must be identical
        np.testing.assert_allclose(
            b_new_from_legacy_hmat, b_legacy, atol=1e-12, rtol=1e-12,
            err_msg="Pipeline: compaction from legacy hmat mismatch",
        )

        # Now run new head solver too; head matches except last step
        hmat_new, _ = solve_head_equation_elastic_inelastic(
            dt, t, dx, x, bc, ic, k_e, k_i,
        )
        np.testing.assert_allclose(
            hmat_new[:, :-1], hmat_legacy[:, :-1],
            atol=1e-12, rtol=1e-12,
            err_msg="Pipeline: elastic-inelastic head mismatch",
        )

        # Full new pipeline compaction should be very close.  The small
        # difference at the last timestep of hmat (from the off-by-one fix)
        # propagates into a ~2e-5 compaction difference at the last step.
        b_new, _ = solve_compaction_elastic_inelastic(hmat_new, Sske, Sskv, dx)
        np.testing.assert_allclose(
            b_new, b_legacy, atol=1e-4,
            err_msg="Pipeline: end-to-end compaction too different",
        )
