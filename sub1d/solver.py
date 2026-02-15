"""Solver module for the SUB1D 1D land subsidence model.

Provides FTCS explicit, Crank-Nicolson implicit, and compaction solvers
for modelling 1D diffusion (head equation) and resulting compaction in
aquitard systems.  Inner loops are fully vectorised with NumPy; an
optional Numba JIT path is available for the explicit FTCS time-stepper.

Functions
---------
solve_head_equation_single
    Explicit FTCS solver with a single (constant) diffusivity.
solve_head_equation_elastic_inelastic
    Explicit FTCS solver with elastic/inelastic diffusivity switching.
solve_head_equation_crank_nicolson
    Implicit Crank-Nicolson solver with a single diffusivity.
solve_head_equation_cn_elastic_inelastic
    Implicit Crank-Nicolson solver with elastic/inelastic switching.
solve_compaction_elastic_inelastic
    Compaction solver for elastic-inelastic aquitard deformation.
"""

import logging
from typing import Optional

import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.sparse

from sub1d.exceptions import SolverError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:

    @njit
    def _ftcs_loop_numba(
        h: np.ndarray,
        hmat: np.ndarray,
        bc: np.ndarray,
        cfl: float,
        n_t: int,
    ) -> np.ndarray:
        """Run the explicit FTCS time loop using Numba.

        Parameters
        ----------
        h : np.ndarray
            Initial head vector of shape ``(n_x,)``.
        hmat : np.ndarray
            Pre-allocated output array of shape ``(n_x, n_t)``.  Column 0
            must already be filled with the initial condition.
        bc : np.ndarray
            Boundary conditions of shape ``(2, n_t)``.
        cfl : float
            CFL number ``k * dt / dx**2``.
        n_t : int
            Number of time-steps (columns in *hmat*).

        Returns
        -------
        np.ndarray
            The filled *hmat* array.
        """
        for j in range(1, n_t):
            h_new = np.empty_like(h)
            h_new[0] = bc[0, j]
            h_new[-1] = bc[1, j]
            for c in range(1, len(h) - 1):
                h_new[c] = h[c] + cfl * (h[c - 1] - 2.0 * h[c] + h[c + 1])
            h[:] = h_new
            hmat[:, j] = h
        return hmat

    @njit
    def _ftcs_ei_loop_numba(
        h: np.ndarray,
        hmat: np.ndarray,
        bc: np.ndarray,
        overburden_data: np.ndarray,
        cfl_elastic: float,
        cfl_inelastic: float,
        h_precons: np.ndarray,
        inelastic_flag: np.ndarray,
        n_t: int,
    ) -> tuple:
        """Run the explicit FTCS EI time loop using Numba.

        Handles per-node CFL switching and preconsolidation tracking
        in compiled inner loops.

        Parameters
        ----------
        h : np.ndarray
            Initial head vector of shape ``(n_x,)``.
        hmat : np.ndarray
            Pre-allocated output, shape ``(n_x, n_t)``.  Column 0 filled.
        bc : np.ndarray
            Boundary conditions, shape ``(2, n_t)``.
        overburden_data : np.ndarray
            Overburden stress, shape ``(n_t,)``.
        cfl_elastic, cfl_inelastic : float
            CFL numbers for elastic and inelastic diffusivity.
        h_precons : np.ndarray
            Preconsolidation head, shape ``(n_x, n_t)``.  Column 0 filled.
        inelastic_flag : np.ndarray
            Boolean flag array, shape ``(n_x, n_t)``.
        n_t : int
            Number of time-steps.

        Returns
        -------
        tuple
            ``(hmat, h_precons, inelastic_flag)``
        """
        n_x = len(h)
        for j in range(1, n_t):
            h_new = np.empty_like(h)
            h_new[0] = bc[0, j]
            h_new[-1] = bc[1, j]

            ob_delta = overburden_data[j] - overburden_data[j - 1]
            for c in range(1, n_x - 1):
                if inelastic_flag[c, j - 1]:
                    cfl = cfl_inelastic
                else:
                    cfl = cfl_elastic
                h_new[c] = h[c] + cfl * (h[c - 1] - 2.0 * h[c] + h[c + 1]) + ob_delta

            h[:] = h_new
            hmat[:, j] = h

            # Update preconsolidation and inelastic flag
            for c in range(n_x):
                effective = h[c] - overburden_data[j]
                newly_inel = effective < h_precons[c, j - 1]
                inelastic_flag[c, j] = newly_inel
                if j <= n_t - 2:
                    if newly_inel:
                        h_precons[c, j] = effective
                    else:
                        h_precons[c, j] = h_precons[c, j - 1]

        return hmat, h_precons, inelastic_flag


# ---------------------------------------------------------------------------
# Helper: progress logging at ~10 % intervals
# ---------------------------------------------------------------------------
def _log_progress(step: int, total: int, label: str = "Solver") -> None:
    """Log a progress message at approximately every 10 % of *total*.

    Parameters
    ----------
    step : int
        Current iteration index (0-based).
    total : int
        Total number of iterations.
    label : str, optional
        Prefix for the log message.
    """
    if total <= 0:
        return
    interval = max(total // 10, 1)
    if step % interval == 0 or step == total - 1:
        pct = 100.0 * step / total
        logger.info("%s progress: %5.1f %% (step %d / %d)", label, pct, step, total)


# =========================================================================
# 1. Explicit FTCS  --  single diffusivity
# =========================================================================
def solve_head_equation_single(
    dt: float,
    t: np.ndarray,
    dx: float,
    x: np.ndarray,
    bc: np.ndarray,
    ic: float,
    k: float,
    mass_balance_check: bool = False,
) -> np.ndarray:
    """Solve the 1-D diffusion (head) equation with the explicit FTCS scheme.

    Parameters
    ----------
    dt : float
        Time-step size.
    t : np.ndarray
        1-D array of time values, shape ``(n_t,)``.
    dx : float
        Spatial grid spacing.
    x : np.ndarray
        1-D array of spatial node positions, shape ``(n_x,)``.
    bc : np.ndarray
        Boundary conditions, shape ``(2, n_t)``.  ``bc[0, :]`` is the top
        (first node) and ``bc[1, :]`` the bottom (last node) boundary.
    ic : float
        Scalar initial condition applied uniformly across all nodes.
    k : float
        Diffusivity (hydraulic conductivity divided by specific storage).

    Returns
    -------
    np.ndarray
        Head matrix *hmat* of shape ``(n_x, n_t)``.

    Raises
    ------
    SolverError
        If the CFL number ``k * dt / dx**2`` is >= 0.5.
    """
    n_x = len(x)
    n_t = len(t)

    cfl = k * dt / dx ** 2

    logger.info("FTCS explicit solver (single diffusivity)")
    logger.info("  n_x=%d  x=[%.4f, %.4f]  dx=%.6f", n_x, x.min(), x.max(), dx)
    logger.info("  n_t=%d  t=[%.4f, %.4f]  dt=%.6f", n_t, t.min(), t.max(), dt)
    logger.info("  k=%.6g  CFL=%.6g", k, cfl)

    if cfl >= 0.5:
        raise SolverError(
            f"CFL condition failed: k*dt/dx^2 = {cfl:.6f} >= 0.5. "
            "Reduce dt or increase dx."
        )

    # Initialise head vector and output matrix
    h = ic * np.ones(n_x)
    h[0] = bc[0, 0]
    h[-1] = bc[1, 0]

    hmat = np.zeros((n_x, n_t), order='F')
    hmat[:, 0] = h

    # Optional Numba fast-path
    if HAS_NUMBA:
        logger.info("  Using Numba-accelerated FTCS loop.")
        hmat = _ftcs_loop_numba(h.copy(), hmat, bc, cfl, n_t)
        logger.info("Solver complete.")
        return hmat

    # Vectorised NumPy fallback
    mb_interval = max(n_t // 20, 1) if mass_balance_check else 0
    for j in range(1, n_t):
        _log_progress(j, n_t, "FTCS-single")

        h_new = np.empty_like(h)
        h_new[0] = bc[0, j]
        h_new[-1] = bc[1, j]
        h_new[1:-1] = h[1:-1] + cfl * (h[:-2] - 2.0 * h[1:-1] + h[2:])
        h = h_new
        hmat[:, j] = h

        if mb_interval and j % mb_interval == 0:
            from sub1d.diagnostics import check_mass_balance
            check_mass_balance(hmat, dx, dt, k, bc, j)

    logger.info("Solver complete.")
    return hmat


# =========================================================================
# 2. Explicit FTCS  --  elastic / inelastic switching
# =========================================================================
def solve_head_equation_elastic_inelastic(
    dt: float,
    t: np.ndarray,
    dx: float,
    x: np.ndarray,
    bc: np.ndarray,
    ic: float,
    k_elastic: float,
    k_inelastic: float,
    overburden_stress: bool = False,
    overburden_data: Optional[np.ndarray] = None,
    initial_precons: bool = False,
    initial_condition_precons: Optional[np.ndarray] = None,
    smoothing_width: float = 0.0,
    mass_balance_check: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1-D head equation with elastic/inelastic diffusivity switching.

    Parameters
    ----------
    dt : float
        Time-step size.
    t : np.ndarray
        1-D array of time values, shape ``(n_t,)``.
    dx : float
        Spatial grid spacing.
    x : np.ndarray
        1-D array of spatial node positions, shape ``(n_x,)``.
    bc : np.ndarray
        Boundary conditions, shape ``(2, n_t)``.
    ic : float
        Scalar initial condition applied uniformly across all nodes.
    k_elastic : float
        Elastic diffusivity.
    k_inelastic : float
        Inelastic diffusivity.
    overburden_stress : bool, optional
        Whether to include overburden stress effects (default ``False``).
    overburden_data : np.ndarray or None, optional
        1-D array of overburden stress values at each time step, shape
        ``(n_t,)``.  Required when *overburden_stress* is ``True``.
    initial_precons : bool, optional
        Whether a preset preconsolidation stress is provided.
    initial_condition_precons : np.ndarray or None, optional
        1-D array of initial preconsolidation head values per node,
        shape ``(n_x,)``.
    smoothing_width : float, optional
        Sigmoid blend width for elastic-inelastic transition.  When > 0,
        the CFL number is a smooth blend between elastic and inelastic
        values.  When 0 (default), the original sharp ``np.where`` logic
        is used.
    mass_balance_check : bool, optional
        If ``True``, periodically check mass balance (default ``False``).

    Returns
    -------
    hmat : np.ndarray
        Head matrix of shape ``(n_x, n_t)``.
    inelastic_flag : np.ndarray
        Boolean array of shape ``(n_x, n_t)`` indicating which nodes are
        in inelastic (virgin compression) state at each time step.

    Raises
    ------
    SolverError
        If either the elastic or inelastic CFL number is >= 0.5.
    """
    n_x = len(x)
    n_t = len(t)

    # Handle overburden data
    if overburden_stress:
        if overburden_data is None:
            overburden_data = np.zeros(n_t)
        logger.info("Solving with overburden stress included.")
    else:
        overburden_data = np.zeros(n_t)

    cfl_elastic = k_elastic * dt / dx ** 2
    cfl_inelastic = k_inelastic * dt / dx ** 2

    logger.info("FTCS explicit solver (elastic/inelastic)")
    logger.info("  n_x=%d  x=[%.4f, %.4f]  dx=%.6f", n_x, x.min(), x.max(), dx)
    logger.info("  n_t=%d  t=[%.4f, %.4f]  dt=%.6f", n_t, t.min(), t.max(), dt)
    logger.info("  k_elastic=%.6g   CFL_elastic=%.6g", k_elastic, cfl_elastic)
    logger.info("  k_inelastic=%.6g CFL_inelastic=%.6g", k_inelastic, cfl_inelastic)

    # CFL check for BOTH elastic and inelastic
    if cfl_elastic >= 0.5:
        raise SolverError(
            f"CFL condition failed (elastic): k_elastic*dt/dx^2 = {cfl_elastic:.6f} >= 0.5. "
            "Reduce dt or increase dx."
        )
    if cfl_inelastic >= 0.5:
        raise SolverError(
            f"CFL condition failed (inelastic): k_inelastic*dt/dx^2 = {cfl_inelastic:.6f} >= 0.5. "
            "Reduce dt or increase dx."
        )

    # Preconsolidation head
    h_precons = np.zeros((n_x, n_t), order='F')
    if initial_precons and initial_condition_precons is not None:
        h_precons[:, 0] = initial_condition_precons
        logger.info("  Using preset preconsolidation stress.")
    else:
        h_precons[:, 0] = ic

    inelastic_flag = np.zeros((n_x, n_t), dtype=bool, order='F')

    # Initialise head vector and output matrix
    h = ic * np.ones(n_x)
    h[0] = bc[0, 0]
    h[-1] = bc[1, 0]

    hmat = np.zeros((n_x, n_t), order='F')
    hmat[:, 0] = h

    # Optional Numba fast-path (only for sharp switching, no smoothing)
    if HAS_NUMBA and smoothing_width == 0.0:
        logger.info("  Using Numba-accelerated FTCS EI loop.")
        hmat, h_precons, inelastic_flag = _ftcs_ei_loop_numba(
            h.copy(), hmat, bc, overburden_data,
            cfl_elastic, cfl_inelastic,
            h_precons, inelastic_flag, n_t,
        )
        logger.info("Solver complete.")
        return hmat, inelastic_flag

    # Vectorised NumPy fallback
    for j in range(1, n_t):
        _log_progress(j, n_t, "FTCS-elastic-inelastic")

        h_new = np.empty_like(h)
        h_new[0] = bc[0, j]
        h_new[-1] = bc[1, j]

        # Per-node CFL selection based on previous-step state
        if smoothing_width > 0 and j > 1:
            # Sigmoid blend: smooth transition near preconsolidation
            eff_prev = h[1:-1] - overburden_data[j - 1]
            precons_prev = h_precons[1:-1, j - 1]
            blend = 1.0 / (1.0 + np.exp((eff_prev - precons_prev) / smoothing_width))
            cfl_node = blend * cfl_inelastic + (1.0 - blend) * cfl_elastic
        else:
            cfl_node = np.where(
                inelastic_flag[1:-1, j - 1], cfl_inelastic, cfl_elastic
            )
        ob_delta = overburden_data[j] - overburden_data[j - 1]
        h_new[1:-1] = (
            h[1:-1]
            + cfl_node * (h[:-2] - 2.0 * h[1:-1] + h[2:])
            + ob_delta
        )

        h = h_new

        # Store heads
        hmat[:, j] = h

        # Vectorised preconsolidation / inelastic flag update
        effective = h - overburden_data[j]
        newly_inelastic = effective < h_precons[:, j - 1]

        if j <= n_t - 2:
            # Update preconsolidation head for next step
            idx_inel = np.where(newly_inelastic)[0]
            idx_elas = np.where(~newly_inelastic)[0]
            h_precons[idx_inel, j] = effective[idx_inel]
            h_precons[idx_elas, j] = h_precons[idx_elas, j - 1]
        # Also handle very last usable step
        if j == n_t - 1:
            # No j+1 to write to; just set flags
            pass

        inelastic_flag[:, j] = newly_inelastic

    logger.info("Solver complete.")
    return hmat, inelastic_flag


# =========================================================================
# 3. Crank-Nicolson implicit  --  single diffusivity
# =========================================================================
def solve_head_equation_crank_nicolson(
    dt: float,
    t: np.ndarray,
    dx: float,
    x: np.ndarray,
    bc: np.ndarray,
    ic: float,
    k: float,
    mass_balance_check: bool = False,
) -> np.ndarray:
    """Solve the 1-D diffusion equation with the Crank-Nicolson scheme.

    This is an implicit, unconditionally stable scheme (no CFL restriction).
    The tridiagonal system is solved in O(n) time per step via
    ``scipy.linalg.solve_banded``.

    Parameters
    ----------
    dt : float
        Time-step size.
    t : np.ndarray
        1-D array of time values, shape ``(n_t,)``.
    dx : float
        Spatial grid spacing.
    x : np.ndarray
        1-D array of spatial node positions, shape ``(n_x,)``.
    bc : np.ndarray
        Boundary conditions, shape ``(2, n_t)``.
    ic : float
        Scalar initial condition.
    k : float
        Diffusivity.

    Returns
    -------
    np.ndarray
        Head matrix of shape ``(n_x, n_t)``.
    """
    n_x = len(x)
    n_t = len(t)
    r = k * dt / dx ** 2

    logger.info("Crank-Nicolson implicit solver (single diffusivity)")
    logger.info("  n_x=%d  dx=%.6f  n_t=%d  dt=%.6f", n_x, dx, n_t, dt)
    logger.info("  k=%.6g  r=%.6g", k, r)

    # Interior nodes only (indices 1 .. n_x-2)
    m = n_x - 2  # number of interior nodes

    # Initialise
    h = ic * np.ones(n_x)
    h[0] = bc[0, 0]
    h[-1] = bc[1, 0]

    hmat = np.zeros((n_x, n_t), order='F')
    hmat[:, 0] = h

    if m == 0:
        # Only boundary nodes -- no interior to solve
        for j in range(1, n_t):
            h[0] = bc[0, j]
            h[-1] = bc[1, j]
            hmat[:, j] = h
        logger.info("Solver complete (no interior nodes).")
        return hmat

    # Build the LHS banded matrix:  (I - 0.5*r*A)
    # A is tridiagonal with  1, -2, 1  on sub/main/super diagonals.
    # LHS main diagonal:  1 + r   (since -0.5*r*(-2) = r)
    # LHS off-diagonals:  -0.5*r  (since -0.5*r*(1) = -0.5*r)
    # Banded storage for solve_banded: shape (3, m) with rows
    #   [0] = super-diagonal, [1] = main diagonal, [2] = sub-diagonal

    ab_lhs = np.zeros((3, m))
    ab_lhs[0, 1:] = -0.5 * r          # super-diagonal
    ab_lhs[1, :] = 1.0 + r            # main diagonal
    ab_lhs[2, :-1] = -0.5 * r         # sub-diagonal

    # Time-stepping
    for j in range(1, n_t):
        _log_progress(j, n_t, "CN-single")

        h_int = h[1:-1]  # interior values from previous step

        # RHS = (I + 0.5*r*A) h^n_interior  + boundary contributions
        rhs = np.empty(m)
        rhs[0] = (1.0 - r) * h_int[0] + 0.5 * r * h_int[1] + 0.5 * r * (bc[0, j - 1] + bc[0, j])
        rhs[-1] = (1.0 - r) * h_int[-1] + 0.5 * r * h_int[-2] + 0.5 * r * (bc[1, j - 1] + bc[1, j])

        if m > 2:
            rhs[1:-1] = (
                (1.0 - r) * h_int[1:-1]
                + 0.5 * r * h_int[:-2]
                + 0.5 * r * h_int[2:]
            )
        elif m == 2:
            # Both entries already filled above
            pass

        # Solve the tridiagonal system
        h_int_new = scipy.linalg.solve_banded((1, 1), ab_lhs, rhs)

        if mass_balance_check and j % max(n_t // 20, 1) == 0:
            from sub1d.diagnostics import check_cn_residual, check_mass_balance
            check_cn_residual(ab_lhs, h_int_new, rhs, j)

        h[0] = bc[0, j]
        h[-1] = bc[1, j]
        h[1:-1] = h_int_new
        hmat[:, j] = h

        if mass_balance_check and j % max(n_t // 20, 1) == 0:
            from sub1d.diagnostics import check_mass_balance
            check_mass_balance(hmat, dx, dt, k, bc, j)

    logger.info("Solver complete.")
    return hmat


# =========================================================================
# 4. Crank-Nicolson implicit  --  elastic / inelastic switching
# =========================================================================
def solve_head_equation_cn_elastic_inelastic(
    dt: float,
    t: np.ndarray,
    dx: float,
    x: np.ndarray,
    bc: np.ndarray,
    ic: float,
    k_elastic: float,
    k_inelastic: float,
    overburden_stress: bool = False,
    overburden_data: Optional[np.ndarray] = None,
    initial_precons: bool = False,
    initial_condition_precons: Optional[np.ndarray] = None,
    smoothing_width: float = 0.0,
    mass_balance_check: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Crank-Nicolson solver with elastic/inelastic diffusivity switching.

    At each time step the tridiagonal coefficients are rebuilt based on the
    per-node inelastic flag from the previous step.  The system remains
    tridiagonal so the O(n) banded solve is retained.

    Parameters
    ----------
    dt : float
        Time-step size.
    t : np.ndarray
        1-D array of time values, shape ``(n_t,)``.
    dx : float
        Spatial grid spacing.
    x : np.ndarray
        1-D array of spatial node positions, shape ``(n_x,)``.
    bc : np.ndarray
        Boundary conditions, shape ``(2, n_t)``.
    ic : float
        Scalar initial condition.
    k_elastic : float
        Elastic diffusivity.
    k_inelastic : float
        Inelastic diffusivity.
    overburden_stress : bool, optional
        Include overburden stress effects (default ``False``).
    overburden_data : np.ndarray or None, optional
        1-D array of overburden stress values, shape ``(n_t,)``.
    initial_precons : bool, optional
        Whether a preset preconsolidation stress is supplied.
    initial_condition_precons : np.ndarray or None, optional
        Initial preconsolidation head per node, shape ``(n_x,)``.

    Returns
    -------
    hmat : np.ndarray
        Head matrix of shape ``(n_x, n_t)``.
    inelastic_flag : np.ndarray
        Boolean flag array of shape ``(n_x, n_t)``.
    """
    n_x = len(x)
    n_t = len(t)
    m = n_x - 2  # interior node count

    if overburden_stress:
        if overburden_data is None:
            overburden_data = np.zeros(n_t)
        logger.info("CN solver with overburden stress included.")
    else:
        overburden_data = np.zeros(n_t)

    r_elastic = k_elastic * dt / dx ** 2
    r_inelastic = k_inelastic * dt / dx ** 2

    logger.info("Crank-Nicolson solver (elastic/inelastic)")
    logger.info("  n_x=%d  dx=%.6f  n_t=%d  dt=%.6f", n_x, dx, n_t, dt)
    logger.info("  k_elastic=%.6g  r_elastic=%.6g", k_elastic, r_elastic)
    logger.info("  k_inelastic=%.6g  r_inelastic=%.6g", k_inelastic, r_inelastic)

    # Preconsolidation head
    h_precons = np.zeros((n_x, n_t), order='F')
    if initial_precons and initial_condition_precons is not None:
        h_precons[:, 0] = initial_condition_precons
        logger.info("  Using preset preconsolidation stress.")
    else:
        h_precons[:, 0] = ic

    inelastic_flag = np.zeros((n_x, n_t), dtype=bool, order='F')

    # Initialise
    h = ic * np.ones(n_x)
    h[0] = bc[0, 0]
    h[-1] = bc[1, 0]

    hmat = np.zeros((n_x, n_t), order='F')
    hmat[:, 0] = h

    if m == 0:
        for j in range(1, n_t):
            h[0] = bc[0, j]
            h[-1] = bc[1, j]
            hmat[:, j] = h
        logger.info("Solver complete (no interior nodes).")
        return hmat, inelastic_flag

    for j in range(1, n_t):
        _log_progress(j, n_t, "CN-elastic-inelastic")

        # Per-node r value for interior nodes based on previous state
        if smoothing_width > 0 and j > 1:
            eff_prev = h[1:-1] - overburden_data[j - 1]
            precons_prev = h_precons[1:-1, j - 1]
            blend = 1.0 / (1.0 + np.exp((eff_prev - precons_prev) / smoothing_width))
            r_node = blend * r_inelastic + (1.0 - blend) * r_elastic
        else:
            r_node = np.where(
                inelastic_flag[1:-1, j - 1], r_inelastic, r_elastic
            )

        ob_delta = overburden_data[j] - overburden_data[j - 1]

        # Build LHS banded matrix: (I - 0.5 * r_i * A_i)
        # For node i:  main = 1 + r_i,  off-diag = -0.5 * r_i
        # But off-diag coupling between nodes i and i+1 uses different r.
        # We use the average of adjacent r values for the off-diagonal terms
        # to maintain symmetry, or more precisely:
        # The standard CN for spatially varying diffusivity uses:
        #   LHS:  h_i^{n+1} - 0.5*r_i*(h_{i-1}^{n+1} - 2*h_i^{n+1} + h_{i+1}^{n+1})
        # This gives:  main_i = 1 + r_i,  sub_i = -0.5*r_i,  super_i = -0.5*r_i

        ab_lhs = np.zeros((3, m))
        ab_lhs[1, :] = 1.0 + r_node                    # main diagonal
        ab_lhs[0, 1:] = -0.5 * r_node[1:]              # super-diagonal
        ab_lhs[2, :-1] = -0.5 * r_node[:-1]            # sub-diagonal

        # RHS = (I + 0.5*r_i*A) h^n_interior + boundary + overburden
        h_int = h[1:-1]
        rhs = np.empty(m)

        # General formula for interior:
        # rhs_i = (1 - r_i)*h_i + 0.5*r_i*h_{i-1} + 0.5*r_i*h_{i+1} + ob_delta
        rhs[:] = (1.0 - r_node) * h_int
        # Add contributions from neighbours
        # Left neighbour contribution
        rhs[0] += 0.5 * r_node[0] * h[0]               # h[0] is boundary
        if m > 1:
            rhs[1:] += 0.5 * r_node[1:] * h_int[:-1]
        # Right neighbour contribution
        rhs[-1] += 0.5 * r_node[-1] * h[-1]             # h[-1] is boundary
        if m > 1:
            rhs[:-1] += 0.5 * r_node[:-1] * h_int[1:]

        # Boundary contributions from new time level on LHS (move to RHS)
        rhs[0] += 0.5 * r_node[0] * bc[0, j]
        rhs[-1] += 0.5 * r_node[-1] * bc[1, j]

        # Overburden delta
        rhs += ob_delta

        # Solve tridiagonal system
        h_int_new = scipy.linalg.solve_banded((1, 1), ab_lhs, rhs)

        if mass_balance_check and j % max(n_t // 20, 1) == 0:
            from sub1d.diagnostics import check_cn_residual
            check_cn_residual(ab_lhs, h_int_new, rhs, j)

        h[0] = bc[0, j]
        h[-1] = bc[1, j]
        h[1:-1] = h_int_new
        hmat[:, j] = h

        # Preconsolidation / inelastic flag update (same logic as explicit)
        effective = h - overburden_data[j]
        newly_inelastic = effective < h_precons[:, j - 1]

        if j <= n_t - 2:
            idx_inel = np.where(newly_inelastic)[0]
            idx_elas = np.where(~newly_inelastic)[0]
            h_precons[idx_inel, j] = effective[idx_inel]
            h_precons[idx_elas, j] = h_precons[idx_elas, j - 1]

        inelastic_flag[:, j] = newly_inelastic

    logger.info("Solver complete.")
    return hmat, inelastic_flag


# =========================================================================
# 5. Compaction solver  --  elastic / inelastic
# =========================================================================
def solve_compaction_elastic_inelastic(
    hmat: np.ndarray,
    Sske: float,
    Sskv: float,
    dz: float,
    n_midpoints: int = 1,
    overburden: bool = False,
    unconfined: bool = False,
    overburden_data: float | np.ndarray = 0,
    endnodes: bool = False,
    preset_precons: bool = False,
    ic_precons: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute elastic-inelastic compaction from a solved head field.

    Parameters
    ----------
    hmat : np.ndarray
        Head matrix of shape ``(n_z, n_t)`` from one of the head solvers.
    Sske : float
        Elastic skeletal specific storage.
    Sskv : float
        Inelastic (virgin) skeletal specific storage.
    dz : float
        Vertical spacing between head nodes.
    n_midpoints : int, optional
        Interpolation multiplier.  ``n_midpoints=1`` evaluates stress at
        the standard mid-points; larger values refine the integration grid
        (default ``1``).
    overburden : bool, optional
        Whether overburden stress is included (default ``False``).
    unconfined : bool, optional
        Whether the system is unconfined (default ``False``).
    overburden_data : float or np.ndarray, optional
        Overburden time series, shape ``(n_t,)`` or scalar ``0``.
    endnodes : bool, optional
        If ``True``, skip mid-point interpolation and use *hmat* directly
        (default ``False``).
    preset_precons : bool, optional
        Whether preset preconsolidation stresses are provided.
    ic_precons : np.ndarray or None, optional
        Initial preconsolidation values per node, shape ``(n_z,)``.

    Returns
    -------
    b : np.ndarray
        1-D compaction time series of shape ``(n_t,)``.
    inelastic_flag_midpoints : np.ndarray
        Boolean flag array at mid-points, shape ``(n_mid, n_t)``.
    """
    n_z = hmat.shape[0]
    n_t = hmat.shape[1]

    logger.info(
        "Compaction solver. overburden=%s, unconfined=%s, endnodes=%s",
        overburden, unconfined, endnodes,
    )

    # ------------------------------------------------------------------
    # Mid-point interpolation (sparse weight matrix — built once, O(1) per step)
    # ------------------------------------------------------------------
    if not endnodes:
        n_interp = n_z * (2 * n_midpoints) - (2 * n_midpoints - 1)

        # Original node positions
        z_orig = np.linspace(0, (n_z - 1) * dz, n_z)

        # Fine grid positions
        z_fine = np.linspace(0, (n_z - 1) * dz, n_interp)

        if len(hmat[:, 0]) != len(z_orig):
            raise SolverError(
                f"hmat column length ({hmat.shape[0]}) does not match "
                f"expected node count ({len(z_orig)}).  Check that dz_clays "
                "is consistent with the layer thickness."
            )

        # Build sparse interpolation weight matrix W such that
        # hmat_interp = W @ hmat  (shape: n_interp x n_z)
        # For linear interp on equispaced grid, each fine point is a
        # weighted average of at most 2 coarse points.
        dz_orig = dz  # spacing between coarse nodes
        rows = []
        cols = []
        vals = []
        for fi, zf in enumerate(z_fine):
            # Find the coarse interval containing zf
            idx_lo = int(zf / dz_orig)
            idx_lo = min(idx_lo, n_z - 2)  # clamp to valid range
            idx_hi = idx_lo + 1
            frac = (zf - z_orig[idx_lo]) / dz_orig
            # Weight for lower node
            rows.append(fi)
            cols.append(idx_lo)
            vals.append(1.0 - frac)
            # Weight for upper node
            if frac > 0:
                rows.append(fi)
                cols.append(idx_hi)
                vals.append(frac)

        W = scipy.sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_interp, n_z)
        )

        # Single sparse matmul replaces the entire per-timestep loop
        hmat_interp = W @ hmat

        if n_midpoints != 1:
            hmat_midpoints = hmat_interp[1:-1, :]
        else:
            hmat_midpoints = hmat_interp[1::2, :]
    else:
        logger.info("Not using midpoints for head solver.")
        hmat_midpoints = hmat

    n_mid = hmat_midpoints.shape[0]

    # ------------------------------------------------------------------
    # Overburden stress at mid-points
    # ------------------------------------------------------------------
    if overburden:
        if unconfined:
            logger.info("Solving with overburden stress (unconfined).")
        else:
            logger.info("Solving with overburden stress (confined).")
        overburden_data_midpoints = np.tile(
            overburden_data, (n_mid, 1)
        )
    else:
        overburden_data_midpoints = np.zeros_like(hmat_midpoints)

    stress_midpoints = overburden_data_midpoints - hmat_midpoints

    # ------------------------------------------------------------------
    # Preconsolidation stress tracking
    # ------------------------------------------------------------------
    stress_midpoints_precons = np.zeros_like(hmat_midpoints)
    inelastic_flag_midpoints = np.zeros_like(hmat_midpoints, dtype=bool)

    if preset_precons and ic_precons is not None:
        logger.info("Preset preconsolidation found; interpolating to midpoints.")
        z_orig_full = np.linspace(0, (n_z - 1) * dz, n_z)
        dz_fine_full = dz / (2 * n_midpoints)
        z_fine_full = np.linspace(0, (n_z - 1) * dz, n_z * (2 * n_midpoints) - (2 * n_midpoints - 1))

        interp_precons = scipy.interpolate.interp1d(z_orig_full, ic_precons)
        ic_precons_interp = interp_precons(z_fine_full)
        ic_precons_initial = ic_precons_interp[1::2]
        logger.info("  Interpolated preconsolidation to %d midpoints.", len(ic_precons_initial))
        stress_midpoints_precons[:, 0] = ic_precons_initial
    else:
        stress_midpoints_precons[:, 0] = (
            overburden_data_midpoints[:, 0] - hmat_midpoints[:, 0]
        )

    # Vectorised loop: iterate over time, but vectorise over space
    for i in range(n_t - 1):
        _log_progress(i, n_t - 1, "Compaction-precons")

        exceeds = stress_midpoints[:, i] > stress_midpoints_precons[:, i]
        stress_midpoints_precons[exceeds, i + 1] = stress_midpoints[exceeds, i]
        stress_midpoints_precons[~exceeds, i + 1] = stress_midpoints_precons[~exceeds, i]
        inelastic_flag_midpoints[:, i] = exceeds

    inelastic_flag_midpoints = np.array(inelastic_flag_midpoints, dtype=bool)

    # ------------------------------------------------------------------
    # Adjust dz for refined grid
    # ------------------------------------------------------------------
    if n_midpoints != 1:
        dz_eff = dz / (2 * n_midpoints)
    else:
        dz_eff = dz

    # ------------------------------------------------------------------
    # Compute compaction time series (fully vectorised)
    # ------------------------------------------------------------------
    # Sum over spatial axis for each timestep
    precons_col_sum = stress_midpoints_precons.sum(axis=0)  # shape (n_t,)
    precons_init_sum = stress_midpoints_precons[:, 0].sum()  # scalar
    stress_col_sum = stress_midpoints.sum(axis=0)            # shape (n_t,)

    b = -dz_eff * (
        Sskv * (precons_col_sum - precons_init_sum)
        - Sske * (precons_col_sum - stress_col_sum)
    )
    b[0] = 0.0

    logger.info("Compaction solver complete.")
    return b, inelastic_flag_midpoints


# =========================================================================
# 6. Adaptive timestepping explicit solver
# =========================================================================
def solve_head_equation_adaptive(
    bc_func,
    t_start: float,
    t_end: float,
    dt_init: float,
    dx: float,
    x: np.ndarray,
    ic: float,
    k: float,
    tol: float = 1e-4,
    dt_min: float = 1e-10,
    dt_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1-D diffusion equation with adaptive timestepping.

    Uses Richardson error estimation: run each step at ``dt`` and ``dt/2``,
    compare, and adjust step size.  The CFL condition is enforced as an
    upper bound on ``dt``.

    Parameters
    ----------
    bc_func : callable
        ``bc_func(t) -> (h_top, h_bot)`` returning boundary values at time *t*.
    t_start, t_end : float
        Simulation time window.
    dt_init : float
        Initial timestep guess.
    dx : float
        Spatial grid spacing.
    x : np.ndarray
        Spatial node positions, shape ``(n_x,)``.
    ic : float
        Uniform initial condition.
    k : float
        Diffusivity.
    tol : float
        Error tolerance for Richardson extrapolation.
    dt_min : float
        Minimum allowed timestep (raises SolverError if reached).
    dt_max : float or None
        Maximum allowed timestep (default: CFL limit ``0.45 * dx**2 / k``).

    Returns
    -------
    t_out : np.ndarray
        1-D array of actual time values used.
    hmat : np.ndarray
        Head matrix of shape ``(n_x, n_steps)``.

    Raises
    ------
    SolverError
        If the timestep shrinks below ``dt_min``.
    """
    n_x = len(x)
    cfl_limit = 0.45 * dx**2 / k
    if dt_max is None:
        dt_max = cfl_limit
    dt_max = min(dt_max, cfl_limit)

    logger.info("Adaptive FTCS solver: t=[%.4f, %.4f], dx=%.6f, k=%.6g", t_start, t_end, dx, k)
    logger.info("  dt_init=%.6g, tol=%.3e, dt_max=%.6g", dt_init, tol, dt_max)

    def _ftcs_step(h, dt_step, t_now):
        """Single FTCS step."""
        cfl = k * dt_step / dx**2
        h_top, h_bot = bc_func(t_now + dt_step)
        h_new = np.empty_like(h)
        h_new[0] = h_top
        h_new[-1] = h_bot
        h_new[1:-1] = h[1:-1] + cfl * (h[:-2] - 2.0 * h[1:-1] + h[2:])
        return h_new

    # Initialise
    h = ic * np.ones(n_x)
    h_top0, h_bot0 = bc_func(t_start)
    h[0] = h_top0
    h[-1] = h_bot0

    t_list = [t_start]
    h_list = [h.copy()]

    t_now = t_start
    dt = min(dt_init, dt_max)

    while t_now < t_end - 1e-15:
        # Don't overshoot
        if t_now + dt > t_end:
            dt = t_end - t_now

        # Enforce CFL
        dt = min(dt, dt_max)

        # Full step
        h_full = _ftcs_step(h, dt, t_now)

        # Two half-steps
        h_half1 = _ftcs_step(h, dt / 2, t_now)
        h_half2 = _ftcs_step(h_half1, dt / 2, t_now + dt / 2)

        # Error estimate (Richardson: 2nd order scheme)
        err = np.max(np.abs(h_half2 - h_full)) / 3.0

        if err <= tol or dt <= dt_min:
            # Accept step (use the more accurate half-step result)
            h = h_half2
            t_now += dt
            t_list.append(t_now)
            h_list.append(h.copy())

            # Try to grow dt
            if err > 0:
                dt_new = dt * min(2.0, max(0.5, 0.9 * (tol / err) ** 0.5))
            else:
                dt_new = dt * 2.0
            dt = min(dt_new, dt_max)
        else:
            # Reject step, shrink dt
            dt_new = dt * max(0.25, 0.9 * (tol / err) ** 0.5)
            if dt_new < dt_min:
                raise SolverError(
                    f"Adaptive solver: timestep {dt_new:.3e} fell below "
                    f"dt_min={dt_min:.3e} at t={t_now:.6f}."
                )
            dt = dt_new

    t_out = np.array(t_list)
    hmat = np.column_stack(h_list).T.T  # shape (n_x, n_steps)
    # h_list is a list of (n_x,) arrays; stack into (n_x, n_steps)
    hmat = np.array(h_list).T

    logger.info("Adaptive solver complete: %d steps taken.", len(t_list))
    return t_out, hmat
