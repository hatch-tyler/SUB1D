"""Diagnostics for SUB1D solvers: mass balance, CN residual, Richardson error.

Functions
---------
check_mass_balance
    Verify boundary flux vs storage change for the diffusion equation.
check_cn_residual
    Verify that the Crank-Nicolson banded solve residual is small.
estimate_spatial_error
    Richardson extrapolation to estimate spatial discretisation error.
"""

import logging

import numpy as np
import scipy.linalg

logger = logging.getLogger(__name__)


def check_mass_balance(
    hmat: np.ndarray,
    dx: float,
    dt: float,
    k: float,
    bc: np.ndarray,
    step: int,
    threshold: float = 1e-6,
) -> float:
    """Check mass balance at a given timestep.

    For the 1-D diffusion equation  dh/dt = k * d²h/dx², the discrete
    mass balance is:

        storage_change  = dx * sum(h^{n+1} - h^n)
        boundary_flux   = k * dt * [(h[1]-h[0])/dx - (h[-1]-h[-2])/dx]

    The relative imbalance is ``|storage - flux| / max(|storage|, |flux|, eps)``.

    Parameters
    ----------
    hmat : np.ndarray
        Head matrix, shape ``(n_x, n_t)``.
    dx, dt, k : float
        Grid spacing, timestep, and diffusivity.
    bc : np.ndarray
        Boundary conditions, shape ``(2, n_t)``.
    step : int
        Timestep index to check (must be >= 1).
    threshold : float
        Warning threshold for relative imbalance.

    Returns
    -------
    float
        Relative mass-balance error at the given step.
    """
    h_new = hmat[:, step]
    h_old = hmat[:, step - 1]

    storage_change = dx * np.sum(h_new - h_old)

    # Boundary fluxes (positive = into domain)
    flux_left = k * dt * (h_new[1] - h_new[0]) / dx
    flux_right = k * dt * (h_new[-2] - h_new[-1]) / dx
    boundary_flux = flux_left + flux_right

    denom = max(abs(storage_change), abs(boundary_flux), 1e-30)
    rel_error = abs(storage_change - boundary_flux) / denom

    if rel_error > threshold:
        logger.warning(
            "Mass balance warning at step %d: relative error = %.3e "
            "(threshold %.1e). storage=%.6e, flux=%.6e",
            step, rel_error, threshold, storage_change, boundary_flux,
        )

    return rel_error


def check_cn_residual(
    ab_lhs: np.ndarray,
    h_int_new: np.ndarray,
    rhs: np.ndarray,
    step: int,
    threshold: float = 1e-10,
) -> float:
    """Check Crank-Nicolson solve residual: ``||A*x - b|| / ||b||``.

    Parameters
    ----------
    ab_lhs : np.ndarray
        Banded LHS matrix, shape ``(3, m)``.
    h_int_new : np.ndarray
        Computed interior head solution, shape ``(m,)``.
    rhs : np.ndarray
        Right-hand side vector, shape ``(m,)``.
    step : int
        Timestep index (for logging).
    threshold : float
        Warning threshold for relative residual.

    Returns
    -------
    float
        Relative residual ``||A*x - b||_2 / ||b||_2``.
    """
    # Reconstruct A*x using the banded storage
    m = len(h_int_new)
    ax = np.empty(m)

    # Main diagonal
    ax[:] = ab_lhs[1, :] * h_int_new

    # Super-diagonal (ab_lhs[0, 1:] couples to h_int_new[1:])
    if m > 1:
        ax[:-1] += ab_lhs[0, 1:] * h_int_new[1:]
        # Sub-diagonal (ab_lhs[2, :-1] couples to h_int_new[:-1])
        ax[1:] += ab_lhs[2, :-1] * h_int_new[:-1]

    residual = np.linalg.norm(ax - rhs, 2)
    rhs_norm = np.linalg.norm(rhs, 2)
    rel_residual = residual / max(rhs_norm, 1e-30)

    if rel_residual > threshold:
        logger.warning(
            "CN residual warning at step %d: relative residual = %.3e "
            "(threshold %.1e)",
            step, rel_residual, threshold,
        )

    return rel_residual


def estimate_spatial_error(
    solver_func,
    solver_kwargs: dict,
    dz: float,
) -> dict:
    """Estimate spatial discretisation error via Richardson extrapolation.

    Runs the solver at resolution ``dz`` and ``dz/2``, then estimates the
    error assuming second-order spatial accuracy (FTCS and CN are both
    second-order in space).

    Parameters
    ----------
    solver_func : callable
        One of the head solver functions (e.g. ``solve_head_equation_single``).
    solver_kwargs : dict
        Keyword arguments for the solver.  Must include ``dx``, ``x``, and
        ``bc``.  These will be modified for the refined run.
    dz : float
        Coarse grid spacing.

    Returns
    -------
    dict
        ``{"coarse": hmat_coarse, "fine": hmat_fine, "error_estimate": float,
          "max_pointwise_error": float}``
    """
    import copy

    # Coarse run
    kwargs_coarse = copy.deepcopy(solver_kwargs)
    result_coarse = solver_func(**kwargs_coarse)
    if isinstance(result_coarse, tuple):
        hmat_coarse = result_coarse[0]
    else:
        hmat_coarse = result_coarse

    # Fine run at dz/2
    kwargs_fine = copy.deepcopy(solver_kwargs)
    dx_fine = dz / 2.0
    n_x_fine = 2 * (len(solver_kwargs["x"]) - 1) + 1
    x_fine = np.linspace(solver_kwargs["x"][0], solver_kwargs["x"][-1], n_x_fine)

    # Refine boundary conditions (interpolate to fine time grid — same times)
    bc_coarse = solver_kwargs["bc"]
    n_t = bc_coarse.shape[1]

    kwargs_fine["dx"] = dx_fine
    kwargs_fine["x"] = x_fine
    kwargs_fine["bc"] = bc_coarse  # BCs don't change shape (still 2 x n_t)

    # Adjust dt for CFL if needed (for explicit solvers)
    if "k" in kwargs_fine:
        k = kwargs_fine["k"]
        cfl_fine = k * kwargs_fine["dt"] / dx_fine**2
        if cfl_fine >= 0.5:
            kwargs_fine["dt"] = 0.4 * dx_fine**2 / k
            n_t_fine = int((solver_kwargs["t"][-1] - solver_kwargs["t"][0]) / kwargs_fine["dt"]) + 1
            kwargs_fine["t"] = np.arange(n_t_fine) * kwargs_fine["dt"] + solver_kwargs["t"][0]
            # Resize BCs
            bc_new = np.zeros((2, n_t_fine))
            bc_new[0, :] = np.interp(kwargs_fine["t"], solver_kwargs["t"], bc_coarse[0, :])
            bc_new[1, :] = np.interp(kwargs_fine["t"], solver_kwargs["t"], bc_coarse[1, :])
            kwargs_fine["bc"] = bc_new

    result_fine = solver_func(**kwargs_fine)
    if isinstance(result_fine, tuple):
        hmat_fine = result_fine[0]
    else:
        hmat_fine = result_fine

    # Compare at coarse grid points (every other fine node) at final time
    hmat_fine_at_coarse = hmat_fine[::2, -1]
    hmat_coarse_final = hmat_coarse[:, -1]

    # Trim to common length (in case n_t differs)
    n_common = min(len(hmat_fine_at_coarse), len(hmat_coarse_final))
    pointwise_diff = np.abs(hmat_fine_at_coarse[:n_common] - hmat_coarse_final[:n_common])
    max_error = np.max(pointwise_diff)

    # Richardson error estimate (2nd order): error ~ (4/3) * |coarse - fine|
    error_estimate = (4.0 / 3.0) * max_error

    logger.info(
        "Richardson extrapolation: max pointwise diff = %.3e, "
        "estimated error = %.3e",
        max_error, error_estimate,
    )

    return {
        "coarse": hmat_coarse,
        "fine": hmat_fine,
        "error_estimate": error_estimate,
        "max_pointwise_error": max_error,
    }
