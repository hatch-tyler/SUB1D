"""Performance benchmarks: legacy vs new solvers.

Compares wall-clock time for:
1. Single-value FTCS head solver
2. Elastic-inelastic FTCS head solver
3. Compaction solver

Each is run at multiple problem sizes (n_x nodes x n_t timesteps).
Legacy solvers use pure Python inner loops; new solvers use vectorised NumPy.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import types

import numpy as np

# ---------------------------------------------------------------------------
# Load legacy solver
# ---------------------------------------------------------------------------
_LEGACY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "legacy")
)


def _load_legacy_solver():
    utils_shim = types.ModuleType("utils")
    utils_shim.printProgressBar = lambda *args, **kwargs: None
    saved = sys.modules.get("utils")
    sys.modules["utils"] = utils_shim

    spec = importlib.util.spec_from_file_location(
        "solver_legacy",
        os.path.join(_LEGACY_DIR, "solver_legacy.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if saved is None:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = saved
    return module


legacy = _load_legacy_solver()

# Add project root to path for new solver imports
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir)))

from sub1d.solver import (
    solve_head_equation_single,
    solve_head_equation_elastic_inelastic,
    solve_compaction_elastic_inelastic,
)

# Suppress logging from new solvers during benchmarks
import logging
logging.getLogger("sub1d").setLevel(logging.WARNING)

# Suppress print output from legacy solvers
import io


class SuppressStdout:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def _time_fn(fn, n_runs=1):
    """Time a function, return median wall-clock seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def _setup_head(n_x, n_t, k):
    """Create common inputs for head solvers."""
    L = 10.0
    dx = L / (n_x - 1)
    x = np.linspace(0, L, n_x)
    dt = 0.4 * dx ** 2 / k
    t = np.arange(n_t) * dt
    ic = 50.0
    bc = np.zeros((2, n_t))
    bc[0, :] = np.linspace(50.0, 40.0, n_t)
    bc[1, :] = np.linspace(50.0, 40.0, n_t)
    return dt, t, dx, x, bc, ic


def _setup_compaction(n_z, n_t):
    """Create common inputs for compaction solver."""
    hmat = np.zeros((n_z, n_t))
    for i in range(n_z):
        hmat[i, :] = np.linspace(50.0, 40.0, n_t)
    dz = 0.5
    Sske = 1e-4
    Sskv = 1e-3
    return hmat, dz, Sske, Sskv


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
SEPARATOR = "-" * 78

# Problem sizes: (n_x, n_t, description)
HEAD_SIZES = [
    (21, 1_000, "small (21 nodes, 1K steps)"),
    (51, 5_000, "medium (51 nodes, 5K steps)"),
    (101, 10_000, "large (101 nodes, 10K steps)"),
    (101, 50_000, "xlarge (101 nodes, 50K steps)"),
]

COMPACTION_SIZES = [
    (21, 1_000, "small (21 nodes, 1K steps)"),
    (51, 5_000, "medium (51 nodes, 5K steps)"),
    (101, 10_000, "large (101 nodes, 10K steps)"),
    (101, 50_000, "xlarge (101 nodes, 50K steps)"),
]


def bench_single_head():
    """Benchmark: single-value FTCS head solver."""
    print("\n" + SEPARATOR)
    print("BENCHMARK 1: Single-value FTCS head solver")
    print(SEPARATOR)
    print(f"{'Size':>40s}  {'Legacy':>10s}  {'New':>10s}  {'Speedup':>8s}")
    print(f"{'':>40s}  {'(sec)':>10s}  {'(sec)':>10s}  {'':>8s}")
    print("-" * 78)

    k = 0.05
    results = []

    for n_x, n_t, desc in HEAD_SIZES:
        dt, t, dx, x, bc, ic = _setup_head(n_x, n_t, k)

        with SuppressStdout():
            t_legacy = _time_fn(
                lambda: legacy.solve_head_equation_singlevalue(dt, t, dx, x, bc, ic, k)
            )

        t_new = _time_fn(
            lambda: solve_head_equation_single(dt, t, dx, x, bc, ic, k)
        )

        speedup = t_legacy / t_new if t_new > 0 else float("inf")
        results.append((desc, t_legacy, t_new, speedup))
        print(f"{desc:>40s}  {t_legacy:10.4f}  {t_new:10.4f}  {speedup:7.1f}x")

    return results


def bench_elastic_inelastic_head():
    """Benchmark: elastic-inelastic FTCS head solver."""
    print("\n" + SEPARATOR)
    print("BENCHMARK 2: Elastic-inelastic FTCS head solver")
    print(SEPARATOR)
    print(f"{'Size':>40s}  {'Legacy':>10s}  {'New':>10s}  {'Speedup':>8s}")
    print(f"{'':>40s}  {'(sec)':>10s}  {'(sec)':>10s}  {'':>8s}")
    print("-" * 78)

    k_e, k_i = 0.05, 0.02
    results = []

    for n_x, n_t, desc in HEAD_SIZES:
        dt, t, dx, x, bc, ic = _setup_head(n_x, n_t, max(k_e, k_i))

        with SuppressStdout():
            t_legacy = _time_fn(
                lambda: legacy.solve_head_equation_elasticinelastic(
                    dt, t, dx, x, bc, ic, k_e, k_i
                )
            )

        t_new = _time_fn(
            lambda: solve_head_equation_elastic_inelastic(
                dt, t, dx, x, bc, ic, k_e, k_i
            )
        )

        speedup = t_legacy / t_new if t_new > 0 else float("inf")
        results.append((desc, t_legacy, t_new, speedup))
        print(f"{desc:>40s}  {t_legacy:10.4f}  {t_new:10.4f}  {speedup:7.1f}x")

    return results


def bench_compaction():
    """Benchmark: compaction solver."""
    print("\n" + SEPARATOR)
    print("BENCHMARK 3: Compaction solver (elastic-inelastic)")
    print(SEPARATOR)
    print(f"{'Size':>40s}  {'Legacy':>10s}  {'New':>10s}  {'Speedup':>8s}")
    print(f"{'':>40s}  {'(sec)':>10s}  {'(sec)':>10s}  {'':>8s}")
    print("-" * 78)

    results = []

    for n_z, n_t, desc in COMPACTION_SIZES:
        hmat, dz, Sske, Sskv = _setup_compaction(n_z, n_t)

        with SuppressStdout():
            t_legacy = _time_fn(
                lambda: legacy.subsidence_solver_aquitard_elasticinelastic(
                    hmat, Sske, Sskv, dz
                )
            )

        t_new = _time_fn(
            lambda: solve_compaction_elastic_inelastic(hmat, Sske, Sskv, dz)
        )

        speedup = t_legacy / t_new if t_new > 0 else float("inf")
        results.append((desc, t_legacy, t_new, speedup))
        print(f"{desc:>40s}  {t_legacy:10.4f}  {t_new:10.4f}  {speedup:7.1f}x")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("SUB1D PERFORMANCE BENCHMARKS")
    print("Legacy (pure Python loops) vs New (vectorised NumPy)")
    print("=" * 78)
    print(f"NumPy version: {np.__version__}")
    print(f"Python: {sys.version}")

    try:
        from numba import __version__ as numba_ver
        print(f"Numba: {numba_ver} (JIT acceleration available)")
    except ImportError:
        print("Numba: not installed (pure NumPy path)")

    all_results = {}

    all_results["single_head"] = bench_single_head()
    all_results["elastic_inelastic_head"] = bench_elastic_inelastic_head()
    all_results["compaction"] = bench_compaction()

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, results in all_results.items():
        speedups = [r[3] for r in results]
        print(f"  {name:30s}  min={min(speedups):.1f}x  max={max(speedups):.1f}x  "
              f"median={np.median(speedups):.1f}x")

    print("\nNotes:")
    print("  - Speedup = legacy_time / new_time")
    print("  - Legacy uses pure Python for-loops over spatial nodes")
    print("  - New uses vectorised NumPy (single array operation per timestep)")
    print("  - The elastic-inelastic solver also vectorises the per-node CFL selection")
    print("  - The compaction solver vectorises both the preconsolidation tracking")
    print("    and the midpoint interpolation")
