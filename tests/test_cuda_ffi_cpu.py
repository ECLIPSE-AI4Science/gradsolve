"""CPU smoke for the ``cuda_tsit5`` ``jax.ffi`` bridge — no nvcc, no GPU.

``gradsolve/cuda/_build.py::build_cpu_so`` compiles the FFI handler as a HOST C++ shared lib
(``-DGRADSOLVE_FFI``, the handler loops on the CPU), so the full ``jax.ffi`` bridge and its
numerics are exercisable on a machine with only a host C++ compiler. This checks, in a
minimal-dependency CI, that the FFI bridge builds and runs (the CUDA path is the same handler,
compiled with nvcc on a GPU machine).

The smoke builds the host ``.so`` for the ``linear_<D>`` field (dy/dt = 1.01*y, analytic
y(t) = y0 * exp(1.01*t)), invokes it through ``_ffi_bridge.solve_ffi(device="cpu")``, and
checks the result is finite and matches the closed form. Skips (does not fail) when no host
C++ compiler is on PATH.

Imports only ``gradsolve`` (+ numpy/pytest).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

# Skip cleanly on a box with no host C++ compiler (the only environmental prerequisite the
# CPU FFI build has — jax.ffi headers come from jaxlib, which gradsolve already depends on).
if shutil.which("c++") is None and shutil.which("clang++") is None and shutil.which("g++") is None:
    pytest.skip("no host C++ compiler (c++/clang++/g++) for the CPU FFI build",
                allow_module_level=True)

from gradsolve.cuda import _build  # noqa: E402
from gradsolve.cuda._ffi_bridge import solve_ffi  # noqa: E402


class _LinearLadder:
    """dy/dt = 1.01*y elementwise (param ignored). ``name`` routes to the ``linear_<D>``
    registered field, so ``_ffi_bridge`` has a codegen'd handler to compile + call."""

    dim = 2
    name = "linear_ladder_2"
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, p):  # present for the Problem contract; unused by the FFI lane
        return 1.01 * y


def test_build_cpu_so_produces_a_shared_lib():
    so = _build.build_cpu_so("linear_2", 2, "float64")
    assert so.endswith(".so")
    from pathlib import Path
    assert Path(so).exists(), f"build_cpu_so returned a missing path: {so}"


def test_cpu_ffi_forward_matches_analytic():
    prob = _LinearLadder()
    n = 3
    y0 = np.array([[1.0, 2.0], [0.5, 1.5], [3.0, 0.25]], dtype=np.float64)
    # linear_<D> ignores its scalar param; a length-n (n, 1) param array is the right shape.
    params = np.ones((n, 1), dtype=np.float64)

    yf, nsteps = solve_ffi(prob, y0, params, rtol=1e-8, atol=1e-10, device="cpu")

    assert yf.shape == (n, prob.dim), yf.shape
    assert bool(np.all(np.isfinite(yf))), yf
    assert nsteps.shape == (n,)
    assert bool(np.all(np.asarray(nsteps) > 0)), nsteps

    analytic = y0 * np.exp(1.01 * (prob.t1 - prob.t0))
    rel_err = float(np.max(np.abs(yf - analytic) / np.abs(analytic)))
    assert rel_err < 1e-5, f"CPU FFI forward vs analytic rel err too large: {rel_err:.3e}"
