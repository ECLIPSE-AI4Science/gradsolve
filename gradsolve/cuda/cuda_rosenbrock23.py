"""cuda_rosenbrock23 — hand-written CUDA forward-only fused adaptive Rosenbrock23 (ode23s)
stiff engine (``jax.ffi``). The stiff analogue of ``cuda_tsit5``.

Why: the fused Warp stiff kernel ``warp_rosenbrock`` pays a fixed per-trajectory
codegen-startup cost on the forward-only stiff solve that a hand-written kernel avoids.
This kernel integrates one trajectory per thread with the Rosenbrock23 pair of Shampine &
Reichelt (1997), shared with the JAX solvers in ``gradsolve/solvers/``, using an in-kernel
analytic Jacobian; W = I - d*h*J is factored once per step (partial-pivot LU) and reused
across the three stage solves.

**Forward-only by design**: no ``_api_reverse`` -> ``grad_closure(engine="cuda_rosenbrock23")``
raises "no reverse closure" (the contract). The reverse stiff lane stays on the record-and-
replay ``rodas5p_replay`` / ``warp_rosenbrock`` adjoint.

Status: disabled by default. ``dispatch.CUDA_ROSENBROCK23_ENABLED=False``, so the stiff +
low-NVAR + forward cell still routes to ``warp_rosenbrock`` and no existing routing changes;
this engine is reachable only by name (``engine="cuda_rosenbrock23"``).
``supports()`` is False on any machine without nvcc, so ``solve()`` is never reached there
and the router falls back.
"""
from __future__ import annotations

import shutil

from gradsolve.cuda._ffi_bridge import _stiff_field_for

name = "cuda_rosenbrock23"

#: Best-effort CUDA availability for routing / ``supports()``: nvcc on PATH (False
#: without nvcc -> ``supports()`` False -> the router falls back, no raise). The real
#: nvcc-major vs ``jax.devices()`` version-check happens in ``_build._check_cuda_jax_match``
#: before any GPU compile.
_CUDA_AVAILABLE = shutil.which("nvcc") is not None


def supports(problem) -> bool:
    """True iff nvcc is available (so the CUDA kernel can be built and run) and the problem is
    a registered stiff field (robertson, hires, linstiff_<D>). False without nvcc -> the
    router falls back to the general stiff engine (no raise), exactly like
    ``warp_rosenbrock.supports`` without Warp."""
    return _CUDA_AVAILABLE and _stiff_field_for(problem) is not None


def solve(problem, y0, params, *, rtol=1e-6, atol=1e-9, device="cuda"):
    """Forward-only fused adaptive Rosenbrock23 -> ``SolveResult`` via the codegen'd
    ``jax.ffi`` handler (``_ffi_bridge.solve_ffi_rosenbrock``: compiles the stiff kernel for
    the device's platform, runs it, returns final states + accepted-step counts).
    ``device="cuda"`` = the production GPU path; ``device="cpu"`` builds the host-loop FFI
    target (the CPU validation path — no nvcc needed). ``rejected_steps`` is reported as zeros
    (the kernel tracks accepted steps only)."""
    import numpy as np

    from gradsolve.base import SolveResult
    from gradsolve.cuda._ffi_bridge import solve_ffi_rosenbrock

    yf, nsteps = solve_ffi_rosenbrock(problem, y0, params, rtol=rtol, atol=atol,
                                      device=device)
    n = int(yf.shape[0])
    return SolveResult(
        y_final=yf,
        accepted_steps=np.asarray(nsteps),
        rejected_steps=np.zeros(n, dtype=np.int64),
        solver=name,
    )
