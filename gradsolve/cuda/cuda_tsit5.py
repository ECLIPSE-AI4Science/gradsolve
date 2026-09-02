"""cuda_tsit5 — hand-written CUDA forward-only fused adaptive Tsit5 engine (``jax.ffi``).

The fast forward-throughput lane: a hand-written CUDA kernel that integrates one
trajectory per thread with the Tsit5 tableau of Tsitouras (2011), shared with the JAX
solvers in ``gradsolve/solvers/``. **Forward-only by design**: the register residency
that makes it fast breaks the adjoint, so there is no ``_api_reverse`` and
``grad_closure(engine="cuda_tsit5")`` raises "no reverse closure" (that is the
contract). The record-and-replay reverse lane stays on ``warp_replay``.

Routing: reachable via ``engine="cuda_tsit5"`` and, when ``dispatch.CUDA_TSIT5_ENABLED``
is set, as the nonstiff low-NVAR forward-only cell in ``choose_engine`` (otherwise that
cell routes to ``warp_ode``). ``supports()`` is False on any machine without nvcc, so
``solve()`` is never reached there (the router falls back to a scan engine) and behavior is
unchanged.
"""
from __future__ import annotations

import shutil

# Reuse warp_ode's problem->field routing (single source; a pure name-parser, warp-free at
# import). The registered nonstiff low-NVAR forward set: lorenz/diffeqgpu_lorenz, vdp,
# linear_ladder_<D> (lorenz96_<D> too). _param_of/_rho_of are wired in with the solve path.
from gradsolve.warp.warp_ode import _field_for

name = "cuda_tsit5"

#: Best-effort CUDA availability for routing/``supports()``. Proxy: nvcc on PATH
#: (False without nvcc -> ``supports()`` False -> the router falls back, no raise).
#: The build step (``_build.py``) refines this with the real nvcc-major vs ``jax.devices()``
#: version-check before any compile.
_CUDA_AVAILABLE = shutil.which("nvcc") is not None


def supports(problem) -> bool:
    """True iff nvcc is available (so the CUDA kernel can be built and run) and the problem
    has a registered field. False without nvcc -> the router falls back to a scan engine (no
    raise), exactly like ``warp_ode.supports`` without Warp."""
    return _CUDA_AVAILABLE and _field_for(problem) is not None


def solve(problem, y0, params, *, rtol=1e-6, atol=1e-9, device="cuda"):
    """Forward-only fused adaptive Tsit5 -> ``SolveResult`` via the codegen'd ``jax.ffi``
    handler (``_ffi_bridge.solve_ffi``: compiles the kernel for the device's platform, runs
    it, returns final states + accepted-step counts). ``device="cuda"`` = the production GPU
    path; ``device="cpu"`` builds the host-loop FFI target (validation). Reached
    only when ``supports()`` is True (a machine with nvcc); the router otherwise falls
    back. rejected_steps is reported as zeros (the kernel tracks accepted steps only)."""
    import numpy as np

    from gradsolve.base import SolveResult
    from gradsolve.cuda._ffi_bridge import solve_ffi

    yf, nsteps = solve_ffi(problem, y0, params, rtol=rtol, atol=atol, device=device)
    n = int(yf.shape[0])
    return SolveResult(
        y_final=yf,
        accepted_steps=np.asarray(nsteps),
        rejected_steps=np.zeros(n, dtype=np.int64),
        solver=name,
    )
