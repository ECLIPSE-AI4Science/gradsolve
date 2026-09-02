"""gradsolve — differentiable ensemble solvers for differential equations on GPUs, in JAX.

Structured fixed-step ``lax.scan`` solvers (RK4 / Tsit5 / IMEX), the fused NVIDIA-Warp
adaptive + Rosenbrock kernels with their record-and-replay adjoint, the
``SolveResult``/``Backend``/``Problem`` contract, and the ``choose_engine`` decision map.

Importing this package enables float64 in JAX (accuracy work assumes double precision),
unless ``GRADSOLVE_X64=0`` is set before import, which keeps JAX at float32.
"""

from __future__ import annotations

import os

# Enable float64 in JAX as early as possible (must precede any jax array creation), unless the
# caller opted into float32 via GRADSOLVE_X64=0.
_X64 = os.environ.get("GRADSOLVE_X64", "1") != "0"
try:
    import jax

    jax.config.update("jax_enable_x64", _X64)
except Exception:  # pragma: no cover - jax is a required dependency, always importable
    pass

__version__ = "0.2.1"

# Routed public API (imported last: api.py pulls in the engine roster, which needs x64 set).
from gradsolve.api import grad_closure, solve  # noqa: E402


def register_jax_field(name, f_jax, dim, n_params, *, stiff=False):
    """Register a translated ``f_jax(t, y, p)`` into the fused-kernel registries so a
    ``Problem`` of that ``name`` routes to the fused engines like a built-in.

    Thin lazy wrapper over ``gradsolve.warp.jax_field.register_jax_field``: the import is
    deferred to the first call because ``jax_field`` compiles through NVIDIA Warp, so
    ``import gradsolve`` works without Warp installed. See that function for the full
    contract.
    """
    from gradsolve.warp.jax_field import register_jax_field as _impl

    return _impl(name, f_jax, dim, n_params, stiff=stiff)


__all__ = ["solve", "grad_closure", "register_jax_field"]
