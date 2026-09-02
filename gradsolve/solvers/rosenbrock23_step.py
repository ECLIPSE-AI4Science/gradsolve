"""Single pure-JAX Rosenbrock23 (ode23s) step — the replay core of the stiff family.

The stiff counterpart of ``gradsolve/solvers/tsit5_step.py``. Used by the
record-and-replay adjoint (``gradsolve/warp/warp_replay.py::
make_rosenbrock_replay_closure``) to re-integrate a recorded accepted-dt sequence with
exactly the Rosenbrock23 arithmetic the fused Warp kernel ran — but in pure JAX, so the
whole replay is reverse-differentiable by construction (the Warp kernel needs no backward
pass).

Method core (Shampine & Reichelt 1997, ode23s; autonomous specialization — the d(f)/dt
term vanishes; the built-in stiff problems are autonomous). For ``dy/dt = f(t, y, p)`` with
``J = df/dy`` (here ``jax.jacfwd(f, argnums=1)`` — exact AD, no finite-difference):

    d   = 1 / (2 + sqrt(2)) ;  e32 = 6 + sqrt(2)
    W   = I - dt*d*J
    F0  = f(t, y) ;            k1 = solve(W, F0)
    F1  = f(t, y + 0.5*dt*k1); k2 = solve(W, F1 - k1) + k1
    ynew = y + dt*k2

The embedded 3rd-order error estimate is not computed here — the replay only needs the
propagated order-2 solution (``ynew``); the step sizes are frozen data from the recorded
adaptive run, so no controller / error norm is evaluated in the replay.

Padding contract for the replay adjoint: ``dt == 0`` is exactly an identity step in both
value and gradient. With ``dt = 0``: ``W = I``, so ``k1 = F0``, ``k2 = (F1 - k1) + k1 =
F1``, and ``ynew = y + 0*k2 = y``. ``d ynew/dy = I`` and ``d ynew/dp = 0`` (the only dt
factor multiplying any p-dependence is zero), so values and gradients flow correctly
through the zero-padded tail of the recorded dt array — identical to ``tsit5_step``'s
contract.

Pure JAX (jit/grad/vmap-safe). float64 (gradsolve enables jax x64 at import).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Rosenbrock23 coefficients (autonomous ode23s). Single source: the replay step here and
# the hand-CUDA cuda_rosenbrock23 kernel both read d (and e32) from THIS module, so the
# coefficient is never scattered as a magic literal. e32 is only used by the full adaptive
# solver / the kernel's embedded 3rd-order error estimate (the replay needs the propagated
# order-2 solution only, so it does not use e32).
_D_COEF = 1.0 / (2.0 + jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64)))
_E32_COEF = 6.0 + jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))


def rosenbrock23_step(f, t, y, dt, p):
    """One Rosenbrock23 step: ``y(t) -> y(t + dt)`` for ``dy/dt = f(t, y, p)``.

    ``f`` is the problem RHS (``problem.f_jax``); the Jacobian is taken by
    ``jax.jacfwd(f, argnums=1)`` at ``(t, y, p)`` (exact forward-mode AD). The two
    W-solves use ``jnp.linalg.solve`` (LU). ``dt == 0`` is an exact identity step (see
    the module docstring). Traceable under jit/grad/vmap.
    """
    D = y.shape[-1]
    eye = jnp.eye(D, dtype=y.dtype)
    dcoef = jnp.asarray(_D_COEF, dtype=y.dtype)

    J = jax.jacfwd(f, argnums=1)(t, y, p)  # (D, D) df/dy at (t, y, p)
    W = eye - dt * dcoef * J

    F0 = f(t, y, p)
    k1 = jnp.linalg.solve(W, F0)
    F1 = f(t, y + 0.5 * dt * k1, p)
    k2 = jnp.linalg.solve(W, F1 - k1) + k1
    return y + dt * k2
