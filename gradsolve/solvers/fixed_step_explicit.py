"""Fixed-step explicit Runge-Kutta (classical RK4) — the *non-stiff* spine member.

Sibling of ``gradsolve/solvers/fixed_step_imex.py``: a *fixed* number of steps run as one
``jax.lax.scan``, vmapped over the ensemble. Chosen for the same three properties the
unified-library spine needs:

  1. Fixed step count  -> step_cv == 0  (no SIMT warp divergence by construction).
  2. ``lax.scan`` body -> natively reverse-mode differentiable (jax.grad works),
     no custom adjoint, no while_loop.
  3. Explicit RK4 -> 4th-order accurate per step; the cheap non-stiff workhorse
     (the stiff member is the linearly-implicit IMEX sibling).

Per trajectory, with h = (t1 - t0)/n_steps:
    k1 = f(t,       y);        k2 = f(t + h/2, y + h/2 k1)
    k3 = f(t + h/2, y + h/2 k2); k4 = f(t + h,   y + h k3)
    y <- y + (h/6)(k1 + 2 k2 + 2 k3 + k4);   t <- t + h
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gradsolve.base import SolveResult

if TYPE_CHECKING:  # gradsolve imports standalone; Problem is annotation-only
    from gradsolve.base import Problem

name = "fixed_step_explicit"

DEFAULT_N_STEPS = 10_000


def supports(problem: Problem) -> bool:
    """Report whether this explicit RK4 spine member can solve `problem`.

    Parameters
    ----------
    problem : Problem
        Candidate problem; only `problem.is_stiff` is inspected.

    Returns
    -------
    bool
        ``True`` iff `problem` is non-stiff. Explicit RK4 has no implicit
        solve and is not L-stable, so stiff problems are rejected (use the
        ``imex`` spine member instead).
    """
    return not problem.is_stiff  # explicit RK4: non-stiff only.


def _step_factory(f, p, h):
    """Build one `jax.lax.scan`-compatible RK4 step closure over a fixed ``h``.

    Parameters
    ----------
    f : callable
        Problem RHS ``f(t, y, p) -> dy/dt`` (``problem.f_jax``).
    p : array_like
        Parameter vector for the single trajectory being integrated, closed
        over by the returned ``step``.
    h : float
        Fixed step size, ``(t1 - t0) / n_steps``.

    Returns
    -------
    callable
        ``step(carry, _) -> (carry_next, None)`` with ``carry = (t, y)``,
        the classical 4-stage RK4 update (stage formulas in the module
        docstring). The trailing ``None``/``_`` are `jax.lax.scan`'s
        per-step output/input slots (unused: this scan carries no stacked
        per-step output).
    """
    def step(carry, _):
        t, y = carry
        k1 = f(t, y, p)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1, p)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2, p)
        k4 = f(t + h, y + h * k3, p)
        y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (t + h, y_next), None

    return step


def solve_jax(problem, y0, params, *, n_steps: int = DEFAULT_N_STEPS):
    """Pure-JAX differentiable solve: (y0[n,dim], params[n,P]) -> y_final[n,dim]."""
    import jax
    import jax.numpy as jnp

    f = problem.f_jax
    t0, t1 = problem.t0, problem.t1
    h = (t1 - t0) / n_steps

    def solve_one(y0i, pi):
        step = _step_factory(f, pi, h)
        (_, yf), _ = jax.lax.scan(step, (jnp.asarray(t0), y0i), None, length=n_steps)
        return yf

    return jax.vmap(solve_one)(y0, params)


def solve(
    problem: Problem,
    y0: np.ndarray,
    params: np.ndarray,
    *,
    rtol: float,
    atol: float,
    device: str = "cpu",
    n_steps: int = DEFAULT_N_STEPS,
) -> SolveResult:
    """Numpy Backend-protocol entry point.

    Notes (this is a *fixed-work* method, unlike the adaptive diffrax backend):
      - ``rtol``/``atol`` are accepted for Backend-protocol conformance but are **nominal**
        here — accuracy is governed solely by ``n_steps`` (raise it for tighter accuracy).
        A tolerance sweep must drive this backend by ``n_steps``, not rtol/atol.
      - ``accepted_steps`` is a structural constant (``== n_steps``): the fixed step
        schedule of the scan, not a measured adaptive count. It does not reflect
        solve success — check ``np.isfinite(y_final)`` for that (the scan cannot early-exit
        or report failure on its own).
    """
    import jax
    import jax.numpy as jnp

    dev_kind = "gpu" if device in ("cuda", "gpu") else "cpu"
    dev = jax.devices(dev_kind)[0]
    y0j = jax.device_put(jnp.asarray(y0, dtype=jnp.float64), dev)
    pj = jax.device_put(jnp.asarray(params, dtype=jnp.float64), dev)

    yf = jax.jit(lambda a, b: solve_jax(problem, a, b, n_steps=n_steps))(y0j, pj)
    yf = np.asarray(jax.block_until_ready(yf))

    n = y0.shape[0]
    accepted = np.full(n, n_steps, dtype=np.int64)   # fixed -> step_cv == 0
    rejected = np.zeros(n, dtype=np.int64)
    return SolveResult(y_final=yf, accepted_steps=accepted, rejected_steps=rejected,
                       solver=f"rk4[{n_steps}]")
