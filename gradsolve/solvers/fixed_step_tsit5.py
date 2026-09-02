"""Fixed-step Tsit5 — the order-5 *non-stiff* spine member.

Classical Tsitouras (2011) Tsit5: a 7-stage, 5th-order explicit Runge-Kutta pair,
run here as a *fixed* number of steps in one ``jax.lax.scan``, vmapped over the
ensemble. Sibling of ``gradsolve/solvers/fixed_step_explicit.py`` (the RK4 member);
same three spine properties:

  1. Fixed step count  -> step_cv == 0  (no SIMT warp divergence by construction).
  2. ``lax.scan`` body -> natively reverse-mode differentiable (jax.grad works),
     no custom adjoint, no while_loop.
  3. Tsit5 -> 5th-order accurate per step; the same tableau as diffrax's ``Tsit5``.
     What differs from an adaptive solver is the execution model (scan-vmap with a fixed
     step count versus a per-trajectory adaptive loop), not the method.

The Butcher tableau and the 7-stage step block live in
``gradsolve/solvers/tsit5_step.py`` (factored out so the record-and-replay adjoint can
reuse the exact same arithmetic; constants re-exported here for backwards
compatibility). The high-precision constants are transcribed from Tsitouras (2011) and
cross-checked against diffrax's ``Tsit5`` tableau in the tests; this solver does not
call diffrax.
Reference:

    Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first
    column simplifying assumption", Comput. Math. Appl. 62(2), 770-775, 2011.

FSAL is not exploited (irrelevant for fixed-step): all 7 stages are computed each
step. Per trajectory, with h = (t1 - t0)/n_steps:
    k1 = f(t, y)
    k2 = f(t + c2 h, y + h a21 k1)
    ...
    k7 = f(t + c7 h, y + h sum_j a7j kj)
    y <- y + h * sum_i b_i k_i;   t <- t + h
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gradsolve.base import SolveResult

if TYPE_CHECKING:  # gradsolve imports standalone; Problem is annotation-only
    from gradsolve.base import Problem
from gradsolve.solvers.tsit5_step import (  # noqa: F401  -- tableau re-exported from tsit5_step.py
    _A21,
    _A31,
    _A32,
    _A41,
    _A42,
    _A43,
    _A51,
    _A52,
    _A53,
    _A54,
    _A61,
    _A62,
    _A63,
    _A64,
    _A65,
    _B1,
    _B2,
    _B3,
    _B4,
    _B5,
    _B6,
    _B7,
    _C,
    _E1,
    _E2,
    _E3,
    _E4,
    _E5,
    _E6,
    _E7,
    tsit5_step,
)

name = "fixed_step_tsit5"

DEFAULT_N_STEPS = 10_000


def supports(problem: Problem) -> bool:
    """Report whether this explicit Tsit5 spine member can solve `problem`.

    Parameters
    ----------
    problem : Problem
        Candidate problem; only `problem.is_stiff` is inspected.

    Returns
    -------
    bool
        ``True`` iff `problem` is non-stiff. Explicit Tsit5 has no implicit
        solve and is not L-stable, so stiff problems are rejected (use the
        ``imex`` spine member instead).
    """
    return not problem.is_stiff  # explicit Tsit5: non-stiff only.


def _step_factory(f, p, h):
    """Build one `jax.lax.scan`-compatible Tsit5 step closure over a fixed ``h``.

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
        delegating the 7-stage Tsit5 update to `tsit5_step`. The trailing
        ``None``/``_`` are `jax.lax.scan`'s per-step output/input slots
        (unused: this scan carries no stacked per-step output).
    """
    def step(carry, _):
        t, y = carry
        return (t + h, tsit5_step(f, t, y, h, p)), None

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


def solve_jax_saveat(problem, y0, params, saveat, *, n_steps: int = DEFAULT_N_STEPS):
    """Dense-output form of :func:`solve_jax`: ``-> (y_final[n,dim], ys[n,k,dim])``.

    Same fixed grid and same per-step arithmetic (``tsit5_step``) as ``solve_jax``; the
    states at ``saveat`` are produced by re-stepping from each bracketing step to the exact
    requested time (see ``gradsolve.solvers.dense``), so they carry the same accuracy as the
    endpoints. ``saveat`` must already be validated against the domain.
    Kept as a separate entry point so ``solve_jax``'s hot path is byte-for-byte unchanged.
    """
    import jax.numpy as jnp

    from gradsolve.solvers.dense import vmap_saveat

    h = (problem.t1 - problem.t0) / n_steps
    dts = jnp.full((n_steps,), h)  # shared grid: vmap broadcasts it, never tiles it
    return vmap_saveat(problem.f_jax, problem.t0, y0, params, dts, saveat, tsit5_step)


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
                       solver=f"tsit5[{n_steps}]")
