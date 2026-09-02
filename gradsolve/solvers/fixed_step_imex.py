"""Fixed-step linearly-implicit Euler backend — the *fixed-step* strategy.

This is a generic instance of the structured fixed-step IMEX strategy: a *fixed* number of
steps run as one ``jax.lax.scan``. It is
deliberately the simplest faithful instance — order-1 linearly-implicit (Rosenbrock-)
Euler — chosen for three properties:

  1. Fixed step count  -> step_cv == 0  (no SIMT warp divergence by construction).
  2. ``lax.scan`` body -> natively reverse-mode differentiable (jax.grad works),
     no custom adjoint, no while_loop.
  3. Per-step linear solve with the Jacobian -> L-stable, survives stiff problems.

It is not high-order; accuracy comes from step count. A higher-order Rosenbrock method
(ROS2) would be the upgrade if accuracy per step matters.

Per trajectory, with h = (t1 - t0)/n_steps and J = df/dy at (t, y):
    (I - h J) dy = h f(t, y);   y <- y + dy;   t <- t + h
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gradsolve.base import SolveResult

if TYPE_CHECKING:  # gradsolve imports standalone; Problem is annotation-only
    from gradsolve.base import Problem

name = "fixed_step_imex"

DEFAULT_N_STEPS = 10_000


def supports(problem: Problem) -> bool:
    """Report whether this linearly-implicit Euler spine member can solve `problem`.

    Parameters
    ----------
    problem : Problem
        Candidate problem (unused beyond the protocol check).

    Returns
    -------
    bool
        Always ``True``: the per-step dense-Jacobian linear solve is
        L-stable, so both stiff and non-stiff problems are supported
        (subject to the Jacobian being small/dense enough for
        ``jnp.linalg.solve``).
    """
    return True  # stiff and non-stiff both fine (L-stable, small dense Jacobian)


def _step_factory(f, p, h):
    """Build one `jax.lax.scan`-compatible linearly-implicit Euler step closure.

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
        ``step(carry, _) -> (carry_next, None)`` with ``carry = (t, y)``.
        Each call takes the Jacobian ``J = df/dy`` via `jax.jacfwd` at
        ``(t, y, p)`` and solves the dense linear system
        ``(I - h J) dy = h f(t, y)`` (see the module docstring) for one
        L-stable Rosenbrock-Euler step. The trailing ``None``/``_`` are
        `jax.lax.scan`'s per-step output/input slots (unused).
    """
    def step(carry, _):
        t, y = carry
        return (t + h, imex_euler_step(f, t, y, h, p)), None

    return step


def imex_euler_step(f, t, y, dt, p):
    """One linearly-implicit (Rosenbrock-)Euler step: ``(f, t, y, dt, p) -> y_next``.

    The verbatim step body previously inlined in ``_step_factory``, factored out for the
    same reason ``tsit5_step`` was: the dense-output lane
    (``gradsolve.solvers.dense``) must re-integrate with exactly the same arithmetic as the
    scan spine, so there is one source of it rather than two that can drift apart.

    Takes the Jacobian ``J = df/dy`` via ``jax.jacfwd`` at ``(t, y, p)`` and solves the
    dense system ``(I - dt*J) dy = dt*f(t, y)`` (see the module docstring). ``dt == 0``
    gives ``a = I`` and ``dy = 0`` — an exact identity step, which is what makes zero-padded
    replay meshes safe.
    """
    import jax
    import jax.numpy as jnp

    fy = f(t, y, p)
    jac = jax.jacfwd(lambda yy: f(t, yy, p))(y)  # (dim, dim), dim is small
    a = jnp.eye(y.shape[-1]) - dt * jac
    dy = jnp.linalg.solve(a, dt * fy)
    return y + dy


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

    Same fixed grid and same per-step arithmetic (``imex_euler_step``) as ``solve_jax``.
    This lane is order-1, so its saved states inherit the same global error as its final
    state — the dense-output test threshold for this lane is set accordingly (1e-2), not at
    the replay lanes' 1e-6.
    """
    import jax.numpy as jnp

    from gradsolve.solvers.dense import vmap_saveat

    h = (problem.t1 - problem.t0) / n_steps
    dts = jnp.full((n_steps,), h)  # shared grid: vmap broadcasts it, never tiles it
    return vmap_saveat(problem.f_jax, problem.t0, y0, params, dts, saveat, imex_euler_step)


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
                       solver=f"li-euler[{n_steps}]")
