"""gradsolve diffrax general adaptive engine.

Universal catch-all Backend wrapping ``jax.vmap(diffrax.diffeqsolve(...))`` over
the ensemble.  Solver selection follows the standard stiffness split:
  - stiff problems  -> Kvaerno5 (ESDIRK)
  - all others      -> Tsit5

Exposes the module-level Backend protocol (``name``, ``supports``, ``solve``) plus
``_api_reverse`` for the routed API's reverse-mode path (returns a ``jax.grad``-able
closure ``params -> y_final`` via diffrax's RecursiveCheckpointAdjoint).
"""
from __future__ import annotations

import importlib.util

import numpy as np

from gradsolve.base import SolveResult

name = "diffrax"


def supports(problem) -> bool:
    """Universal catch-all — for every problem, whenever diffrax is installed.

    diffrax is an optional extra (``pip install gradsolve[diffrax]``). Deciding here rather than at
    call time lets ``api._fallback`` skip diffrax instead of raising ``ModuleNotFoundError``
    mid-solve.
    """
    return importlib.util.find_spec("diffrax") is not None


def _solver_for(problem):
    import diffrax as dfx

    if problem.is_stiff:
        return dfx.Kvaerno5(), "Kvaerno5"
    return dfx.Tsit5(), "Tsit5"


def solve(
    problem,
    y0: np.ndarray,
    params: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    device: str = "cpu",
    max_steps: int = 200_000,
) -> SolveResult:
    """Ensemble ODE solve via ``jax.vmap(diffrax.diffeqsolve)``."""
    import diffrax as dfx
    import jax
    import jax.numpy as jnp

    solver, solver_name = _solver_for(problem)
    f = problem.f_jax
    t0, t1 = problem.t0, problem.t1
    term = dfx.ODETerm(f)
    controller = dfx.PIDController(rtol=rtol, atol=atol)
    saveat = dfx.SaveAt(t1=True)

    def solve_one(y0i, pi):
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            None,  # dt0=None -> controller selects the initial step
            y0i,
            args=pi,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=max_steps,
            throw=False,
        )
        return (
            sol.ys[-1],
            sol.stats["num_accepted_steps"],
            sol.stats["num_rejected_steps"],
        )

    dev_kind = "gpu" if device in ("cuda", "gpu") else "cpu"
    dev = jax.devices(dev_kind)[0]
    y0j = jax.device_put(jnp.asarray(y0, dtype=jnp.float64), dev)
    pj = jax.device_put(jnp.asarray(params, dtype=jnp.float64), dev)

    yf, acc, rej = jax.jit(jax.vmap(solve_one))(y0j, pj)
    yf = np.asarray(jax.block_until_ready(yf))
    acc = np.asarray(acc, dtype=np.int64)
    rej = np.asarray(rej, dtype=np.int64)
    return SolveResult(y_final=yf, accepted_steps=acc, rejected_steps=rej, solver=solver_name)


def _build_solve_one(problem, *, rtol, atol, max_steps):
    """Shared body of the reverse closure/kernel: ``(y0_i, p_i) -> y_final_i``.

    ``RecursiveCheckpointAdjoint`` propagates reverse-mode AD through the adaptive step
    sequence without storing the full trajectory. Note the semantics this implies for
    both arguments: unlike gradsolve's replay lanes there is no frozen mesh here — the
    step-size controller re-runs on every call, so a perturbed ``y0`` (or ``params``) is
    an adaptive re-solve with its own step sequence.
    """
    import diffrax as dfx

    solver, _ = _solver_for(problem)
    f = problem.f_jax
    t0, t1 = problem.t0, problem.t1
    term = dfx.ODETerm(f)
    controller = dfx.PIDController(rtol=rtol, atol=atol)
    saveat = dfx.SaveAt(t1=True)
    adjoint = dfx.RecursiveCheckpointAdjoint()

    def solve_one(y0i, pi):
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            None,
            y0i,
            args=pi,
            stepsize_controller=controller,
            saveat=saveat,
            max_steps=max_steps,
            throw=False,
            adjoint=adjoint,
        )
        return sol.ys[-1]

    return solve_one


def _api_reverse_kernel(problem, y0, params, *, rtol, atol, device, max_steps: int = 200_000):
    """Return ``(y0, params) -> y_final[n, dim]``, jax.grad-able in both arguments.

    The two-argument form of ``_api_reverse`` (the routed API's ``wrt=`` builder). ``y0``
    and ``params`` here are only the *recording point* in the other lanes' sense — diffrax
    has no mesh to freeze, so this kernel simply ignores them and takes both live.
    """
    del y0, params  # no mesh to record: diffrax re-solves adaptively on every call
    import jax
    import jax.numpy as jnp

    solve_one = _build_solve_one(problem, rtol=rtol, atol=atol, max_steps=max_steps)
    dev_kind = "gpu" if device in ("cuda", "gpu") else "cpu"
    dev = jax.devices(dev_kind)[0]

    def kernel(z, q):
        zj = jax.device_put(jnp.asarray(z, dtype=jnp.float64), dev)
        qj = jax.device_put(jnp.asarray(q, dtype=jnp.float64), dev)
        return jax.jit(jax.vmap(solve_one))(zj, qj)

    return kernel


def _api_reverse(problem, y0, params, *, rtol, atol, device, max_steps: int = 200_000):
    """Return a ``jax.grad``-able closure ``params -> y_final[n, dim]``.

    Uses ``diffrax.RecursiveCheckpointAdjoint`` so reverse-mode AD propagates
    through the adaptive step sequence without storing the full trajectory.
    """
    import jax
    import jax.numpy as jnp

    solve_one = _build_solve_one(problem, rtol=rtol, atol=atol, max_steps=max_steps)
    dev_kind = "gpu" if device in ("cuda", "gpu") else "cpu"
    dev = jax.devices(dev_kind)[0]
    y0j = jax.device_put(jnp.asarray(y0, dtype=jnp.float64), dev)

    def closure(q):
        qj = jax.device_put(jnp.asarray(q, dtype=jnp.float64), dev)
        return jax.jit(jax.vmap(solve_one))(y0j, qj)

    return closure
