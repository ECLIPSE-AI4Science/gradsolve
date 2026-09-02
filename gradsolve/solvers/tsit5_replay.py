"""Pure-JAX Tsit5 record-and-replay for an arbitrary (nonstiff) RHS — the reference
implementation of gradsolve's record-and-replay reverse-mode adjoint.

The fused Warp Tsit5 engine only records a mesh for problems that have a *registered*
analytic field (Lorenz/Van der Pol/linear ladder), so it cannot differentiate through a
user's own ``f_jax`` or record at perturbed parameters (which a parameter-recovery fit
needs). This module records the accepted adaptive Tsit5 step mesh in plain numpy at
arbitrary parameters, then replays it as a fixed-length ``lax.scan`` via the grad-safe
``replay_solve_jax`` — no custom kernel, and it works on any right-hand side.

* ``tsit5_adaptive`` — the concrete adaptive Tsit5 + I-controller loop (also the semantic
  spec the Warp kernel matches).
* ``record_tsit5_jax`` — record the accepted mesh per trajectory, padded to a common length.
* ``make_tsit5_frozen_mesh_closure`` — record once at ``params0``; return a ``jax.grad``-able
  ``closure(p) -> y_final[n, D]`` that replays the frozen mesh, differentiating w.r.t. every
  parameter column through ``problem.f_jax``.

Gradient semantics (same as the fused replay): the closure differentiates the realized
trajectory with the step-size controller held fixed — the timesteps are data, not a function
of the params — so the gradient is the exact discrete adjoint of the replayed frozen-step
integration, valid near ``params0``. float64 (gradsolve enables jax x64 at import).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.base import SolveResult
from gradsolve.solvers.method import TSIT5_METHOD
from gradsolve.solvers.record_jax import choose_recorder, record_adaptive_jax
from gradsolve.solvers.tsit5_step import ERR_FLOOR, FAC_MAX, FAC_MIN, ORDER_EXP, SAFETY
from gradsolve.warp.warp_replay import replay_solve_jax

name = "tsit5_replay"


def supports(problem) -> bool:
    """Explicit Tsit5 on any nonstiff ``f_jax`` (no registered field needed); stiff problems
    take ``rodas5p_replay``."""
    return not problem.is_stiff


def solve(problem, y0, params, *, rtol, atol, device="cpu") -> SolveResult:
    """Backend-protocol forward solve: eager record + replay -> final state with true
    per-trajectory accepted-step counts (mirrors ``vern7_replay.solve``). ``device`` is
    accepted for protocol parity: the recorder is host code, the replay runs on the default
    JAX device."""
    del device
    y0 = np.asarray(y0, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    _yf, dts, n_acc, n_rej = record_tsit5_jax(
        problem.f_jax, y0, params, problem.t0, problem.t1, rtol=rtol, atol=atol,
        return_rejected=True)
    y_final = np.asarray(replay_solve_jax(
        problem, jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts)))
    return SolveResult(
        y_final=y_final, accepted_steps=np.asarray(n_acc, dtype=np.int64),
        rejected_steps=np.asarray(n_rej, dtype=np.int64), solver=name)


def tsit5_adaptive(f, y0, t0, t1, rtol, atol, dt0, max_steps):
    """Adaptive Tsit5 with a WRMS I-controller (Hairer--Norsett--Wanner II.4).

    Returns ``(y_final, dts[accepted], n_rejected, status)`` with ``status==0`` iff the
    integration reached ``t1``. FSAL: k1 of the next step reuses k7 of the accepted step.
    """
    from gradsolve.solvers.fixed_step_tsit5 import (
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
        _C,
        _E1,
        _E2,
        _E3,
        _E4,
        _E5,
        _E6,
        _E7,
    )
    _C2, _C3, _C4, _C5, _C6 = _C[0], _C[1], _C[2], _C[3], _C[4]
    y, t, dt = np.array(y0, dtype=np.float64), float(t0), float(dt0)
    k1 = f(t, y)
    dts, n_rej = [], 0
    for _ in range(max_steps):
        if t >= t1:
            break
        h = min(dt, t1 - t)
        k2 = f(t + _C2 * h, y + h * (_A21 * k1))
        k3 = f(t + _C3 * h, y + h * (_A31 * k1 + _A32 * k2))
        k4 = f(t + _C4 * h, y + h * (_A41 * k1 + _A42 * k2 + _A43 * k3))
        k5 = f(t + _C5 * h, y + h * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4))
        k6 = f(t + _C6 * h, y + h * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5))
        y5 = y + h * (_B1 * k1 + _B2 * k2 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)  # _B7 == 0
        k7 = f(t + h, y5)
        e = h * (_E1 * k1 + _E2 * k2 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = float(np.sqrt(np.mean((e / sc) ** 2)))
        if err <= 1.0:
            t, y, k1 = t + h, y5, k7
            dts.append(h)
        else:
            n_rej += 1
        dt = h * min(max(SAFETY * max(err, ERR_FLOOR) ** ORDER_EXP, FAC_MIN), FAC_MAX)
    status = 0 if t >= t1 else 1
    return y, np.array(dts), n_rej, status


def record_tsit5_jax(f, y0, params, t0, t1, *, rtol, atol, dt0=None, max_steps=50000,
                     return_rejected=False, recorder="auto"):
    """Record the accepted adaptive Tsit5 mesh for an arbitrary RHS ``f(t, y, p)``.

    Returns ``(y_finals[n, D], dts_padded[n, S], n_acc[n])`` with ``S = max(n_acc)``;
    zero-padded rows replay as identity steps. Raises ``RuntimeError`` if any trajectory
    exhausts ``max_steps`` before ``t1`` (replaying an incomplete mesh would be silently wrong).

    ``return_rejected=True`` appends the per-trajectory rejected step counts the controller
    already produces, giving ``(y_finals, dts_padded, n_acc, n_rej)`` — what ``solve()`` needs
    to fill ``SolveResult.rejected_steps`` with the true counts instead of zeros. Default False so
    every existing three-tuple caller is unchanged.

    ``recorder`` selects the mesh recorder (``choose_recorder``): ``"host"`` is this numpy
    loop, ``"jax"`` is the vmapped device loop ``record_adaptive_jax``, and the default
    ``"auto"`` picks the device recorder on a GPU or for ``n >= 32``.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if params.ndim == 1:
        params = params[None, :]
    n, D = y0.shape
    if dt0 is None:
        dt0 = (t1 - t0) / 100.0

    if choose_recorder(n, recorder) == "jax":
        yf, dts, n_acc, n_rej = record_adaptive_jax(
            TSIT5_METHOD, f, y0, params, t0, t1, rtol=rtol, atol=atol, dt0=dt0, max_steps=max_steps)
        return (yf, dts, n_acc, n_rej) if return_rejected else (yf, dts, n_acc)

    f_jit = jax.jit(lambda t, yy, p: f(t, yy, p))  # explicit RK -> no Jacobian needed

    y_finals = np.empty((n, D), dtype=np.float64)
    dts_list = []
    n_acc = np.empty(n, dtype=np.int64)
    n_rej = np.empty(n, dtype=np.int64)
    for i in range(n):
        p_j = jnp.asarray(params[i])

        def f_eval(t, yv, p_j=p_j):
            return np.asarray(f_jit(jnp.asarray(t), jnp.asarray(yv), p_j), dtype=np.float64)

        yf, dts, n_rej_i, status = tsit5_adaptive(
            f_eval, y0[i], t0, t1, rtol, atol, dt0, max_steps)
        if status != 0:
            raise RuntimeError(
                f"record_tsit5_jax: trajectory {i} exhausted max_steps={max_steps} before "
                f"t1={t1} — the recorded dt sequence is incomplete and a frozen replay of it "
                "would be silently wrong. Increase max_steps, loosen the tol, or shorten the horizon.")
        y_finals[i] = yf
        dts_list.append(dts)
        n_acc[i] = dts.shape[0]
        n_rej[i] = n_rej_i

    S = int(n_acc.max()) if n else 0
    dts_padded = np.zeros((n, S), dtype=np.float64)
    for i, dts in enumerate(dts_list):
        dts_padded[i, : dts.shape[0]] = dts
    if return_rejected:
        return y_finals, dts_padded, n_acc, n_rej
    return y_finals, dts_padded, n_acc


def make_tsit5_frozen_mesh_kernel(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                                  max_steps=50000, saveat=None):
    """Record the Tsit5 mesh once at ``(y0, params0)``; return ``(y0, params) -> y_final``.

    The two-argument form of ``make_tsit5_frozen_mesh_closure``: the frozen mesh is baked
    in but both the initial state and the parameters stay free, so the routed API can build
    the ``params``-only, ``y0``-only, or joint gradient closure from one recorded mesh
    (``gradsolve.api.grad_closure``'s ``wrt=``). Differentiating w.r.t. ``y0`` carries exactly
    the same frozen-controller caveat as ``params`` (see the module docstring).

    ``saveat`` (validated output times, or None): when set the kernel returns
    ``(y_final[n,dim], ys[n,k,dim])`` — dense states streamed off the same frozen mesh.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    params0 = np.asarray(params0, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if params0.ndim == 1:
        params0 = params0[None, :]
    _yf, dts_padded, _n_acc = record_tsit5_jax(
        problem.f_jax, y0, params0, problem.t0, problem.t1, rtol=rtol, atol=atol, max_steps=max_steps)
    dts_j = jnp.asarray(dts_padded)
    if saveat is not None:
        from gradsolve.warp.warp_replay import replay_solve_saveat_jax
        ts_j = jnp.asarray(saveat)
        return lambda z, p: replay_solve_saveat_jax(
            problem, jnp.asarray(z), jnp.asarray(p), dts_j, ts_j)
    return lambda z, p: replay_solve_jax(problem, jnp.asarray(z), jnp.asarray(p), dts_j)


def make_tsit5_frozen_mesh_closure(problem, y0, params0, *, rtol=1e-6, atol=1e-9, max_steps=50000):
    """Record the Tsit5 mesh once at ``params0``; return a ``jax.grad``-able replay closure.

    ``closure(p) -> y_final[n, D]`` replays the recorded mesh via ``replay_solve_jax`` (pure
    JAX, jit/grad/vmap-safe), differentiating w.r.t. all columns of ``p`` through
    ``problem.f_jax``. ``problem`` must expose ``.f_jax(t, y, p)``, ``.t0``, ``.t1``.
    """
    kernel = make_tsit5_frozen_mesh_kernel(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps)
    y0_j = jnp.asarray(np.atleast_2d(np.asarray(y0, dtype=np.float64)))
    return lambda p: kernel(y0_j, p)
