"""Pure-JAX Vern7 record-and-replay for an arbitrary nonstiff RHS — the high-order sibling of
gradsolve/solvers/tsit5_replay.py.

Records the accepted adaptive Vern7 mesh in plain numpy at arbitrary parameters (via the
method-generic recorder gradsolve/solvers/method.record_adaptive instantiated with VERN7_METHOD),
then replays it as a fixed-length lax.scan of vern7_advance — grad-safe, any RHS. Exposed as
the opt-in engine ``vern7_replay`` (reachable only by explicit engine="vern7_replay"; not an
auto-routing target).

Gradient semantics (identical to tsit5_replay): the closure differentiates the realized
trajectory with the step-size controller held fixed — timesteps are data, not a function of
the params — so the gradient is the exact discrete adjoint of the replayed frozen-step
integration, valid near params0. float64 throughout (the general-RHS recorder is f64-only;
precision='float32' is rejected upstream by the router, mirroring tsit5_replay).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.base import SolveResult
from gradsolve.solvers.method import record_adaptive
from gradsolve.solvers.record_jax import choose_recorder, record_adaptive_jax
from gradsolve.solvers.vern7_step import VERN7_METHOD, vern7_advance

name = "vern7_replay"


def supports(problem) -> bool:
    """Vern7 is an explicit nonstiff method; stiff problems are rejected (use the stiff
    lanes or ``rodas5p_replay``)."""
    return not problem.is_stiff


def record_vern7(f, y0, params, t0, t1, *, rtol, atol, dt0=None, max_steps=50000, recorder="auto"):
    """Record the accepted adaptive Vern7 mesh for an arbitrary RHS ``f(t, y, p)``.

    Returns ``(y_finals[n, D], dts_padded[n, S], n_acc[n])`` with ``S = max(n_acc)``;
    zero-padded rows replay as identity steps. Raises ``RuntimeError`` if any trajectory
    exhausts ``max_steps`` before ``t1`` (replaying an incomplete mesh would be silently wrong).
    Mirrors record_tsit5_jax exactly, only the recorder method differs.

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
        yf, dts, n_acc, _n_rej = record_adaptive_jax(
            VERN7_METHOD, f, y0, params, t0, t1, rtol=rtol, atol=atol, dt0=dt0, max_steps=max_steps)
        return yf, dts, n_acc

    f_jit = jax.jit(lambda t, yy, p: f(t, yy, p))

    y_finals = np.empty((n, D), dtype=np.float64)
    dts_list = []
    n_acc = np.empty(n, dtype=np.int64)
    for i in range(n):
        p_j = jnp.asarray(params[i])

        def f_eval(t, yv, p_j=p_j):
            return np.asarray(f_jit(jnp.asarray(t), jnp.asarray(yv), p_j), dtype=np.float64)

        yf, dts, _n_rej, status = record_adaptive(
            VERN7_METHOD, f_eval, y0[i], t0, t1, rtol, atol, dt0, max_steps)
        if status != 0:
            raise RuntimeError(
                f"record_vern7: trajectory {i} exhausted max_steps={max_steps} before "
                f"t1={t1} — the recorded dt sequence is incomplete and a frozen replay of it "
                "would be silently wrong. Increase max_steps, loosen the tol, or shorten the horizon.")
        y_finals[i] = yf
        dts_list.append(dts)
        n_acc[i] = dts.shape[0]

    S = int(n_acc.max()) if n else 0
    dts_padded = np.zeros((n, S), dtype=np.float64)
    for i, dts in enumerate(dts_list):
        dts_padded[i, : dts.shape[0]] = dts
    return y_finals, dts_padded, n_acc


def replay_vern7_jax(problem, y0, params, dts_padded, remat=False):
    """Pure-JAX Vern7 replay: ``(y0[n,D], params[n,P], dts_padded[n,S]) -> y_final[n,D]``.

    Zero-padded rows replay as identity steps (vern7_advance dt==0 identity contract). Safe
    under jit/grad/vmap. Own function (does not touch warp_replay.py's proven Tsit5 replay).
    """
    f = problem.f_jax

    def one(y0_j, p_j, dts_j):
        def body(carry, dt):
            t, y = carry
            return (t + dt, vern7_advance(f, t, y, dt, p_j)), None
        step = jax.checkpoint(body) if remat else body
        t0 = jnp.asarray(problem.t0, dtype=y0_j.dtype)
        (_t, yf), _ = jax.lax.scan(step, (t0, y0_j), dts_j)
        return yf

    return jax.vmap(one)(jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts_padded))


def make_vern7_frozen_mesh_kernel(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                                  max_steps=50000, saveat=None):
    """Record once at ``(y0, params0)``; return ``(y0, params) -> y_final`` (both free).

    ``saveat`` (validated times, or None): when set, returns ``(y_final, ys[n,k,dim])`` off the
    same frozen mesh via the shared dense machinery (gradsolve/solvers/dense.py), stepping with
    vern7_advance. Mirrors make_tsit5_frozen_mesh_kernel.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    params0 = np.asarray(params0, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if params0.ndim == 1:
        params0 = params0[None, :]
    _yf, dts_padded, _n = record_vern7(
        problem.f_jax, y0, params0, problem.t0, problem.t1, rtol=rtol, atol=atol,
        max_steps=max_steps)
    dts_j = jnp.asarray(dts_padded)
    if saveat is not None:
        from gradsolve.solvers.dense import vmap_saveat
        ts_j = jnp.asarray(saveat)
        return lambda z, p: vmap_saveat(
            problem.f_jax, problem.t0, jnp.asarray(z), jnp.asarray(p), dts_j, ts_j, vern7_advance)
    return lambda z, p: replay_vern7_jax(problem, jnp.asarray(z), jnp.asarray(p), dts_j)


def make_vern7_frozen_mesh_closure(problem, y0, params0, *, rtol=1e-6, atol=1e-9, max_steps=50000):
    """Record once at ``params0``; return a jax.grad-able ``params -> y_final`` closure
    (y0 closed over) — the default reverse contract, mirroring make_tsit5_frozen_mesh_closure."""
    kernel = make_vern7_frozen_mesh_kernel(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps)
    y0_j = jnp.asarray(np.atleast_2d(np.asarray(y0, dtype=np.float64)))
    return lambda p: kernel(y0_j, p)


def solve(problem, y0, params, *, rtol, atol, device="cpu") -> SolveResult:
    """Backend-protocol forward solve: eager record + replay -> final state with true
    per-trajectory accepted-step counts. device is accepted for protocol parity (the recorder
    is host code; the replay runs on the default JAX device)."""
    del device
    y0 = np.asarray(y0, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    yf_rec, dts, n_acc = record_vern7(
        problem.f_jax, y0, params, problem.t0, problem.t1, rtol=rtol, atol=atol)
    y_final = np.asarray(replay_vern7_jax(problem, y0, params, jnp.asarray(dts)))
    return SolveResult(
        y_final=y_final, accepted_steps=np.asarray(n_acc, dtype=np.int64),
        rejected_steps=np.zeros(y0.shape[0], dtype=np.int64), solver=name)
