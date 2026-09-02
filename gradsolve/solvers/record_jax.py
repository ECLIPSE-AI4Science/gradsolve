"""Device-resident adaptive mesh recorder for the general path (any ``f_jax``).

A ``jax.lax.while_loop`` over the method's trial step, vmapped over the ensemble, that
reproduces the host recorders' arithmetic (``tsit5_replay.tsit5_adaptive`` /
``method.record_adaptive`` / ``rodas5p_replay._record_one``) so it records the same mesh,
only on the device and for all trajectories at once. The replay is untouched.

The compiled loop reproduces the host meshes up to fused-multiply-add arithmetic: identical
accepted/rejected step counts and final states, with step sizes agreeing to about 1e-8 relative
on the built-in problems (XLA fuses the trial-step sums, the numpy host loops do not).
"""
from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.solvers.method import ERR_FLOOR, FAC_MAX, FAC_MIN, SAFETY, Method

_EPS = float(np.finfo(np.float64).eps)


def choose_recorder(n: int, recorder: str = "auto") -> str:
    """'host' | 'jax' from the request, the ``GRADSOLVE_RECORDER`` override, and the device.

    ``"host"``/``"jax"`` are honoured verbatim. ``"auto"`` consults ``GRADSOLVE_RECORDER``
    (``host``/``jax`` take precedence) and otherwise picks the device recorder off a GPU or for large
    ensembles (``n >= 32``), falling back to the host loop on a small CPU ensemble.
    """
    if recorder not in ("auto", "host", "jax"):
        raise ValueError(f"recorder must be 'auto', 'host' or 'jax', got {recorder!r}")
    if recorder != "auto":
        return recorder
    env = os.environ.get("GRADSOLVE_RECORDER", "").strip().lower()
    if env in ("host", "jax"):
        return env
    return "jax" if (jax.default_backend() != "cpu" or n >= 32) else "host"


def controller_update_jnp(method: Method, err_prev, accepted_any, err, h):
    """Traceable twin of ``method.controller_update``; returns ``(dt_next, err_prev, accepted_any)``."""
    e = jnp.maximum(err, ERR_FLOOR)
    prev = jnp.where(accepted_any, err_prev, 1.0)
    factor = SAFETY * e ** (-method.beta1) * prev ** method.beta2
    factor = jnp.minimum(jnp.maximum(factor, FAC_MIN), FAC_MAX)
    return h * factor, err, jnp.bool_(True)


def controller_reject_jnp(method: Method, err_prev, accepted_any, err, h):
    """Traceable twin of ``method.controller_reject``: dt shrinks, history unchanged."""
    e = jnp.maximum(err, ERR_FLOOR)
    factor = SAFETY * e ** (-method.beta1)
    factor = jnp.minimum(jnp.maximum(factor, FAC_MIN), FAC_MAX)
    return h * factor, err_prev, accepted_any


class _Carry(NamedTuple):
    t: object          # scalar f64
    y: object          # (D,)
    dt: object         # scalar f64, next proposed step
    k1: object         # (D,) FSAL stage (zeros and unused when method.fsal is False)
    err_prev: object   # scalar f64
    accepted_any: object   # bool
    n_acc: object      # int32
    n_rej: object      # int32
    attempts: object   # int32
    dts: object        # (cap,) f64 accepted step sizes
    status: object     # int32: 0 running/reached, 1 exhausted, 2 underflow/non-finite, 3 buffer full


def _record_one_traj(method: Method, f, t0, t1, rtol, atol, dt0, max_steps, cap, floor):
    """Build the per-trajectory while_loop for one method; returns a function of (y0, p)."""
    fsal = bool(method.fsal)
    needs_jacobian = bool(method.needs_jacobian)

    def cond(c: _Carry):
        return (c.t < t1) & (c.status == 0)

    def body(c: _Carry, p):
        h = jnp.minimum(c.dt, t1 - c.t)
        # Rodas5P host semantics only (status 2): the explicit host loops have no underflow or
        # non-finite handling at all, so the check is gated on the METHOD, not on the floor's sign.
        degenerate = needs_jacobian & ((c.t + h == c.t) | (h <= floor))
        sr = method.trial_step(f, c.t, c.y, h, p, c.k1) if fsal else method.trial_step(f, c.t, c.y, h, p)
        y_next, e = sr.y_next, sr.y_err
        finite = jnp.all(jnp.isfinite(y_next)) & jnp.all(jnp.isfinite(e))
        sc = atol + rtol * jnp.maximum(jnp.abs(c.y), jnp.abs(y_next))
        err = jnp.sqrt(jnp.mean((e / sc) ** 2))
        err_ok = finite & jnp.isfinite(err)
        accept = err_ok & (err <= 1.0) & ~degenerate
        # accepted branch
        dt_acc, prev_acc, any_acc = controller_update_jnp(method, c.err_prev, c.accepted_any, err, h)
        # rejected branch (finite error): controller_reject; non-finite: bounded shrink, no update
        dt_rej, prev_rej, any_rej = controller_reject_jnp(method, c.err_prev, c.accepted_any, err, h)
        # non-finite trial: Rodas5P host loop shrinks by FAC_MIN without touching the controller
        # history; the explicit host loops would spin on a NaN step size until max_steps, which the
        # user sees as "exhausted max_steps" — report that outcome directly (status 1 below).
        dt_rej = jnp.where(err_ok, dt_rej, h * FAC_MIN)
        buffer_full = accept & (c.n_acc >= cap)
        idx = jnp.minimum(c.n_acc, cap - 1)
        # O(1) masked single-element write: a full-buffer ``jnp.where(..., dts.at[idx].set(h), dts)``
        # is an O(cap) select every iteration (an (n, cap) select under vmap). Rewriting the one
        # slot with its own value when ``write`` is false is identical in effect.
        write = accept & ~buffer_full
        dts_new = c.dts.at[idx].set(jnp.where(write, h, c.dts[idx]))
        new = _Carry(
            t=jnp.where(accept, c.t + h, c.t),
            y=jnp.where(accept, y_next, c.y),
            dt=jnp.where(accept, dt_acc, jnp.where(degenerate, c.dt, dt_rej)),
            k1=(jnp.where(accept, sr.fsal, c.k1) if fsal else c.k1),
            err_prev=jnp.where(accept, prev_acc, jnp.where(degenerate, c.err_prev, prev_rej)),
            accepted_any=jnp.where(accept, any_acc, jnp.where(degenerate, c.accepted_any, any_rej)),
            n_acc=c.n_acc + jnp.where(accept & ~buffer_full, 1, 0),
            n_rej=c.n_rej + jnp.where(accept | degenerate, 0, 1),
            attempts=c.attempts + jnp.where(degenerate, 0, 1),
            dts=dts_new,
            status=c.status,
        )
        status = jnp.where(degenerate, 2, new.status)
        status = jnp.where(buffer_full, 3, status)
        status = jnp.where((new.attempts >= max_steps) & (new.t < t1) & (status == 0), 1, status)
        status = jnp.where((~err_ok) & (not needs_jacobian) & (status == 0), 1, status)
        return new._replace(status=status)

    def run(y0, p):
        D = y0.shape[0]
        k1 = f(jnp.float64(t0), y0, p) if fsal else jnp.zeros((D,), jnp.float64)
        c0 = _Carry(t=jnp.float64(t0), y=y0, dt=jnp.float64(dt0), k1=k1,
                    err_prev=jnp.float64(1.0), accepted_any=jnp.bool_(False),
                    n_acc=jnp.int32(0), n_rej=jnp.int32(0), attempts=jnp.int32(0),
                    dts=jnp.zeros((cap,), jnp.float64), status=jnp.int32(0))
        c = jax.lax.while_loop(cond, lambda c: body(c, p), c0)
        return c.y, c.dts, c.n_acc, c.n_rej, c.status

    return run


# Compiled ``jax.jit(jax.vmap(...))`` kernels, keyed on the static closure constants so that
# repeated records of the same (method, f, tols, horizon, cap) reuse the compiled loop instead of
# re-tracing and re-compiling on every call. Without this each call builds a fresh jit object, which
# XLA cannot recognise as the same program, so the whole compile cost lands inside every timed
# call. The value keeps strong references to ``method`` and ``f`` so their ids
# cannot be recycled onto a different object while the entry lives.
# NOTE: plain unbounded dict; the number of distinct (method, f, tol) combos in any real program
# is a handful. Swap for a WeakValueDictionary/LRU if a caller ever churns closures.
_JIT_CACHE: dict = {}


def _cached_run(method: Method, f, t0, t1, rtol, atol, dt0, max_steps, cap, floor):
    """Return the vmapped, jitted per-trajectory loop for these constants, building it once.

    ``f`` is keyed on its underlying function and bound instance, not its object id: a bound method
    (``prob.f_jax``) is a fresh object on every attribute access, so an id key would miss even for
    the same RHS. The cache value keeps ``method``/``fn``/``inst`` alive so those ids cannot be
    recycled onto a different object while the entry lives.
    """
    fn = getattr(f, "__func__", f)
    inst = getattr(f, "__self__", None)
    key = (id(method), id(fn), id(inst), t0, t1, rtol, atol, dt0, max_steps, cap, floor)
    hit = _JIT_CACHE.get(key)
    if hit is not None:
        return hit[-1]
    run = jax.jit(jax.vmap(_record_one_traj(method, f, t0, t1, rtol, atol, dt0, max_steps, cap, floor)))
    _JIT_CACHE[key] = (method, fn, inst, run)
    return run


def record_adaptive_jax(method: Method, f, y0, params, t0, t1, *, rtol, atol, dt0=None,
                        max_steps=50000, cap0=1024):
    """Record the accepted adaptive mesh of ``method`` for every trajectory on the device.

    Same returns as the host recorders: ``(y_finals[n,D], dts_padded[n,S], n_acc[n], n_rej[n])``
    with ``S = max(n_acc)``; zero rows replay as identity steps. Raises ``RuntimeError`` with
    "exhausted max_steps" or "underflow floor" exactly as the host recorders do.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if params.ndim == 1:
        params = params[None, :]
    n, D = y0.shape
    if n == 0:
        return (np.empty((0, D)), np.zeros((0, 0)), np.empty(0, np.int64), np.empty(0, np.int64))
    t0, t1 = float(t0), float(t1)
    if dt0 is None:
        dt0 = (t1 - t0) / 100.0
    floor = 4.0 * _EPS * max(abs(t0), abs(t1 - t0)) if method.needs_jacobian else -1.0
    cap = int(min(cap0, max_steps))
    while True:
        run = _cached_run(method, f, t0, t1, float(rtol), float(atol),
                          float(dt0), int(max_steps), cap, floor)
        yf, dts, n_acc, n_rej, status = (np.asarray(a) for a in run(jnp.asarray(y0), jnp.asarray(params)))
        if not np.any(status == 3) or cap >= max_steps:
            break
        cap = int(min(2 * cap, max_steps))
    bad = np.flatnonzero(status != 0)
    if bad.size:
        i = int(bad[0])
        if status[i] == 2:
            raise RuntimeError(
                f"record: trajectory {i} reached the step-size underflow floor before t1={t1} — a "
                "singular or non-finite field (check the RHS; loosen the tol or shorten the horizon); "
                "the recorded dt sequence is degenerate and a frozen replay of it would be silently wrong.")
        raise RuntimeError(
            f"record: trajectory {i} exhausted max_steps={max_steps} before t1={t1} — the recorded dt "
            "sequence is incomplete and a frozen replay of it would be silently wrong. Increase "
            "max_steps, loosen the tol, or shorten the horizon.")
    S = int(n_acc.max()) if n else 0
    return yf, np.ascontiguousarray(dts[:, :S]), n_acc.astype(np.int64), n_rej.astype(np.int64)
