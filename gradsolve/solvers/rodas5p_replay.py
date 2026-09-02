"""Pure-JAX Rodas5P record-and-replay for any jax.jacfwd-compatible stiff RHS — the stiff high-order sibling of
gradsolve/solvers/vern7_replay.py.

record_rodas5p records the accepted adaptive Rodas5P mesh with a dedicated jitted host loop (the
Rosenbrock trial_step needs jax.jacfwd on a traceable RHS, so the numpy record_adaptive is not
usable — RODAS5P_METHOD.needs_jacobian=True). One jitted step is built per call and each parameter
vector is passed dynamically (no per-trajectory recompile). The loop reuses the shared controller
(controller_update/controller_reject/ControllerState) and is robust to a singular/non-finite W: a
non-finite stage or error triggers a bounded shrink, and a step-size underflow raises with status.
It returns true per-trajectory accepted and rejected counts. Replay is a fixed-length lax.scan of
rodas5p_advance — reverse-diff by construction, including through the per-step AD Jacobian.
Exposed as the opt-in engine ``rodas5p_replay``.

Differentiability contract: record_rodas5p probe-traces the Rodas5P step
once, so a RHS that is not jax.jacfwd-compatible for the forward solve surfaces as a clear
rodas5p_replay-specific error (not a raw JAX trace error). The frozen-replay gradient additionally
needs second-order AD; the probe does not preflight that, so a RHS that traces forward but lacks
second-order-AD rules (e.g. a custom_vjp without a JVP rule) will fail later during jax.grad with a
raw JAX error — this narrower promise is documented, not wrapped. float64 throughout;
precision='float32' is rejected upstream.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.base import SolveResult
from gradsolve.solvers.method import (
    FAC_MIN,
    ControllerState,
    controller_reject,
    controller_update,
)
from gradsolve.solvers.record_jax import choose_recorder, record_adaptive_jax
from gradsolve.solvers.rodas5p_step import (
    RODAS5P_METHOD,
    rodas5p_advance,
    rodas5p_trial_step,
)

name = "rodas5p_replay"
_EPS = float(np.finfo(np.float64).eps)


def supports(problem) -> bool:
    """True — Rodas5P is reachable only by explicit opt-in. Note the differentiability contract
    (module docstring): the RHS must be jax.jacfwd-compatible and support second-order AD for the
    frozen-replay gradient; supports() cannot cheaply verify that, so an incompatible RHS raises a
    clear rodas5p_replay-specific error at record time rather than returning False here."""
    del problem
    return True


def _record_one(step, y0_i, t0, t1, rtol, atol, dt0, max_steps):
    """Host controller loop for one trajectory. ``step(t, y, dt)`` is the pre-jitted trial step
    (params already bound for this trajectory). Returns (y_final, dts[accepted], n_rej, status).
    status: 0 reached t1; 1 max_steps exhausted; 2 step underflow / non-finite."""
    y = jnp.asarray(y0_i, dtype=jnp.float64)
    t, dt = float(t0), float(dt0)
    cstate = ControllerState(err_prev=1.0, accepted_any=False)
    dts, n_rej = [], 0
    floor = 4.0 * _EPS * max(abs(t0), abs(t1 - t0))   # scale-aware underflow floor
    for _ in range(max_steps):
        if t >= t1:
            break
        h = min(dt, t1 - t)
        if t + h == t or h <= floor:   # step no longer usefully advances t -> singular/degenerate
            return np.asarray(y, dtype=np.float64), np.array(dts, dtype=np.float64), n_rej, 2
        sr = step(t, y, h)
        y_next = sr.y_next
        finite = bool(jnp.all(jnp.isfinite(y_next)) & jnp.all(jnp.isfinite(sr.y_err)))
        if not finite:
            n_rej += 1
            dt = h * FAC_MIN            # bounded shrink on a non-finite step (do NOT feed NaN err)
            continue
        sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_next))
        err = float(jnp.sqrt(jnp.mean((sr.y_err / sc) ** 2)))
        if not np.isfinite(err):
            n_rej += 1
            dt = h * FAC_MIN
            continue
        if err <= 1.0:
            t, y = t + h, y_next
            dts.append(h)
            dt, cstate = controller_update(RODAS5P_METHOD, cstate, err, h)
        else:
            n_rej += 1
            dt, cstate = controller_reject(cstate, err, h, RODAS5P_METHOD)
    status = 0 if t >= t1 else 1
    return np.asarray(y, dtype=np.float64), np.array(dts, dtype=np.float64), n_rej, status


def record_rodas5p(f, y0, params, t0, t1, *, rtol, atol, dt0=None, max_steps=50000, recorder="auto"):
    """Record the accepted adaptive Rodas5P mesh for any jax.jacfwd-compatible stiff RHS ``f(t, y, p)``.
    Returns ``(y_finals[n,D], dts_padded[n,S], n_acc[n], n_rej[n])``. Raises ``RuntimeError`` on
    max_steps exhaustion, step-size underflow, or a non-finite field (a frozen replay of an
    incomplete/degenerate mesh would be silently wrong).

    ``recorder`` selects the mesh recorder (``choose_recorder``): ``"host"`` is this jitted host
    loop, ``"jax"`` is the vmapped device loop ``record_adaptive_jax``, and the default ``"auto"``
    picks the device recorder on a GPU or for ``n >= 32``. The switch sits after the jacfwd probe,
    so an AD-incompatible RHS still raises the clear rodas5p_replay contract error on either path."""
    y0 = np.asarray(y0, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if params.ndim == 1:
        params = params[None, :]
    n, D = y0.shape
    if n == 0:   # empty ensemble: nothing to record; guard the probe (which indexes y0[0])
        return (np.empty((0, D), dtype=np.float64), np.zeros((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if dt0 is None:
        dt0 = (t1 - t0) / 100.0

    step = jax.jit(lambda t, y, dt, p: rodas5p_trial_step(f, t, y, dt, p))  # ONE jit for all traj
    try:  # probe-trace once so an AD-incompatible RHS surfaces as a clear engine error
        _ = step(float(t0), jnp.asarray(y0[0]), 0.0, jnp.asarray(params[0]))
    except Exception as e:  # noqa: BLE001 -- re-raised as an engine-specific contract error
        raise ValueError(
            "rodas5p_replay requires a jax.jacfwd-compatible RHS (and second-order-AD support for "
            f"the frozen-replay gradient); tracing the Rodas5P step of this f_jax failed: {e!r}") from e

    if choose_recorder(n, recorder) == "jax":
        return record_adaptive_jax(
            RODAS5P_METHOD, f, y0, params, t0, t1, rtol=rtol, atol=atol, dt0=dt0, max_steps=max_steps)

    y_finals = np.empty((n, D), dtype=np.float64)
    dts_list = []
    n_acc = np.empty(n, dtype=np.int64)
    n_rej = np.empty(n, dtype=np.int64)
    for i in range(n):
        p_i = jnp.asarray(params[i])
        yf, dts, rej, status = _record_one(
            lambda t, y, dt, p_i=p_i: step(t, y, dt, p_i),
            y0[i], t0, t1, rtol, atol, dt0, max_steps)
        if status != 0:
            # Distinct keywords per status so a test cannot accidentally match the wrong path
            # status 1 -> "exhausted max_steps"; status 2 -> "underflow floor".
            detail = {
                1: f"exhausted max_steps={max_steps} (increase max_steps or loosen the tol)",
                2: ("the step size collapsed to the underflow floor — a singular or non-finite "
                    "field (check the RHS; loosen the tol or shorten the horizon)"),
            }[status]
            raise RuntimeError(
                f"record_rodas5p: trajectory {i} {detail} before reaching t1={t1}; the recorded dt "
                "sequence is incomplete/degenerate and a frozen replay of it would be silently wrong.")
        y_finals[i] = yf
        dts_list.append(dts)
        n_acc[i] = dts.shape[0]
        n_rej[i] = rej

    S = int(n_acc.max()) if n else 0
    dts_padded = np.zeros((n, S), dtype=np.float64)
    for i, dts in enumerate(dts_list):
        dts_padded[i, : dts.shape[0]] = dts
    return y_finals, dts_padded, n_acc, n_rej


def replay_rodas5p_jax(problem, y0, params, dts_padded, remat=False):
    """Pure-JAX Rodas5P replay: ``(y0[n,D], params[n,P], dts_padded[n,S]) -> y_final[n,D]``.
    Zero-padded rows replay as identity steps. Safe under jit/grad/vmap (grad flows through the
    per-step AD Jacobian). Own function — warp_replay.py untouched."""
    f = problem.f_jax

    def one(y0_j, p_j, dts_j):
        def body(carry, dt):
            t, y = carry
            return (t + dt, rodas5p_advance(f, t, y, dt, p_j)), None
        step = jax.checkpoint(body) if remat else body
        t0 = jnp.asarray(problem.t0, dtype=y0_j.dtype)
        (_t, yf), _ = jax.lax.scan(step, (t0, y0_j), dts_j)
        return yf

    return jax.vmap(one)(jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts_padded))


def make_rodas5p_frozen_mesh_kernel(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                                    max_steps=50000, saveat=None, remat=False, dense=False):
    """Record once at ``(y0, params0)``; return ``(y0, params) -> y_final`` (both free). ``saveat``
    returns ``(y_final, ys[n,k,dim])`` off the frozen mesh. ``remat`` threads jax.checkpoint into
    the final-state replay scan; it is ignored when ``saveat`` is supplied (the dense path goes
    through the shared ``dense.vmap_saveat``, which takes no remat argument).

    ``dense=True`` (requires ``saveat``) opts into Rodas5P's own order-4 continuous extension
    instead of bracket-and-re-step: a saved state is then evaluated from the bracketing step's
    interpolating polynomial rather than integrated to the requested time, which deletes the
    ``k`` extra solver steps. The trade-off, both well inside the record tolerance, is that
    the polynomial is less accurate than a genuine re-step on the same mesh (tested in
    tests/test_rodas5p_dense_saveat.py). ``dense=False`` (the default) uses bracket-and-re-step.

    With ``dense=True`` the concrete recorded mesh is checked once here, at build time, by
    ``validate_dense_thetas`` — outside [0, 1] the extension extrapolates, and a traced guard
    could not raise from inside the scan.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    params0 = np.asarray(params0, dtype=np.float64)
    if y0.ndim == 1:
        y0 = y0[None, :]
    if params0.ndim == 1:
        params0 = params0[None, :]
    if dense and saveat is None:
        raise ValueError(
            "dense=True is a saveat option (it replaces the per-save re-step with the step's own "
            "continuous extension); pass saveat=..., or drop dense.")
    _yf, dts_padded, _na, _nr = record_rodas5p(
        problem.f_jax, y0, params0, problem.t0, problem.t1, rtol=rtol, atol=atol, max_steps=max_steps)
    dts_j = jnp.asarray(dts_padded)
    if saveat is not None:
        ts_j = jnp.asarray(saveat)
        if dense:
            from gradsolve.solvers.dense import validate_dense_thetas, vmap_saveat_dense
            from gradsolve.solvers.rodas5p_step import (
                rodas5p_dense_eval,
                rodas5p_stages_and_dense,
            )
            # Pass saveat through unconverted: validate_dense_thetas no-ops on tracers and
            # does its own host conversion otherwise. Forcing np.asarray here would be the
            # only saveat factory in the library that rejects a traced save-time vector.
            validate_dense_thetas(dts_padded, saveat, problem.t0)
            return lambda z, p: vmap_saveat_dense(
                problem.f_jax, problem.t0, jnp.asarray(z), jnp.asarray(p), dts_j, ts_j,
                rodas5p_stages_and_dense, rodas5p_dense_eval)
        from gradsolve.solvers.dense import vmap_saveat
        return lambda z, p: vmap_saveat(
            problem.f_jax, problem.t0, jnp.asarray(z), jnp.asarray(p), dts_j, ts_j, rodas5p_advance)
    return lambda z, p: replay_rodas5p_jax(problem, jnp.asarray(z), jnp.asarray(p), dts_j, remat=remat)


def make_rodas5p_frozen_mesh_closure(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                                     max_steps=50000, remat=False):
    """Record once at ``params0``; return a jax.grad-able ``params -> y_final`` closure."""
    kernel = make_rodas5p_frozen_mesh_kernel(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps, remat=remat)
    y0_j = jnp.asarray(np.atleast_2d(np.asarray(y0, dtype=np.float64)))
    return lambda p: kernel(y0_j, p)


def solve(problem, y0, params, *, rtol, atol, device="cpu") -> SolveResult:
    """Backend-protocol forward solve: record + replay -> final state with true per-trajectory
    accepted and rejected step counts."""
    del device
    y0 = np.asarray(y0, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    _yf, dts, n_acc, n_rej = record_rodas5p(
        problem.f_jax, y0, params, problem.t0, problem.t1, rtol=rtol, atol=atol)
    y_final = np.asarray(replay_rodas5p_jax(problem, y0, params, jnp.asarray(dts)))
    return SolveResult(
        y_final=y_final, accepted_steps=np.asarray(n_acc, dtype=np.int64),
        rejected_steps=np.asarray(n_rej, dtype=np.int64), solver=name)
