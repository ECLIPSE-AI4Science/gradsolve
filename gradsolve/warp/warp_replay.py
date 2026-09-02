"""Record-and-replay adjoint: fused Warp forward records accepted dts;
gradient = re-integrating those dts as a fixed-step ``lax.scan`` in pure JAX
(reverse-diff by construction — the Warp kernel needs NO backward pass).

The replay re-runs the EXACT same Tsit5 arithmetic (``gradsolve/solvers/tsit5_step.py``,
bit-parity with the scan spine) over the recorded accepted-dt sequence of each
trajectory. Padding with dt=0.0 is exactly an identity step
(``y + 0*sum(b_i k_i) = y``, ``d y_next/dy = I``, ``d y_next/dp = 0``), so values AND
gradients flow correctly through the zero-padded tail of the ``[n, S]`` dt array.

Scope: this differentiates the realized trajectory with the step controller
frozen — dt is data, not a differentiable function of parameters. The gradient is
the exact discrete adjoint of the replayed (frozen-step-sequence) integration, not
of the adaptive solver as a parameter-dependent algorithm: the dts were accepted at
the recording parameters, and perturbing the parameters does not re-run the
controller. ``make_replay_closure`` therefore records once at concrete ``params0``
and returns a pure-JAX closure valid in a neighbourhood of ``params0``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.solvers.rosenbrock23_step import rosenbrock23_step
from gradsolve.solvers.tsit5_step import tsit5_step
from gradsolve.warp import warp_ode


def replay_solve_jax(problem, y0, params, dts_padded, n_steps=None, remat=False):
    """Pure-JAX replay: ``(y0[n,dim], params[n,P], dts_padded[n,S]) -> y_final[n,dim]``.

    ``dts_padded``: per-trajectory accepted-dt sequences, zero-padded to the common
    length S (zero rows replay as identity steps — see module docstring). Fully
    traceable: safe under ``jit``/``grad``/``vmap`` — this is the closure body that
    the gradient-closure factory in ``gradsolve.api`` traces.

    ``n_steps``: optional per-trajectory accepted-step counts (``n_acc``); accepted
    for interface clarity but UNUSED — zero-padding is exactly an identity step, so
    replaying the full padded array gives identical values and gradients.
    """
    del n_steps  # padding == identity step; per-trajectory lengths are not needed
    f = problem.f_jax

    def one_traj(y0_j, p_j, dts_j):
        def body(carry, dt):
            t, y = carry
            return (t + dt, tsit5_step(f, t, y, dt, p_j)), None

        step = jax.checkpoint(body) if remat else body  # remat: recompute fwd step in bwd
        t0 = jnp.asarray(problem.t0, dtype=y0_j.dtype)  # concrete dtype: scan-carry stable
        (_t_fin, y_fin), _ = jax.lax.scan(step, (t0, y0_j), dts_j)
        return y_fin

    return jax.vmap(one_traj)(jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts_padded))


def replay_solve_saveat_jax(problem, y0, params, dts_padded, ts, remat=False):
    """Dense-output form of :func:`replay_solve_jax`: ``-> (y_final[n,dim], ys[n,k,dim])``.

    Replays the SAME frozen mesh with the SAME per-step arithmetic (``tsit5_step``) and
    emits the states at ``ts`` by re-stepping from each bracketing step to the exact
    requested time (see ``gradsolve.solvers.dense``). The
    zero-padded tail is skipped by construction — a padding step spans dt == 0 and so
    brackets no output time. ``ts`` must already be validated against the domain.

    ``remat`` is accepted for interface parity with ``replay_solve_jax`` and is currently
    NOT applied here: the dense lane's carry holds the k outputs, which a per-step
    ``jax.checkpoint`` would recompute rather than save, and the tradeoff has not been
    measured. Stated rather than silently ignored.
    """
    del remat
    from gradsolve.solvers.dense import vmap_saveat

    return vmap_saveat(problem.f_jax, problem.t0, y0, params, dts_padded, ts, tsit5_step)


def replay_solve_rosenbrock_saveat_jax(problem, y0, params, dts_padded, ts, remat=False):
    """Stiff analog of :func:`replay_solve_saveat_jax` (``rosenbrock23_step``)."""
    del remat  # see replay_solve_saveat_jax
    from gradsolve.solvers.dense import vmap_saveat

    return vmap_saveat(problem.f_jax, problem.t0, y0, params, dts_padded, ts,
                       rosenbrock23_step)


_PRECISIONS = ("float64", "float32")


def _run_dtype(precision):
    """Validate ``precision`` and return its NumPy dtype."""
    if precision not in _PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; known: {list(_PRECISIONS)}")
    return np.float64 if precision == "float64" else np.float32


def _require_x64_for_f64(precision, what):
    """A float64 record+replay is only correct in an x64 process.

    The Warp kernel would happily record float64, but in a non-x64 process the JAX replay
    silently downcasts to float32 (it diverges ~4e-6 from the f64 warp forward even on a
    short horizon) while still reporting "float64" — so a finite-difference check would
    compare against a mislabelled lane. f32 is a first-class choice rather than an
    accident: ask for it explicitly and it works in any process; ask for f64 without x64 and
    this refuses instead of quietly giving you f32.
    """
    if precision != "float64":
        return
    try:
        x64 = bool(jax.config.read("jax_enable_x64"))
    except Exception:  # pragma: no cover - accessor name drift across jax versions
        x64 = bool(getattr(jax.config, "jax_enable_x64", True))
    if not x64:
        raise RuntimeError(
            f"{what} at precision='float64' needs a jax x64 process, but jax_enable_x64 is "
            "off: the replay would silently downcast to float32 and report float64. Run in "
            "the f64 process (GRADSOLVE_X64=1), or pass precision='float32' explicitly."
        )


def record(problem, y0, params, *, rtol=1e-6, atol=1e-9, max_steps=4096, device="cpu",
           record_mode="host", record_transport="dlpack", precision="float64"):
    """Concrete (non-traced) Warp forward with ``record=1``.

    Host code — must NOT run inside a jit/grad trace; call it at closure-build time
    with concrete ``params`` (see ``make_replay_closure``). Field + dim routing is
    ``warp_ode._field_for``; the per-trajectory scalar is ``warp_ode._param_of`` (the
    single shared source of the mapping rules — rho for Lorenz, mu for vdp, ignored for
    linear); ``dt0 = (t1 - t0)/100`` matches ``warp_ode.solve``'s default. General-D:
    the recorded ``dt`` sequence is dimension-independent, so the replay (which uses
    ``problem.f_jax`` directly) works for any registered field at any small D.

    Returns ``(y_warp[n,dim], n_acc[n], dts_padded[n,S])`` with ``S = max(n_acc)``:
    row j is trajectory j's accepted-dt sequence, zero-padded past ``n_acc[j]``.
    ``y_warp``/``n_acc`` are host numpy in BOTH modes (tiny; needed for accuracy
    checks). ``dts_padded``:

    * ``record_mode="host"`` (default, unchanged): the kernel's ``[max_steps, n]``
      record round-trips through host numpy, trimmed + transposed here -> numpy.
    * ``record_mode="device"``: ``warp_ode._record_handoff`` hands the
      trimmed ``[S, n]`` record to JAX without the full-buffer D2H
      (``record_transport`` = "dlpack" zero-copy | "pinned" trimmed copy); the
      transpose to ``[n, S]`` happens IN JAX (``jnp.transpose`` — a device op on
      CUDA) -> jax.Array, bit-identical to host mode's values.
    """
    np_dtype = _run_dtype(precision)
    _require_x64_for_f64(precision, "record-and-replay adjoint")
    routed = warp_ode._field_for(problem)
    if routed is None:
        raise ValueError(
            f"warp_replay does not support problem {problem.name!r}; supported: "
            "'diffeqgpu_lorenz', 'lorenz', 'vdp', 'linear_ladder_<D>'."
        )
    field_key, dim = routed
    y0_np = np.asarray(y0, dtype=np_dtype)
    rho_np = warp_ode._param_of(problem, params)  # validates in f64; _launch casts to run dtype
    dt0 = (problem.t1 - problem.t0) / 100.0

    y_warp, n_acc, _n_rej, status, dts = warp_ode._launch(
        y0_np, rho_np,
        t0=problem.t0, t1=problem.t1,
        rtol=rtol, atol=atol, dt0=dt0,
        max_steps=max_steps, record=1, device=device,
        field_key=field_key, dim=dim, precision=precision,
        record_mode=record_mode, record_transport=record_transport,
    )
    if np.any(status != 0):
        bad = np.flatnonzero(status != 0)
        raise RuntimeError(
            f"warp_replay.record: {bad.size} trajectories (rows {bad[:8].tolist()}...) "
            f"exhausted max_steps={max_steps} before t1 — the recorded dt sequence is "
            "incomplete and a frozen replay of it would be silently wrong."
        )
    if record_mode == "device":
        # dts is already the trimmed [S, n] handoff (jax zero-copy or pinned numpy
        # view); transpose IN JAX — a device op, no host round-trip. jnp.asarray
        # is a no-op for the dlpack jax array; for the pinned numpy view it is the
        # one (trimmed, ~6.4 MB) H2D upload on CUDA.
        dts_padded = jnp.transpose(jnp.asarray(dts))  # [n, S] jax.Array
    else:
        s = int(np.max(n_acc)) if n_acc.size else 0
        dts_padded = np.ascontiguousarray(dts[:s, :].T)  # [n, S], zero-padded per row
    return y_warp, n_acc, dts_padded


def make_replay_kernel(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                       max_steps=4096, device="cpu",
                       record_mode="host", record_transport="dlpack", remat=False,
                       saveat=None, precision="float64"):
    """Record ONCE at concrete ``(y0, params0)``; return ``(y0, params) -> y_final``.

    The two-argument form of ``make_replay_closure`` — the frozen mesh is baked in, but
    BOTH the initial state and the parameters stay free, so the routed API can build the
    ``params``-only, ``y0``-only, or joint gradient closure from one recorded mesh
    (``gradsolve.api.grad_closure``'s ``wrt=``). Differentiating w.r.t. ``y0`` has exactly
    the same scope as w.r.t. ``params``: the recorded dts are data either way, so the result is the
    discrete adjoint of the replayed frozen-step integration, valid near the recording
    point (see the module docstring).

    ``saveat`` (validated output times, or None): when set the kernel returns
    ``(y_final[n,dim], ys[n,k,dim])`` — the dense states are streamed off the SAME frozen
    mesh, so a time-series loss differentiates exactly as a final-state loss does.

    ``precision``: "float64" (default) or "float32". f32 records the mesh with the f32 Warp
    kernel (exactly half the record buffer's bytes) and replays in f32; the kernel casts its
    inputs to the run precision, so the arithmetic is the requested one rather than whatever
    the caller's dtypes happened to promote to."""
    dt = _run_dtype(precision)
    _, _n_acc, dts_padded = record(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps,
        device=device, record_mode=record_mode, record_transport=record_transport,
        precision=precision)
    dts_j = jnp.asarray(dts_padded, dtype=dt)  # no-op when device mode already returned jax
    if saveat is not None:
        ts_j = jnp.asarray(saveat, dtype=dt)
        return lambda z, p: replay_solve_saveat_jax(
            problem, jnp.asarray(z, dtype=dt), jnp.asarray(p, dtype=dt), dts_j, ts_j)
    return lambda z, p: replay_solve_jax(
        problem, jnp.asarray(z, dtype=dt), jnp.asarray(p, dtype=dt), dts_j, remat=remat)


def make_replay_closure(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                        max_steps=4096, device="cpu",
                        record_mode="host", record_transport="dlpack", remat=False):
    """Entry point for the gradient-closure factory in ``gradsolve.api``: record once at
    concrete ``params0``, return a pure-JAX closure ``p -> replay(y0, p, dts_frozen)``. The
    closure is jit/grad-safe; gradients are exact for the frozen step sequence recorded at
    ``params0`` (frozen-controller discrete adjoint — see the module docstring's scope note).
    ``max_steps`` sizes the record buffer (GPU grad cells may want 2048).
    ``record_mode``/``record_transport`` select the device-resident record
    path (see ``record``); the frozen dts are bit-identical across modes, so the
    closure's values AND gradients are too."""
    kernel = make_replay_kernel(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps, device=device,
        record_mode=record_mode, record_transport=record_transport, remat=remat)
    y0_j = jnp.asarray(y0)
    return lambda p: kernel(y0_j, p)


def solve_jax(problem, y0, params, *, rtol=1e-6, atol=1e-9, device="cpu"):
    """Eager record+replay (concrete params only — NOT traceable: the record step is
    host code). Forward value via the replay, so it inherits the replay arithmetic."""
    _, _n_acc, dts_padded = record(
        problem, y0, params, rtol=rtol, atol=atol, device=device)
    return replay_solve_jax(problem, y0, params, dts_padded)


# =====================================================================================
# Stiff record-and-replay adjoint: the fused Rosenbrock23 Warp kernel
# records the accepted-dt sequence; the gradient re-integrates those frozen dts as a
# pure-JAX ``lax.scan`` of ``rosenbrock23_step`` (reverse-diff by construction — the
# Warp kernel needs NO backward pass). EXACTLY parallels the explicit Tsit5 path above:
# only the kernel (warp_rosenbrock vs warp_ode) and the per-step function
# (rosenbrock23_step vs tsit5_step) differ.
#
# Scope (identical to the Tsit5 note): this differentiates the realized trajectory
# with the step controller FROZEN — dt is data, not a differentiable function of the
# parameters. The gradient is the exact discrete adjoint of the replayed
# (frozen-step-sequence) Rosenbrock23 integration, NOT of the adaptive solver as a
# parameter-dependent algorithm: the dts were accepted at the recording parameters, and
# perturbing the parameters does not re-run the controller.
# ``make_rosenbrock_replay_closure`` therefore records ONCE at concrete ``params0`` and
# returns a pure-JAX closure valid in a neighbourhood of ``params0``.
# =====================================================================================


def replay_solve_rosenbrock_jax(problem, y0, params, dts_padded, n_steps=None, remat=False):
    """Pure-JAX stiff replay: ``(y0[n,dim], params[n,P], dts_padded[n,S]) -> y_final``.

    The Rosenbrock23 analog of ``replay_solve_jax``: re-integrates the frozen accepted-dt
    sequence with ``rosenbrock23_step`` (in-step ``jax.jacfwd`` Jacobian + ``jnp.linalg.
    solve`` W-solves). ``dts_padded`` is zero-padded to the common length S; zero rows
    replay as exact identity steps (dt=0 => W=I, ynew=y — see ``rosenbrock23_step``'s
    docstring), so values AND gradients flow through the padded tail. Fully traceable
    (jit/grad/vmap-safe); this is the closure body the stiff grad cell traces.

    ``n_steps``: accepted for interface parity with ``replay_solve_jax`` but UNUSED
    (zero-padding is exactly identity)."""
    del n_steps  # padding == identity step; per-trajectory lengths are not needed
    f = problem.f_jax

    def one_traj(y0_j, p_j, dts_j):
        def body(carry, dt):
            t, y = carry
            return (t + dt, rosenbrock23_step(f, t, y, dt, p_j)), None

        step = jax.checkpoint(body) if remat else body  # remat: recompute fwd step in bwd
        t0 = jnp.asarray(problem.t0, dtype=y0_j.dtype)  # concrete dtype: scan-carry stable
        (_t_fin, y_fin), _ = jax.lax.scan(step, (t0, y0_j), dts_j)
        return y_fin

    return jax.vmap(one_traj)(jnp.asarray(y0), jnp.asarray(params),
                              jnp.asarray(dts_padded))


def record_rosenbrock(problem, y0, params, *, rtol=1e-6, atol=1e-9, max_steps=50000,
                      device="cpu", precision="float64"):
    """Concrete (non-traced) fused Rosenbrock23 forward with ``record=1``.

    Stiff analog of ``record``: host code — must NOT run inside a jit/grad trace; call it
    at closure-build time with concrete ``params``. Field + dim routing is
    ``warp_rosenbrock._field_for``; the per-trajectory param vector is
    ``warp_rosenbrock._params_of`` (k1,k2,k3 for Robertson; ignored for linstiff);
    ``dt0 = (t1 - t0)/100`` matches ``warp_rosenbrock.solve``'s default. ``max_steps``
    defaults higher than the explicit path (stiff solves take many steps — Robertson
    ~3700 accepted at rtol=1e-6).

    Returns ``(y_warp[n,dim], n_acc[n], dts_padded[n,S])`` with ``S = max(n_acc)``: row j
    is trajectory j's accepted-dt sequence, zero-padded past ``n_acc[j]`` (host numpy).
    """
    np_dtype = _run_dtype(precision)
    _require_x64_for_f64(precision, "stiff record-and-replay adjoint")
    from gradsolve.warp import warp_rosenbrock

    routed = warp_rosenbrock._field_for(problem)
    if routed is None:
        raise ValueError(
            f"warp_replay (stiff) does not support problem {problem.name!r}; supported: "
            "'robertson', 'hires' and registered 'linstiff_*'.")
    field_key, dim = routed
    y0_np = np.asarray(y0, dtype=np_dtype)
    params_np = warp_rosenbrock._params_of(problem, params)  # f64-validated; _launch casts
    dt0 = (problem.t1 - problem.t0) / 100.0

    y_warp, n_acc, _n_rej, status, dts = warp_rosenbrock._launch(
        y0_np, params_np,
        t0=problem.t0, t1=problem.t1,
        rtol=rtol, atol=atol, dt0=dt0,
        max_steps=max_steps, record=1, device=device,
        field_key=field_key, dim=dim, precision=precision,
    )
    if np.any(status != 0):
        bad = np.flatnonzero(status != 0)
        raise RuntimeError(
            f"warp_replay.record_rosenbrock: {bad.size} trajectories "
            f"(rows {bad[:8].tolist()}...) exhausted max_steps={max_steps} before t1 — "
            "the recorded dt sequence is incomplete and a frozen replay of it would be "
            "silently wrong.")
    s = int(np.max(n_acc)) if n_acc.size else 0
    dts_padded = np.ascontiguousarray(dts[:s, :].T)  # [n, S], zero-padded per row
    return y_warp, n_acc, dts_padded


def make_rosenbrock_replay_closure(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                                   max_steps=50000, device="cpu", remat=False):
    """Stiff grad entry: record ONCE at concrete ``params0`` via the fused Rosenbrock23
    kernel, return a pure-JAX closure ``p -> replay(y0, p, dts_frozen)``. The closure is
    jit/grad-safe; gradients are exact for the FROZEN step sequence recorded at
    ``params0`` (frozen-controller discrete adjoint).

    Scope (identical to ``make_replay_closure``'s note): dt is data, not a
    differentiable function of the parameters — the dts were accepted by the adaptive
    controller at ``params0`` and perturbing the parameters does not re-run the
    controller. The returned closure is the exact discrete adjoint of the replayed
    frozen-step Rosenbrock23 integration, valid in a neighbourhood of ``params0``.
    ``max_steps`` sizes the record buffer (stiff solves take many steps)."""
    kernel = make_rosenbrock_replay_kernel(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps, device=device,
        remat=remat)
    y0_j = jnp.asarray(y0)
    return lambda p: kernel(y0_j, p)


def make_rosenbrock_replay_kernel(problem, y0, params0, *, rtol=1e-6, atol=1e-9,
                                  max_steps=50000, device="cpu", remat=False,
                                  saveat=None, precision="float64"):
    """Stiff analog of ``make_replay_kernel``: record ONCE at concrete ``(y0, params0)``
    via the fused Rosenbrock23 kernel; return ``(y0, params) -> y_final`` replaying the
    frozen mesh with both arguments free (the routed API's ``wrt=`` builder).

    ``saveat`` (validated output times, or None): when set the kernel returns
    ``(y_final[n,dim], ys[n,k,dim])`` streamed off the same frozen mesh.
    ``precision``: see ``make_replay_kernel``."""
    dt = _run_dtype(precision)
    _, _n_acc, dts_padded = record_rosenbrock(
        problem, y0, params0, rtol=rtol, atol=atol, max_steps=max_steps, device=device,
        precision=precision)
    dts_j = jnp.asarray(dts_padded, dtype=dt)
    if saveat is not None:
        ts_j = jnp.asarray(saveat, dtype=dt)
        return lambda z, p: replay_solve_rosenbrock_saveat_jax(
            problem, jnp.asarray(z, dtype=dt), jnp.asarray(p, dtype=dt), dts_j, ts_j)
    return lambda z, p: replay_solve_rosenbrock_jax(
        problem, jnp.asarray(z, dtype=dt), jnp.asarray(p, dtype=dt), dts_j, remat=remat)
