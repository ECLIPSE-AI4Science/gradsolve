"""Named, override-only fused-backward reverse engines.

Promotes the genuine single-kernel fused Warp backward to a ``jax.grad``-able
reverse closure so ``jax.grad`` flows through the fused stiff Rosenbrock23 kernel (record
the frozen mesh once at the closure's params/tol, forward = the fused Warp kernel, backward
= ``wp.Tape`` through the custom-adjoint solve). Registered under
``"fused_rosenbrock_backward"`` — selectable by explicit override only, not the ``auto``
default (the fused backward is conditioning-sensitive and performs about the same as the
record-and-replay path; it is an explicit opt-in engine, deliberately not in
``choose_engine``).

Mechanics (mirrors ``warp_replay.make_rosenbrock_replay_closure``'s frozen-controller
note):
- The frozen step mesh ``dts`` is recorded once at concrete ``params0`` and the given
  ``rtol``/``atol`` (stiff replay needs a fine mesh — robertson rtol~1e-12, atol~1e-14 —
  else the forward blows up).
- The forward ``f(params) -> y_final[n, D]`` runs the fused Warp kernel; Warp is not
  JAX-traceable, so it is wrapped in ``jax.pure_callback``.
- The backward seeds the incoming output cotangent ``ybar`` onto ``y_traj.grad[:, S, :]``
  via ``fused_vjp`` (the generalized cotangent seed, not the hard-coded sum(y**2) one),
  runs ``tape.backward()``, and returns ``(grad_y0, grad_pvec)`` — also via
  ``pure_callback``. Only the param gradient is propagated (``y0`` is a closed-over
  constant, like ``dts``/``field_key``/``dim``/``t0``).
- ``gpvec[n, D]`` (padded layout) is mapped back to the native param shape ``[n, P]`` by
  the inverse of ``warp_rosenbrock._params_of`` + the kernel's pad/truncate-to-D: take the
  leading ``P`` columns (robertson native P=3 == D=3; for fields whose native P < D the
  extra columns were zero-padded and carry no native gradient; P > D is truncated by the
  kernel, so those native columns are unreachable and reported zero).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.warp import fused_rosenbrock as _fused_rb
from gradsolve.warp import warp_replay as _warp_replay
from gradsolve.warp import warp_rosenbrock

# The fused Rosenbrock23 backward needs NVIDIA Warp. ``fused_rosenbrock`` imports as a stub in
# a warp-less environment and exposes a single shared availability flag; reuse it here so
# ``import gradsolve`` succeeds everywhere and the fused lane reports itself unavailable (rather
# than importable-but-broken) when Warp is absent. The fused_rosenbrock_backward engine is
# override-only, so deferring the hard failure to actual use keeps the package importable while
# still erroring clearly if invoked without Warp.
_FUSED_AVAILABLE = _fused_rb._FUSED_AVAILABLE


def fused_rosenbrock_grad_closure(problem, y0, params, *, rtol=1e-12, atol=1e-14,
                                  device="cpu"):
    """Record the frozen stiff mesh once, return a ``jax.grad``-able closure
    ``params_jax -> y_final[n, D]`` whose reverse pass is the genuine single-kernel fused
    Warp Rosenbrock23 backward (``wp.Tape`` + the custom analytic solve adjoint).

    Frozen-controller scope (identical to ``make_rosenbrock_replay_closure``): the dts
    are recorded at concrete ``params`` and the controller is not re-run when params are
    perturbed — the gradient is the exact discrete adjoint of the replayed frozen-step
    integration, valid in a neighbourhood of ``params``. Override-only engine; not
    added to ``choose_engine``.
    """
    if not _FUSED_AVAILABLE:
        raise RuntimeError(
            "fused_rosenbrock_backward requires NVIDIA Warp (the fused kernel); "
            "warp is not importable in this environment. Use engine='warp_rosenbrock' "
            "(pure-JAX record-replay reverse) instead.")
    if not warp_rosenbrock.supports(problem):
        raise ValueError(
            f"fused_rosenbrock_backward does not support problem {problem.name!r}; "
            "supported: 'robertson', 'hires', and any 'linstiff_*' field you register "
            "first via warp_rosenbrock.register_linstiff(key, A).")

    field_key, dim = warp_rosenbrock._field_for(problem)
    t0 = float(problem.t0)
    y0_np = np.asarray(y0, dtype=np.float64)
    params_np = np.asarray(params, dtype=np.float64)
    P = params_np.shape[1] if params_np.ndim == 2 else 1

    # Record the frozen STABLE mesh ONCE at concrete params + the closure's tolerances.
    # record_rosenbrock applies _params_of internally; we keep the RAW native params here
    # and re-map to the padded pvec inside the callbacks (so jax.grad sees native shape).
    _yw, _nacc, dts = _warp_replay.record_rosenbrock(
        problem, y0_np, params_np, rtol=rtol, atol=atol, device=device)
    dts = np.asarray(dts, dtype=np.float64)
    n = y0_np.shape[0]

    def _pvec_of(params_native):
        # native [n, P] -> padded pvec [n, D] (exactly the kernel's _inputs pad/truncate,
        # via the backend's _params_of for the field-specific mapping).
        return np.asarray(warp_rosenbrock._params_of(problem, params_native),
                          dtype=np.float64)

    def _forward_np(params_native):
        pvec = _pvec_of(np.asarray(params_native, dtype=np.float64))
        yf = _fused_rb.fused_forward(field_key, dim, y0_np, pvec, dts, t0, device=device)
        return np.asarray(yf, dtype=np.float64)

    def _bwd_np(params_native, ybar):
        pvec = _pvec_of(np.asarray(params_native, dtype=np.float64))
        gy = np.asarray(ybar, dtype=np.float64)
        _gy0, gpvec = _fused_rb.fused_vjp(
            field_key, dim, y0_np, pvec, dts, t0, gy, device=device)
        # map padded gpvec[n, D] back to native params shape [n, P]: leading P columns
        # (the inverse of _params_of + pad/truncate-to-D for these autonomous/full fields).
        gpvec = np.asarray(gpvec, dtype=np.float64)
        gp_native = np.zeros((n, P), dtype=np.float64)
        cols = min(P, dim)
        gp_native[:, :cols] = gpvec[:, :cols]
        return gp_native

    out_shape = jax.ShapeDtypeStruct((n, dim), jnp.float64)
    grad_shape = jax.ShapeDtypeStruct((n, P), jnp.float64)

    @jax.custom_vjp
    def f(params_jax):
        return jax.pure_callback(_forward_np, out_shape, params_jax)

    def f_fwd(params_jax):
        y_final = jax.pure_callback(_forward_np, out_shape, params_jax)
        return y_final, (params_jax,)

    def f_bwd(res, ybar):
        (params_jax,) = res
        grad_params = jax.pure_callback(_bwd_np, grad_shape, params_jax, ybar)
        return (grad_params,)

    f.defvjp(f_fwd, f_bwd)

    return f
