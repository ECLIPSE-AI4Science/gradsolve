"""Method-abstraction layer for gradsolve's record-and-replay solvers.

The abstraction a second adaptive method needs, factored out of the Tsit5/Rosenbrock23-specific
code. Two tiers:

* ``advance(f, t, y, dt, p) -> y_next``   value-only; what the replay scan and ``saveat``
  use. For the built-in methods this is the existing single-step function (``tsit5_step`` /
  ``rosenbrock23_step``) by identity, so the proven value path cannot drift.
* ``trial_step(f, t, y, dt, p, k1=None) -> StepResult`` value + embedded error + optional
  FSAL trailing stage; what the adaptive recorder uses. Kept separate so the value path never
  pays for the error/stage work.

The PI-controller history is not a step output — it lives in a recorder-owned
``ControllerState``, so a numerical method is not coupled to a specific controller.

``record_adaptive`` is the method-generic accepted-mesh recorder. Instantiated with
``TSIT5_METHOD`` it reproduces ``gradsolve.solvers.tsit5_replay.tsit5_adaptive`` byte-for-byte
(a test pins this); the nonstiff high-order lane instantiates it with ``VERN7_METHOD``.

Pure host/numpy arithmetic in the recorder; ``advance``/``trial_step`` are array-library
agnostic (pure arithmetic on the inputs), so they are equally valid on numpy and traceable
under jit/grad/vmap. float64 (gradsolve enables jax x64 at import).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import numpy as np

from gradsolve.solvers.rosenbrock23_step import rosenbrock23_step
from gradsolve.solvers.tsit5_step import tsit5_step

# --- adaptive-controller constants (shared with tsit5_step; Hairer-Norsett-Wanner II.4) ---
SAFETY, ERR_FLOOR, FAC_MIN, FAC_MAX = 0.9, 1e-10, 0.2, 5.0


class StepResult(NamedTuple):
    """One trial step's outputs — value, embedded error vector, optional FSAL stage.

    Deliberately does not carry controller state: PI history belongs to the recorder's
    ControllerState, because it depends on prior accept/reject decisions, not on the method.
    """
    y_next: object            # (dim,) propagated solution — byte-identical to advance(...)
    y_err: object             # (dim,) embedded error estimate (already dt-scaled)
    fsal: Optional[object]    # trailing-stage derivative to reuse as next k1, or None


class ControllerState(NamedTuple):
    """Recorder-owned PI-controller history."""
    err_prev: float           # previous accepted normalized error (1.0 before any accept)
    accepted_any: bool        # whether any step has been accepted yet (PI kicks in after)


@dataclass(frozen=True)
class Method:
    """A one-step method on the two-tier contract.

    Attributes
    ----------
    name : str
    order : int              order p of the propagated solution (local error ~ dt**(p+1)).
    error_order : int        order of the embedded estimate; the I/PI exponent uses
                             ``1/(error_order+1)`` (Tsit5: 4 -> 0.2, matching ORDER_EXP=-0.2).
    fsal : bool              whether trial_step returns a reusable trailing stage.
    advance : Callable       (f,t,y,dt,p) -> y_next   [value-only; the existing step fn].
    trial_step : Callable    (f,t,y,dt,p,k1=None) -> StepResult.
    beta1, beta2 : float     PI gains. beta2==0 reduces to the I-controller; Tsit5 = (0.2, 0.0)
                             reproduces the current I-control bit-for-bit.
    needs_jacobian : bool    whether trial_step calls jax.jacfwd (the Rosenbrock family), which
                             requires a traceable JAX RHS and hence a dedicated jitted recorder.
    """
    name: str
    order: int
    error_order: int
    fsal: bool
    advance: Callable
    trial_step: Callable
    beta1: float
    beta2: float
    needs_jacobian: bool = False   # True => the trial_step calls jax.jacfwd; record via the
    #                                dedicated jitted recorder (record_rodas5p), NOT record_adaptive.


def controller_update(method: Method, cstate: ControllerState, err: float, h: float):
    """One PI/I step-size update; returns ``(dt_next, cstate_next)`` for an accepted step.

    Byte-compatible with ``tsit5_adaptive`` when ``method.beta2 == 0`` and
    ``method.beta1 == 0.2``:
        factor = SAFETY * max(err, ERR_FLOOR)**(-beta1) * err_prev**beta2
        dt_next = h * clip(factor, FAC_MIN, FAC_MAX)
    With beta2 == 0, ``err_prev**0.0 == 1.0`` exactly and ``x * 1.0 == x`` exactly, so the PI
    form collapses to ``SAFETY * max(err, ERR_FLOOR)**(-0.2)`` bit-for-bit.
    """
    e = max(err, ERR_FLOOR)
    prev = cstate.err_prev if cstate.accepted_any else 1.0
    factor = SAFETY * e ** (-method.beta1) * prev ** method.beta2
    factor = min(max(factor, FAC_MIN), FAC_MAX)
    return h * factor, ControllerState(err_prev=err, accepted_any=True)


def controller_reject(cstate: ControllerState, err: float, h: float, method: Method):
    """Step-size update for a rejected step. Matches tsit5_adaptive: the controller history is
    not advanced on rejection (err_prev/accepted_any unchanged), only dt shrinks."""
    e = max(err, ERR_FLOOR)
    factor = SAFETY * e ** (-method.beta1)   # I-form on rejection (no PI memory advance)
    factor = min(max(factor, FAC_MIN), FAC_MAX)
    return h * factor, cstate


def record_adaptive(method: Method, f, y0, t0, t1, rtol, atol, dt0, max_steps):
    """Method-generic accepted-mesh recorder (host/numpy).

    Returns ``(y_final, dts[accepted], n_rejected, status)`` with ``status==0`` iff the
    integration reached ``t1``. Mirrors ``tsit5_adaptive`` exactly: WRMS I/PI control, the
    ``h = min(dt, t1 - t)`` clamp to hit t1, FSAL reuse when ``method.fsal``.
    """
    if method.needs_jacobian:
        raise ValueError(
            f"record_adaptive is the numpy host recorder for explicit methods; {method.name!r} "
            "sets needs_jacobian=True (its trial_step calls jax.jacfwd) and must record via its "
            "dedicated jitted recorder (e.g. record_rodas5p).")
    # The host recorder RHS is 2-arg ``f(t, y)`` (params baked in, matching tsit5_adaptive /
    # record_tsit5_jax's f_eval). trial_step uses the 3-arg ``f(t, y, p)`` convention (gradsolve's
    # f_jax), so adapt here: the p slot is unused (None) and the wrapper adds no arithmetic, so
    # byte-identity with tsit5_adaptive is preserved.
    def f3(t, y, p):
        return f(t, y)
    y, t, dt = np.array(y0, dtype=np.float64), float(t0), float(dt0)
    k1 = f(t, y) if method.fsal else None
    cstate = ControllerState(err_prev=1.0, accepted_any=False)
    dts, n_rej = [], 0
    for _ in range(max_steps):
        if t >= t1:
            break
        h = min(dt, t1 - t)
        sr = method.trial_step(f3, t, y, h, None, k1) if method.fsal \
            else method.trial_step(f3, t, y, h, None)
        y_next, e = sr.y_next, sr.y_err
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        err = float(np.sqrt(np.mean((e / sc) ** 2)))
        if err <= 1.0:
            t, y = t + h, y_next
            if method.fsal:
                k1 = sr.fsal
            dts.append(h)
            dt, cstate = controller_update(method, cstate, err, h)
        else:
            n_rej += 1
            dt, cstate = controller_reject(cstate, err, h, method)
    status = 0 if t >= t1 else 1
    return y, np.array(dts), n_rej, status


# ---------------------------------------------------------------------------
# trial_step implementations for the built-in methods
# ---------------------------------------------------------------------------

def _tsit5_trial_step(f, t, y, dt, p, k1=None):
    """Tsit5 7-stage trial step -> StepResult(y5, dt*sum(E_i k_i), k7).

    Same arithmetic as tsit5_step (advance) plus the embedded E-weighted error. y_next is
    byte-identical to tsit5_step: tsit5_step adds ``_B7*k7`` with _B7==0.0 (i.e. +0.0), and y5
    here omits that exact +0.0 term — a test asserts the two agree bit-for-bit.
    """
    from gradsolve.solvers.tsit5_step import (
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
    if k1 is None:
        k1 = f(t, y, p)
    k2 = f(t + _C[0] * dt, y + dt * (_A21 * k1), p)
    k3 = f(t + _C[1] * dt, y + dt * (_A31 * k1 + _A32 * k2), p)
    k4 = f(t + _C[2] * dt, y + dt * (_A41 * k1 + _A42 * k2 + _A43 * k3), p)
    k5 = f(t + _C[3] * dt, y + dt * (_A51 * k1 + _A52 * k2 + _A53 * k3 + _A54 * k4), p)
    k6 = f(t + _C[4] * dt,
           y + dt * (_A61 * k1 + _A62 * k2 + _A63 * k3 + _A64 * k4 + _A65 * k5), p)
    y5 = y + dt * (_B1 * k1 + _B2 * k2 + _B3 * k3 + _B4 * k4 + _B5 * k5 + _B6 * k6)
    k7 = f(t + _C[5] * dt, y5, p)
    e = dt * (_E1 * k1 + _E2 * k2 + _E3 * k3 + _E4 * k4 + _E5 * k5 + _E6 * k6 + _E7 * k7)
    return StepResult(y_next=y5, y_err=e, fsal=k7)


def _rosenbrock23_trial_step(f, t, y, dt, p, k1=None):
    """Rosenbrock23 trial step -> StepResult(ynew, err_vec, None).

    Requires jax (in-step jacfwd + linear solves); the k3 correction gives the order-3
    embedded estimate (Shampine & Reichelt 1997). Not FSAL. ``k1`` is ignored (kept for
    signature parity). NB: the stiff lane's recorder is the fused Warp kernel; this trial_step
    exists for the trial_step-consistency test, not to replace that recorder.
    """
    import jax
    import jax.numpy as jnp
    del k1
    d = 1.0 / (2.0 + np.sqrt(2.0))
    e32 = 6.0 + np.sqrt(2.0)
    D = y.shape[-1]
    eye = jnp.eye(D, dtype=y.dtype)
    J = jax.jacfwd(f, argnums=1)(t, y, p)
    W = eye - dt * d * J
    F0 = f(t, y, p)
    k1s = jnp.linalg.solve(W, F0)
    F1 = f(t, y + 0.5 * dt * k1s, p)
    k2 = jnp.linalg.solve(W, F1 - k1s) + k1s
    ynew = y + dt * k2
    F2 = f(t, ynew, p)
    k3 = jnp.linalg.solve(W, F2 - e32 * (k2 - F1) - 2.0 * (k1s - F0))
    err_vec = (dt / 6.0) * (k1s - 2.0 * k2 + k3)
    return StepResult(y_next=ynew, y_err=err_vec, fsal=None)


TSIT5_METHOD = Method(
    name="tsit5", order=5, error_order=4, fsal=True,
    advance=tsit5_step, trial_step=_tsit5_trial_step, beta1=0.2, beta2=0.0,
)

ROSENBROCK23_METHOD = Method(
    name="rosenbrock23", order=2, error_order=3, fsal=False,
    advance=rosenbrock23_step, trial_step=_rosenbrock23_trial_step, beta1=1.0 / 3.0, beta2=0.0,
    needs_jacobian=True,
)
