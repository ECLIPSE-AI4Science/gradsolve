"""Vern7 single-step core (gradsolve/solvers/vern7_step.py).

Implemented from Verner (2010): the "most efficient" RK7(6) pair, 10-stage main method (the
lazy interpolant is out of scope — gradsolve keeps re-step saveat). Correctness gates:
  1. node-consistency: every A-row sums to its c-node.
  2. local order 8: single-step error ~ dt**8 (log-log slope), on three linear systems whose
     exact flow is expm(A dt) y0.
  3. embedded error order ~7 and shrinks monotonically with dt.
  4. dt == 0 is an exact identity step in value, JVP and VJP (the replay padding contract).

No diffrax cross-check (diffrax 0.7.2 has no Vern7); scipy DOP853 backs the trajectory
reference in test_vern7_replay.py. Imports only gradsolve (+ numpy/jax/scipy/pytest).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import gradsolve  # noqa: F401
from gradsolve.solvers.vern7_step import (
    VERN7_A,
    VERN7_C,
    vern7_advance,
    vern7_trial_step,
)

_SYSTEMS = {
    "spiral": (np.array([[-1.0, 2.0], [-3.0, -1.0]]), np.array([1.0, 0.5])),
    "diag": (np.array([[-2.0, 0.0], [0.0, -5.0]]), np.array([1.0, 0.5])),
    "nonnormal3": (np.array([[-1.0, 1.0, 0.0], [0.0, -2.0, 1.0], [0.0, 0.0, -3.0]]),
                   np.array([1.0, -0.3, 0.7])),
}
_NAMES = sorted(_SYSTEMS)
_P = jnp.zeros(0, dtype=jnp.float64)
_ORDER_DTS = (0.2, 0.1, 0.05, 0.025, 0.0125)


def _rhs(A):
    Aj = jnp.asarray(A, dtype=jnp.float64)

    def f(t, y, p):
        del t, p
        return Aj @ y
    return f


def _exact(A, y0, dt):
    return expm(np.asarray(A) * dt) @ np.asarray(y0)


def _slope(dts, errs):
    return float(np.polyfit(np.log(dts), np.log(errs), 1)[0])


# --- 1. node consistency (cheap transcription check) --------------------------------

def test_vern7_node_consistency():
    """Every lower-triangular A-row must sum to its c-node (a necessary Runge-Kutta
    consistency condition; catches most transcription typos immediately)."""
    for i, c_i in enumerate(VERN7_C):
        row_sum = sum(VERN7_A[i][:i])   # stages 0..i-1 feed stage i
        assert abs(row_sum - c_i) < 1e-13, f"row {i}: sum(A)={row_sum} != c={c_i}"


# --- 2. local order 8 -----------------------------------------------------------------
#
# Why the order test is not a plain log-log slope. A 5-point log-log error-vs-dt slope on
# three linear systems, requiring slope in [7.6, 8.8], is fragile for an order-8 method in
# double precision, and a correct tableau fails it: (a) at dt=0.0125 the local error (~dt**8 ~ 6e-16) underflows to exactly 0, so np.log(0)
# gives a NaN slope; (b) the clean asymptotic decade (error above the ~1e-15 roundoff floor yet
# small enough to be asymptotic) is narrow and sits at a different dt for each system's
# error-coefficient magnitude, so no fixed grid lands all three on a slope-8 fit (the fast
# eigenvalue of diag(-2,-5) is pre-asymptotic wherever it is also above roundoff -> slope ~6.8).
# The order is instead verified exactly via the linear Runge-Kutta order conditions (roundoff-free, and a
# stricter transcription gate than the slope), then confirm the actual step function reaches
# order ~8 on one well-conditioned trajectory. Independently, the recorder-vs-DOP853 parity
# test in test_vern7_replay.py exercises full-trajectory order-8 accuracy.


def _order_residuals(weights):
    """tau_k = w . A^(k-1) . 1 - 1/k!  for k=1..8.

    Applied to y' = A y, one Verner step is R(A dt) y0 with R(w) = 1 + sum_k (w.A^(k-1).1) w^k,
    so a method with weights w has order p on linear systems iff w.A^(k-1).1 == 1/k! for
    k=1..p. A single wrong b_i (k=1 term) or A_ij (k>=2 terms) breaks one residual immediately.
    """
    import math
    S = len(VERN7_C)
    Amat = np.zeros((S, S))
    for i in range(S):
        for j in range(len(VERN7_A[i])):
            Amat[i, j] = VERN7_A[i][j]
    w = np.asarray(weights, dtype=np.float64)
    v = np.ones(S)                       # A^0 . 1
    res = []
    for k in range(1, 9):
        res.append(float(w @ v) - 1.0 / math.factorial(k))
        v = Amat @ v                     # advance to A^k . 1
    return res


def test_vern7_propagated_weights_are_exact_order_seven():
    """Propagated b satisfies the order-7 linear conditions exactly (local error O(dt**8)) and
    breaks at order 8 — the roundoff-free order-8 proof and the transcription gate."""
    from gradsolve.solvers.vern7_step import VERN7_B
    res = _order_residuals(VERN7_B)
    for k in range(1, 8):                # matched through 1/7!
        assert abs(res[k - 1]) < 1e-12, f"vern7 b breaks order condition k={k}: {res[k-1]:.2e}"
    assert abs(res[7]) > 1e-9, f"vern7 b unexpectedly satisfies order 8 (resid {res[7]:.2e})"


def test_vern7_embedded_weights_are_exact_order_six():
    """Embedded bhat satisfies the order-6 linear conditions exactly and breaks at order 7."""
    from gradsolve.solvers.vern7_step import VERN7_BHAT
    res = _order_residuals(VERN7_BHAT)
    for k in range(1, 7):                # matched through 1/6!
        assert abs(res[k - 1]) < 1e-12, f"vern7 bhat breaks order condition k={k}: {res[k-1]:.2e}"
    assert abs(res[6]) > 1e-9, f"vern7 bhat unexpectedly satisfies order 7 (resid {res[6]:.2e})"


def test_vern7_advance_reaches_order_eight_on_a_trajectory():
    """The actual step function (stage recursion + b-weights, not just the tableau constants)
    reaches order ~8 on a well-conditioned linear system (spiral, |eig|~2.65), over a dt grid
    kept above the roundoff floor and inside the asymptotic regime."""
    A, y0 = _SYSTEMS["spiral"]
    f = _rhs(A)
    yj = jnp.asarray(y0, dtype=jnp.float64)
    dts = (0.2, 0.15, 0.1, 0.07, 0.05)
    errs = [float(np.linalg.norm(np.asarray(vern7_advance(f, 0.3, yj, dt, _P)) - _exact(A, y0, dt)))
            for dt in dts]
    assert min(errs) > 1e-15, f"error underflowed the roundoff floor: {errs}"
    slope = _slope(dts, errs)
    assert 7.4 <= slope <= 8.8, f"vern7 trajectory order {slope:.3f} far from 8"
    assert errs[0] < 1e-6, f"vern7 dt=0.2 error {errs[0]:.3e} too large"


# --- 3. embedded error order ----------------------------------------------------------

@pytest.mark.parametrize("system", _NAMES)
def test_vern7_embedded_error_shrinks_and_has_right_order(system):
    A, y0 = _SYSTEMS[system]
    f = _rhs(A)
    yj = jnp.asarray(y0, dtype=jnp.float64)
    norms = [float(jnp.linalg.norm(vern7_trial_step(f, 0.3, yj, dt, _P).y_err))
             for dt in _ORDER_DTS]
    assert all(a > b for a, b in zip(norms, norms[1:])), f"not monotone: {norms}"
    slope = _slope(_ORDER_DTS, norms)
    assert 6.3 <= slope <= 8.2, f"vern7 embedded order {slope:.3f} off ({system})"


@pytest.mark.parametrize("system", _NAMES)
def test_vern7_trial_step_value_byte_identical_to_advance(system):
    A, y0 = _SYSTEMS[system]
    f = _rhs(A)
    yj = jnp.asarray(y0, dtype=jnp.float64)
    adv = vern7_advance(f, 0.3, yj, 0.05, _P)
    tri = vern7_trial_step(f, 0.3, yj, 0.05, _P)
    np.testing.assert_array_equal(np.asarray(tri.y_next), np.asarray(adv))


# --- 4. dt == 0 identity in value / JVP / VJP (replay padding contract) ---------

@pytest.mark.parametrize("system", _NAMES)
def test_vern7_dt_zero_value_identity(system):
    A, y0 = _SYSTEMS[system]
    f = _rhs(A)
    yj = jnp.asarray(y0, dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(vern7_advance(f, 0.3, yj, 0.0, _P)), np.asarray(y0))


def test_vern7_dt_zero_jvp_is_identity_in_y_zero_in_p():
    """d y_next/dy == I and d y_next/dp == 0 at dt==0 (forward mode)."""
    A = np.array([[-1.0, 2.0], [-3.0, -1.0]])
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.array([0.7], dtype=jnp.float64)

    def g(t, y, p):  # p-dependent field so the p-derivative is meaningful
        return jnp.asarray(A) @ y + p[0] * y

    yt = jnp.array([0.3, -0.9], dtype=jnp.float64)
    _, jvp_y = jax.jvp(lambda yy: vern7_advance(g, 0.3, yy, 0.0, p), (y,), (yt,))
    np.testing.assert_allclose(np.asarray(jvp_y), np.asarray(yt), rtol=0, atol=0)
    pt = jnp.array([1.0], dtype=jnp.float64)
    _, jvp_p = jax.jvp(lambda pp: vern7_advance(g, 0.3, y, 0.0, pp), (p,), (pt,))
    np.testing.assert_array_equal(np.asarray(jvp_p), np.zeros_like(np.asarray(y)))


def test_vern7_dt_zero_vjp_passes_cotangent_through_y_zero_in_p():
    A = np.array([[-1.0, 2.0], [-3.0, -1.0]])
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.array([0.7], dtype=jnp.float64)

    def g(t, y, p):
        return jnp.asarray(A) @ y + p[0] * y

    out, pull = jax.vjp(lambda yy, pp: vern7_advance(g, 0.3, yy, 0.0, pp), y, p)
    ct = jnp.array([1.0, -2.0], dtype=jnp.float64)
    gy, gp = pull(ct)
    np.testing.assert_array_equal(np.asarray(gy), np.asarray(ct))   # identity in y
    np.testing.assert_array_equal(np.asarray(gp), np.zeros_like(np.asarray(p)))  # zero in p
