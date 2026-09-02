"""Method-order + embedded-error-estimate checks for the single-step cores.

Covers ``gradsolve/solvers/tsit5_step.py`` (Tsitouras 2011, 7-stage explicit 5(4) pair)
and ``gradsolve/solvers/rosenbrock23_step.py`` (Shampine & Reichelt 1997 ``ode23s``,
linearly-implicit 2(3) pair) against the one system with a known closed-form flow any
RK/Rosenbrock method can be checked against exactly: the linear autonomous system
``dy/dt = A y``, whose exact solution is ``y(t+dt) = expm(A*dt) @ y(t)``
(``scipy.linalg.expm``).

Two properties per solver, both promised in the module docstrings:
  1. a single step matches the exact flow to the solver's *local* order (error ~
     C * dt**(p+1)), verified via a log-log convergence-rate fit across a range of
     dt -- not just an absolute-threshold check at one dt;
  2. the embedded error estimate shrinks monotonically with dt, at its own
     theoretical rate (one order below the propagated solution's local order).

Neither step function returns its embedded estimate -- both only propagate the
accepted solution (see the module docstrings: Tsit5 keeps the b7=0 k7 term but
never forms ``b - bhat``; Rosenbrock23 stops after the order-2 ``ynew``, no k3).
So each estimate is recomputed here from arithmetic the module already exposes:
Tsit5's private tableau (``_C``/``_A**``/``_B*``/``_E*``, already bit-verified
against diffrax in ``tests/test_tsit5_error_weights.py``) run through the identical
7-stage block; Rosenbrock23's published k3 correction (Shampine & Reichelt 1997 eq.
3 / Hairer-Wanner II, the same arithmetic as
``tests/_oracles/gradsolve_g2_spike_oracle.py::rosenbrock23_adaptive``'s single-step
core), built on top of the same W/k1/F1/k2 the module computes -- cross-checked
below to reproduce the module's own ``ynew`` bit-for-bit.

Both step functions take a bare ``f(t, y, p)`` callable (see their signatures in
``gradsolve/solvers/tsit5_step.py`` / ``rosenbrock23_step.py`` and the ``Problem``
contract in ``gradsolve/base.py``) -- no ``Problem`` object is needed at the
single-step level, so the linear test fields below are tiny inline closures, not
duck-typed ``Problem`` instances.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import gradsolve  # noqa: F401 -- import side effect: enables jax float64 before any array op
from gradsolve.solvers import tsit5_step as _tsit5_mod
from gradsolve.solvers.rosenbrock23_step import rosenbrock23_step
from gradsolve.solvers.tsit5_step import tsit5_step

# --- linear test systems: dy/dt = A y, exact flow = expm(A dt) y0 ------------------
# Three matrices spanning a damped spiral (complex eigenvalues), a diagonal (real,
# well-separated eigenvalues) and a non-normal triangular system, so the order fits
# below aren't an artifact of one particular A.
_SYSTEMS = {
    "spiral": (
        jnp.array([[-1.0, 2.0], [-3.0, -1.0]], dtype=jnp.float64),
        jnp.array([1.0, 0.5], dtype=jnp.float64),
    ),
    "diag": (
        jnp.array([[-2.0, 0.0], [0.0, -5.0]], dtype=jnp.float64),
        jnp.array([1.0, 0.5], dtype=jnp.float64),
    ),
    "nonnormal3": (
        jnp.array([[-1.0, 1.0, 0.0], [0.0, -2.0, 1.0], [0.0, 0.0, -3.0]], dtype=jnp.float64),
        jnp.array([1.0, -0.3, 0.7], dtype=jnp.float64),
    ),
}

_P = jnp.zeros(0, dtype=jnp.float64)  # both problems below carry no free params
_SYSTEM_NAMES = sorted(_SYSTEMS)


def _linear_rhs(A):
    """dy/dt = A y as a bare f(t, y, p) closure -- exactly what both step functions expect."""
    def f(t, y, p):
        del t, p
        return A @ y
    return f


def _exact_flow(A, y0, dt):
    return expm(np.asarray(A, dtype=np.float64) * dt) @ np.asarray(y0, dtype=np.float64)


def _loglog_slope(dts, errs):
    """Least-squares convergence rate: err ~ C * dt**slope, fit over log(dt)/log(err)."""
    return float(np.polyfit(np.log(dts), np.log(errs), 1)[0])


# A geometric dt ladder: large enough that dt=0.2 stays well inside both methods'
# stability region for these A's, small enough (down to 0.0125) that the fit is deep
# in the asymptotic regime without yet hitting float64 roundoff noise (checked
# empirically -- the smallest per-step error above is ~2e-11, ~1e4 above eps-scale).
_ORDER_DTS = (0.2, 0.1, 0.05, 0.025, 0.0125)


# =====================================================================================
# Property 1: a single step matches exp(A dt) y0 to the method's local order.
# =====================================================================================

@pytest.mark.parametrize("system", _SYSTEM_NAMES)
def test_tsit5_step_matches_matrix_exponential_to_order(system):
    """Tsit5 is a 5th-order pair -> local (single-step) error ~ dt**6 (Tsitouras 2011)."""
    A, y0 = _SYSTEMS[system]
    f = _linear_rhs(A)
    errs = [
        np.linalg.norm(np.asarray(tsit5_step(f, 0.3, y0, dt, _P)) - _exact_flow(A, y0, dt))
        for dt in _ORDER_DTS
    ]
    slope = _loglog_slope(_ORDER_DTS, errs)
    assert 5.7 <= slope <= 6.8, f"tsit5 local order {slope:.3f} far from theoretical 6 ({system})"
    # belt-and-suspenders absolute check: the coarsest step is still accurate to ~4-5 digits.
    assert errs[0] < 5e-4, f"tsit5 step error {errs[0]:.3e} too large at dt=0.2 ({system})"


@pytest.mark.parametrize("system", _SYSTEM_NAMES)
def test_rosenbrock23_step_matches_matrix_exponential_to_order(system):
    """Rosenbrock23 propagates the 2nd-order solution -> local error ~ dt**3 (Shampine & Reichelt 1997)."""
    A, y0 = _SYSTEMS[system]
    f = _linear_rhs(A)
    errs = [
        np.linalg.norm(np.asarray(rosenbrock23_step(f, 0.3, y0, dt, _P)) - _exact_flow(A, y0, dt))
        for dt in _ORDER_DTS
    ]
    slope = _loglog_slope(_ORDER_DTS, errs)
    assert 2.5 <= slope <= 3.3, f"rosenbrock23 local order {slope:.3f} far from theoretical 3 ({system})"
    assert errs[0] < 2e-2, f"rosenbrock23 step error {errs[0]:.3e} too large at dt=0.2 ({system})"


# =====================================================================================
# Property 2: the embedded error estimate shrinks with dt, at its own theoretical rate.
# =====================================================================================

def _tsit5_error_estimate(f, t, y, dt, p):
    """dt * sum(E_i k_i): the 7-stage block re-run with tsit5_step's OWN (imported, not
    re-derived) tableau constants, to recover the b-bhat difference the step function
    computes internally but does not return."""
    m = _tsit5_mod
    k1 = f(t, y, p)
    k2 = f(t + m._C[0] * dt, y + dt * (m._A21 * k1), p)
    k3 = f(t + m._C[1] * dt, y + dt * (m._A31 * k1 + m._A32 * k2), p)
    k4 = f(t + m._C[2] * dt, y + dt * (m._A41 * k1 + m._A42 * k2 + m._A43 * k3), p)
    k5 = f(t + m._C[3] * dt,
           y + dt * (m._A51 * k1 + m._A52 * k2 + m._A53 * k3 + m._A54 * k4), p)
    k6 = f(t + m._C[4] * dt,
           y + dt * (m._A61 * k1 + m._A62 * k2 + m._A63 * k3 + m._A64 * k4 + m._A65 * k5), p)
    k7 = f(t + m._C[5] * dt,
           y + dt * (m._B1 * k1 + m._B2 * k2 + m._B3 * k3 + m._B4 * k4 + m._B5 * k5 + m._B6 * k6),
           p)
    return dt * (m._E1 * k1 + m._E2 * k2 + m._E3 * k3 + m._E4 * k4 + m._E5 * k5
                  + m._E6 * k6 + m._E7 * k7)


def _rosenbrock23_error_estimate(f, t, y, dt, p):
    """3rd-order embedded correction (Shampine & Reichelt 1997): recomputes the SAME
    W/F0/k1/F1/k2/ynew arithmetic rosenbrock23_step runs internally (same d = 1/(2+sqrt2)
    coefficient), then adds the k3 correction stage to form the error vector. Returns
    (err_vec, ynew) so the caller can cross-check ynew against the module's own output."""
    d = 1.0 / (2.0 + jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64)))
    e32 = 6.0 + jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    D = y.shape[-1]
    eye = jnp.eye(D, dtype=y.dtype)
    J = jax.jacfwd(f, argnums=1)(t, y, p)
    W = eye - dt * d * J
    F0 = f(t, y, p)
    k1 = jnp.linalg.solve(W, F0)
    F1 = f(t, y + 0.5 * dt * k1, p)
    k2 = jnp.linalg.solve(W, F1 - k1) + k1
    ynew = y + dt * k2
    F2 = f(t, ynew, p)
    k3 = jnp.linalg.solve(W, F2 - e32 * (k2 - F1) - 2.0 * (k1 - F0))
    err_vec = (dt / 6.0) * (k1 - 2.0 * k2 + k3)
    return err_vec, ynew


@pytest.mark.parametrize("system", _SYSTEM_NAMES)
def test_tsit5_error_estimate_shrinks_with_dt(system):
    A, y0 = _SYSTEMS[system]
    f = _linear_rhs(A)
    norms = [
        float(jnp.linalg.norm(_tsit5_error_estimate(f, 0.3, y0, dt, _P)))
        for dt in _ORDER_DTS
    ]
    assert all(a > b for a, b in zip(norms, norms[1:])), f"not monotone shrinking: {norms}"
    slope = _loglog_slope(_ORDER_DTS, norms)
    # the b - bhat embedded difference is O(dt**5) (one order below the propagated dt**6).
    assert 4.5 <= slope <= 5.5, f"tsit5 error-estimate order {slope:.3f} far from theoretical 5 ({system})"


@pytest.mark.parametrize("system", _SYSTEM_NAMES)
def test_rosenbrock23_error_estimate_shrinks_with_dt(system):
    A, y0 = _SYSTEMS[system]
    f = _linear_rhs(A)
    norms = []
    for dt in _ORDER_DTS:
        err_vec, ynew = _rosenbrock23_error_estimate(f, 0.3, y0, dt, _P)
        # cross-check: the recomputed k1/k2/ynew arithmetic reproduces the module's own
        # propagated value bit-for-bit (same operations, same order) -- so the error
        # estimate is anchored to the actual step, not a drifted re-derivation.
        np.testing.assert_array_equal(
            np.asarray(ynew), np.asarray(rosenbrock23_step(f, 0.3, y0, dt, _P))
        )
        norms.append(float(jnp.linalg.norm(err_vec)))
    assert all(a > b for a, b in zip(norms, norms[1:])), f"not monotone shrinking: {norms}"
    slope = _loglog_slope(_ORDER_DTS, norms)
    # the embedded estimate approximates the order-2 method's own local error -> O(dt**3).
    assert 2.5 <= slope <= 3.3, f"rosenbrock23 error-estimate order {slope:.3f} far from theoretical 3 ({system})"


# =====================================================================================
# Bonus: the dt == 0 identity-step contract stated explicitly in both module docstrings
# ("dt == 0 is exactly an identity step in both value and gradient") -- cheap, exact
# (not just close), and directly documents a padding contract the replay adjoin relies on.
# =====================================================================================

@pytest.mark.parametrize("system", _SYSTEM_NAMES)
def test_tsit5_step_dt_zero_is_exact_identity(system):
    A, y0 = _SYSTEMS[system]
    f = _linear_rhs(A)
    y1 = tsit5_step(f, 0.3, y0, 0.0, _P)
    np.testing.assert_array_equal(np.asarray(y1), np.asarray(y0))


@pytest.mark.parametrize("system", _SYSTEM_NAMES)
def test_rosenbrock23_step_dt_zero_is_exact_identity(system):
    A, y0 = _SYSTEMS[system]
    f = _linear_rhs(A)
    y1 = rosenbrock23_step(f, 0.3, y0, 0.0, _P)
    np.testing.assert_array_equal(np.asarray(y1), np.asarray(y0))
