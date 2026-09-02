"""Rodas5P dense output — the method-specific continuous extension.

Gates in this file are ordered so a transcription or conversion error fails early and
specifically:
  1. shape + exact-value sentinels on the stored H;
  2. the two consistency invariants (H_stored @ d == 0 and sum(m) == 1) -- these are order
     conditions of the extension, not coincidences, and a wrong conversion breaks them;
  3. driver parity: our converted weights reproduce the stage vectors of a numpy reference
     driver in the transformed (a, C) Rosenbrock formulation of Steinebach (2023);
  4. the endpoint identities b(1) == m and b(0) == 0, bitwise;
  5. exactness on a constant field (the direct consequence of invariant 2);
  6. observed order -- global (rate ~5) and local (interior rate ~5 => extension order 4,
     endpoint rate ~6 => step order 5). The local pair is what pins q=4; the global test
     alone cannot distinguish q=4 from q>=5.
  7. dt == 0 identity in value, JVP and VJP.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gradsolve  # noqa: F401
from gradsolve.solvers.rodas5p_step import (  # noqa: F401  (H/HD reached via `rs.` below)
    _A_STORED,  # (s, s) stored a, already transcribed -- no new numbers here
    _B_STORED,  # (s,) stored b = last a row + 1
    _C_STORED,  # (s, s) stored C, padded
    _H_STORED,  # (3, s) stored dense-output weights H, as published
    RODAS5P_C,  # (s,) stored abscissae
    RODAS5P_D,  # (s,) stored df/dt weights
    RODAS5P_GAMMA,  # (s, s) converted, lower, diagonal == gamma
    RODAS5P_GAMMA_DIAG,  # scalar gamma
    RODAS5P_HD,  # (3, s) dense weights converted to OUR k_i convention
    RODAS5P_M,  # (s,) converted propagated weights
    _rodas5p_stages,
    rodas5p_advance,
    rodas5p_dense_eval,
    rodas5p_dense_weights,
    rodas5p_stages_and_dense,
)

_P = jnp.zeros(0, dtype=jnp.float64)
_S = 8
_SUM_ABS_H_STORED = 415.89274467417624
_SUM_ABS_HD       = 102.01062464738597


def test_dense_weight_shapes_and_sentinels():
    from gradsolve.solvers import rodas5p_step as rs
    assert rs._H_STORED.shape == (3, _S)
    assert rs.RODAS5P_HD.shape == (3, _S)
    assert rs._H_STORED[0, 0] == 25.948786856663858
    assert rs._H_STORED[1, 3] == -24.495224566215796
    assert rs._H_STORED[2, 7] == 4.883087185713722
    np.testing.assert_allclose(np.abs(rs._H_STORED).sum(), _SUM_ABS_H_STORED, rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.abs(rs.RODAS5P_HD).sum(), _SUM_ABS_HD, rtol=0, atol=1e-11)


def test_dense_consistency_invariants():
    """H_stored @ d == 0 and sum(m) == 1 are order conditions of the continuous extension:
    on f == const every k_i equals dt*c, so the theta*(1-theta)*(...) block must contribute
    nothing at any theta. A wrong H->hd conversion breaks these even though the raw
    transcription is fine. Observed residuals: 1.9e-13 and 1.95e-14."""
    from gradsolve.solvers import rodas5p_step as rs
    np.testing.assert_allclose(rs._H_STORED @ rs.RODAS5P_D, np.zeros(3), rtol=0, atol=1e-11)
    np.testing.assert_allclose(rs.RODAS5P_HD.sum(axis=1), np.zeros(3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(rs.RODAS5P_M.sum(), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.13, 0.5, 0.77, 1.0])
def test_dense_weights_sum_to_theta(theta):
    """sum_i b_i(theta) == theta exactly follows from the two invariants above. Observed
    worst deviation over these thetas: 2.51e-14."""
    b = rodas5p_dense_weights(theta)
    np.testing.assert_allclose(float(np.sum(b)), theta, rtol=0, atol=1e-12)


def test_dense_endpoint_identity_is_bitwise():
    """b_i(1) == m_i exactly (the (1-theta) factor is exactly 0.0 at theta==1.0), so a save
    landing on a mesh node reproduces the step output bit for bit and the dense lane cannot
    introduce a discontinuity at node boundaries."""
    np.testing.assert_array_equal(np.asarray(rodas5p_dense_weights(1.0)), np.asarray(RODAS5P_M))
    np.testing.assert_array_equal(np.asarray(rodas5p_dense_weights(0.0)), np.zeros(_S))


# --- the conversion is a MEASUREMENT, not an assertion -----------------------------
# Nonlinear, non-autonomous, stiff field -- exercises alpha, Gamma and d simultaneously, so a
# wrong split in any one of the three shows up in the stage vectors.

def _parity_f_jnp(t, y, p):
    del p
    return jnp.array([-30.0 * (y[0] - y[1] ** 2) + 0.4 * jnp.sin(3.0 * t),
                      y[0] - y[1] - 0.7 * y[1] ** 3 + 0.2 * jnp.cos(t)])


def _parity_f_np(t, y):
    return np.array([-30.0 * (y[0] - y[1] ** 2) + 0.4 * np.sin(3.0 * t),
                     y[0] - y[1] - 0.7 * y[1] ** 3 + 0.2 * np.cos(t)])


def _parity_jac_np(t, y):
    del t
    return np.array([[-30.0, 60.0 * y[1]], [1.0, -1.0 - 2.1 * y[1] ** 2]])


def _parity_dfdt_np(t, y):
    del y
    return np.array([1.2 * np.cos(3.0 * t), -0.2 * np.sin(t)])


def _ordinarydiffeq_stage_vectors(t, y, dt):
    """A numpy reference driver in the transformed (a, C) Rosenbrock formulation of
    Steinebach (2023), built from the already-transcribed a/C tableau (so this test
    introduces no new numbers):
        W = J - I/(dt*gamma);  dtC = C/dt;  dtd = dt*d
        u_i           = uprev + sum_{j<i} a_ij * ks_j
        linsolve_tmp  = f(t + c_i*dt, u_i) + dtd_i * dT + sum_{j<i} dtC_ij * ks_j
        ks_i          = W \\ -linsolve_tmp
    J and dT are the analytic Jacobian/time-derivative at the step start, matching what our
    driver gets from jax.jacfwd to roundoff."""
    n = y.shape[0]
    W = _parity_jac_np(t, y) - np.eye(n) / (dt * RODAS5P_GAMMA_DIAG)
    dT = _parity_dfdt_np(t, y)
    ks = []
    for i in range(_S):
        u = y.copy()
        for j in range(i):
            u = u + _A_STORED[i, j] * ks[j]
        lt = _parity_f_np(t + RODAS5P_C[i] * dt, u) + (dt * RODAS5P_D[i]) * dT
        for j in range(i):
            lt = lt + (_C_STORED[i, j] / dt) * ks[j]
        ks.append(np.linalg.solve(W, -lt))
    return np.array(ks)


def test_dense_conversion_matches_ordinarydiffeq_stage_vectors():
    """u = Gamma @ k is the whole content of hd = H @ Gamma. Run a numpy reference driver in
    the transformed (a, C) Rosenbrock formulation (W = J - I/(dt*gamma); dtC = C/dt;
    dtd = dt*d; ks = W \\ -linsolve_tmp) from the already-transcribed a/C tableau, and check
    its stage vectors against Gamma @ k from our alpha/Gamma driver on a nonlinear,
    non-autonomous, stiff field (exercises alpha, Gamma and d at once).
    Observed: 1.492e-16 absolute, 4.113e-15 relative."""
    t, dt = 0.3, 0.02
    y = jnp.array([0.4, 0.6], dtype=jnp.float64)
    y_np = np.asarray(y)

    ks = _ordinarydiffeq_stage_vectors(t, y_np, dt)
    k = np.stack([np.asarray(ki) for ki in _rodas5p_stages(_parity_f_jnp, t, y, dt, _P)])
    gamma_k = RODAS5P_GAMMA @ k

    # Amplitude-invariant relative scale: max|.| over the whole stage block, NOT elementwise
    # (one stage vector is ~8e-9 here, and dividing by it would report pure roundoff as 1e-9).
    abs_err = float(np.abs(ks - gamma_k).max())
    rel_err = abs_err / float(np.abs(ks).max())
    assert abs_err <= 1e-14, f"u == Gamma @ k broken: {abs_err:.3e} abs (expected ~1.5e-16)"
    assert rel_err <= 1e-13, f"u == Gamma @ k broken: {rel_err:.3e} rel (expected ~4e-15)"

    # ...and the step output the two conventions build from those vectors agrees too.
    y1_ode = y_np + _B_STORED @ ks
    y1_ours = np.asarray(rodas5p_advance(_parity_f_jnp, t, y, dt, _P))
    y1_err = float(np.abs(y1_ode - y1_ours).max())
    assert y1_err <= 1e-14, f"y1 disagrees: {y1_err:.3e} (expected ~3e-16)"


def test_dense_is_exact_on_a_constant_field():
    """The direct consequence of H_stored @ d == 0. On f == const the Jacobian and df/dt both
    vanish, so every k_i is exactly dt*c (max|k_i - dt*c| == 0.0) and the exact flow
    is y0 + theta*dt*c. The theta*(1-theta) block must therefore contribute nothing at any
    theta -- which it can only do if the hd row sums are zero. Observed worst
    deviation: 1.32e-14 over 21 thetas in [0, 1] on this field/dt."""
    c = np.array([1.3, -0.7])
    c_j = jnp.asarray(c)

    def f(t, y, p):
        del t, y, p
        return c_j

    t, dt = 0.3, 0.4
    y0 = jnp.array([0.4, 0.6], dtype=jnp.float64)
    y0_np = np.asarray(y0)

    k = np.stack([np.asarray(ki) for ki in _rodas5p_stages(f, t, y0, dt, _P)])
    np.testing.assert_array_equal(k, np.broadcast_to(dt * c, k.shape))   # bitwise, not allclose

    worst = max(
        float(np.abs((y0_np + np.asarray(rodas5p_dense_weights(float(th))) @ k)
                     - (y0_np + th * dt * c)).max())
        for th in np.linspace(0.0, 1.0, 21)
    )
    assert worst <= 1e-12, f"constant-field exactness broken: {worst:.3e} (expected ~1e-14)"


# --- the 5-vector Horner bundle, WELDED to the b_i(theta) contract --------------

@pytest.mark.parametrize("theta", [0.0, 0.13, 0.5, 0.77, 1.0])
def test_dense_bundle_agrees_with_the_weight_contract(theta):
    """rodas5p_dense_eval on the (y_l, y_r, D1, D2, D3) bundle and the contract form
    y_l + sum_i b_i(theta) k_i are the same polynomial written two ways -- the bundle pays the
    three D_r contractions once per step instead of once per requested time. They are welded
    here so the fast form can never drift from the form the value gates pin. Not bitwise (the
    factor orderings differ), but agreement is at roundoff."""
    t, dt = 0.3, 0.02
    y = jnp.array([0.4, 0.6], dtype=jnp.float64)

    _y_next, coeffs = rodas5p_stages_and_dense(_parity_f_jnp, t, y, dt, _P)
    from_bundle = np.asarray(rodas5p_dense_eval(coeffs, theta))

    k = np.stack([np.asarray(ki) for ki in _rodas5p_stages(_parity_f_jnp, t, y, dt, _P)])
    from_weights = np.asarray(y) + np.asarray(rodas5p_dense_weights(theta)) @ k

    err = float(np.abs(from_bundle - from_weights).max())
    assert err <= 1e-15, f"bundle vs weight contract disagree by {err:.3e} at theta={theta}"


def test_dense_bundle_y_next_is_byte_identical_to_advance():
    """Opting into the dense lane must not perturb the mesh: the bundle's y_next uses the same
    weights in the same accumulation order as rodas5p_advance, so it is bitwise equal."""
    t, dt = 0.3, 0.02
    y = jnp.array([0.4, 0.6], dtype=jnp.float64)
    y_next, _coeffs = rodas5p_stages_and_dense(_parity_f_jnp, t, y, dt, _P)
    np.testing.assert_array_equal(np.asarray(y_next),
                                  np.asarray(rodas5p_advance(_parity_f_jnp, t, y, dt, _P)))


def test_dense_bundle_endpoints_are_bitwise():
    """theta == 1.0 makes (1-theta) exactly 0.0, so the bundle returns y_r -- the step output
    itself, bit for bit -- and theta == 0.0 returns y_l. This is the property that stops a save
    landing on a mesh node from introducing a discontinuity at the node boundary."""
    t, dt = 0.3, 0.02
    y = jnp.array([0.4, 0.6], dtype=jnp.float64)
    y_next, coeffs = rodas5p_stages_and_dense(_parity_f_jnp, t, y, dt, _P)
    np.testing.assert_array_equal(np.asarray(rodas5p_dense_eval(coeffs, 1.0)), np.asarray(y_next))
    np.testing.assert_array_equal(np.asarray(rodas5p_dense_eval(coeffs, 0.0)), np.asarray(y))


def test_dense_bundle_broadcasts_a_column_of_thetas():
    """A (k, 1) theta column evaluates all k requested times of one step at once -- the shape
    the streaming saveat lane relies on to stay ~5*k*dim per step."""
    t, dt = 0.3, 0.02
    y = jnp.array([0.4, 0.6], dtype=jnp.float64)
    _y_next, coeffs = rodas5p_stages_and_dense(_parity_f_jnp, t, y, dt, _P)
    ths = np.array([0.0, 0.25, 0.5, 1.0])
    many = np.asarray(rodas5p_dense_eval(coeffs, jnp.asarray(ths)[:, None]))
    assert many.shape == (4, 2)
    for j, th in enumerate(ths):
        np.testing.assert_array_equal(many[j], np.asarray(rodas5p_dense_eval(coeffs, float(th))))


# --- observed order, GLOBAL and LOCAL ----------------------------------------------
# One nonlinear, non-autonomous 2D field for both, so the rooted-tree AND the df/dt order
# conditions are exercised (a linear autonomous field would certify neither).

def _order_f_jnp(t, y, p):
    del p
    return jnp.array([0.3 * jnp.sin(2.0 * t) - y[0] - 0.2 * y[0] ** 2 + 0.1 * y[1],
                      -0.5 * y[1] + 0.3 * y[0] ** 2 + 0.2 * jnp.cos(1.5 * t)])


def _order_f_np(t, y):
    return np.array([0.3 * np.sin(2.0 * t) - y[0] - 0.2 * y[0] ** 2 + 0.1 * y[1],
                     -0.5 * y[1] + 0.3 * y[0] ** 2 + 0.2 * np.cos(1.5 * t)])


_ORDER_Y0 = np.array([0.5, -0.3])
_ORDER_T = 2.0
_ROUNDOFF_FLOOR = 1e-12
# The LOCAL test needs its own, lower floor. Its endpoint curve is O(h^6) and so dives much
# faster than anything in the global test: by j=4 it is at 1.6e-13 while still tracking the
# rate cleanly (5.90, against 5.81 at the pair before it). The DOP853 reference is itself good
# to ~1e-14 at |y| ~ 0.5, so 1e-13 is the right floor -- and reusing the global 1e-12 here
# would discard exactly the pair whose 5.90 rate is the p=5 evidence.
_LOCAL_FLOOR = 1e-13


def _dense_state(f, t, y, dt, p, theta):
    """y(t + theta*dt) from the step's own continuous extension: y + sum_i b_i(theta) * k_i."""
    k = _rodas5p_stages(f, t, y, dt, p)
    b = rodas5p_dense_weights(theta)
    out = y
    for i in range(_S):
        out = out + b[i] * k[i]
    return out


def _ref(t0, y0, t1):
    """Tight DOP853 reference state at t1 (integrated to t1 directly -- no dense output, so
    the reference carries no interpolation error of its own)."""
    return solve_ivp(_order_f_np, (t0, t1), y0, method="DOP853", rtol=1e-13, atol=1e-15).y[:, -1]


def _pairwise_rates(errs):
    """log2 refinement rates, dropping any pair that has fallen to the roundoff floor."""
    return [np.log(errs[i] / errs[i + 1]) / np.log(2.0) for i in range(len(errs) - 1)
            if errs[i] > _ROUNDOFF_FLOOR and errs[i + 1] > _ROUNDOFF_FLOOR]


@functools.lru_cache(maxsize=None)
def _fixed_step_to_final(n):
    """Fixed-step march to the start of the final step. theta-independent, so it is computed
    once per n and shared by the five theta cases (pure runtime; the arrays are identical)."""
    h = _ORDER_T / n
    y = jnp.asarray(_ORDER_Y0)
    t = 0.0
    for _ in range(n - 1):
        y = rodas5p_advance(_order_f_jnp, t, y, h, _P)
        t += h
    return t, y, h


@pytest.mark.parametrize("theta", [0.25, 1.0 / 3.0, 0.5, 0.75, 0.9])
def test_dense_global_order_is_five(theta):
    """Fixed-step integration, then evaluate the interpolant interior to the final step and
    score against DOP853 at the same instant. Observed rates, for calibration:
        theta=0.25 -> 5.14/5.11/5.07;  1/3 -> 4.99/5.03/5.02;  0.5 -> 4.90/4.97/4.99;
        0.75 -> 4.92/4.97/4.99;        0.9 -> 5.00/5.00/5.00.
    NB this test alone cannot distinguish q=4 from q>=5 -- the accumulated order-5 global error
    and an O(h^5) local interpolation error are the same size. See the local test below."""
    errs = []
    for n in (8, 16, 32, 64, 128, 256):
        t, y, h = _fixed_step_to_final(n)
        y_th = np.asarray(_dense_state(_order_f_jnp, t, y, h, _P, theta))
        errs.append(float(np.linalg.norm(y_th - _ref(0.0, _ORDER_Y0, t + theta * h))))
    rates = _pairwise_rates(errs)
    assert len(rates) >= 3, f"too few above-floor levels to gauge order: {errs}"
    for r in rates:
        assert 4.6 <= r <= 5.6, f"theta={theta}: rate {r:.2f} not ~5 (rates {rates}, errs {errs})"


def test_dense_local_order_pins_extension_order_four():
    """The global test cannot distinguish q=4 from q>=5 (both give rate 5, since the
    accumulated order-5 error and an O(h^5) local interpolation error are the same size).
    The local one-step test separates them:
        endpoint local error ~ O(h^(p+1)) -> measured rate -> 5.90 -> p = 5
        interior local error ~ O(h^(q+1)) -> measured rate -> 4.97-5.00 -> q = 4
    Assert the interior rate is in [4.6, 5.4] and strictly below the endpoint rate at the
    same refinement -- that asymmetry is the order-4 signature and is what justifies a
    tolerance-referenced accuracy gate rather than a constant-factor one."""
    t0 = 0.4
    y0 = np.array([0.5, -0.3])
    y0_j = jnp.asarray(y0)
    thetas = (0.5, 0.8)

    end_errs = []
    int_errs = {th: [] for th in thetas}
    for j in range(8):
        h = 0.4 / 2 ** j
        y_end = np.asarray(rodas5p_advance(_order_f_jnp, t0, y0_j, h, _P))
        end_errs.append(float(np.linalg.norm(y_end - _ref(t0, y0, t0 + h))))
        for th in thetas:
            y_th = np.asarray(_dense_state(_order_f_jnp, t0, y0_j, h, _P, th))
            int_errs[th].append(float(np.linalg.norm(y_th - _ref(t0, y0, t0 + th * h))))

    for th in thetas:
        # Finest refinement at which BOTH curves are still above the roundoff floor, so the
        # two rates are compared at the SAME pair and the asymmetry is not a floor artifact.
        alive = [i for i in range(7)
                 if min(end_errs[i], end_errs[i + 1],
                        int_errs[th][i], int_errs[th][i + 1]) > _LOCAL_FLOOR]
        assert len(alive) >= 3, f"theta={th}: too few above-floor pairs (end {end_errs}, " \
                                f"int {int_errs[th]})"
        i = alive[-1]
        r_end = float(np.log(end_errs[i] / end_errs[i + 1]) / np.log(2.0))
        r_int = float(np.log(int_errs[th][i] / int_errs[th][i + 1]) / np.log(2.0))
        assert 4.6 <= r_int <= 5.4, (
            f"theta={th}: interior local rate {r_int:.2f} not ~5 (=> extension order 4); "
            f"endpoint {r_end:.2f}; end {end_errs}, int {int_errs[th]}")
        assert r_int < r_end, (
            f"theta={th}: interior rate {r_int:.2f} is NOT strictly below the endpoint rate "
            f"{r_end:.2f} at the same refinement -- the order-4-on-order-5 asymmetry is gone")


@pytest.mark.parametrize("theta", [0.0, 0.37, 1.0])
def test_dense_dt_zero_identity_value_jvp_vjp(theta):
    """The single-step core proves every k_i is exactly zero at dt == 0, and
    y(theta) = y0 + sum_i b_i(theta) k_i, so the interpolant is the identity at every theta --
    in value and in both differentiation modes. Replay meshes are zero-padded, so this is the
    property that keeps a padded trajectory from poisoning the whole gradient. Mirrors the single-step
    test_rodas5p_dt_zero_jvp_vjp_identity, asserted bitwise."""
    A = np.array([[-100.0, 1.0], [0.0, -1.0]])
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.array([0.7], dtype=jnp.float64)

    def g(t, y, p):
        del t
        return jnp.asarray(A) @ y + p[0] * y

    def ev(yy, pp):
        return _dense_state(g, 0.3, yy, 0.0, pp, theta)

    np.testing.assert_array_equal(np.asarray(ev(y, p)), np.asarray(y))

    yt = jnp.array([0.3, -0.9], dtype=jnp.float64)
    _, jy = jax.jvp(lambda yy: ev(yy, p), (y,), (yt,))
    np.testing.assert_array_equal(np.asarray(jy), np.asarray(yt))

    _out, pull = jax.vjp(ev, y, p)
    ct = jnp.array([1.0, -2.0], dtype=jnp.float64)
    gy, gp = pull(ct)
    np.testing.assert_array_equal(np.asarray(gy), np.asarray(ct))
    np.testing.assert_array_equal(np.asarray(gp), np.zeros_like(np.asarray(p)))
