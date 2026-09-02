"""Rodas5P single-step core (gradsolve/solvers/rodas5p_step.py).

Implemented from Steinebach (2023): the Rodas5P Rosenbrock-Wanner order-5(4) pair, transcribed
in the stored a/C form and converted to alpha/Gamma. Gates:
  1. shapes + stage count == 8; node checksum sum_j alpha_ij == c_i; Gamma-rowsum == d_i (these
     pin the alpha/gamma split, so the alpha+delta/gamma-delta gauge cannot pass).
  2. exact ROW stability-function series (roundoff-free transcription tripwire): propagated R(z)
     matches exp through z**5 (NB: it actually matches through z**6 -- the linear stability
     function is order >=6; do not assert a k=6 break); embedded Rhat matches through z**4 and
     breaks at z**5 (order 4).
  3. observed order 5 on one non-stiff, nonlinear, non-autonomous problem (vs a tight DOP853
     reference) via roundoff-filtered pairwise refinement rates -- this single field exercises the
     nonlinear rooted-tree and the df/dt conditions the linear series cannot certify. (Stiff
     correctness is covered separately by the nonlinear-stiff Radau parity in test_rodas5p_replay.)
  4. dt == 0 identity in value, JVP and VJP.
  5. trial_step.y_next byte-identical to advance.

Imports only gradsolve (+ numpy/jax/scipy/pytest).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gradsolve  # noqa: F401
from gradsolve.solvers.rodas5p_step import (
    RODAS5P_ALPHA,  # (s, s) converted, strictly lower
    RODAS5P_C,  # (s,) stored abscissae (alpha row sums)
    RODAS5P_D,  # (s,) stored df/dt weights (Gamma row sums)
    RODAS5P_GAMMA,  # (s, s) converted, lower, diagonal == gamma
    RODAS5P_GAMMA_DIAG,  # scalar gamma
    RODAS5P_M,  # (s,) converted propagated weights
    RODAS5P_MHAT,  # (s,) converted embedded weights
    rodas5p_advance,
    rodas5p_trial_step,
)

_P = jnp.zeros(0, dtype=jnp.float64)
_S = 8


def test_rodas5p_shapes_and_stage_count():
    assert len(RODAS5P_C) == _S and len(RODAS5P_D) == _S
    assert RODAS5P_ALPHA.shape == (_S, _S) and RODAS5P_GAMMA.shape == (_S, _S)
    assert len(RODAS5P_M) == _S and len(RODAS5P_MHAT) == _S
    assert np.allclose(np.triu(RODAS5P_ALPHA), 0.0)             # alpha strictly lower
    assert np.allclose(np.diag(RODAS5P_GAMMA), RODAS5P_GAMMA_DIAG)  # constant gamma diagonal


def test_rodas5p_node_and_gamma_rowsum_checksums():
    """sum_j alpha_ij == c_i and sum_j gamma_ij == d_i — the conversion+transcription checksums
    that pin the alpha/gamma split (a wrong split fails these even though beta is unchanged)."""
    np.testing.assert_allclose(RODAS5P_ALPHA.sum(axis=1), np.asarray(RODAS5P_C), rtol=0, atol=1e-11)
    np.testing.assert_allclose(RODAS5P_GAMMA.sum(axis=1), np.asarray(RODAS5P_D), rtol=0, atol=1e-11)


def _stability_series(m_weights, order):
    """Taylor coefficients of R(z)=1+m.k(z), k_i=z/(1-gamma z)(1+sum_{j<i} beta_ij k_j),
    beta=alpha+gamma (strictly lower). Truncated power series; roundoff-free."""
    N = order + 2
    g = RODAS5P_GAMMA_DIAG
    inv = np.array([g ** n for n in range(N)])          # 1/(1-gz) series
    one = np.zeros(N)
    one[0] = 1.0

    def mul(a, b):
        return np.convolve(a, b)[:N]

    def shift(a):
        return np.concatenate([[0.0], a[:-1]])

    k = [np.zeros(N) for _ in range(_S)]
    for i in range(_S):
        acc = one.copy()
        for j in range(i):
            beta = RODAS5P_ALPHA[i, j] + RODAS5P_GAMMA[i, j]
            acc = acc + beta * k[j]
        k[i] = mul(shift(inv), acc)
    R = one.copy()
    for i in range(_S):
        R = R + m_weights[i] * k[i]
    return R


def test_rodas5p_propagated_series_matches_through_order_five():
    """The propagated stability function R(z) must match exp(z) through z**5 (necessary for
    order 5). NB (numerically verified against the published tableau): Rodas5P's linear stability
    function actually matches through z**6 (R[6] residual ~1e-18) — the method's order-5 limit is
    a nonlinear order condition, not visible here. So this is a transcription tripwire (a wrong
    alpha/gamma constant breaks the through-k=5 match), and the observed nonlinear order test
    below is the real order-5 gate. Do not assert a k=6 break (there is none)."""
    R = _stability_series(RODAS5P_M, order=6)
    for k in range(6):
        assert abs(R[k] - 1.0 / math.factorial(k)) < 1e-11, f"R breaks at z^{k}: {R[k]:.3e}"


def test_rodas5p_embedded_series_is_order_four():
    """The embedded R_hat matches through z**4 and breaks at z**5 (order 4); the R_hat[5]
    residual is ~ -1.2e-4."""
    Rhat = _stability_series(RODAS5P_MHAT, order=5)
    for k in range(5):
        assert abs(Rhat[k] - 1.0 / math.factorial(k)) < 1e-11, f"Rhat breaks at z^{k}"
    assert abs(Rhat[5] - 1.0 / math.factorial(5)) > 1e-9, "Rhat unexpectedly order >=5"


# --- exact-value sentinel transcription checks --------------------------
# A handful of stored coefficients pinned to their EXACT published values (independent of the
# alpha-rowsum==c / Gamma-rowsum==d invariants, which coordinated transcription errors could
# preserve). These are the published Rodas5P coefficients (Steinebach 2023) in the stored a/C form.

def test_rodas5p_stored_coefficient_sentinels():
    from gradsolve.solvers import rodas5p_step as rs
    assert rs.RODAS5P_GAMMA_DIAG == 0.21193756319429014
    assert rs._A_STORED[1, 0] == 3.0
    assert rs._A_STORED[7, 5] == 1.0 and rs._A_STORED[7, 6] == 1.0        # b = last A row + 1
    assert rs._C_STORED[1, 0] == -14.155112264123755
    assert rs._C_STORED[7, 6] == -9.48861652309627
    np.testing.assert_array_equal(rs._BTILDE_STORED, np.eye(8)[7])         # btilde == e8
    assert float(RODAS5P_C[1]) == 0.6358126895828704                       # abscissa sentinel
    assert float(RODAS5P_D[0]) == 0.21193756319429014                     # d1 == gamma


def test_rodas5p_conversion_reconstruction_residual():
    """The stored->alpha/Gamma inversion round-trips: Gamma @ (I/gamma - C) == I. (cond(Gamma^-1)
    ~ 2557 — not trivially well-conditioned by triangularity; the residual bound is measured.)"""
    from gradsolve.solvers import rodas5p_step as rs
    ginv = np.eye(8) / rs.RODAS5P_GAMMA_DIAG - rs._C_STORED
    np.testing.assert_allclose(np.asarray(RODAS5P_GAMMA) @ ginv, np.eye(8), rtol=0, atol=1e-11)
    assert np.linalg.cond(ginv) < 5e3      # measured ~2557; guards against a wildly wrong C


# --- OBSERVED order 5 on a NON-STIFF nonlinear NON-autonomous problem ------------------
# A global slope on a STIFF problem is invalid (coarse superconvergence + stiff
# order reduction + roundoff give slope ~8.6 on a correct method). Verified numerically: a
# NON-stiff (linear part -1) nonlinear (y**2) non-autonomous (sin(2t)) problem gives clean
# pairwise rates 5.22, 5.11, 5.06, 5.03, 4.99 -> 5.0. This single field exercises BOTH the
# nonlinear rooted-tree conditions AND the df/dt term. (Stiff correctness is covered separately
# by the recorder-vs-Radau parity on a nonlinear STIFF field in test_rodas5p_replay.py.)

def test_rodas5p_observed_order_five_nonlinear_nonautonomous_nonstiff():
    def f(t, y, p):
        del p
        return jnp.array([0.3 * jnp.sin(2.0 * t) - y[0] - 0.2 * y[0] ** 2])
    def f_np(t, y):
        return np.array([0.3 * np.sin(2.0 * t) - y[0] - 0.2 * y[0] ** 2])
    y0 = np.array([0.5])
    T = 2.0
    ref = solve_ivp(f_np, (0.0, T), y0, method="DOP853", rtol=1e-13, atol=1e-15).y[:, -1]
    ns = (4, 8, 16, 32, 64, 128)
    errs = []
    for n in ns:
        y = jnp.asarray(y0)
        h = T / n
        t = 0.0
        for _ in range(n):
            y = rodas5p_advance(f, t, y, h, _P)
            t += h
        errs.append(float(np.linalg.norm(np.asarray(y) - ref)))
    # Pairwise rates on consecutive refinements, keeping only pairs above the roundoff floor.
    rates = [np.log(errs[i] / errs[i + 1]) / np.log(2.0) for i in range(len(errs) - 1)
             if errs[i] > 1e-11 and errs[i + 1] > 1e-11]
    assert len(rates) >= 3, f"too few above-floor levels to gauge order: {errs}"
    for r in rates:
        assert 4.6 <= r <= 5.6, f"observed pairwise order {r:.2f} not ~5 (rates {rates})"


def test_rodas5p_method_needs_jacobian():
    """RODAS5P_METHOD.needs_jacobian is True — its trial_step calls jax.jacfwd, so it must record
    via the dedicated jitted recorder (record_rodas5p), not the numpy record_adaptive."""
    from gradsolve.solvers.rodas5p_step import RODAS5P_METHOD
    assert RODAS5P_METHOD.needs_jacobian is True


def _lin(A):
    Aj = jnp.asarray(A, dtype=jnp.float64)
    def f(t, y, p):
        del t, p
        return Aj @ y
    return f


def test_rodas5p_trial_step_value_byte_identical_to_advance():
    f = _lin(np.array([[-100.0, 1.0], [0.0, -1.0]]))
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    adv = rodas5p_advance(f, 0.3, y, 0.02, _P)
    tri = rodas5p_trial_step(f, 0.3, y, 0.02, _P)
    np.testing.assert_array_equal(np.asarray(tri.y_next), np.asarray(adv))


@pytest.mark.parametrize("A", [np.array([[-100.0, 1.0], [0.0, -1.0]]),
                               np.array([[-2.0, 0.0], [0.0, -50.0]])])
def test_rodas5p_dt_zero_value_identity(A):
    f = _lin(A)
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    np.testing.assert_array_equal(np.asarray(rodas5p_advance(f, 0.3, y, 0.0, _P)), np.asarray(y))


def test_rodas5p_dt_zero_identity_with_singular_df_dt():
    """The dt==0 padding identity must hold unconditionally, including where df/dt is not
    finite at the padded node.

    A padded row sits at t == t1. Rodas5P evaluates df/dt (unlike the explicit lanes), and
    a successful record does not imply df/dt is finite there: the last real step only ever
    evaluates it at its left endpoint. Without masking, dt * dt * f_t = 0 * -inf = NaN
    destroys every shorter trajectory of a heterogeneous ensemble.
    """
    def g(t, y, p):
        # finite on [0, 1] (g(1, .) = -p*y), but df/dt -> -inf as t -> 1
        return -p[0] * y + jnp.sqrt(jnp.maximum(1.0 - t, 0.0))

    y = jnp.array([0.5], dtype=jnp.float64)
    p = jnp.array([1.0], dtype=jnp.float64)
    t1 = 1.0
    np.testing.assert_array_equal(np.asarray(rodas5p_advance(g, t1, y, 0.0, p)), np.asarray(y))
    tri = rodas5p_trial_step(g, t1, y, 0.0, p)
    np.testing.assert_array_equal(np.asarray(tri.y_next), np.asarray(y))
    assert np.all(np.isfinite(np.asarray(tri.y_err)))
    # gradient identity too: d(out)/dy == I, d(out)/dp == 0
    _out, pull = jax.vjp(lambda yy, pp: rodas5p_advance(g, t1, yy, 0.0, pp), y, p)
    gy, gp = pull(jnp.array([1.0], dtype=jnp.float64))
    np.testing.assert_array_equal(np.asarray(gy), np.array([1.0]))
    np.testing.assert_array_equal(np.asarray(gp), np.zeros_like(np.asarray(p)))


def test_rodas5p_dt_zero_jvp_vjp_identity():
    A = np.array([[-100.0, 1.0], [0.0, -1.0]])
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.array([0.7], dtype=jnp.float64)
    def g(t, y, p):
        return jnp.asarray(A) @ y + p[0] * y
    yt = jnp.array([0.3, -0.9], dtype=jnp.float64)
    _, jy = jax.jvp(lambda yy: rodas5p_advance(g, 0.3, yy, 0.0, p), (y,), (yt,))
    np.testing.assert_allclose(np.asarray(jy), np.asarray(yt), rtol=0, atol=0)
    _out, pull = jax.vjp(lambda yy, pp: rodas5p_advance(g, 0.3, yy, 0.0, pp), y, p)
    ct = jnp.array([1.0, -2.0], dtype=jnp.float64)
    gy, gp = pull(ct)
    np.testing.assert_array_equal(np.asarray(gy), np.asarray(ct))
    np.testing.assert_array_equal(np.asarray(gp), np.zeros_like(np.asarray(p)))
