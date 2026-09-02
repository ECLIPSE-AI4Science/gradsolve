"""Vern7 record-and-replay (gradsolve/solvers/vern7_replay.py).

record_vern7 (accepted-mesh recorder, via record_adaptive(VERN7_METHOD)); replay_vern7_jax
(fixed-mesh lax.scan replay of vern7_advance); the frozen-mesh closure. Gates: recorder
shapes/padding, mesh-parity vs a tight scipy DOP853 reference, dt==0 padding identity in
value/JVP/VJP, and the frozen-mesh FD gradient gate. Imports only gradsolve
(+ numpy/jax/scipy/pytest); tiny inline duck-typed Problem.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gradsolve  # noqa: F401
from gradsolve.solvers.vern7_replay import (
    make_vern7_frozen_mesh_closure,
    record_vern7,
    replay_vern7_jax,
)


class _Decay:
    """dy/dt = -p*y, dim=1. Analytic: y(t) = y0*exp(-p*t)."""
    name = "vern7_test_decay"
    dim = 1
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        return -params[0] * y


def _f_decay(t, y, p):
    return -p[0] * y


# --- recorder shapes + padding --------------------------------------------------------

def test_record_vern7_shapes_and_padding():
    y0 = np.array([[1.0], [1.0], [1.0]])
    params = np.array([[0.2], [2.0], [8.0]])
    yf, dts_padded, n_acc = record_vern7(_f_decay, y0, params, 0.0, 1.0, rtol=1e-6, atol=1e-9)
    n = 3
    assert yf.shape == (n, 1) and yf.dtype == np.float64
    S = int(n_acc.max())
    assert dts_padded.shape == (n, S) and dts_padded.dtype == np.float64
    for i in range(n):
        acc = dts_padded[i][: n_acc[i]]
        tail = dts_padded[i][n_acc[i]:]
        assert np.all(acc > 0.0) and np.all(tail == 0.0)
        np.testing.assert_allclose(acc.sum(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(yf, np.exp(-params * 1.0), rtol=2e-6, atol=1e-9)


def test_record_vern7_raises_on_max_steps_exhaustion():
    with pytest.raises(RuntimeError, match="exhausted max_steps"):
        record_vern7(_f_decay, np.array([[1.0]]), np.array([[8.0]]), 0.0, 1.0,
                     rtol=1e-6, atol=1e-9, max_steps=2)


# --- mesh parity vs a tight reference (scipy DOP853, order 8) --------------------------

def test_vern7_recorded_final_state_matches_dop853():
    """The recorded Vern7 final state matches a tight-tolerance DOP853 solve (an independent
    order-8 integrator) — the recorder-vs-reference parity gate."""
    A = np.array([[-1.0, 2.0], [-3.0, -1.0]])

    def f(t, y, p):
        return A @ y
    y0 = np.array([[1.0, 0.5]])
    p = np.array([[0.0]])
    yf, _dts, _n = record_vern7(f, y0, p, 0.0, 2.0, rtol=1e-9, atol=1e-12)
    ref = solve_ivp(lambda t, y: A @ y, (0.0, 2.0), y0[0], method="DOP853",
                    rtol=1e-12, atol=1e-14)
    assert ref.success
    np.testing.assert_allclose(yf[0], ref.y[:, -1], rtol=1e-7, atol=1e-9)


# --- dt == 0 padding identity in value / JVP / VJP -----------------------------

def test_replay_vern7_zero_padding_value_jvp_vjp_identity():
    """A zero-padded mesh row must replay identically to its unpadded prefix in value and
    gradient — extend a real mesh with zero columns and confirm value + grad are unchanged."""
    problem = _Decay()
    y0 = np.array([[1.0]])
    params = np.array([[2.0]])
    _yf, dts, _n = record_vern7(problem.f_jax, y0, params, 0.0, 1.0, rtol=1e-6, atol=1e-9)
    dts_pad = np.concatenate([dts, np.zeros((1, 5))], axis=1)  # 5 identity steps appended

    def loss(p, mesh):
        return jnp.sum(replay_vern7_jax(problem, jnp.asarray(y0), p, jnp.asarray(mesh)) ** 2)

    v0 = loss(jnp.asarray(params), dts)
    v1 = loss(jnp.asarray(params), dts_pad)
    np.testing.assert_array_equal(np.asarray(v0), np.asarray(v1))
    g0 = jax.grad(loss)(jnp.asarray(params), dts)
    g1 = jax.grad(loss)(jnp.asarray(params), dts_pad)
    np.testing.assert_array_equal(np.asarray(g0), np.asarray(g1))


# --- frozen-mesh FD gradient gate ---------------------------------------------

def test_vern7_grad_matches_frozen_mesh_finite_difference():
    """jax.grad of a scalar functional of the frozen replay closure equals central finite
    differences of the same frozen closure (re-recording per perturbation would test a
    different function). Adaptive re-recording is a separate stability test below."""
    problem = _Decay()
    y0 = np.array([[1.0], [2.0]])
    params0 = np.array([[1.3], [0.7]])
    closure = make_vern7_frozen_mesh_closure(problem, y0, params0, rtol=1e-8, atol=1e-11)

    def loss(p):
        return jnp.sum(closure(p) ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(params0)))
    eps = 1e-6
    fd = np.zeros_like(params0)
    for i in range(params0.shape[0]):
        pp = params0.copy()
        pp[i, 0] += eps
        pm = params0.copy()
        pm[i, 0] -= eps
        fd[i, 0] = (float(loss(jnp.asarray(pp))) - float(loss(jnp.asarray(pm)))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


@pytest.mark.slow
def test_vern7_adaptive_rerecord_stability_separate_from_frozen_gate():
    """Separate stability experiment: re-recording the mesh at nearby params keeps the
    final state close — the validity envelope of the frozen-mesh adjoint. Not the gradient gate."""
    problem = _Decay()
    y0 = np.array([[1.0]])
    yf_a, _, _ = record_vern7(problem.f_jax, y0, np.array([[1.30]]), 0.0, 1.0, rtol=1e-8, atol=1e-11)
    yf_b, _, _ = record_vern7(problem.f_jax, y0, np.array([[1.30001]]), 0.0, 1.0, rtol=1e-8, atol=1e-11)
    assert abs(float(yf_a[0, 0]) - float(yf_b[0, 0])) < 1e-4
