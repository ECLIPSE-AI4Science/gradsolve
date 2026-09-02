"""record-and-replay adjoint (``gradsolve/warp/warp_replay.py``): reverse-mode gradients
match finite differences on a tiny toy ensemble -- nonstiff (Van der Pol / Tsit5,
``replay_solve_jax`` + ``make_replay_closure``) and stiff (Robertson-kinetics /
Rosenbrock23, ``replay_solve_rosenbrock_jax`` + ``make_rosenbrock_replay_closure``).

Problems are defined inline (duck-typed against ``gradsolve.base.Problem``'s
name/dim/t0/t1/is_stiff + ``f_jax(t, y, p)`` contract -- y is (dim,), p is (P,) per
trajectory, see ``gradsolve/solvers/tsit5_step.py``/``gradsolve/base.py``): ``vdp`` and
``robertson`` are the exact field names the fused Warp kernels
(``gradsolve/warp/_warp_kernel.py::make_vdp_field``,
``gradsolve/warp/_warp_rosenbrock.py::make_robertson``) route by ``problem.name``, so the
``f_jax``/``f_np`` below reproduce those fields' arithmetic exactly -- needed for the
record-vs-numpy-oracle and record-vs-replay round-trip checks below. Both are standard
textbook systems (Van der Pol; Robertson (1966) chemical kinetics), defined inline here.

CPU-only throughout (``device="cpu"``): Warp's CPU backend needs no NVIDIA GPU (see
``gradsolve/warp/warp_ode.py``'s device selection).
Guarded with ``pytest.importorskip("warp")`` so a warp-less env skips cleanly.
"""
from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")

import jax  # noqa: E402 (after importorskip)
import jax.numpy as jnp  # noqa: E402

from gradsolve.warp import warp_replay  # noqa: E402
from tests._oracles.gradsolve_g2_spike_oracle import (  # noqa: E402
    robertson_jac,
    rosenbrock23_adaptive,
)
from tests._oracles.tier0_fused_adaptive_oracle import tsit5_adaptive  # noqa: E402

RTOL, ATOL = 1e-6, 1e-9
DEVICE = "cpu"


# ------------------------------------------------------------------------------------
# Inline toy problems (duck-typed; defined here).
# ------------------------------------------------------------------------------------


class VdPProblem:
    """Van der Pol oscillator: ``dx/dt = v``, ``dv/dt = mu*((1-x^2)*v - x)``; p = [mu].

    Matches ``gradsolve.warp._warp_kernel.make_vdp_field`` verbatim, so
    ``problem.name == 'vdp'`` routes ``warp_replay.record`` to the fused kernel's real
    VdP field."""

    name = "vdp"
    dim = 2
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    @staticmethod
    def f_jax(t, y, p):
        x, v = y[0], y[1]
        mu = p[0]
        return jnp.stack([v, mu * ((1.0 - x * x) * v - x)])

    @staticmethod
    def f_np(t, y, p):
        x, v = float(y[0]), float(y[1])
        mu = float(p[0])
        return np.array([v, mu * ((1.0 - x * x) * v - x)], dtype=np.float64)


class RobertsonishProblem:
    """Robertson (1966) chemical-kinetics triple; p = [k1, k2, k3]:

        d0 = -k1*y0 + k3*y1*y2 ; d1 = k1*y0 - k3*y1*y2 - k2*y1^2 ; d2 = k2*y1^2

    Matches ``gradsolve.warp._warp_rosenbrock.make_robertson`` verbatim, so
    ``problem.name == 'robertson'`` routes ``warp_replay.record_rosenbrock`` to the
    fused kernel's real Robertson field. Rate constants are kept modest (not the
    canonical 3e7-stiff triple) so this toy ensemble integrates in a handful of
    accepted steps -- a fast correctness/autodiff check, not a stiffness stress test.
    """

    name = "robertson"
    dim = 3
    t0 = 0.0
    t1 = 0.5
    is_stiff = True

    @staticmethod
    def f_jax(t, y, p):
        k1, k2, k3 = p[0], p[1], p[2]
        y0, y1, y2 = y[0], y[1], y[2]
        d0 = -k1 * y0 + k3 * y1 * y2
        d1 = k1 * y0 - k3 * y1 * y2 - k2 * y1 * y1
        d2 = k2 * y1 * y1
        return jnp.stack([d0, d1, d2])

    @staticmethod
    def f_np(t, y, p):
        k1, k2, k3 = float(p[0]), float(p[1]), float(p[2])
        y0, y1, y2 = float(y[0]), float(y[1]), float(y[2])
        d0 = -k1 * y0 + k3 * y1 * y2
        d1 = k1 * y0 - k3 * y1 * y2 - k2 * y1 * y1
        d2 = k2 * y1 * y1
        return np.array([d0, d1, d2], dtype=np.float64)


def _central_fd_grad(loss, x0, h_scale=1e-6):
    """Central finite-difference gradient of a scalar JAX ``loss`` at ``x0`` (numpy)."""
    x0 = np.asarray(x0, dtype=np.float64)
    g = np.empty_like(x0)
    for i in range(x0.size):
        h = h_scale * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (float(loss(jnp.asarray(xp))) - float(loss(jnp.asarray(xm)))) / (2.0 * h)
    return g


def _rel_err(g, g_fd):
    return float(np.max(np.abs(g - g_fd) / np.maximum(np.abs(g_fd), 1e-8)))


# ------------------------------------------------------------------------------------
# NONSTIFF: Van der Pol / fused Tsit5 kernel.
# ------------------------------------------------------------------------------------


def _vdp_batch():
    y0 = np.array([[2.0, 0.0], [1.5, -0.5], [0.6, 1.0]], dtype=np.float64)
    mu = np.array([0.6, 1.2, 0.9], dtype=np.float64)
    params = mu.reshape(-1, 1)
    return y0, params


def test_vdp_warp_forward_matches_numpy_oracle():
    """Ground truth anchor: the warp forward is not just self-consistent -- it agrees
    with an independent numpy adaptive-Tsit5 integrator (``tests/_oracles``)."""
    prob = VdPProblem()
    y0, params = _vdp_batch()
    n = y0.shape[0]
    dt0 = (prob.t1 - prob.t0) / 100.0

    y_warp, n_acc, _dts_padded = warp_replay.record(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=4096, device=DEVICE)

    for j in range(n):
        f = lambda t, y, p=params[j]: prob.f_np(t, y, p)  # noqa: E731
        y_ref, dts_ref, _n_rej, status = tsit5_adaptive(
            f, y0[j], prob.t0, prob.t1, RTOL, ATOL, dt0, max_steps=4096)
        assert status == 0, f"oracle traj {j} did not reach t1"
        assert int(n_acc[j]) == len(dts_ref), (
            f"traj {j}: accepted {n_acc[j]} != oracle {len(dts_ref)}")
        np.testing.assert_allclose(
            y_warp[j], y_ref, rtol=0, atol=1e-8,
            err_msg=f"traj {j}: warp forward vs numpy oracle")


def test_vdp_replay_reproduces_warp_forward():
    """``record``'s padded dts replay (in pure JAX) back to the warp forward value."""
    prob = VdPProblem()
    y0, params = _vdp_batch()

    y_warp, _n_acc, dts_padded = warp_replay.record(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=4096, device=DEVICE)
    y_replay = np.asarray(warp_replay.replay_solve_jax(prob, y0, params, dts_padded))
    np.testing.assert_allclose(
        y_replay, y_warp, rtol=0, atol=1e-9,
        err_msg="replay_solve_jax does not reproduce the warp forward")


def test_vdp_replay_solve_jax_grad_wrt_y0_matches_fd():
    """``replay_solve_jax``: reverse-mode grad w.r.t. y0 through the
    frozen-mesh replay matches central finite differences to tight tolerance."""
    prob = VdPProblem()
    y0, params = _vdp_batch()
    n, dim = y0.shape

    _y_warp, _n_acc, dts_padded = warp_replay.record(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=4096, device=DEVICE)

    def loss(y0_flat):
        y0_ = y0_flat.reshape(n, dim)
        yf = warp_replay.replay_solve_jax(prob, y0_, params, dts_padded)
        return jnp.sum(yf ** 2)

    x0 = y0.reshape(-1)
    g = np.asarray(jax.grad(loss)(jnp.asarray(x0)))
    g_fd = _central_fd_grad(loss, x0)
    rel = _rel_err(g, g_fd)
    assert rel < 1e-6, f"replay_solve_jax grad wrt y0 rel err {rel:.3e} >= 1e-6"


def test_vdp_make_replay_closure_grad_wrt_params_matches_fd():
    """``make_replay_closure``: reverse-mode grad w.r.t. params through the
    record-once-replay-many closure matches central finite differences to tight
    tolerance."""
    prob = VdPProblem()
    y0, params = _vdp_batch()
    n, p_dim = params.shape

    closure = warp_replay.make_replay_closure(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=4096, device=DEVICE)

    def loss(p_flat):
        p = p_flat.reshape(n, p_dim)
        yf = closure(p)
        return jnp.sum(yf ** 2)

    x0 = params.reshape(-1)
    g = np.asarray(jax.grad(loss)(jnp.asarray(x0)))
    g_fd = _central_fd_grad(loss, x0)
    rel = _rel_err(g, g_fd)
    assert rel < 1e-6, f"make_replay_closure grad wrt params rel err {rel:.3e} >= 1e-6"


# ------------------------------------------------------------------------------------
# STIFF: Robertson-kinetics / fused Rosenbrock23 kernel.
# ------------------------------------------------------------------------------------


def _robertson_batch():
    y0 = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.05, 0.05], [0.8, 0.1, 0.1]], dtype=np.float64)
    params = np.array(
        [[0.4, 20.0, 5.0], [0.5, 25.0, 6.0], [0.35, 15.0, 4.0]], dtype=np.float64)
    return y0, params


def test_robertsonish_warp_forward_matches_numpy_oracle():
    """Ground truth anchor: the fused Rosenbrock23 forward agrees with an independent
    numpy adaptive-Rosenbrock23 integrator (``tests/_oracles``)."""
    prob = RobertsonishProblem()
    y0, params = _robertson_batch()
    n = y0.shape[0]
    dt0 = (prob.t1 - prob.t0) / 100.0

    y_warp, n_acc, _dts_padded = warp_replay.record_rosenbrock(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=50000, device=DEVICE)

    for j in range(n):
        p = params[j]
        f = lambda t, y, p=p: prob.f_np(t, y, p)  # noqa: E731
        jac = lambda y, _p, p=p: robertson_jac(y, p)  # noqa: E731
        y_ref, dts_ref, _n_rej, status = rosenbrock23_adaptive(
            f, jac, y0[j], prob.t0, prob.t1, RTOL, ATOL, dt0, max_steps=50000)
        assert status == 0, f"oracle traj {j} did not reach t1"
        assert int(n_acc[j]) == len(dts_ref), (
            f"traj {j}: accepted {n_acc[j]} != oracle {len(dts_ref)}")
        np.testing.assert_allclose(
            y_warp[j], y_ref, rtol=0, atol=1e-7,
            err_msg=f"traj {j}: warp forward vs numpy oracle")


def test_robertsonish_replay_reproduces_warp_forward():
    """``record_rosenbrock``'s padded dts replay (in pure JAX) back to the warp
    forward value, within the documented LU/FMA floor."""
    prob = RobertsonishProblem()
    y0, params = _robertson_batch()

    y_warp, _n_acc, dts_padded = warp_replay.record_rosenbrock(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=50000, device=DEVICE)
    y_replay = np.asarray(
        warp_replay.replay_solve_rosenbrock_jax(prob, y0, params, dts_padded))
    np.testing.assert_allclose(
        y_replay, y_warp, rtol=0, atol=1e-8,
        err_msg="replay_solve_rosenbrock_jax does not reproduce the warp forward")


def test_robertsonish_replay_solve_rosenbrock_jax_grad_wrt_y0_matches_fd():
    """``replay_solve_rosenbrock_jax``: reverse-mode grad w.r.t. y0 through
    the frozen-mesh stiff replay matches central finite differences to tight
    tolerance."""
    prob = RobertsonishProblem()
    y0, params = _robertson_batch()
    n, dim = y0.shape

    _y_warp, _n_acc, dts_padded = warp_replay.record_rosenbrock(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=50000, device=DEVICE)

    def loss(y0_flat):
        y0_ = y0_flat.reshape(n, dim)
        yf = warp_replay.replay_solve_rosenbrock_jax(prob, y0_, params, dts_padded)
        return jnp.sum(yf[:, 1] ** 2)  # depend on the stiff intermediate species

    x0 = y0.reshape(-1)
    g = np.asarray(jax.grad(loss)(jnp.asarray(x0)))
    g_fd = _central_fd_grad(loss, x0)
    rel = _rel_err(g, g_fd)
    assert rel < 1e-5, (
        f"replay_solve_rosenbrock_jax grad wrt y0 rel err {rel:.3e} >= 1e-5")


def test_robertsonish_make_rosenbrock_replay_closure_grad_wrt_params_matches_fd():
    """``make_rosenbrock_replay_closure``: reverse-mode grad w.r.t. params
    through the record-once-replay-many stiff closure matches central finite
    differences to tight tolerance."""
    prob = RobertsonishProblem()
    y0, params = _robertson_batch()
    n, p_dim = params.shape

    closure = warp_replay.make_rosenbrock_replay_closure(
        prob, y0, params, rtol=RTOL, atol=ATOL, max_steps=50000, device=DEVICE)

    def loss(p_flat):
        p = p_flat.reshape(n, p_dim)
        yf = closure(p)
        return jnp.sum(yf[:, 1] ** 2)

    x0 = params.reshape(-1)
    g = np.asarray(jax.grad(loss)(jnp.asarray(x0)))
    g_fd = _central_fd_grad(loss, x0)
    rel = _rel_err(g, g_fd)
    assert rel < 1e-5, (
        f"make_rosenbrock_replay_closure grad wrt params rel err {rel:.3e} >= 1e-5")
