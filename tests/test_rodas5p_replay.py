"""Rodas5P record-and-replay (gradsolve/solvers/rodas5p_replay.py)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gradsolve  # noqa: F401
from gradsolve.solvers.rodas5p_replay import (
    make_rodas5p_frozen_mesh_kernel,
    record_rodas5p,
    replay_rodas5p_jax,
)


class _NLStiff:
    """dy/dt = [-p0*(y0 - y1**2), y0 - y1 - p1*y1**3]. Jacobian depends on y and p (exercises the
    dJ/dy higher-order AD path). Stiff via large p0."""
    name = "rodas5p_nl_stiff"
    dim = 2
    t0 = 0.0
    t1 = 0.05
    is_stiff = True
    def f_jax(self, t, y, p):
        return jnp.array([-p[0] * (y[0] - y[1] ** 2), y[0] - y[1] - p[1] * y[1] ** 3])


_PROB = _NLStiff()   # record via the JAX f_jax (record_rodas5p jits + jacfwd's it)


def _f_nl(t, y, p):
    """SciPy/Radau reference only (returns np.array) — never pass to record_rodas5p, which jits the
    step and would raise TracerArrayConversionError on the np.array of traced inputs."""
    return np.array([-p[0] * (y[0] - y[1] ** 2), y[0] - y[1] - p[1] * y[1] ** 3])


def test_record_rodas5p_shapes_padding_and_rejections():
    y0 = np.array([[0.4, 0.6], [0.4, 0.6]])
    params = np.array([[500.0, 1.0], [1000.0, 1.0]])
    yf, dts_padded, n_acc, n_rej = record_rodas5p(_PROB.f_jax, y0, params, 0.0, 0.05, rtol=1e-6, atol=1e-9)
    n = 2
    assert yf.shape == (n, 2) and yf.dtype == np.float64
    assert n_rej.shape == (n,) and n_rej.dtype == np.int64 and np.all(n_rej >= 0)  # true per-trajectory counts
    S = int(n_acc.max())
    assert dts_padded.shape == (n, S)
    for i in range(n):
        assert np.all(dts_padded[i][: n_acc[i]] > 0.0) and np.all(dts_padded[i][n_acc[i]:] == 0.0)
        np.testing.assert_allclose(dts_padded[i][: n_acc[i]].sum(), 0.05, rtol=0, atol=1e-10)


def test_record_rodas5p_raises_on_max_steps_exhaustion():
    with pytest.raises(RuntimeError, match="exhausted max_steps"):
        record_rodas5p(_PROB.f_jax, np.array([[0.4, 0.6]]), np.array([[1e5, 1.0]]), 0.0, 0.05,
                       rtol=1e-10, atol=1e-13, max_steps=3)


def test_record_rodas5p_reports_underflow_on_nonfinite_field():
    """A permanently non-finite RHS: the recorder catches the non-finite stage, shrinks bounded,
    hits the scale-aware underflow floor within ~19 rejections, and raises status-2 —
    the message says 'underflow floor' (distinct from the max_steps message so the match is exact)."""
    def f_bad(t, y, p):
        return jnp.array([jnp.nan, jnp.nan])   # constant -> jit-safe; every trial non-finite
    with pytest.raises(RuntimeError, match="underflow floor"):
        record_rodas5p(f_bad, np.array([[1.0, 1.0]]), np.array([[1.0, 1.0]]), 0.0, 1.0,
                       rtol=1e-6, atol=1e-9, max_steps=200)


def test_rodas5p_replay_rejects_ad_incompatible_rhs():
    """A forward-jacfwd-incompatible RHS surfaces as a clear rodas5p_replay error via the probe,
    not a raw JAX TracerArrayConversionError."""
    def f_bad(t, y, p):
        return np.asarray(y) * -1.0            # np.asarray of a tracer -> not jacfwd-traceable
    with pytest.raises(ValueError, match="jax.jacfwd-compatible"):
        record_rodas5p(f_bad, np.array([[1.0]]), np.array([[1.0]]), 0.0, 1.0, rtol=1e-6, atol=1e-9)


def test_record_rodas5p_recovers_from_singular_first_trial():
    """A singular first trial (lambda = 1/(gamma*dt0) makes W = I - dt0*gamma*J exactly singular),
    with an unclamped dt0 (t1 >> dt0), forces a non-finite trial; the recorder shrinks bounded and
    still completes (status 0, n_rej>=1, finite) — the recovery path. Here lambda ~ 471.8 and
    the run completes with n_rej=3."""
    from gradsolve.solvers.rodas5p_step import RODAS5P_GAMMA_DIAG
    dt0 = 0.01
    lam = 1.0 / (RODAS5P_GAMMA_DIAG * dt0)     # W singular at the first trial
    def f(t, y, p):
        return lam * y                          # growing mode; short horizon keeps y finite
    yf, dts, n_acc, n_rej = record_rodas5p(
        f, np.array([[1.0]]), np.array([[0.0]]), 0.0, 0.03, rtol=1e-8, atol=1e-11, dt0=dt0)
    assert n_acc[0] > 0 and n_rej[0] >= 1 and np.all(np.isfinite(yf))


def test_rodas5p_recorded_final_state_matches_radau_nonlinear():
    p = np.array([[500.0, 1.0]])
    y0 = np.array([[0.4, 0.6]])
    yf, _dts, _n, _r = record_rodas5p(_PROB.f_jax, y0, p, 0.0, 0.05, rtol=1e-9, atol=1e-12)  # JAX RHS
    ref = solve_ivp(lambda t, y: _f_nl(t, y, p[0]), (0.0, 0.05), y0[0],   # numpy RHS for SciPy only
                    method="Radau", rtol=1e-11, atol=1e-13)
    assert ref.success
    np.testing.assert_allclose(yf[0], ref.y[:, -1], rtol=1e-6, atol=1e-9)


def test_replay_rodas5p_zero_padding_value_and_grad_identity():
    problem = _NLStiff()
    y0 = np.array([[0.4, 0.6]])
    params = np.array([[500.0, 1.0]])
    _yf, dts, _n, _r = record_rodas5p(problem.f_jax, y0, params, 0.0, 0.05, rtol=1e-6, atol=1e-9)
    dts_pad = np.concatenate([dts, np.zeros((1, 4))], axis=1)
    def loss(pp, mesh):
        return jnp.sum(replay_rodas5p_jax(problem, jnp.asarray(y0), pp, jnp.asarray(mesh)) ** 2)
    np.testing.assert_array_equal(np.asarray(loss(jnp.asarray(params), dts)),
                                  np.asarray(loss(jnp.asarray(params), dts_pad)))
    g0 = jax.grad(loss)(jnp.asarray(params), dts)
    g1 = jax.grad(loss)(jnp.asarray(params), dts_pad)
    np.testing.assert_array_equal(np.asarray(g0), np.asarray(g1))


def _fd_grad(loss, x0, eps=1e-5):
    g = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x0.copy()
        xp[idx] += eps
        xm = x0.copy()
        xm[idx] -= eps
        g[idx] = (float(loss(jnp.asarray(xp))) - float(loss(jnp.asarray(xm)))) / (2 * eps)
    return g


def test_rodas5p_grad_matches_fd_params_y0_joint_nonlinear():
    """Frozen-mesh FD gradient gate on a nonlinear stiff field whose Jacobian depends on y and p
    (exercises reverse-through-jacfwd incl. the dJ/dy path) — wrt params, y0, and joint."""
    problem = _NLStiff()
    y0 = np.array([[0.4, 0.6]])
    params0 = np.array([[300.0, 1.2]])
    kern = make_rodas5p_frozen_mesh_kernel(problem, y0, params0, rtol=1e-8, atol=1e-11)
    gp = np.asarray(jax.grad(lambda p: jnp.sum(kern(jnp.asarray(y0), p) ** 2))(jnp.asarray(params0)))
    np.testing.assert_allclose(gp, _fd_grad(lambda p: jnp.sum(kern(jnp.asarray(y0), p) ** 2), params0),
                               rtol=2e-4, atol=1e-6)
    gz = np.asarray(jax.grad(lambda z: jnp.sum(kern(z, jnp.asarray(params0)) ** 2))(jnp.asarray(y0)))
    np.testing.assert_allclose(gz, _fd_grad(lambda z: jnp.sum(kern(z, jnp.asarray(params0)) ** 2), y0),
                               rtol=2e-4, atol=1e-6)
    # joint
    gzj, gpj = jax.grad(lambda z, p: jnp.sum(kern(z, p) ** 2), argnums=(0, 1))(
        jnp.asarray(y0), jnp.asarray(params0))
    np.testing.assert_allclose(np.asarray(gpj), gp, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gzj), gz, rtol=2e-4, atol=1e-6)


def test_rodas5p_remat_replay_matches_non_remat():
    problem = _NLStiff()
    y0 = np.array([[0.4, 0.6]])
    params = np.array([[500.0, 1.0]])
    _yf, dts, _n, _r = record_rodas5p(problem.f_jax, y0, params, 0.0, 0.05, rtol=1e-7, atol=1e-10)
    a = replay_rodas5p_jax(problem, jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts), remat=False)
    b = replay_rodas5p_jax(problem, jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts), remat=True)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)


@pytest.mark.slow
def test_rodas5p_rerecord_stability_relative_on_scaled_component():
    """Re-record at a nearby p; require the frozen replay to track the re-recorded replay to small
    relative error on an O(1) component (not a vacuous abs threshold near exp(-large))."""
    problem = _NLStiff()
    y0 = np.array([[0.4, 0.6]])
    p0 = np.array([[300.0, 1.2]])
    p1 = np.array([[300.3, 1.2]])
    kern = make_rodas5p_frozen_mesh_kernel(problem, y0, p0, rtol=1e-8, atol=1e-11)
    frozen = np.asarray(kern(jnp.asarray(y0), jnp.asarray(p1)))         # frozen mesh, params moved
    rerec, _, _, _2 = record_rodas5p(problem.f_jax, y0, p1, 0.0, 0.05, rtol=1e-8, atol=1e-11)
    rel = np.linalg.norm(frozen[0] - rerec[0]) / np.linalg.norm(rerec[0])
    assert rel < 1e-3, f"frozen-vs-re-recorded relative error {rel:.2e} too large"
