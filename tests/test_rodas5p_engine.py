"""rodas5p_replay as an opt-in routed engine (integration matrix).

Asserts: engine="rodas5p_replay" is accepted by solve() and grad_closure(); grad wrt params, y0
and (y0, params) on a nonlinear stiff field checked against finite differences; saveat dense
output + dense grad; rejected_steps are propagated (non-zero on a genuinely rejecting dense run);
precision='float32' raises (the general-RHS recorder is f64-only, like vern7_replay); remat=True
routes and matches remat=False; the engine is registered/re-exported/discoverable; .route
observability; and choose_engine/DECISION_MAP name rodas5p_replay only for stiff
gradient cells beyond the fused kernel (dim > NVAR_CEILING, or the gate-off/no-field
alternative): never for nonstiff or forward-only requests. Imports only gradsolve.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve
import gradsolve.api as api
from gradsolve.dispatch import DECISION_MAP, NVAR_CEILING, choose_engine


class _StiffDecay:
    """Linear stiff decay with a closed-form solution: y(t) = y0 * exp(-p0 t)."""
    name = "rodas5p_engine_decay"
    dim = 1
    t0 = 0.0
    t1 = 1.0
    is_stiff = True

    def f_jax(self, t, y, params):
        return -params[0] * y


class _NLStiff:
    """dy/dt = [-p0*(y0 - y1**2), y0 - y1 - p1*y1**3] — Jacobian depends on y and p."""
    name = "rodas5p_engine_nl_stiff"
    dim = 2
    t0 = 0.0
    t1 = 0.05
    is_stiff = True

    def f_jax(self, t, y, p):
        return jnp.array([-p[0] * (y[0] - y[1] ** 2), y[0] - y[1] - p[1] * y[1] ** 3])


Y0 = np.array([[1.0], [2.0]])
P0 = np.array([[1.3], [0.7]])

NL_Y0 = np.array([[0.4, 0.6]])
NL_P0 = np.array([[300.0, 1.2]])


def _analytic(y0, p, t):
    return y0 * np.exp(-p * t)


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


# --- dispatch: the stiff GRADIENT cells only ------------------------------------------

def test_choose_engine_returns_rodas5p_replay_only_for_stiff_gradient_cells():
    """rodas5p_replay is the stiff gradient engine beyond the fused kernel (dim > 64), and
    the gate-off alternative below it — never a nonstiff or forward-only target."""
    for dim in (1, 3, 16, 64, 65, 1000):
        for stiff in (False, True):
            for need_grad in (False, True):
                got = choose_engine(dim=dim, stiff=stiff, need_grad=need_grad)
                assert (got == "rodas5p_replay") is (stiff and need_grad and dim > NVAR_CEILING)
                gate_off = choose_engine(dim=dim, stiff=stiff, need_grad=need_grad,
                                         stiff_fused_enabled=False)
                assert (gate_off == "rodas5p_replay") is (stiff and need_grad)


def test_decision_map_names_rodas5p_replay_on_the_stiff_gradient_rows_only():
    assert len(DECISION_MAP) == 9
    shipped = [(r["nvar"], r["stiff"], r["need_grad"]) for r in DECISION_MAP
               if r["engine"] == "rodas5p_replay"]
    assert shipped == [("high", True, True)]
    gated = [(r["nvar"], r["stiff"], r["need_grad"]) for r in DECISION_MAP
             if r["gated_engine"] == "rodas5p_replay"]
    assert gated == [("low", True, True), ("high", True, False)]


# --- registry / discoverability ------------------------------------------------------

def test_rodas5p_replay_registered_as_engine():
    assert "rodas5p_replay" in api.ENGINE_REGISTRY
    spec = api.ENGINE_REGISTRY["rodas5p_replay"]
    assert spec.name == "rodas5p_replay"
    assert spec.supports(_StiffDecay()) is True


def test_rodas5p_modules_reexported_from_solvers_but_not_in_spine_registry():
    """Re-exported for discoverability; deliberately not spine REGISTRY members (those are the
    fixed-step solve_jax modules) — the same contract vern7_replay has."""
    from gradsolve import solvers
    assert solvers.rodas5p_replay.name == "rodas5p_replay"
    assert hasattr(solvers.rodas5p_step, "RODAS5P_METHOD")
    assert "rodas5p_replay" in solvers.__all__ and "rodas5p_step" in solvers.__all__
    assert "rodas5p_replay" not in solvers.REGISTRY


# --- forward solve -------------------------------------------------------------------

def test_solve_rodas5p_replay_forward():
    res = gradsolve.solve(_StiffDecay(), Y0, P0, engine="rodas5p_replay")
    assert res.solver == "rodas5p_replay"
    np.testing.assert_allclose(res.y_final, _analytic(Y0, P0, 1.0), rtol=2e-6, atol=1e-9)
    assert np.all(res.accepted_steps > 0)  # true per-trajectory adaptive counts


def test_solve_rodas5p_replay_reports_rejected_steps():
    """The recorder's true rejection counts reach SolveResult. p0=5000 at the
    default tol genuinely rejects (n_rej=2), so this is not a vacuous >=0 check."""
    res = gradsolve.solve(_NLStiff(), NL_Y0, np.array([[5000.0, 1.0]]), engine="rodas5p_replay")
    assert res.rejected_steps.shape == (1,)
    assert np.any(res.rejected_steps > 0), f"expected a rejecting run, got {res.rejected_steps}"


# --- grad wrt params / y0 / joint (NONLINEAR STIFF field, vs FD) ---------------------

def test_grad_closure_rodas5p_replay_params():
    c = gradsolve.grad_closure(_NLStiff(), NL_Y0, NL_P0, engine="rodas5p_replay", wrt="params",
                             rtol=1e-8, atol=1e-11)
    assert c.route.actual == "rodas5p_replay"

    def loss(p):
        return jnp.sum(c(p) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(NL_P0)))
    np.testing.assert_allclose(g, _fd_grad(loss, NL_P0), rtol=2e-4, atol=1e-6)


def test_grad_closure_rodas5p_replay_y0():
    c = gradsolve.grad_closure(_NLStiff(), NL_Y0, NL_P0, engine="rodas5p_replay", wrt="y0",
                             rtol=1e-8, atol=1e-11)
    assert c.route.actual == "rodas5p_replay"

    def loss(z):
        return jnp.sum(c(z) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(NL_Y0)))
    np.testing.assert_allclose(g, _fd_grad(loss, NL_Y0), rtol=2e-4, atol=1e-6)


def test_grad_closure_rodas5p_replay_joint():
    c = gradsolve.grad_closure(_NLStiff(), NL_Y0, NL_P0, engine="rodas5p_replay",
                             wrt=("y0", "params"), rtol=1e-8, atol=1e-11)
    assert c.route.actual == "rodas5p_replay"
    gz, gp = jax.grad(lambda z, p: jnp.sum(c(z, p) ** 2), argnums=(0, 1))(
        jnp.asarray(NL_Y0), jnp.asarray(NL_P0))
    np.testing.assert_allclose(
        np.asarray(gp), _fd_grad(lambda p: jnp.sum(c(jnp.asarray(NL_Y0), p) ** 2), NL_P0),
        rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(gz), _fd_grad(lambda z: jnp.sum(c(z, jnp.asarray(NL_P0)) ** 2), NL_Y0),
        rtol=2e-4, atol=1e-6)


# --- remat routing (final-state path) ------------------------------------------------

def test_grad_closure_rodas5p_replay_remat_matches_non_remat():
    """remat is threaded into the final-state replay scan; values (and grads) must be unchanged."""
    kw = dict(engine="rodas5p_replay", wrt="params", rtol=1e-8, atol=1e-11)
    c_off = gradsolve.grad_closure(_NLStiff(), NL_Y0, NL_P0, remat=False, **kw)
    c_on = gradsolve.grad_closure(_NLStiff(), NL_Y0, NL_P0, remat=True, **kw)
    assert c_on.route.actual == "rodas5p_replay"
    np.testing.assert_allclose(np.asarray(c_on(jnp.asarray(NL_P0))),
                               np.asarray(c_off(jnp.asarray(NL_P0))), rtol=0, atol=0)
    go = np.asarray(jax.grad(lambda p: jnp.sum(c_off(p) ** 2))(jnp.asarray(NL_P0)))
    gn = np.asarray(jax.grad(lambda p: jnp.sum(c_on(p) ** 2))(jnp.asarray(NL_P0)))
    np.testing.assert_allclose(gn, go, rtol=1e-10, atol=1e-12)


# --- saveat (dense) on the rodas5p replay engine -------------------------------------

def test_solve_rodas5p_replay_saveat():
    ts = np.array([0.25, 0.5, 1.0])
    res = gradsolve.solve(_StiffDecay(), Y0, P0, engine="rodas5p_replay", saveat=ts)
    assert res.solver == "rodas5p_replay"
    assert res.y_saved.shape == (2, 3, 1)
    np.testing.assert_allclose(res.y_saved[:, -1, :], res.y_final, rtol=0, atol=0)
    np.testing.assert_allclose(res.y_saved[:, 1, 0], (Y0 * np.exp(-P0 * 0.5))[:, 0],
                               rtol=2e-6, atol=1e-9)


def test_solve_rodas5p_replay_saveat_reports_rejected_steps():
    """_dense_result(rejected=) passes the recorder's counts through the dense lane too
    — the other dense lanes keep their zero default."""
    ts = np.array([0.025, 0.05])
    res = gradsolve.solve(_NLStiff(), NL_Y0, np.array([[5000.0, 1.0]]),
                        engine="rodas5p_replay", saveat=ts)
    assert np.all(res.accepted_steps > 0)
    assert np.any(res.rejected_steps > 0), f"expected a rejecting run, got {res.rejected_steps}"


def test_grad_closure_rodas5p_replay_saveat_timeseries():
    ts = np.array([0.5, 1.0])
    c = gradsolve.grad_closure(_StiffDecay(), Y0, P0, engine="rodas5p_replay", saveat=ts,
                             wrt="params")
    out = np.asarray(c(jnp.asarray(P0)))
    assert out.shape == (2, 2, 1)  # (n, k, dim)


def test_grad_closure_rodas5p_replay_saveat_is_differentiable():
    ts = np.array([0.5, 1.0])
    c = gradsolve.grad_closure(_StiffDecay(), Y0, P0, engine="rodas5p_replay", saveat=ts,
                             wrt="params")
    g = np.asarray(jax.grad(lambda p: jnp.sum(c(p) ** 2))(jnp.asarray(P0)))
    # d/dp sum_k (y0 e^{-p t_k})^2 = sum_k -2 t_k y0^2 e^{-2 p t_k}
    fd = sum(-2.0 * t * (Y0 ** 2) * np.exp(-2.0 * P0 * t) for t in ts)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


# --- precision f32 rejected (general-RHS recorder is f64-only) ------------------------

def test_rodas5p_replay_rejects_float32():
    with pytest.raises(ValueError, match="float32"):
        gradsolve.grad_closure(_StiffDecay(), Y0, P0, engine="rodas5p_replay",
                             precision="float32")
