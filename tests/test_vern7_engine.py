"""vern7_replay as an opt-in routed engine (integration matrix).

Asserts: engine="vern7_replay" is accepted by solve() and grad_closure(); grad wrt params, y0
and (y0,params); saveat dense output + dense grad; precision='float32' raises (general-RHS
recorder is f64-only, like tsit5_replay); the engine is discoverable; .route observability; and
— critically — choose_engine/DECISION_MAP are unchanged (vern7_replay is never auto-routed).
Imports only gradsolve.
"""
from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve
import gradsolve.api as api
from gradsolve.dispatch import DECISION_MAP, choose_engine


class _Decay:
    name = "vern7_engine_decay"
    dim = 1
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        return -params[0] * y


Y0 = np.array([[1.0], [2.0]])
P0 = np.array([[1.3], [0.7]])

#: The forward-only general stiff engine on this interpreter: diffrax is an optional extra,
#: and without it api._fallback serves the forward request from the record-and-replay lane.
_FWD_STIFF = "diffrax" if importlib.util.find_spec("diffrax") is not None else "rodas5p_replay"


def _analytic(y0, p, t):
    return y0 * np.exp(-p * t)


# --- dispatch is UNCHANGED (the opt-in guarantee) -------------------------------

def test_choose_engine_never_returns_vern7_replay():
    for dim in (1, 3, 16, 64, 65, 1000):
        for stiff in (False, True):
            for need_grad in (False, True):
                assert choose_engine(dim=dim, stiff=stiff, need_grad=need_grad) != "vern7_replay"


def test_decision_map_unchanged_nine_rows():
    assert len(DECISION_MAP) == 9
    assert all(row["engine"] != "vern7_replay" for row in DECISION_MAP)


# --- registry / discoverability ------------------------------------------------------

def test_vern7_replay_registered_as_engine():
    assert "vern7_replay" in api.ENGINE_REGISTRY
    spec = api.ENGINE_REGISTRY["vern7_replay"]
    assert spec.name == "vern7_replay"
    assert spec.supports(_Decay()) is True


# --- forward solve -------------------------------------------------------------------

def test_solve_vern7_replay_forward():
    res = gradsolve.solve(_Decay(), Y0, P0, engine="vern7_replay")
    assert res.solver == "vern7_replay"
    np.testing.assert_allclose(res.y_final, _analytic(Y0, P0, 1.0), rtol=2e-6, atol=1e-9)
    assert np.all(res.accepted_steps > 0)  # true per-trajectory adaptive counts


# --- grad wrt params / y0 / joint ----------------------------------------------------

def test_grad_closure_vern7_replay_params():
    c = gradsolve.grad_closure(_Decay(), Y0, P0, engine="vern7_replay", wrt="params")
    assert c.route.actual == "vern7_replay"
    g = np.asarray(jax.grad(lambda p: jnp.sum(c(p) ** 2))(jnp.asarray(P0)))
    # d/dp sum( (y0 e^{-p})^2 ) = -2 y0^2 e^{-2p}
    fd = -2.0 * (Y0 ** 2) * np.exp(-2.0 * P0)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_grad_closure_vern7_replay_y0():
    c = gradsolve.grad_closure(_Decay(), Y0, P0, engine="vern7_replay", wrt="y0")
    assert c.route.actual == "vern7_replay"
    g = np.asarray(jax.grad(lambda z: jnp.sum(c(z) ** 2))(jnp.asarray(Y0)))
    fd = 2.0 * Y0 * np.exp(-2.0 * P0)
    np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-6)


def test_grad_closure_vern7_replay_joint():
    c = gradsolve.grad_closure(_Decay(), Y0, P0, engine="vern7_replay", wrt=("y0", "params"))
    gz, gp = jax.grad(lambda z, p: jnp.sum(c(z, p) ** 2), argnums=(0, 1))(
        jnp.asarray(Y0), jnp.asarray(P0))
    np.testing.assert_allclose(np.asarray(gz), 2.0 * Y0 * np.exp(-2.0 * P0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gp), -2.0 * (Y0 ** 2) * np.exp(-2.0 * P0), rtol=1e-4, atol=1e-6)


# --- saveat (dense) on the vern7 replay engine ---------------------------------------

def test_solve_vern7_replay_saveat():
    ts = np.array([0.25, 0.5, 1.0])
    res = gradsolve.solve(_Decay(), Y0, P0, engine="vern7_replay", saveat=ts)
    assert res.y_saved.shape == (2, 3, 1)
    np.testing.assert_allclose(res.y_saved[:, -1, :], res.y_final, rtol=0, atol=0)
    np.testing.assert_allclose(res.y_saved[:, 1, 0], (Y0 * np.exp(-P0 * 0.5))[:, 0],
                               rtol=2e-6, atol=1e-9)


def test_grad_closure_vern7_replay_saveat_timeseries():
    ts = np.array([0.5, 1.0])
    c = gradsolve.grad_closure(_Decay(), Y0, P0, engine="vern7_replay", saveat=ts, wrt="params")
    out = np.asarray(c(jnp.asarray(P0)))
    assert out.shape == (2, 2, 1)  # (n, k, dim)


# --- precision f32 rejected (general-RHS recorder is f64-only) ------------------------

def test_vern7_replay_on_a_stiff_problem_is_silently_rerouted():
    """A stiff problem requested with engine='vern7_replay' is rerouted to the general stiff
    engine (diffrax when installed, else rodas5p_replay, on the forward path; rodas5p_replay
    for gradients). ``SolveResult.solver`` and ``.route`` report the engine actually used,
    and the reverse path fills ``.route.reason``.
    """
    class _StiffDecay(_Decay):
        name = "vern7_engine_stiff_decay"
        is_stiff = True

    res = gradsolve.solve(_StiffDecay(), Y0, P0, engine="vern7_replay", rtol=1e-10, atol=1e-13)
    assert res.solver == _FWD_STIFF, (
        "reroute target changed; this identity check assumes _fallback()")
    assert res.solver != "vern7_replay"
    # and the reverse path names the reason
    c = gradsolve.grad_closure(_StiffDecay(), Y0, P0, engine="vern7_replay")
    assert c.route.requested == "vern7_replay"
    assert c.route.actual == "rodas5p_replay"
    assert "does-not-support" in (c.route.reason or "")

    # nonstiff: the request IS honoured (the guard above must not fire on the good path)
    ok = gradsolve.solve(_Decay(), Y0, P0, engine="vern7_replay")
    assert ok.solver == "vern7_replay"


def test_vern7_replay_rejects_float32():
    with pytest.raises(ValueError, match="float32"):
        gradsolve.grad_closure(_Decay(), Y0, P0, engine="vern7_replay", precision="float32")
