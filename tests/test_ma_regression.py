"""Regression tripwire: the value/replay path must be byte-identical and must not
gain RHS evaluations or degrade its jaxpr. Building the Tsit5 replay via TSIT5_METHOD.advance
must be indistinguishable from building it via tsit5_step directly — same values, same jaxpr,
same RHS-call count. Imports only gradsolve.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import gradsolve  # noqa: F401 -- x64 on import
from gradsolve.solvers import method as m
from gradsolve.solvers.tsit5_step import tsit5_step


class _Decay:
    name = "ma_regression_decay"
    dim = 1
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        return -params[0] * y


def _replay_with(step_fn, problem, y0, params, dts):
    """One-trajectory fixed-mesh replay parameterized by the step function (the replay body
    the record-and-replay adjoint runs)."""
    f = problem.f_jax

    def one(y0_j, p_j, dts_j):
        def body(carry, dt):
            t, y = carry
            return (t + dt, step_fn(f, t, y, dt, p_j)), None
        t0 = jnp.asarray(problem.t0, dtype=y0_j.dtype)
        (_t, yf), _ = jax.lax.scan(body, (t0, y0_j), dts_j)
        return yf
    return jax.vmap(one)(jnp.asarray(y0), jnp.asarray(params), jnp.asarray(dts))


def test_advance_replay_jaxpr_identical_to_direct_step():
    """Replaying through TSIT5_METHOD.advance yields the same jaxpr as replaying through
    tsit5_step directly — no extra primitives, no degraded reverse graph."""
    prob = _Decay()
    y0 = np.array([[1.0]])
    params = np.array([[2.0]])
    dts = np.full((1, 32), 1.0 / 32)

    jp_direct = jax.make_jaxpr(
        lambda p: _replay_with(tsit5_step, prob, y0, p, dts))(jnp.asarray(params))
    jp_method = jax.make_jaxpr(
        lambda p: _replay_with(m.TSIT5_METHOD.advance, prob, y0, p, dts))(jnp.asarray(params))
    assert str(jp_direct) == str(jp_method)


def test_advance_replay_values_byte_identical():
    prob = _Decay()
    y0 = np.array([[1.0], [2.0]])
    params = np.array([[0.5], [3.0]])
    dts = np.full((2, 40), 1.0 / 40)
    a = np.asarray(_replay_with(tsit5_step, prob, y0, params, dts))
    b = np.asarray(_replay_with(m.TSIT5_METHOD.advance, prob, y0, params, dts))
    np.testing.assert_array_equal(a, b)


def test_trial_step_rhs_eval_count_is_seven():
    """A Tsit5 trial step evaluates the RHS exactly 7 times (FSAL reuse notwithstanding, the
    trial step recomputes k1 when none is supplied). Pinned so a refactor cannot silently add
    stage work — the reverse-cost tripwire's cheap proxy."""
    calls = {"n": 0}

    def f(t, y, p):
        calls["n"] += 1
        return -y
    y = jnp.array([1.0])
    p = jnp.zeros(0)
    m.TSIT5_METHOD.trial_step(f, 0.0, y, 0.1, p)
    assert calls["n"] == 7
    calls["n"] = 0
    m.TSIT5_METHOD.trial_step(f, 0.0, y, 0.1, p, k1=f(0.0, y, p))  # supply FSAL k1
    assert calls["n"] == 1 + 6  # the supplied-k1 call + 6 fresh stages
