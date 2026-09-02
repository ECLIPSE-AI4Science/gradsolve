"""``gradsolve.solvers.tsit5_replay``: mesh-recording shapes/padding + frozen-mesh forward value.

Covers:
* ``record_tsit5_jax`` — per-trajectory accepted dts + counts, zero-padded to a common
  length ``S = max(n_acc)``; 1-D (single-trajectory) inputs promoted to a batch of 1;
  ``RuntimeError`` when ``max_steps`` is exhausted before ``t1``.
* ``make_tsit5_frozen_mesh_closure`` — the closure's forward value (recorded once at
  ``params0``, replayed) matches a direct dense solve (tight-tolerance ``scipy.
  integrate.solve_ivp``) and the closed-form analytic solution, on a tiny inline
  ensemble.

Imports only gradsolve (+ numpy/scipy/pytest); the
test problem is defined inline as a tiny duck-typed ``Problem`` (see the Protocol in
``gradsolve/base.py``): ``name``/``dim``/``t0``/``t1``/``is_stiff`` + ``f_jax(t, y, params)``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gradsolve.solvers.tsit5_replay import (
    make_tsit5_frozen_mesh_closure,
    record_tsit5_jax,
)


class _Decay:
    """``dy/dt = -p*y``, dim=1. Analytic: ``y(t) = y0 * exp(-p*(t - t0))``."""

    name = "test_linear_decay"
    dim = 1
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        return -params[0] * y


def _f_decay(t, y, p):
    """Bare RHS matching ``_Decay.f_jax`` — ``record_tsit5_jax`` takes ``f`` directly,
    not a ``Problem``."""
    return -p[0] * y


def _analytic_decay(y0, params, t0, t1):
    return y0 * np.exp(-params * (t1 - t0))


# ---------------------------------------------------------------------------------
# record_tsit5_jax
# ---------------------------------------------------------------------------------

def test_record_tsit5_jax_shapes_and_padding():
    """Trajectories with different params accept different step counts; dts are
    zero-padded to the common length ``S = max(n_acc)``, tail-zero, and each row's
    non-padded dts sum exactly to the integration horizon."""
    y0 = np.array([[1.0], [1.0], [1.0]])
    params = np.array([[0.2], [2.0], [8.0]])  # spread -> distinct accepted-step counts
    t0, t1 = 0.0, 1.0

    y_finals, dts_padded, n_acc = record_tsit5_jax(
        _f_decay, y0, params, t0, t1, rtol=1e-6, atol=1e-9)

    n = 3
    assert y_finals.shape == (n, 1)
    assert y_finals.dtype == np.float64
    assert n_acc.shape == (n,)
    S = int(n_acc.max())
    assert dts_padded.shape == (n, S)
    assert dts_padded.dtype == np.float64
    # the whole point of padding-to-common-length: step counts actually differ here
    assert len(set(n_acc.tolist())) == n
    assert S == max(n_acc.tolist())

    for i in range(n):
        row = dts_padded[i]
        accepted = row[: n_acc[i]]
        tail = row[n_acc[i]:]
        assert np.all(accepted > 0.0)
        assert np.all(tail == 0.0)
        # accepted dts must exactly cover [t0, t1] (last step clipped to hit t1 exactly)
        np.testing.assert_allclose(accepted.sum(), t1 - t0, rtol=0, atol=1e-12)

    analytic = _analytic_decay(y0, params, t0, t1)
    np.testing.assert_allclose(y_finals, analytic, rtol=2e-6, atol=1e-9)


def test_record_tsit5_jax_promotes_1d_inputs_to_single_trajectory():
    """1-D ``y0``/``params`` (no explicit batch axis) are promoted to a batch of 1."""
    y0 = np.array([2.0])
    params = np.array([1.5])
    t0, t1 = 0.0, 0.5

    y_finals, dts_padded, n_acc = record_tsit5_jax(
        _f_decay, y0, params, t0, t1, rtol=1e-6, atol=1e-9)

    assert y_finals.shape == (1, 1)
    assert n_acc.shape == (1,)
    # single trajectory -> S == n_acc[0] exactly (no padding to add)
    assert dts_padded.shape == (1, int(n_acc[0]))
    np.testing.assert_allclose(
        y_finals[0, 0], 2.0 * np.exp(-1.5 * 0.5), rtol=2e-6, atol=1e-9)


def test_record_tsit5_jax_raises_on_max_steps_exhaustion():
    """A too-small ``max_steps`` must raise rather than silently return an incomplete
    (and therefore silently-wrong-if-replayed) mesh."""
    y0 = np.array([[1.0]])
    params = np.array([[8.0]])  # needs ~30 accepted steps at rtol=1e-6 over [0, 1]
    with pytest.raises(RuntimeError, match="exhausted max_steps"):
        record_tsit5_jax(_f_decay, y0, params, 0.0, 1.0, rtol=1e-6, atol=1e-9, max_steps=2)


# ---------------------------------------------------------------------------------
# make_tsit5_frozen_mesh_closure
# ---------------------------------------------------------------------------------

def test_frozen_mesh_closure_matches_dense_solve():
    """``closure(params0)`` forward value matches a direct dense solve (tight-tolerance
    ``scipy.integrate.solve_ivp``) on a tiny ensemble."""
    problem = _Decay()
    y0 = np.array([[1.0], [2.0], [0.5]])
    params0 = np.array([[0.2], [2.0], [8.0]])

    closure = make_tsit5_frozen_mesh_closure(problem, y0, params0, rtol=1e-6, atol=1e-9)
    y_final = np.asarray(closure(jnp.asarray(params0)))

    assert y_final.shape == (3, 1)
    assert y_final.dtype == np.float64

    dense = np.empty_like(y_final)
    for i in range(3):
        p = float(params0[i, 0])
        sol = solve_ivp(
            lambda t, y, p=p: -p * y, (problem.t0, problem.t1), y0[i],
            method="RK45", rtol=1e-12, atol=1e-14)
        assert sol.success
        dense[i, 0] = sol.y[0, -1]

    np.testing.assert_allclose(y_final, dense, rtol=5e-6, atol=1e-9)


def test_frozen_mesh_closure_matches_analytic():
    """Cross-check against the closed-form decay solution (independent of scipy)."""
    problem = _Decay()
    y0 = np.array([[1.0], [3.0]])
    params0 = np.array([[0.5], [4.0]])

    closure = make_tsit5_frozen_mesh_closure(problem, y0, params0)  # default rtol/atol
    y_final = np.asarray(closure(jnp.asarray(params0)))
    analytic = _analytic_decay(y0, params0, problem.t0, problem.t1)

    np.testing.assert_allclose(y_final, analytic, rtol=2e-6, atol=1e-9)


def test_frozen_mesh_closure_replay_is_deterministic():
    """Replaying the same frozen closure twice at the same params gives bit-identical
    values (pure-JAX replay of a fixed dt mesh has no hidden state/randomness)."""
    problem = _Decay()
    y0 = np.array([[1.0], [2.0]])
    params0 = np.array([[1.0], [3.0]])

    closure = make_tsit5_frozen_mesh_closure(problem, y0, params0)
    y_a = np.asarray(closure(jnp.asarray(params0)))
    y_b = np.asarray(closure(jnp.asarray(params0)))

    np.testing.assert_array_equal(y_a, y_b)
