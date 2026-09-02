"""The method-abstraction layer contract (gradsolve/solvers/method.py).

Two-tier step contract: advance(...) is value-only; trial_step(...) returns a
StepResult carrying the embedded error and the optional FSAL trailing stage — and not the
controller state, which the recorder owns separately. Plus the method-generic adaptive
recorder validated against the proven Tsit5 recorder (gradsolve/solvers/tsit5_replay.py::
tsit5_adaptive), which the method-abstraction layer must not modify.

Imports only gradsolve (+ numpy/jax/pytest); tiny inline duck-typed fields.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve  # noqa: F401 -- enables jax x64 on import
from gradsolve.solvers import method as m
from gradsolve.solvers.method import (
    ControllerState,
    StepResult,
    record_adaptive,
)
from gradsolve.solvers.tsit5_step import tsit5_step

# --- inline linear field: dy/dt = A y (numpy + jax forms) --------------------------
_A = np.array([[-1.0, 2.0], [-3.0, -1.0]], dtype=np.float64)


def _f(t, y, p):
    del t, p
    return _A @ y


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_stepresult_fields_and_no_controller_state():
    """StepResult carries value + embedded error + optional FSAL stage only — the PI
    controller history is deliberately not a field here."""
    sr = StepResult(y_next=np.zeros(2), y_err=np.zeros(2), fsal=None)
    assert set(sr._fields) == {"y_next", "y_err", "fsal"}
    assert "err_prev" not in sr._fields
    assert "controller" not in sr._fields


def test_controller_state_holds_pi_history():
    cs = ControllerState(err_prev=1.0, accepted_any=False)
    assert cs.err_prev == 1.0 and cs.accepted_any is False


def test_tsit5_method_advance_is_the_existing_step_fn_by_identity():
    """advance must be the very same object as tsit5_step: the value/replay path cannot
    diverge from the proven arithmetic (the value/replay byte-identity guarantee)."""
    assert m.TSIT5_METHOD.advance is tsit5_step
    assert m.TSIT5_METHOD.name == "tsit5"
    assert m.TSIT5_METHOD.order == 5 and m.TSIT5_METHOD.error_order == 4


def test_rosenbrock23_method_registered():
    from gradsolve.solvers.rosenbrock23_step import rosenbrock23_step
    assert m.ROSENBROCK23_METHOD.advance is rosenbrock23_step
    assert m.ROSENBROCK23_METHOD.order == 2 and m.ROSENBROCK23_METHOD.error_order == 3


# ---------------------------------------------------------------------------
# Two-tier consistency: trial_step.y_next is byte-identical to advance
# ---------------------------------------------------------------------------

def _fj(t, y, p):
    del t, p
    return jnp.asarray(_A) @ y


@pytest.mark.parametrize("dt", [0.2, 0.05, 0.013])
def test_tsit5_trial_step_value_is_byte_identical_to_advance(dt):
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.zeros(0, dtype=jnp.float64)
    adv = m.TSIT5_METHOD.advance(_fj, 0.3, y, dt, p)
    tri = m.TSIT5_METHOD.trial_step(_fj, 0.3, y, dt, p)
    np.testing.assert_array_equal(np.asarray(tri.y_next), np.asarray(adv))
    assert tri.fsal is not None  # FSAL trailing stage present


@pytest.mark.parametrize("dt", [0.2, 0.05, 0.013])
def test_tsit5_trial_step_error_matches_reference_estimate(dt):
    """The StepResult error equals dt*sum(E_i k_i) recomputed independently from the tableau
    (the same arithmetic tests/test_steps.py already verifies against diffrax)."""
    from gradsolve.solvers import tsit5_step as ts
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.zeros(0, dtype=jnp.float64)
    k1 = _fj(0.3, y, p)
    k2 = _fj(0.3 + ts._C[0] * dt, y + dt * (ts._A21 * k1), p)
    k3 = _fj(0.3 + ts._C[1] * dt, y + dt * (ts._A31 * k1 + ts._A32 * k2), p)
    k4 = _fj(0.3 + ts._C[2] * dt, y + dt * (ts._A41 * k1 + ts._A42 * k2 + ts._A43 * k3), p)
    k5 = _fj(0.3 + ts._C[3] * dt,
             y + dt * (ts._A51 * k1 + ts._A52 * k2 + ts._A53 * k3 + ts._A54 * k4), p)
    k6 = _fj(0.3 + ts._C[4] * dt,
             y + dt * (ts._A61 * k1 + ts._A62 * k2 + ts._A63 * k3 + ts._A64 * k4 + ts._A65 * k5), p)
    y5 = y + dt * (ts._B1 * k1 + ts._B2 * k2 + ts._B3 * k3 + ts._B4 * k4 + ts._B5 * k5 + ts._B6 * k6)
    k7 = _fj(0.3 + ts._C[5] * dt, y5, p)
    e_ref = dt * (ts._E1 * k1 + ts._E2 * k2 + ts._E3 * k3 + ts._E4 * k4 + ts._E5 * k5
                  + ts._E6 * k6 + ts._E7 * k7)
    tri = m.TSIT5_METHOD.trial_step(_fj, 0.3, y, dt, p)
    np.testing.assert_array_equal(np.asarray(tri.y_err), np.asarray(e_ref))


def test_rosenbrock23_trial_step_value_is_byte_identical_to_advance():
    from gradsolve.solvers.rosenbrock23_step import rosenbrock23_step
    y = jnp.array([1.0, 0.5], dtype=jnp.float64)
    p = jnp.zeros(0, dtype=jnp.float64)
    adv = rosenbrock23_step(_fj, 0.3, y, 0.05, p)
    tri = m.ROSENBROCK23_METHOD.trial_step(_fj, 0.3, y, 0.05, p)
    np.testing.assert_array_equal(np.asarray(tri.y_next), np.asarray(adv))


# ---------------------------------------------------------------------------
# record_adaptive(TSIT5_METHOD) reproduces tsit5_adaptive byte-for-byte
# ---------------------------------------------------------------------------

def test_generic_recorder_reproduces_tsit5_adaptive_bitwise():
    """The method-generic recorder, instantiated with TSIT5_METHOD, must produce the same
    accepted-dt mesh, final state, rejection count and status as the proven tsit5_adaptive —
    bit-for-bit. This is the guarantee vern7_replay's record_vern7 relies on."""
    from gradsolve.solvers.tsit5_replay import tsit5_adaptive

    def f_np(t, y):  # numpy host RHS, dim=2
        return _A @ y

    y0 = np.array([1.0, 0.5], dtype=np.float64)
    t0, t1, dt0 = 0.0, 3.0, 0.03
    rtol, atol, max_steps = 1e-6, 1e-9, 50000

    yf_ref, dts_ref, nrej_ref, st_ref = tsit5_adaptive(
        f_np, y0, t0, t1, rtol, atol, dt0, max_steps)
    yf_g, dts_g, nrej_g, st_g = record_adaptive(
        m.TSIT5_METHOD, f_np, y0, t0, t1, rtol, atol, dt0, max_steps)

    assert st_g == st_ref == 0
    assert nrej_g == nrej_ref
    assert dts_g.shape == dts_ref.shape
    np.testing.assert_array_equal(dts_g, dts_ref)   # BYTE-identical accepted mesh
    np.testing.assert_array_equal(yf_g, yf_ref)     # BYTE-identical final state


def test_generic_recorder_status_flags_incomplete_mesh():
    def f_np(t, y):
        return _A @ y
    y0 = np.array([1.0, 0.5], dtype=np.float64)
    yf, dts, nrej, status = record_adaptive(
        m.TSIT5_METHOD, f_np, y0, 0.0, 3.0, 1e-9, 1e-12, 0.03, max_steps=3)
    assert status == 1  # did not reach t1


# ---------------------------------------------------------------------------
# needs_jacobian metadata (True for BOTH Rosenbrock methods) + record_adaptive guard
# ---------------------------------------------------------------------------

def test_needs_jacobian_flag_matches_whether_trial_step_uses_jacfwd():
    assert m.TSIT5_METHOD.needs_jacobian is False                 # explicit, numpy recorder OK
    assert m.ROSENBROCK23_METHOD.needs_jacobian is True           # calls jax.jacfwd -> jitted recorder
    # (Rodas5P is asserted True in test_rodas5p_step.py once it exists.)


def test_record_adaptive_rejects_a_needs_jacobian_method():
    """record_adaptive is the numpy host recorder for explicit methods; a needs_jacobian method
    would call jax.jacfwd on a numpy RHS and fail obscurely — so it rejects loudly, routing the
    caller to the dedicated jitted recorder (record_rodas5p)."""
    def f_np(t, y):
        return _A @ y
    with pytest.raises(ValueError, match="needs_jacobian"):
        record_adaptive(m.ROSENBROCK23_METHOD, f_np, np.array([1.0, 0.5]), 0.0, 1.0,
                        1e-6, 1e-9, 0.03, 1000)
