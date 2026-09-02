"""gradsolve/base.py Problem Protocol + SolveResult/Backend contract.

Covers:
  * a minimal duck-typed class (name/dim/t0/t1/is_stiff/f_jax, no inheritance from
    ``gradsolve.base.Problem``) satisfies ``gradsolve.solve()`` and ``gradsolve.grad_closure()``
    in full -- the Protocol is purely structural, never enforced by isinstance.
  * ``SolveResult``'s default step-array contract: the length-0 default is reserved for
    "genuinely unreportable"; a real fixed-step backend must fill the constant accepted
    count (not empty) and zero rejects (not empty), per the dataclass docstring.
  * unknown-engine names raise ``ValueError`` through the same duck-typed problem (the
    routing contract does not care what kind of Problem it is given).

CPU-only, float64, tiny ensembles (n<=4), short horizons ([0, 0.5]) -> seconds to run.
No GPU or warp imports -- everything the solvers touch
on this problem is plain numpy/jax arithmetic on a diagonal linear decay, so every
number here is either a closed-form analytic check or a central finite difference
against the exact same closure being tested.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import gradsolve
from gradsolve.base import Problem, SolveResult

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  (conditional import after importorskip)

_HAS_DIFFRAX = importlib.util.find_spec("diffrax") is not None
# Where an unregistered problem's FORWARD solve lands (api._fallback): diffrax when installed,
# else the record-and-replay lane of the same stiffness class.
_FWD_NONSTIFF = "diffrax" if _HAS_DIFFRAX else "tsit5_replay"
_FWD_STIFF = "diffrax" if _HAS_DIFFRAX else "rodas5p_replay"

# ---------------------------------------------------------------------------------
# Minimal duck-typed Problems -- plain classes, no base class other than `object`, no
# import of `gradsolve.base.Problem` in their definition. Diagonal linear decay
# dy_i/dt = -params_i * y_i so the analytic solution is y(t) = y0 * exp(-params * t)
# and the reverse-mode gradient w.r.t. params has a clean closed form too, but it is
# checked against central finite differences rather than the derivation alone.
# ---------------------------------------------------------------------------------

class _NonstiffDecay:
    """dim=1 exponential decay, is_stiff=False -> routes to diffrax (installed) or tsit5_replay."""

    name = "duck_nonstiff_decay"
    dim = 1
    t0 = 0.0
    t1 = 0.5

    @property
    def is_stiff(self) -> bool:
        return False

    def f_jax(self, t, y, params):
        return -params * y


class _StiffDecay:
    """dim=2 diagonal decay, is_stiff=True -> auto-routes to diffrax (installed) or rodas5p_replay."""

    name = "duck_stiff_decay"
    dim = 2
    t0 = 0.0
    t1 = 0.5

    @property
    def is_stiff(self) -> bool:
        return True

    def f_jax(self, t, y, params):
        return -params * y


# Tiny ensembles, deterministic.
_N = 4
_Y0_1D = np.array([[2.0], [1.0], [0.5], [3.0]], dtype=np.float64)
_K_1D = np.array([[0.7], [1.3], [2.0], [0.4]], dtype=np.float64)

_Y0_2D = np.array([[2.0, 1.0], [1.0, 0.5], [0.5, 2.0], [3.0, 1.5]], dtype=np.float64)
_K_2D = np.array([[0.7, 1.5], [1.3, 0.6], [2.0, 0.9], [0.4, 1.1]], dtype=np.float64)


def _analytic_decay(y0, k, t):
    return y0 * np.exp(-k * t)


# ---------------------------------------------------------------------------------
# Protocol shape: structural, not nominal.
# ---------------------------------------------------------------------------------

def test_duck_problem_satisfies_protocol_with_no_inheritance():
    problem = _NonstiffDecay()

    # No inheritance whatsoever from gradsolve.base.Problem (or anything but object).
    assert type(problem).__bases__ == (object,)
    assert Problem not in type(problem).__mro__

    # Problem is a plain (non-runtime-checkable) Protocol -> isinstance is not even
    # a legal operation on it. That is the point: conformance is purely structural
    # (the six attributes below), never enforced by Python's type system.
    with pytest.raises(TypeError):
        isinstance(problem, Problem)

    for attr in ("name", "dim", "t0", "t1", "is_stiff", "f_jax"):
        assert hasattr(problem, attr), f"missing structural member: {attr}"
    assert problem.is_stiff is False
    assert callable(problem.f_jax)


def test_duck_problem_stiff_flag_is_a_property_not_a_stored_value():
    stiff = _StiffDecay()
    nonstiff = _NonstiffDecay()
    assert stiff.is_stiff is True
    assert nonstiff.is_stiff is False
    assert type(stiff).__bases__ == (object,)


# ---------------------------------------------------------------------------------
# gradsolve.solve() on the duck-typed problem.
# ---------------------------------------------------------------------------------

def test_solve_auto_engine_nonstiff_matches_analytic_and_takes_the_general_engine():
    problem = _NonstiffDecay()
    res = gradsolve.solve(
        problem, _Y0_1D, _K_1D, engine="auto", rtol=1e-10, atol=1e-13, device="cpu",
    )
    assert isinstance(res, SolveResult)

    expected = _analytic_decay(_Y0_1D, _K_1D, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, rtol=1e-8, atol=1e-10)

    # engine="auto" wants cuda_tsit5/warp_ode for this (nonstiff, low-dim) cell, but a duck-typed
    # problem has no registered field for those lanes, so solve() falls back to the general
    # forward engine: diffrax when installed, else the record-and-replay lane.
    assert res.solver == _FWD_NONSTIFF
    assert res.route.actual == _FWD_NONSTIFF

    # SolveResult step-array contract (base.py) for an ADAPTIVE backend: true per-trajectory
    # counts, shape (n,), positive accepted, non-negative rejected.
    n = _Y0_1D.shape[0]
    assert res.accepted_steps.shape == (n,)
    assert res.rejected_steps.shape == (n,)
    assert np.all(res.accepted_steps > 0)
    assert np.all(res.rejected_steps >= 0)


def test_solve_explicit_fixed_step_imex_on_stiff_duck_problem():
    problem = _StiffDecay()
    res = gradsolve.solve(
        problem, _Y0_2D, _K_2D, engine="fixed_step_imex", rtol=1e-8, atol=1e-11, device="cpu",
    )
    expected = _analytic_decay(_Y0_2D, _K_2D, problem.t1)
    # fixed_step_imex is a 1st-order linearly-implicit (backward) Euler scheme; at the
    # modest rates used here (h*k << 1 for h = 0.5/10000) its global error is tiny but
    # not machine precision, hence the looser tolerance than the explicit Tsit5 check.
    np.testing.assert_allclose(res.y_final, expected, rtol=1e-3, atol=1e-6)
    assert res.solver == "fixed_step_imex"

    n = _Y0_2D.shape[0]
    assert res.accepted_steps.shape == (n,)
    assert res.rejected_steps.shape == (n,)
    assert np.all(res.accepted_steps == res.accepted_steps[0])
    assert res.accepted_steps[0] > 0
    np.testing.assert_array_equal(res.rejected_steps, np.zeros(n, dtype=res.rejected_steps.dtype))


def test_solve_auto_engine_stiff_duck_problem_takes_diffrax_or_rodas5p_replay():
    problem = _StiffDecay()
    res = gradsolve.solve(
        problem, _Y0_2D, _K_2D, engine="auto", rtol=1e-8, atol=1e-11, device="cpu",
    )
    # engine="auto" wants warp_rosenbrock for this (stiff, low-dim) cell, but a user-defined duck
    # problem has no registered analytic-Jacobian field, so it falls back to the general stiff
    # engine (diffrax if installed, else rodas5p_replay).
    assert res.solver == _FWD_STIFF
    expected = _analytic_decay(_Y0_2D, _K_2D, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, rtol=1e-6, atol=1e-9)


def test_solve_unknown_engine_raises_value_error():
    problem = _NonstiffDecay()
    with pytest.raises(ValueError):
        gradsolve.solve(problem, _Y0_1D, _K_1D, engine="not_a_real_engine", rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------------
# gradsolve.grad_closure() on the duck-typed problem: jax.grad vs. central FD on the
# same closure (the closure's own frozen-mesh / fixed-step arithmetic is the ground
# truth being differentiated, so FD-on-the-closure is the tightest check).
# ---------------------------------------------------------------------------------

def _fd_gradient(closure, params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(params)
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            pp = params.copy()
            pp[i, j] += eps
            pm = params.copy()
            pm[i, j] -= eps
            lp = float(jnp.sum(closure(jnp.asarray(pp))))
            lm = float(jnp.sum(closure(jnp.asarray(pm))))
            grad[i, j] = (lp - lm) / (2 * eps)
    return grad


def test_grad_closure_auto_engine_nonstiff_matches_fd():
    problem = _NonstiffDecay()
    closure = gradsolve.grad_closure(
        problem, _Y0_1D, _K_1D, engine="auto", rtol=1e-10, atol=1e-13, device="cpu",
    )
    params_j = jnp.asarray(_K_1D)

    y_final = closure(params_j)
    expected = _analytic_decay(_Y0_1D, _K_1D, problem.t1)
    np.testing.assert_allclose(np.asarray(y_final), expected, rtol=1e-6, atol=1e-9)

    grad_auto = np.asarray(jax.grad(lambda p: jnp.sum(closure(p)))(params_j))
    grad_fd = _fd_gradient(closure, _K_1D)
    np.testing.assert_allclose(grad_auto, grad_fd, rtol=1e-4, atol=1e-6)

    # Sanity: gradient sign must be negative (larger decay rate -> smaller y_final).
    assert np.all(grad_auto < 0)


def test_grad_closure_auto_engine_stiff_matches_fd():
    problem = _StiffDecay()
    closure = gradsolve.grad_closure(
        problem, _Y0_2D, _K_2D, engine="auto", rtol=1e-8, atol=1e-11, device="cpu",
    )
    assert closure.route.actual == "rodas5p_replay"
    params_j = jnp.asarray(_K_2D)

    grad_auto = np.asarray(jax.grad(lambda p: jnp.sum(closure(p)))(params_j))
    grad_fd = _fd_gradient(closure, _K_2D)
    np.testing.assert_allclose(grad_auto, grad_fd, rtol=1e-4, atol=1e-6)
    assert np.all(grad_auto < 0)


def test_grad_closure_unknown_engine_raises_value_error():
    problem = _NonstiffDecay()
    with pytest.raises(ValueError):
        gradsolve.grad_closure(problem, _Y0_1D, _K_1D, engine="not_a_real_engine")


# ---------------------------------------------------------------------------------
# SolveResult default contract, isolated from any backend.
# ---------------------------------------------------------------------------------

def test_solveresult_defaults_are_the_length_zero_unreportable_case():
    res = SolveResult(y_final=np.zeros((_N, 1)))
    # The length-0 default is reserved for "a backend whose API exposes no step
    # counts at all" -- distinct from a fixed-step backend's constant-fill contract
    # exercised above.
    assert res.accepted_steps.shape == (0,)
    assert res.rejected_steps.shape == (0,)
    assert res.solver == ""


def test_solveresult_defaults_are_independent_across_instances():
    # dataclass field(default_factory=...) must not share mutable state between
    # instances (the classic mutable-default-argument footgun the factory avoids).
    a = SolveResult(y_final=np.zeros((1, 1)))
    b = SolveResult(y_final=np.zeros((1, 1)))
    a.accepted_steps = np.array([7, 8, 9])
    assert b.accepted_steps.shape == (0,)
