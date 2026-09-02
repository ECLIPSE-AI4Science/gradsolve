"""Tests for gradsolve/api.py: solve() + grad_closure().

Covers the routed public API's contract:
  * engine="auto" routing (delegates to gradsolve.dispatch.choose_engine)
  * explicit engine=<name> override (bypasses choose_engine, still supports()-checked)
  * unsupported->fallback (solve() and grad_closure() fall back to diffrax / the general
    record-and-replay lane when the routed/overridden engine's supports(problem) is False)
  * SolveResult stamping: .solver is the stable registry key (not the engine's internal
    descriptive string), and the accepted/rejected step arrays match the documented
    fixed-step-vs-adaptive contract (gradsolve/base.py's SolveResult docstring)
  * grad_closure()'s value+gradient correctness (vs. an analytic closed form and/or FD)
    across the fixed-step and record-and-replay engines, plus its error contract
    (unknown engine, engine with no reverse path).

Problem fixture: a tiny inline duck-typed linear-decay ODE (see gradsolve/base.py's Problem
protocol and tests/test_tsit5_error_weights.py for the shape conventions). Its ``name``
deliberately does not match any Warp-kernel registered field ("lorenz", "vdp",
"linear_ladder_<D>", "robertson", "hires", "linstiff_*") so every engine that routes by
field-name lookup (warp_ode / warp_rosenbrock / cuda_tsit5 / fused_rosenbrock_backward)
reports supports()==False here — which is what this suite uses to exercise
the unsupported->fallback path deterministically and without any GPU.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve
import gradsolve.api as api_mod
from gradsolve.base import SolveResult

_HAS_DIFFRAX = importlib.util.find_spec("diffrax") is not None
# Where an unregistered problem's FORWARD solve lands (api._fallback): diffrax when installed,
# else the record-and-replay lane of the same stiffness class. Gradients never go to diffrax.
_FWD_NONSTIFF = "diffrax" if _HAS_DIFFRAX else "tsit5_replay"
_FWD_STIFF = "diffrax" if _HAS_DIFFRAX else "rodas5p_replay"

# ---------------------------------------------------------------------------
# Inline duck-typed Problem (Protocol: name, dim, t0, t1, is_stiff, f_jax(t, y, params))
# ---------------------------------------------------------------------------


class LinearDecay:
    """dy/dt = -k*y, batched: y shape (dim,), params shape (1,) -> params[0] = k.

    Analytic solution: y(t1) = y0 * exp(-k*(t1 - t0)). ``is_stiff`` is a constructor
    flag (not a property of the arithmetic) so the same RHS can probe both the
    non-stiff and stiff routing branches of choose_engine.
    """

    name = "unit_test_linear_decay"  # matches no registered Warp field by design
    dim = 2
    t0 = 0.0
    t1 = 1.0

    def __init__(self, stiff: bool = False):
        self._stiff = stiff

    @property
    def is_stiff(self) -> bool:
        return self._stiff

    def f_jax(self, t, y, params):
        k = params[0]
        return -k * y


K = 0.5
Y0_VEC = np.array([1.0, 2.0])


def _make_batch(n: int = 4):
    y0 = np.tile(Y0_VEC, (n, 1))
    params = np.tile(np.array([K]), (n, 1))
    return y0, params


def _analytic_final(y0: np.ndarray, k: float, t0: float, t1: float) -> np.ndarray:
    return y0 * math.exp(-k * (t1 - t0))


def _fd_grad_col0(closure, params: np.ndarray, eps: float = 1e-6) -> float:
    """Central-difference d(sum(closure(p)**2))/d(params[0, 0]), holding all other
    rows/columns fixed. Valid because the closures here vmap over independent
    trajectories (row i's output depends only on row i's params)."""

    def loss(p):
        yf = np.asarray(closure(jnp.asarray(p)))
        return float(np.sum(yf ** 2))

    p_plus = params.copy()
    p_plus[0, 0] += eps
    p_minus = params.copy()
    p_minus[0, 0] -= eps
    return (loss(p_plus) - loss(p_minus)) / (2 * eps)


def _analytic_dloss_dk(y0_row: np.ndarray, k: float, t: float) -> float:
    """d(sum(y**2))/dk for y = y0*exp(-k*t): -2*t*sum(y0**2)*exp(-2*k*t)."""
    return -2.0 * t * float(np.sum(y0_row ** 2)) * math.exp(-2.0 * k * t)


# ---------------------------------------------------------------------------
# solve() — routing / explicit override / unsupported->fallback
# ---------------------------------------------------------------------------


def test_solve_explicit_engine_stamps_registry_key_not_internal_string():
    """.solver must be the ENGINE_REGISTRY key the caller asked for, even though the
    engine's own solve() tags SolveResult with a different descriptive string
    internally (fixed_step_tsit5.solve sets solver=f"tsit5[{n_steps}]")."""
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=4)

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5")

    assert isinstance(res, SolveResult)
    assert res.solver == "fixed_step_tsit5"  # NOT "tsit5[10000]"
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, atol=1e-10, rtol=0)


@pytest.mark.skipif(not _HAS_DIFFRAX, reason="diffrax not installed: an explicit engine='diffrax' "
                    "request is rerouted, so the registry-key contract cannot be exercised")
def test_solve_diffrax_universal_override_stamps_registry_key():
    """'diffrax' is the universal catch-all (supports() True whenever diffrax is installed);
    .solver is stamped to the registry key "diffrax", not diffrax's own solver_name
    ("Tsit5"/"Kvaerno5")."""
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=4)

    res = gradsolve.solve(problem, y0, params, engine="diffrax", rtol=1e-8, atol=1e-10)

    assert res.solver == "diffrax"
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, atol=1e-8, rtol=0)


def test_diffrax_supports_tracks_the_install(monkeypatch):
    """supports() is True exactly when diffrax is importable (it is an optional extra), so the
    dispatcher can skip it without a ModuleNotFoundError mid-solve."""
    from gradsolve.solvers import diffrax_fallback

    assert diffrax_fallback.supports(LinearDecay()) is _HAS_DIFFRAX
    monkeypatch.setitem(sys.modules, "diffrax", None)  # importlib now reports diffrax absent
    assert diffrax_fallback.supports(LinearDecay()) is False


def test_solve_auto_nonstiff_unregistered_takes_diffrax_or_tsit5_replay():
    """choose_engine(dim=2, stiff=False, need_grad=False) routes to 'cuda_tsit5', but
    this problem has no registered Warp field -> cuda_tsit5.supports() is False ->
    solve() must fall back to diffrax when installed, else the general-RHS
    record-and-replay (tolerances honoured either way)."""
    from gradsolve.dispatch import choose_engine

    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=4)

    assert choose_engine(dim=problem.dim, stiff=False, need_grad=False) == "cuda_tsit5"

    res = gradsolve.solve(problem, y0, params, engine="auto", rtol=1e-10, atol=1e-13)

    assert res.solver == _FWD_NONSTIFF
    assert res.route.requested == "auto" and res.route.actual == _FWD_NONSTIFF
    assert res.route.reason.startswith("engine-does-not-support-problem")
    assert ("diffrax-not-installed" in res.route.reason) is (not _HAS_DIFFRAX)
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, atol=1e-10, rtol=0)


def test_solve_auto_stiff_unregistered_takes_diffrax_or_rodas5p_replay():
    """choose_engine(dim=2, stiff=True, need_grad=False) routes to 'warp_rosenbrock',
    but this problem has no registered Rosenbrock field -> falls back to the general stiff
    engine: diffrax when installed, else the Rodas5P record-and-replay."""
    from gradsolve.dispatch import choose_engine

    problem = LinearDecay(stiff=True)
    y0, params = _make_batch(n=4)

    assert choose_engine(dim=problem.dim, stiff=True, need_grad=False) == "warp_rosenbrock"

    res = gradsolve.solve(problem, y0, params, engine="auto", rtol=1e-8, atol=1e-11)

    assert res.solver == _FWD_STIFF
    assert res.route.actual == _FWD_STIFF
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    # Both targets honour rtol/atol (the order-1 fixed scan this replaces needed 1e-3 here).
    np.testing.assert_allclose(res.y_final, expected, atol=1e-6, rtol=0)


def test_solve_explicit_tsit5_replay_forward():
    """tsit5_replay (the general-RHS nonstiff record-and-replay) is a registry engine, so the
    forward fallback can run it when diffrax is absent — and a user can name it."""
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=4)

    res = gradsolve.solve(problem, y0, params, engine="tsit5_replay", rtol=1e-10, atol=1e-13)

    assert res.solver == "tsit5_replay"
    assert res.route.actual == "tsit5_replay" and res.route.reason is None
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, atol=1e-10, rtol=0)
    assert res.accepted_steps.shape == (4,) and np.all(res.accepted_steps > 0)
    # The controller's rejected counts are reported, not zero-filled (step-count metrics
    # read this lane now that it is an auto-routing target).
    assert res.rejected_steps.shape == (4,) and np.all(res.rejected_steps >= 0)
    # stiff -> supports() False -> the forward fallback, never the order-1 fixed scan
    stiff = gradsolve.solve(LinearDecay(stiff=True), y0, params, engine="tsit5_replay")
    assert stiff.solver == _FWD_STIFF


@pytest.mark.parametrize("stiff, replay", [(False, "tsit5_replay"), (True, "rodas5p_replay")])
def test_solve_auto_forward_without_diffrax_takes_the_replay_lane(monkeypatch, stiff, replay):
    """diffrax is an optional extra. With it absent, a forward request the fused kernels cannot
    serve lands on the record-and-replay lane of its stiffness class — never on a
    tolerance-blind fixed scan — and the route says why."""
    problem = LinearDecay(stiff=stiff)
    y0, params = _make_batch(n=3)
    monkeypatch.setitem(sys.modules, "diffrax", None)  # importlib now reports diffrax absent

    res = gradsolve.solve(problem, y0, params, engine="auto", rtol=1e-8, atol=1e-11)

    assert res.solver == replay
    assert res.route.reason == "engine-does-not-support-problem; diffrax-not-installed"
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(res.y_final, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "stiff, override_engine",
    [
        (False, "warp_ode"),
        (False, "cuda_tsit5"),
        (True, "warp_rosenbrock"),
        (True, "cuda_rosenbrock23"),
        (True, "fused_rosenbrock_backward"),
        (True, "tsit5_replay"),
    ],
)
def test_solve_explicit_unsupported_override_falls_back(stiff, override_engine):
    """Explicitly overriding to an engine this problem does not match must still fall back
    (supports() is checked even for an explicit override)."""
    problem = LinearDecay(stiff=stiff)
    y0, params = _make_batch(n=4)

    res = gradsolve.solve(problem, y0, params, engine=override_engine)

    expected = _FWD_STIFF if stiff else _FWD_NONSTIFF
    assert res.solver == expected
    assert res.route.reason.startswith("engine-does-not-support-problem")
    assert np.all(np.isfinite(res.y_final))


def test_solve_unknown_engine_raises_valueerror():
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=2)

    with pytest.raises(ValueError, match="unknown engine"):
        gradsolve.solve(problem, y0, params, engine="not_a_real_engine")


def test_solve_result_step_arrays_fixed_step_contract():
    """Fixed-step engines report the constant n_steps per trajectory (accepted) and
    zero rejected steps, per gradsolve/base.py's SolveResult contract."""
    problem = LinearDecay(stiff=False)
    n = 5
    y0, params = _make_batch(n=n)

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5")

    assert res.accepted_steps.shape == (n,)
    assert res.rejected_steps.shape == (n,)
    np.testing.assert_array_equal(res.accepted_steps, np.full(n, 10_000))
    np.testing.assert_array_equal(res.rejected_steps, np.zeros(n))


def test_solve_result_step_arrays_adaptive_contract():
    """Adaptive (diffrax) engine reports real per-trajectory accepted/rejected
    counts: positive, integer-shaped (n,), and (for identical trajectories) equal
    across the batch."""
    problem = LinearDecay(stiff=False)
    n = 4
    y0, params = _make_batch(n=n)

    res = gradsolve.solve(problem, y0, params, engine="diffrax")

    assert res.accepted_steps.shape == (n,)
    assert res.rejected_steps.shape == (n,)
    assert np.all(res.accepted_steps > 0)
    # identical trajectories -> identical accepted-step count
    np.testing.assert_array_equal(res.accepted_steps, res.accepted_steps[0])


def test_solve_accepts_routing_hint_kwargs_without_error():
    """batch_n / accuracy_target are accepted (not yet branch points) — must not raise."""
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=3)

    res = gradsolve.solve(
        problem, y0, params, engine="fixed_step_tsit5", batch_n=100_000, accuracy_target=1e-8
    )
    assert res.solver == "fixed_step_tsit5"


# ---------------------------------------------------------------------------
# grad_closure() — value/gradient correctness + routing/fallback + error contract
# ---------------------------------------------------------------------------


def test_grad_closure_explicit_fixed_step_tsit5_matches_analytic():
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=3)

    closure = gradsolve.grad_closure(problem, y0, params, engine="fixed_step_tsit5")
    yf = closure(jnp.asarray(params))
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(np.asarray(yf), expected, atol=1e-10, rtol=0)

    def loss(p):
        return jnp.sum(closure(p) ** 2)
    grad = jax.grad(loss)(jnp.asarray(params))
    expected_g = _analytic_dloss_dk(Y0_VEC, K, problem.t1 - problem.t0)
    np.testing.assert_allclose(np.asarray(grad)[:, 0], expected_g, atol=1e-8, rtol=1e-6)

    fd = _fd_grad_col0(closure, params)
    assert math.isclose(float(grad[0, 0]), fd, rel_tol=1e-4)


def test_grad_closure_explicit_fixed_step_imex_matches_fd():
    """fixed_step_imex is a 1st-order approximate scan; check jax.grad against a
    finite-difference of the same closure (self-consistency), and the forward value
    against the analytic solution at a loose tolerance."""
    problem = LinearDecay(stiff=True)
    y0, params = _make_batch(n=3)

    closure = gradsolve.grad_closure(problem, y0, params, engine="fixed_step_imex")
    yf = closure(jnp.asarray(params))
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(np.asarray(yf), expected, atol=1e-3, rtol=0)

    def loss(p):
        return jnp.sum(closure(p) ** 2)
    grad = jax.grad(loss)(jnp.asarray(params))
    fd = _fd_grad_col0(closure, params)
    assert math.isclose(float(grad[0, 0]), fd, rel_tol=1e-4)


@pytest.mark.parametrize("engine", ["auto", "warp_replay"])
def test_grad_closure_nonstiff_needgrad_uses_general_rhs_replay_fallback(engine):
    """need_grad routes non-stiff low-dim problems to 'warp_replay'; since this
    problem has no registered Warp field, grad_closure falls back to the portable
    pure-JAX record-and-replay reference implementation
    (make_tsit5_frozen_mesh_closure) rather than raising."""
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=3)

    closure = gradsolve.grad_closure(problem, y0, params, engine=engine)
    assert closure.route.actual == "tsit5_replay"
    yf = closure(jnp.asarray(params))
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(np.asarray(yf), expected, atol=1e-6, rtol=0)

    def loss(p):
        return jnp.sum(closure(p) ** 2)
    grad = jax.grad(loss)(jnp.asarray(params))
    fd = _fd_grad_col0(closure, params)
    assert math.isclose(float(grad[0, 0]), fd, rel_tol=1e-4)


@pytest.mark.parametrize("engine", ["auto", "warp_rosenbrock", "fused_rosenbrock_backward"])
def test_grad_closure_stiff_needgrad_takes_rodas5p_replay(engine):
    """need_grad routes stiff low-dim problems to 'warp_rosenbrock' (or the
    override-only 'fused_rosenbrock_backward'); with no registered Rosenbrock field
    both must fall back to the general stiff record-and-replay 'rodas5p_replay'
    (tolerance-honouring, exact frozen-mesh adjoint) and produce a finite, working
    closure."""
    problem = LinearDecay(stiff=True)
    y0, params = _make_batch(n=3)

    closure = gradsolve.grad_closure(problem, y0, params, engine=engine)
    assert closure.route.actual == "rodas5p_replay"
    yf = closure(jnp.asarray(params))
    assert np.all(np.isfinite(np.asarray(yf)))
    expected = _analytic_final(y0, K, problem.t0, problem.t1)
    np.testing.assert_allclose(np.asarray(yf), expected, atol=1e-6, rtol=0)

    def loss(p):
        return jnp.sum(closure(p) ** 2)
    grad = jax.grad(loss)(jnp.asarray(params))
    fd = _fd_grad_col0(closure, params)
    assert math.isclose(float(grad[0, 0]), fd, rel_tol=1e-4)


def test_grad_closure_unknown_engine_raises_valueerror():
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=2)

    with pytest.raises(ValueError, match="unknown engine"):
        gradsolve.grad_closure(problem, y0, params, engine="not_a_real_engine")


def test_grad_closure_engine_without_reverse_raises(monkeypatch):
    """cuda_tsit5 is forward-only by design (no _api_reverse -> EngineSpec.reverse is
    None). Without nvcc supports() is already False, so the generic
    unsupported->fallback catches it before the "no reverse" branch is ever reached;
    force supports()->True (monkeypatched EngineSpec, restored automatically) to
    exercise the explicit "engine has no reverse closure implemented" contract."""
    problem = LinearDecay(stiff=False)
    y0, params = _make_batch(n=2)

    orig_spec = api_mod.ENGINE_REGISTRY["cuda_tsit5"]
    forced_spec = dataclasses.replace(orig_spec, supports=lambda _problem: True)
    monkeypatch.setitem(api_mod.ENGINE_REGISTRY, "cuda_tsit5", forced_spec)

    with pytest.raises(ValueError, match="no reverse closure"):
        gradsolve.grad_closure(problem, y0, params, engine="cuda_tsit5")


def test_cuda_rosenbrock23_registered_and_forward_only():
    """The hand-CUDA stiff lane is registered but forward-only: its spec carries no reverse.
    Forcing supports()->True (needed where nvcc is absent) exercises the "no reverse closure"
    contract at grad_closure, exactly like cuda_tsit5."""
    assert "cuda_rosenbrock23" in api_mod.ENGINE_REGISTRY
    assert api_mod.ENGINE_REGISTRY["cuda_rosenbrock23"].reverse is None
    assert api_mod._reverse_for("cuda_rosenbrock23") is None


def test_grad_closure_cuda_rosenbrock23_without_reverse_raises(monkeypatch):
    problem = LinearDecay(stiff=True)
    y0, params = _make_batch(n=2)
    orig_spec = api_mod.ENGINE_REGISTRY["cuda_rosenbrock23"]
    forced_spec = dataclasses.replace(orig_spec, supports=lambda _problem: True)
    monkeypatch.setitem(api_mod.ENGINE_REGISTRY, "cuda_rosenbrock23", forced_spec)
    with pytest.raises(ValueError, match="no reverse closure"):
        gradsolve.grad_closure(problem, y0, params, engine="cuda_rosenbrock23")
