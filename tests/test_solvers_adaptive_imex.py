"""Tests for gradsolve.solvers.adaptive_imex: CFL/accuracy step-count sizing, the
frozen CFL grid, and the generic reverse-diff replay primitive.

Covers:
  * cfl_n_steps / accuracy_n_steps: monotone non-decreasing in k, clamped to
    [n_floor, n_cap] / [n_min, n_cap], CFL condition actually satisfied.
  * cfl_grid: shape == (N(k)+1,), endpoints pinned, CFL-stable spacing.
  * solve_replay: matches an analytic ODE flow over a frozen (uniform and
    non-uniform) grid, and is reverse-differentiable (jax.grad finite, matches
    central finite differences) -- both on a bare step_fn and through a tiny
    inline duck-typed Problem's f_jax (the Problem/Backend contract shape).

Imports only gradsolve (+ numpy/jax/scipy/pytest).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gradsolve.solvers import adaptive_imex as ai

# A long span so the CFL step count is well above the floor; the functions under test are
# generic.
T0, T1 = 0.0, 14173.5
SPAN = T1 - T0


# --------------------------------------------------------------------------- #
# cfl_n_steps: monotone in k, clamped to [n_floor, n_cap], CFL-satisfying.
# --------------------------------------------------------------------------- #

def test_cfl_n_steps_low_k_hits_floor():
    n_floor = 2000
    N = ai.cfl_n_steps(1e-5, T0, T1, cfl=0.30, n_floor=n_floor)
    assert N == n_floor
    assert isinstance(N, int)


def test_cfl_n_steps_high_k_matches_formula_and_exceeds_floor():
    cfl, n_floor = 0.30, 2000
    N = ai.cfl_n_steps(0.5, T0, T1, cfl=cfl, n_floor=n_floor)
    assert N == math.ceil(SPAN * 0.5 / cfl)
    assert N > n_floor


def test_cfl_n_steps_monotone_nondecreasing_in_k():
    cfl, n_floor = 0.30, 2000
    ks = np.geomspace(1e-5, 0.5, 16)
    Ns = [ai.cfl_n_steps(float(k), T0, T1, cfl=cfl, n_floor=n_floor) for k in ks]
    assert all(b >= a for a, b in zip(Ns, Ns[1:]))


def test_cfl_n_steps_clamped_to_n_cap():
    N = ai.cfl_n_steps(0.5, T0, T1, cfl=0.30, n_floor=2000, n_cap=5000)
    assert N == 5000
    # a k so large the CFL-only count would vastly exceed n_cap.
    N2 = ai.cfl_n_steps(1e6, T0, T1, cfl=0.30, n_floor=2000, n_cap=26000)
    assert N2 == 26000


def test_cfl_n_steps_clamped_to_n_floor_never_below():
    # even a vanishingly small k must never produce fewer than n_floor steps.
    N = ai.cfl_n_steps(1e-12, T0, T1, cfl=0.30, n_floor=777)
    assert N == 777


def test_cfl_n_steps_satisfies_cfl_condition_everywhere():
    cfl, n_floor = 0.30, 2000
    ks = np.geomspace(1e-5, 0.5, 12)
    for k in ks:
        N = ai.cfl_n_steps(float(k), T0, T1, cfl=cfl, n_floor=n_floor)
        dtau = SPAN / N
        assert k * dtau <= cfl + 1e-12


@pytest.mark.parametrize("t0,t1", [(5.0, 5.0), (5.0, 1.0)])
def test_cfl_n_steps_rejects_nonpositive_span(t0, t1):
    with pytest.raises(ValueError):
        ai.cfl_n_steps(0.1, t0, t1)


@pytest.mark.parametrize("k", [0.0, -1.0])
def test_cfl_n_steps_rejects_nonpositive_k(k):
    with pytest.raises(ValueError):
        ai.cfl_n_steps(k, T0, T1)


# --------------------------------------------------------------------------- #
# accuracy_n_steps: monotone in k, clamped to [n_min, n_cap].
# --------------------------------------------------------------------------- #

def test_accuracy_n_steps_clamped_to_n_min():
    # a k so small the power-law floor formula dips under n_min -> clamp engages.
    n_min = 999
    N = ai.accuracy_n_steps(1e-30, T0, T1, n_min=n_min)
    assert N == n_min
    assert isinstance(N, int)


def test_accuracy_n_steps_clamped_to_n_cap():
    n_cap = 3000
    N = ai.accuracy_n_steps(1e6, T0, T1, n_cap=n_cap)
    assert N == n_cap


def test_accuracy_n_steps_monotone_nondecreasing_in_k():
    ks = np.geomspace(1e-6, 1.0, 24)
    Ns = [ai.accuracy_n_steps(float(k), T0, T1) for k in ks]
    assert all(b >= a for a, b in zip(Ns, Ns[1:]))


def test_accuracy_n_steps_within_default_bounds():
    ks = np.geomspace(1e-6, 1.0, 24)
    for k in ks:
        N = ai.accuracy_n_steps(float(k), T0, T1)
        assert ai.DEFAULT_N_MIN <= N <= ai.DEFAULT_N_CAP
        assert isinstance(N, int)


@pytest.mark.parametrize("t0,t1", [(5.0, 5.0), (5.0, 1.0)])
def test_accuracy_n_steps_rejects_nonpositive_span(t0, t1):
    with pytest.raises(ValueError):
        ai.accuracy_n_steps(0.1, t0, t1)


@pytest.mark.parametrize("k", [0.0, -1.0])
def test_accuracy_n_steps_rejects_nonpositive_k(k):
    with pytest.raises(ValueError):
        ai.accuracy_n_steps(k, T0, T1)


# --------------------------------------------------------------------------- #
# cfl_grid: shape == N(k)+1, endpoints pinned, CFL-stable spacing.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [1e-5, 3e-4, 0.01, 0.5])
def test_cfl_grid_shape_matches_n_plus_one(k):
    cfl, n_floor = 0.30, 2000
    grid = ai.cfl_grid(k, T0, T1, cfl=cfl, n_floor=n_floor)
    N = ai.cfl_n_steps(k, T0, T1, cfl=cfl, n_floor=n_floor)
    assert grid.shape == (N + 1,)


def test_cfl_grid_endpoints_and_strictly_increasing():
    grid = ai.cfl_grid(0.1, T0, T1, cfl=0.30, n_floor=2000)
    assert grid[0] == pytest.approx(T0)
    assert grid[-1] == pytest.approx(T1)
    assert np.all(np.diff(grid) > 0)


def test_cfl_grid_respects_n_cap_shape():
    grid = ai.cfl_grid(0.5, T0, T1, cfl=0.30, n_floor=2000, n_cap=4321)
    assert grid.shape == (4322,)


def test_cfl_grid_is_cfl_stable_everywhere():
    k, cfl, n_floor = 0.1, 0.30, 2000
    grid = ai.cfl_grid(k, T0, T1, cfl=cfl, n_floor=n_floor)
    assert np.diff(grid).max() <= cfl / k + 1e-9


# --------------------------------------------------------------------------- #
# solve_replay: matches an analytic flow (uniform + non-uniform grid).
# --------------------------------------------------------------------------- #

def test_solve_replay_matches_analytic_exponential_decay():
    jnp = pytest.importorskip("jax.numpy")
    lam = 0.7
    T = 3.0
    y0 = jnp.asarray([2.0, -1.0, 0.5])

    def step_fn(tau, dtau, y):
        return y * jnp.exp(-lam * dtau)

    grid = np.linspace(0.0, T, 41)
    hist = ai.solve_replay(step_fn, grid, y0)
    assert hist.shape == (len(grid), 3)
    np.testing.assert_allclose(np.asarray(hist[0]), np.asarray(y0))
    np.testing.assert_allclose(
        np.asarray(hist[-1]), np.asarray(y0) * np.exp(-lam * T), rtol=1e-10, atol=1e-12
    )


def test_solve_replay_matches_analytic_on_nonuniform_grid():
    jnp = pytest.importorskip("jax.numpy")
    lam = 1.3
    T = 2.0
    y0 = jnp.asarray([1.0, -2.0])

    def step_fn(tau, dtau, y):
        return y * jnp.exp(-lam * dtau)

    rng = np.random.default_rng(0)
    grid = np.sort(np.concatenate([[0.0, T], rng.uniform(0, T, 25)]))
    hist = ai.solve_replay(step_fn, grid, y0)
    assert hist.shape == (len(grid), 2)
    np.testing.assert_allclose(
        np.asarray(hist[-1]), np.asarray(y0) * np.exp(-lam * T), rtol=1e-10, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# solve_replay: reverse-diff (jax.grad finite, matches central FD).
# --------------------------------------------------------------------------- #

def test_solve_replay_reverse_diff_finite_and_matches_fd():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    lam = 0.7
    T = 3.0
    y0 = jnp.asarray([2.0, -1.0, 0.5])

    def step_fn(tau, dtau, y):
        return y * jnp.exp(-lam * dtau)

    grid = jnp.asarray(np.linspace(0.0, T, 41))

    def loss(y0v):
        return jnp.sum(ai.solve_replay(step_fn, grid, y0v) ** 2)

    g = np.asarray(jax.grad(loss)(y0))
    assert np.isfinite(g).all()

    eps = 1e-6
    base = np.asarray(y0)
    g_fd = np.empty_like(g)
    for i in range(base.size):
        tp, tm = base.copy(), base.copy()
        tp[i] += eps
        tm[i] -= eps
        g_fd[i] = (float(loss(jnp.asarray(tp))) - float(loss(jnp.asarray(tm)))) / (2 * eps)

    rel = np.max(np.abs(g - g_fd)) / (np.max(np.abs(g_fd)) + 1e-30)
    assert rel < 1e-5, f"grad-vs-FD rel err {rel:.2e}"


# --------------------------------------------------------------------------- #
# solve_replay through an inline duck-typed Problem's f_jax, over a FROZEN
# (params-independent) cfl_grid: the pattern a fixed-grid backend replays, with a
# controllable linear-decay field so there is a closed-form + scipy reference.
# --------------------------------------------------------------------------- #

class _LinearDecayProblem:
    """Tiny duck-typed Problem: dy/dt = -params[0] * y (scalar linear decay).

    Matches the structural gradsolve.base.Problem contract: name/dim/t0/t1/is_stiff
    + f_jax(t, y, params) with y of shape (dim,) and params of shape (P,).
    """

    name = "linear_decay"
    dim = 1
    t0 = 0.0
    t1 = 5.0

    @property
    def is_stiff(self) -> bool:
        return False

    def f_jax(self, t, y, params):
        import jax.numpy as jnp

        return -params[0] * y * jnp.ones_like(y)


def _rk4_step_fn(problem, params):
    """Classical RK4 single-step over problem.f_jax, closing over fixed params."""

    def step_fn(tau, dtau, y):
        k1 = problem.f_jax(tau, y, params)
        k2 = problem.f_jax(tau + 0.5 * dtau, y + 0.5 * dtau * k1, params)
        k3 = problem.f_jax(tau + 0.5 * dtau, y + 0.5 * dtau * k2, params)
        k4 = problem.f_jax(tau + dtau, y + dtau * k3, params)
        return y + (dtau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return step_fn


def test_solve_replay_through_inline_problem_matches_scipy_reference():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    scipy_integrate = pytest.importorskip("scipy.integrate")

    problem = _LinearDecayProblem()
    k_like = 0.2  # a stand-in "k" sizing the grid; not a physical wavenumber here.
    grid_np = ai.cfl_grid(k_like, problem.t0, problem.t1, cfl=0.05, n_floor=50, n_cap=2000)
    assert grid_np.shape == (51,)  # n_floor dominates at this k -> N == n_floor

    lam = 0.6
    params = jnp.asarray([lam])
    y0 = jnp.asarray([1.0])

    hist = ai.solve_replay(_rk4_step_fn(problem, params), grid_np, y0)
    assert hist.shape == (grid_np.shape[0], problem.dim)

    ref = scipy_integrate.solve_ivp(
        lambda t, y: -lam * y, (problem.t0, problem.t1), np.asarray(y0), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(np.asarray(hist[-1]), ref.y[:, -1], rtol=1e-6, atol=1e-9)


def test_solve_replay_through_inline_problem_reverse_diff_matches_fd():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    problem = _LinearDecayProblem()
    grid_np = ai.cfl_grid(0.2, problem.t0, problem.t1, cfl=0.05, n_floor=50, n_cap=2000)
    y0 = jnp.asarray([1.0])

    def loss(params):
        hist = ai.solve_replay(_rk4_step_fn(problem, params), grid_np, y0)
        return jnp.sum(hist[-1] ** 2)

    lam0 = jnp.asarray([0.6])
    g = float(np.asarray(jax.grad(loss)(lam0))[0])
    assert np.isfinite(g)

    eps = 1e-5
    g_fd = (float(loss(lam0 + eps)) - float(loss(lam0 - eps))) / (2 * eps)
    rel = abs(g - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 1e-4, f"grad-vs-FD rel err {rel:.2e}"
