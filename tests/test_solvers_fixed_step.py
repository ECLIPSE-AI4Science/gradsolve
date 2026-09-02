"""Forward-accuracy + supports()-gating tests for the fixed-step solver spine.

Covers ``gradsolve.solvers.{fixed_step_explicit, fixed_step_tsit5, fixed_step_imex, imex}``:

  * ``solve_jax`` forward accuracy vs an analytic linear-decay solution (tests each
    solver's own convergence order: RK4 ~ machine precision by n_steps=1000, Tsit5
    likewise, order-1 linearly-implicit Euler converges at O(h)).
  * ``solve_jax`` forward accuracy vs a tight scipy (DOP853) reference on a small
    nonstiff Lorenz-like ensemble.
  * the numpy-facing ``solve()`` Backend-protocol entry point: ``SolveResult`` shape /
    dtype / ``accepted_steps == n_steps`` / ``rejected_steps == 0`` contract
    (``gradsolve/base.py``), plus the same accuracy check routed through ``solve()``.
  * ``supports()`` stiff/non-stiff gating: the explicit RK4/Tsit5 members reject a
    problem flagged ``is_stiff=True``; the IMEX member (and its ``imex`` facade) accept
    both.

Test problems are defined inline, duck-typed per ``gradsolve.base.Problem``
(``name``/``dim``/``t0``/``t1``/``is_stiff`` + ``f_jax(t, y, params)`` with per-trajectory
``y`` shape ``(dim,)`` and ``params`` shape ``(P,)`` — see ``gradsolve/base.py`` and
``tests/test_tsit5_error_weights.py``). Imports only gradsolve (+ numpy/jax/scipy/pytest).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gradsolve.base import SolveResult
from gradsolve.solvers import (
    fixed_step_explicit,
    fixed_step_imex,
    fixed_step_tsit5,
    imex,
)

# ---------------------------------------------------------------------------
# Inline test problems (duck-typed gradsolve.base.Problem)
# ---------------------------------------------------------------------------


class _LinearDecay:
    """Decoupled linear decay: dy_i/dt = -k_i * y_i.

    Analytic solution y_i(t1) = y0_i * exp(-k_i * (t1 - t0)) — an exact closed form to
    check each solver's own convergence order against (RK4/Tsit5 -> near machine
    precision by a few thousand steps; order-1 linearly-implicit Euler -> O(h)).
    ``is_stiff`` is a plain constructor flag (duck-typed gating knob only — this
    particular RHS is mild either way; the *gating* tests only check that ``supports()``
    reads the flag, not that the dynamics are actually stiff).
    """

    name = "linear_decay"
    dim = 2
    t0 = 0.0
    t1 = 0.5

    def __init__(self, stiff: bool = False):
        self._stiff = stiff

    @property
    def is_stiff(self) -> bool:
        return self._stiff

    def f_jax(self, t, y, params):
        return -params * y

    def analytic(self, y0: np.ndarray, params: np.ndarray) -> np.ndarray:
        return y0 * np.exp(-params * (self.t1 - self.t0))


class _LorenzLike:
    """Standard Lorenz attractor, short horizon (t1=0.2) to stay well clear of the
    horizon where chaotic error growth would swamp a method-accuracy comparison.
    Nonstiff -> ``is_stiff = False`` (a real value, not just a gating stub)."""

    name = "lorenz_like"
    dim = 3
    t0 = 0.0
    t1 = 0.2
    is_stiff = False

    def f_jax(self, t, y, params):
        sigma, rho, beta = params[0], params[1], params[2]
        x, yv, z = y[0], y[1], y[2]
        dx = sigma * (yv - x)
        dyv = x * (rho - z) - yv
        dz = x * yv - beta * z
        return jnp.stack([dx, dyv, dz])

    def f_np(self, t, y, params):
        sigma, rho, beta = params
        x, yv, z = y
        return np.array([sigma * (yv - x), x * (rho - z) - yv, x * yv - beta * z])


def _linear_batch(stiff: bool = False, n: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    y0 = 1.0 + 0.3 * rng.standard_normal((n, 2))
    k = np.array([1.5, 3.0]) + 0.2 * rng.standard_normal((n, 2))
    return _LinearDecay(stiff=stiff), y0.astype(np.float64), k.astype(np.float64)


def _lorenz_batch(n: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    y0 = np.array([1.0, 1.0, 1.0])[None, :] + 0.02 * rng.standard_normal((n, 3))
    sigma = np.full(n, 10.0)
    beta = np.full(n, 8.0 / 3.0)
    rho = np.full(n, 28.0)
    params = np.stack([sigma, rho, beta], axis=-1)
    return _LorenzLike(), y0.astype(np.float64), params.astype(np.float64)


def _scipy_reference(problem, y0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Per-trajectory tight scipy DOP853 reference (scipy has no native batching)."""
    out = np.empty_like(y0)
    for i in range(y0.shape[0]):
        sol = solve_ivp(
            lambda t, y, p=params[i]: problem.f_np(t, y, p),
            (problem.t0, problem.t1),
            y0[i],
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        assert sol.success
        out[i] = sol.y[:, -1]
    return out


# ---------------------------------------------------------------------------
# solve_jax forward accuracy vs the analytic linear-decay solution
# ---------------------------------------------------------------------------


def test_fixed_step_explicit_linear_decay_accuracy():
    problem, y0, params = _linear_batch()
    y_ref = problem.analytic(y0, params)
    y_hat = np.asarray(fixed_step_explicit.solve_jax(problem, y0, params, n_steps=1000))
    np.testing.assert_allclose(y_hat, y_ref, rtol=1e-8, atol=1e-10)


def test_fixed_step_tsit5_linear_decay_accuracy():
    problem, y0, params = _linear_batch()
    y_ref = problem.analytic(y0, params)
    y_hat = np.asarray(fixed_step_tsit5.solve_jax(problem, y0, params, n_steps=1000))
    np.testing.assert_allclose(y_hat, y_ref, rtol=1e-8, atol=1e-10)


def test_fixed_step_imex_linear_decay_accuracy():
    # Order-1 (linearly-implicit / Rosenbrock-Euler) -> needs many more steps than the
    # 4th/5th-order explicit members for a comparable tolerance (measured: n_steps=20000
    # -> max rel err ~5.7e-5, comfortably inside the tolerance below with margin).
    problem, y0, params = _linear_batch()
    y_ref = problem.analytic(y0, params)
    y_hat = np.asarray(fixed_step_imex.solve_jax(problem, y0, params, n_steps=20_000))
    np.testing.assert_allclose(y_hat, y_ref, rtol=2e-4, atol=1e-6)


def test_imex_facade_matches_fixed_step_imex_linear_decay():
    """``gradsolve.solvers.imex`` is a thin facade -> bit-for-bit same numerics as the
    underlying ``fixed_step_imex`` module it re-exposes, and equally accurate."""
    problem, y0, params = _linear_batch()
    y_ref = problem.analytic(y0, params)
    y_direct = np.asarray(fixed_step_imex.solve_jax(problem, y0, params, n_steps=20_000))
    y_facade = np.asarray(imex.solve_jax(problem, y0, params, n_steps=20_000))
    np.testing.assert_array_equal(y_facade, y_direct)
    np.testing.assert_allclose(y_facade, y_ref, rtol=2e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# solve_jax forward accuracy vs a tight scipy (DOP853) reference (Lorenz-like)
# ---------------------------------------------------------------------------


def test_fixed_step_explicit_lorenz_like_accuracy_vs_scipy():
    problem, y0, params = _lorenz_batch()
    y_ref = _scipy_reference(problem, y0, params)
    y_hat = np.asarray(fixed_step_explicit.solve_jax(problem, y0, params, n_steps=2000))
    np.testing.assert_allclose(y_hat, y_ref, rtol=1e-6, atol=1e-8)


def test_fixed_step_tsit5_lorenz_like_accuracy_vs_scipy():
    problem, y0, params = _lorenz_batch()
    y_ref = _scipy_reference(problem, y0, params)
    y_hat = np.asarray(fixed_step_tsit5.solve_jax(problem, y0, params, n_steps=2000))
    np.testing.assert_allclose(y_hat, y_ref, rtol=1e-6, atol=1e-8)


def test_fixed_step_imex_lorenz_like_accuracy_vs_scipy():
    # Order-1 on a mildly chaotic RHS needs many more steps for a comparable absolute
    # tolerance (measured: n_steps=50000 -> max abs err ~6.0e-4 against |y| ~ O(5-14)).
    problem, y0, params = _lorenz_batch()
    y_ref = _scipy_reference(problem, y0, params)
    y_hat = np.asarray(fixed_step_imex.solve_jax(problem, y0, params, n_steps=50_000))
    np.testing.assert_allclose(y_hat, y_ref, rtol=0.0, atol=2e-3)


# ---------------------------------------------------------------------------
# solve() Backend-protocol entry point: SolveResult contract + accuracy routed
# through the numpy-facing wrapper (device placement, dtype casting, jit).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mod, n_steps, rtol, atol, solver_prefix",
    [
        (fixed_step_explicit, 1000, 1e-8, 1e-10, "rk4"),
        (fixed_step_tsit5, 1000, 1e-8, 1e-10, "tsit5"),
        (fixed_step_imex, 20_000, 2e-4, 1e-6, "li-euler"),
        (imex, 20_000, 2e-4, 1e-6, "li-euler"),  # facade delegates -> same solver tag
    ],
)
def test_solve_result_contract_and_accuracy(mod, n_steps, rtol, atol, solver_prefix):
    problem, y0, params = _linear_batch()
    y_ref = problem.analytic(y0, params)
    result = mod.solve(problem, y0, params, rtol=1e-8, atol=1e-10, device="cpu", n_steps=n_steps)

    assert isinstance(result, SolveResult)
    assert result.y_final.shape == y0.shape
    assert np.all(np.isfinite(result.y_final))
    np.testing.assert_allclose(result.y_final, y_ref, rtol=rtol, atol=atol)

    n = y0.shape[0]
    np.testing.assert_array_equal(result.accepted_steps, np.full(n, n_steps, dtype=np.int64))
    np.testing.assert_array_equal(result.rejected_steps, np.zeros(n, dtype=np.int64))
    assert result.solver.startswith(solver_prefix)


# ---------------------------------------------------------------------------
# supports() stiff/non-stiff gating
# ---------------------------------------------------------------------------


def test_fixed_step_explicit_rejects_stiff_accepts_nonstiff():
    nonstiff = _LinearDecay(stiff=False)
    stiff = _LinearDecay(stiff=True)
    assert fixed_step_explicit.supports(nonstiff) is True
    assert fixed_step_explicit.supports(stiff) is False


def test_fixed_step_tsit5_rejects_stiff_accepts_nonstiff():
    nonstiff = _LinearDecay(stiff=False)
    stiff = _LinearDecay(stiff=True)
    assert fixed_step_tsit5.supports(nonstiff) is True
    assert fixed_step_tsit5.supports(stiff) is False


def test_fixed_step_imex_accepts_stiff_and_nonstiff():
    nonstiff = _LinearDecay(stiff=False)
    stiff = _LinearDecay(stiff=True)
    assert fixed_step_imex.supports(nonstiff) is True
    assert fixed_step_imex.supports(stiff) is True


def test_imex_facade_supports_delegates_and_accepts_stiff_and_nonstiff():
    nonstiff = _LinearDecay(stiff=False)
    stiff = _LinearDecay(stiff=True)
    assert imex.supports(nonstiff) is True
    assert imex.supports(stiff) is True
    # facade genuinely delegates, not hardcoded True -> same verdicts as the underlying
    # module for both flag values.
    assert imex.supports(nonstiff) == fixed_step_imex.supports(nonstiff)
    assert imex.supports(stiff) == fixed_step_imex.supports(stiff)
