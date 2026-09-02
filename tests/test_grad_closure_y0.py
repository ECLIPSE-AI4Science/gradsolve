"""Initial-condition gradients — ``grad_closure(..., wrt=...)`` + the route contract.

Covers the ``wrt=`` gradient API:

  * ``wrt="params"`` (default) — closure ``params -> y_final``.
  * ``wrt="y0"`` — closure ``y0 -> y_final``, params closed over.
  * ``wrt=("y0", "params")`` — closure ``(y0, params) -> y_final``; gradient shapes mirror
    the input shapes.
  * Route observability — every closure carries ``.route`` (requested engine, actual
    engine, reason) so these tests assert the observed route, never the requested one.
  * ``fused_rosenbrock_backward`` stays params-only: any ``wrt`` involving y0 reroutes to
    the stiff replay loudly (``route.reason == "y0-unsupported"``), never silently.

FD gates:

  * scan/replay rows — central differences on the same frozen mesh. The closures returned
    for these lanes replay a mesh recorded once at closure-build time, so calling
    ``closure(y0 +/- eps)`` perturbs y0 *inside the replay* and never re-records. f64,
    step 1e-6 relative, pass rel-L2 <= 1e-6.
  * the diffrax row — has no frozen mesh by construction: diffrax's closure re-runs its
    adaptive step-size controller on every call, so each FD perturbation is an adaptive
    re-solve with its own step sequence, not a replay of one mesh. That semantic
    difference is why its gate is rel-L2 <= 1e-4, two orders looser than the replay rows
    (see ``test_diffrax_y0_grad_matches_adaptive_resolve_fd``).

Warp rows: the registered-field lanes (``warp_replay`` nonstiff/stiff) are skipped — never
silently rerouted to a pure-JAX lane — when Warp is unavailable, so a warp-absent CI cannot
report a green warp row it did not run.

Problem fixtures: ``LinearDecay`` deliberately matches no registered Warp field (exercises
the general-RHS lanes + fallbacks); ``Lorenz``/``Robertson`` do match registered fields
("lorenz"/"robertson") and are the only way to reach the genuine warp record + JAX replay.
"""
from __future__ import annotations

import importlib.util
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve
from gradsolve.warp import warp_ode, warp_rosenbrock

# ---------------------------------------------------------------------------
# Problem fixtures (duck-typed Problem protocol: name, dim, t0, t1, is_stiff, f_jax)
# ---------------------------------------------------------------------------


class LinearDecay:
    """dy/dt = -k*y; params (1,) -> k. Analytic: y(t1) = y0*exp(-k*(t1-t0)).

    ``name`` matches no registered Warp field by design, so field-routed engines report
    supports()==False here and the general-RHS lanes/fallbacks are exercised.
    """

    name = "unit_test_linear_decay"
    dim = 2
    t0 = 0.0
    t1 = 1.0

    def __init__(self, stiff: bool = False):
        self._stiff = stiff

    @property
    def is_stiff(self) -> bool:
        return self._stiff

    def f_jax(self, t, y, params):
        return -params[0] * y


class Lorenz:
    """Registered nonstiff field "lorenz" — reaches the genuine warp record + JAX replay."""

    name = "diffeqgpu_lorenz"
    dim = 3
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        sigma, rho, beta = 10.0, params[0], 8.0 / 3.0
        return jnp.array([
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ])


class Robertson:
    """Registered stiff field "robertson" — reaches the fused Rosenbrock record + replay."""

    name = "robertson"
    dim = 3
    t0 = 0.0
    t1 = 1.0e2
    is_stiff = True

    def f_jax(self, t, y, params):
        k1, k2, k3 = params[0], params[1], params[2]
        return jnp.array([
            -k1 * y[0] + k3 * y[1] * y[2],
            k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2],
            k2 * y[1] ** 2,
        ])


K = 0.5
Y0_VEC = np.array([1.0, 2.0])

_HAS_DIFFRAX = importlib.util.find_spec("diffrax") is not None
_WARP_NONSTIFF = warp_ode.supports(Lorenz())
_WARP_STIFF = warp_rosenbrock.supports(Robertson())
requires_warp_nonstiff = pytest.mark.skipif(
    not _WARP_NONSTIFF, reason="Warp (or its registered 'lorenz' field) unavailable — the "
                               "warp row is SKIPPED, never rerouted to a pure-JAX lane")
requires_warp_stiff = pytest.mark.skipif(
    not _WARP_STIFF, reason="Warp (or its registered 'robertson' field) unavailable — the "
                            "warp row is SKIPPED, never rerouted to a pure-JAX lane")


def _decay_batch(n: int = 3):
    return np.tile(Y0_VEC, (n, 1)), np.tile(np.array([K]), (n, 1))


def _lorenz_batch(n: int = 2):
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    y0[:, 0] += 0.1 * np.arange(n)  # break the degeneracy across trajectories
    return y0, np.tile(np.array([28.0]), (n, 1))


def _robertson_batch(n: int = 2):
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    return y0, np.tile(np.array([0.04, 3.0e7, 1.0e4]), (n, 1))


# ---------------------------------------------------------------------------
# FD helpers — fixed thresholds; f64; central differences; step 1e-6 relative
# ---------------------------------------------------------------------------

FD_REL_STEP = 1e-6
FROZEN_MESH_FD_TOL = 1e-6   # scan/replay rows: same-frozen-mesh FD
ADAPTIVE_FD_TOL = 1e-4      # diffrax row: adaptive re-solve FD (no frozen mesh)


def _loss_of(closure):
    return lambda arr: jnp.sum(closure(jnp.asarray(arr)) ** 2)


def _fd_grad(closure, x: np.ndarray) -> np.ndarray:
    """Central-difference d(sum(closure(x)**2))/dx, componentwise, step 1e-6 relative.

    For the replay/scan lanes the closure holds an already-recorded frozen mesh, so every
    evaluation here perturbs the input inside the replay — the mesh is never re-recorded.
    """
    loss = _loss_of(closure)
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        eps = FD_REL_STEP * max(1.0, abs(float(x[idx])))
        xp = x.copy()
        xp[idx] += eps
        xm = x.copy()
        xm[idx] -= eps
        g[idx] = (float(loss(xp)) - float(loss(xm))) / (2 * eps)
        it.iternext()
    return g


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||b||. A zero reference gives inf/nan, which fails the gate — that is
    intended: an all-zero FD gradient means the closure ignored the input it claims to
    differentiate, not that it agrees perfectly."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _assert_y0_grad_matches_fd(closure, y0: np.ndarray, tol: float) -> np.ndarray:
    g_ad = np.asarray(jax.grad(_loss_of(closure))(jnp.asarray(y0)))
    g_fd = _fd_grad(closure, y0)
    assert g_ad.shape == y0.shape, (  # return shapes mirror input shapes
        f"y0 gradient shape {g_ad.shape} != y0 shape {y0.shape}")
    rel = _rel_l2(g_ad, g_fd)
    assert rel <= tol, (
        f"y0 gradient disagrees with finite differences: rel-L2 {rel:.3e} > {tol:.0e} "
        f"(route={closure.route.actual!r}, |grad_ad|={np.linalg.norm(g_ad):.3e}, "
        f"|grad_fd|={np.linalg.norm(g_fd):.3e})")
    return g_ad


# ---------------------------------------------------------------------------
# Route observability contract
# ---------------------------------------------------------------------------


def test_route_attribute_reports_requested_actual_and_reason():
    """Every closure carries .route so tests assert the observed engine, not the asked-for
    one. A directly-supported engine routes to itself with no reason."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="fixed_step_tsit5")

    assert closure.route.requested == "fixed_step_tsit5"
    assert closure.route.actual == "fixed_step_tsit5"
    assert closure.route.reason is None


def test_route_reports_general_rhs_replay_fallback_for_unregistered_field():
    """engine="warp_replay" on a problem with no registered Warp field must be observably
    the general-RHS pure-JAX replay — the old API silently substituted it."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="warp_replay")

    assert closure.route.requested == "warp_replay"
    assert closure.route.actual == "tsit5_replay"
    assert "no-registered-field" in closure.route.reason


def test_route_reports_unsupported_fallback_to_stiff_replay():
    problem = LinearDecay(stiff=True)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="warp_rosenbrock")

    assert closure.route.requested == "warp_rosenbrock"
    assert closure.route.actual == "rodas5p_replay"
    assert closure.route.reason == "engine-does-not-support-problem"


# ---------------------------------------------------------------------------
# wrt="params" — the default is the engine's own closure
# ---------------------------------------------------------------------------


def test_default_wrt_is_params_and_matches_the_engine_closure_exactly():
    """The default closure must be bit-identical to the engine's own closure, in value and
    gradient — i.e. the wrt= routing adds no wrapper drift, and wrt="params" is not merely
    equivalent to the default but the same code path.
    """
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    from gradsolve.solvers.tsit5_replay import make_tsit5_frozen_mesh_closure
    reference = make_tsit5_frozen_mesh_closure(problem, y0, params, rtol=1e-6, atol=1e-9)

    closure = gradsolve.grad_closure(problem, y0, params, engine="warp_replay")
    explicit = gradsolve.grad_closure(problem, y0, params, engine="warp_replay", wrt="params")

    ref_val = np.asarray(reference(jnp.asarray(params)))
    np.testing.assert_array_equal(np.asarray(closure(jnp.asarray(params))), ref_val)
    np.testing.assert_array_equal(np.asarray(explicit(jnp.asarray(params))), ref_val)

    ref_g = np.asarray(jax.grad(_loss_of(reference))(jnp.asarray(params)))
    np.testing.assert_array_equal(
        np.asarray(jax.grad(_loss_of(closure))(jnp.asarray(params))), ref_g)


def test_invalid_wrt_raises_valueerror():
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=2)

    with pytest.raises(ValueError, match="wrt"):
        gradsolve.grad_closure(problem, y0, params, engine="fixed_step_tsit5", wrt="not_a_thing")


# ---------------------------------------------------------------------------
# wrt="y0" — the lane matrix
# ---------------------------------------------------------------------------


def test_fixed_step_tsit5_y0_grad_matches_analytic_and_fd():
    """Lane row: fixed_step_tsit5. y = y0*exp(-k*t) => d(sum(y**2))/dy0_j =
    2*y0_j*exp(-2*k*t) — checked against the closed form and same-mesh FD."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="fixed_step_tsit5", wrt="y0")

    assert closure.route.actual == "fixed_step_tsit5"
    np.testing.assert_allclose(
        np.asarray(closure(jnp.asarray(y0))),
        y0 * math.exp(-K * (problem.t1 - problem.t0)), atol=1e-10, rtol=0)

    g_ad = _assert_y0_grad_matches_fd(closure, y0, FROZEN_MESH_FD_TOL)
    expected = 2.0 * y0 * math.exp(-2.0 * K * (problem.t1 - problem.t0))
    np.testing.assert_allclose(g_ad, expected, rtol=1e-6, atol=0)


def test_fixed_step_imex_y0_grad_matches_fd():
    """Lane row: fixed_step_imex (order-1 stiff scan; FD self-consistency on its own
    fixed grid — its global error legitimately misses the analytic solution)."""
    problem = LinearDecay(stiff=True)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="fixed_step_imex", wrt="y0")

    assert closure.route.actual == "fixed_step_imex"
    _assert_y0_grad_matches_fd(closure, y0, FROZEN_MESH_FD_TOL)


def test_tsit5_replay_general_rhs_y0_grad_matches_frozen_mesh_fd():
    """Lane row: tsit5_replay (general RHS, pure-JAX record-and-replay)."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="warp_replay", wrt="y0")

    assert closure.route.actual == "tsit5_replay"
    _assert_y0_grad_matches_fd(closure, y0, FROZEN_MESH_FD_TOL)


@requires_warp_nonstiff
def test_warp_replay_nonstiff_registered_field_y0_grad_matches_frozen_mesh_fd():
    """Lane row: warp_replay nonstiff (registered field) — the genuine Warp record +
    JAX replay, not a pure-JAX substitute (asserted via the observed route)."""
    problem = Lorenz()
    y0, params = _lorenz_batch()

    closure = gradsolve.grad_closure(problem, y0, params, engine="warp_replay", wrt="y0")

    assert closure.route.actual == "warp_replay"
    _assert_y0_grad_matches_fd(closure, y0, FROZEN_MESH_FD_TOL)


@requires_warp_stiff
def test_warp_replay_stiff_registered_field_y0_grad_matches_frozen_mesh_fd():
    """Lane row: warp_replay stiff (registered Rosenbrock field)."""
    problem = Robertson()
    y0, params = _robertson_batch()

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="warp_rosenbrock", wrt="y0", rtol=1e-6, atol=1e-9)

    assert closure.route.actual == "warp_rosenbrock"
    _assert_y0_grad_matches_fd(closure, y0, FROZEN_MESH_FD_TOL)


@pytest.mark.skipif(not _HAS_DIFFRAX, reason="diffrax not installed — the diffrax row is SKIPPED, never rerouted")
def test_diffrax_y0_grad_matches_adaptive_resolve_fd():
    """Lane row: diffrax fallback. Semantic difference: diffrax has no
    frozen mesh by construction — its closure re-runs the adaptive step-size controller on
    every call, so each FD perturbation below is an adaptive re-solve with its own step
    sequence rather than a replay of one recorded mesh. The controller's discrete step
    choices shift with y0, so the FD quotient carries that extra noise; the gate is
    therefore 1e-4, not the replay rows' 1e-6."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="diffrax", wrt="y0", rtol=1e-10, atol=1e-12)

    assert closure.route.actual == "diffrax"
    _assert_y0_grad_matches_fd(closure, y0, ADAPTIVE_FD_TOL)


# ---------------------------------------------------------------------------
# fused-fallback row — params-only engine + y0 request reroutes LOUDLY
# ---------------------------------------------------------------------------


@requires_warp_stiff
def test_fused_backward_with_y0_reroutes_to_stiff_replay_loudly():
    """fused_rosenbrock_backward's custom_vjp is params-only. Requesting a y0 gradient
    must reroute to the stiff replay with an observable reason — never silently return a
    closure that ignores the request."""
    problem = Robertson()
    y0, params = _robertson_batch()

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="fused_rosenbrock_backward", wrt="y0",
        rtol=1e-6, atol=1e-9)

    assert closure.route.requested == "fused_rosenbrock_backward"
    assert closure.route.actual == "warp_rosenbrock"
    assert closure.route.reason == "y0-unsupported"
    _assert_y0_grad_matches_fd(closure, y0, FROZEN_MESH_FD_TOL)


@requires_warp_stiff
def test_fused_backward_genuine_lane_is_reached_and_route_stamped_for_params():
    """The fused backward on a registered stiff field is the one lane whose closure is not
    a plain lambda (it is a jax.custom_vjp object). Route stamping must survive that — and
    the params request must actually reach the fused engine rather than being rerouted."""
    problem = Robertson()
    y0, params = _robertson_batch()

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="fused_rosenbrock_backward", wrt="params",
        rtol=1e-6, atol=1e-9)

    assert closure.route.actual == "fused_rosenbrock_backward"
    assert closure.route.reason is None
    assert np.all(np.isfinite(np.asarray(closure(jnp.asarray(params)))))


def test_fused_backward_keeps_params_route_when_wrt_is_params():
    """The reroute is triggered by the y0 request alone — with wrt="params" the fused
    engine is still selected (here it further falls back only because this problem has no
    registered field, which the reason records)."""
    problem = LinearDecay(stiff=True)
    y0, params = _decay_batch(n=2)

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="fused_rosenbrock_backward", wrt="params")

    assert closure.route.requested == "fused_rosenbrock_backward"
    assert "y0-unsupported" not in (closure.route.reason or "")


# ---------------------------------------------------------------------------
# wrt=("y0", "params") — joint closure
# ---------------------------------------------------------------------------


def test_joint_wrt_closure_takes_both_and_grads_mirror_input_shapes():
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="fixed_step_tsit5", wrt=("y0", "params"))

    yf = np.asarray(closure(jnp.asarray(y0), jnp.asarray(params)))
    np.testing.assert_allclose(
        yf, y0 * math.exp(-K * (problem.t1 - problem.t0)), atol=1e-10, rtol=0)

    def loss(z, p):
        return jnp.sum(closure(z, p) ** 2)

    g_y0, g_p = jax.grad(loss, argnums=(0, 1))(jnp.asarray(y0), jnp.asarray(params))
    assert np.asarray(g_y0).shape == y0.shape
    assert np.asarray(g_p).shape == params.shape

    t = problem.t1 - problem.t0
    np.testing.assert_allclose(
        np.asarray(g_y0), 2.0 * y0 * math.exp(-2.0 * K * t), rtol=1e-6, atol=0)
    expected_dk = -2.0 * t * float(np.sum(Y0_VEC ** 2)) * math.exp(-2.0 * K * t)
    np.testing.assert_allclose(np.asarray(g_p)[:, 0], expected_dk, rtol=1e-6, atol=0)


def test_joint_wrt_matches_separate_single_wrt_closures():
    """The joint closure's two partials must agree with the single-wrt closures — one
    frozen mesh, three views of it."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    joint = gradsolve.grad_closure(
        problem, y0, params, engine="warp_replay", wrt=("y0", "params"))
    only_y0 = gradsolve.grad_closure(problem, y0, params, engine="warp_replay", wrt="y0")
    only_p = gradsolve.grad_closure(problem, y0, params, engine="warp_replay", wrt="params")

    g_y0_j, g_p_j = jax.grad(
        lambda z, p: jnp.sum(joint(z, p) ** 2), argnums=(0, 1)
    )(jnp.asarray(y0), jnp.asarray(params))

    np.testing.assert_allclose(
        np.asarray(g_y0_j),
        np.asarray(jax.grad(_loss_of(only_y0))(jnp.asarray(y0))), rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        np.asarray(g_p_j),
        np.asarray(jax.grad(_loss_of(only_p))(jnp.asarray(params))), rtol=1e-12, atol=0)
