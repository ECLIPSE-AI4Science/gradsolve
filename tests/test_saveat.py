"""Dense output on the JAX lanes — ``saveat``.

Covers the saveat dense-output contract on the JAX lanes:

  * ``solve(..., saveat=ts)`` -> ``SolveResult`` gains ``y_saved (n, k, dim)`` +
    ``ts_saved (k,)``, host NumPy (consistent with the existing host-side contract).
  * ``grad_closure(..., saveat=ts)`` -> the closure returns a jax array ``(n, k, dim)`` so
    time-series losses differentiate.
  * ``saveat=None`` -> today's behavior, bit-compatible.
  * Deterministic routing, each row stating its expected actual engine; explicit
    final-state-only engines raise ValueError.
  * Streaming: carry O(n*k*dim), mesh O(n*S), and no O(n*S*dim) state history is ever
    materialized (asserted by an allocation test).

Value gates are two-tier, and the tiers test different things:

  (i) Saving correctness — with ``saveat`` placed on exact mesh points, the saved states
      must equal a same-discretization trajectory oracle (the identical lane re-run,
      emitting its states) to rel <= 1e-12. A mesh point re-steps by exactly its own step's
      dt, so no sub-stepping error can hide here: this isolates the saving code — did
      the right state land in the right output slot? It says nothing about the integrator's
      accuracy.

  (ii) Dense/solution accuracy — at interior (non-mesh) times, versus analytic references,
      at lane-specific declared tolerances. These are per-lane numbers, not one
      constant: the adaptive replay and the 10k-step Tsit5 scan are held to
      1e-6, while the order-1 fixed IMEX lane is held to 1e-2 because its global error
      legitimately misses 1e-6 — gating it at 1e-6 would be testing a fiction.

Tier (ii) is what rejected an earlier design: cubic Hermite interpolation between the
bracketing (y, f) pairs missed the 1e-6 replay gate, because the adaptive controller
takes big steps where the problem is easy. States are now produced by re-stepping from the
bracketing step to the exact requested time with the lane's own step function, which makes
dense accuracy equal endpoint accuracy.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve
from gradsolve.solvers.tsit5_step import tsit5_step
from gradsolve.warp import warp_ode, warp_rosenbrock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class LinearDecay:
    """dy/dt = -k*y; analytic y(t) = y0*exp(-k*(t-t0)). Matches no registered Warp field."""

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
    name = "diffeqgpu_lorenz"
    dim = 3
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        sigma, rho, beta = 10.0, params[0], 8.0 / 3.0
        return jnp.array([sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1],
                          y[0] * y[1] - beta * y[2]])


class Robertson:
    name = "robertson"
    dim = 3
    t0 = 0.0
    t1 = 1.0e2
    is_stiff = True

    def f_jax(self, t, y, params):
        k1, k2, k3 = params[0], params[1], params[2]
        return jnp.array([-k1 * y[0] + k3 * y[1] * y[2],
                          k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2],
                          k2 * y[1] ** 2])


K = 0.5
Y0_VEC = np.array([1.0, 2.0])

requires_warp_nonstiff = pytest.mark.skipif(
    not warp_ode.supports(Lorenz()), reason="Warp/'lorenz' field unavailable — SKIP, never "
                                            "silently reroute to a pure-JAX lane")
requires_warp_stiff = pytest.mark.skipif(
    not warp_rosenbrock.supports(Robertson()), reason="Warp/'robertson' field unavailable")


def _decay_batch(n: int = 3):
    return np.tile(Y0_VEC, (n, 1)), np.tile(np.array([K]), (n, 1))


def _analytic(y0: np.ndarray, ts: np.ndarray, k: float = K) -> np.ndarray:
    """y(t) = y0*exp(-k*t) evaluated on a saveat grid -> (n, k, dim)."""
    return y0[:, None, :] * np.exp(-k * ts)[None, :, None]


SAVING_TOL = 1e-12       # tier (i): the saving code
REPLAY_ACC_TOL = 1e-6    # tier (ii): adaptive replay / 10k-step Tsit5 scan
IMEX_ACC_TOL = 1e-2      # tier (ii): order-1 fixed IMEX — its declared tolerance


# ---------------------------------------------------------------------------
# Tier (i) — SAVING correctness against a same-discretization oracle
# ---------------------------------------------------------------------------


def _tsit5_trajectory_oracle(problem, y0i, pi, dts):
    """Re-run the identical lane arithmetic, emitting every state: (ts[S], ys[S, dim]).

    Materializing the whole history is exactly what the library must not do — but it is
    what a test needs in order to check the library saved the right states.
    """
    f = problem.f_jax

    def body(carry, dt):
        t, y = carry
        y_next = tsit5_step(f, t, y, dt, pi)
        return (t + dt, y_next), (t + dt, y_next)

    _carry, (ts, ys) = jax.lax.scan(body, (jnp.asarray(problem.t0), jnp.asarray(y0i)), dts)
    return np.asarray(ts), np.asarray(ys)


def test_saving_correctness_fixed_step_tsit5_on_exact_mesh_points():
    """Tier (i): saveat on mesh points must reproduce the lane's own states exactly.

    A saveat time that is a mesh point re-steps by exactly that step's dt from that step's
    own left state, so no interpolation or sub-stepping error can hide here. This isolates
    the saving code: did the right state land in the right output slot?
    """
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=2)
    n_steps = 10_000
    h = (problem.t1 - problem.t0) / n_steps
    dts = jnp.full((n_steps,), h)

    ts_oracle, ys_oracle = _tsit5_trajectory_oracle(problem, y0[0], params[0], dts)
    idx = np.array([0, 1, 137, 5000, n_steps - 1])
    saveat = ts_oracle[idx]

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5", saveat=saveat)

    assert res.y_saved.shape == (y0.shape[0], len(saveat), problem.dim)
    np.testing.assert_allclose(res.y_saved[0], ys_oracle[idx], rtol=SAVING_TOL, atol=0)


def test_saving_correctness_replay_lane_on_exact_mesh_points():
    """Tier (i) for the record-and-replay lane: saveat on the recorded mesh's own accepted
    step boundaries must reproduce the replayed states exactly."""
    from gradsolve.solvers.tsit5_replay import record_tsit5_jax

    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=2)
    _yf, dts_padded, n_acc = record_tsit5_jax(
        problem.f_jax, y0, params, problem.t0, problem.t1, rtol=1e-6, atol=1e-9)

    dts0 = jnp.asarray(dts_padded[0])
    ts_oracle, ys_oracle = _tsit5_trajectory_oracle(problem, y0[0], params[0], dts0)
    idx = np.array([0, 2, int(n_acc[0]) // 2, int(n_acc[0]) - 1])
    saveat = ts_oracle[idx]

    res = gradsolve.solve(problem, y0, params, engine="warp_replay", saveat=saveat)

    np.testing.assert_allclose(res.y_saved[0], ys_oracle[idx], rtol=SAVING_TOL, atol=0)


# ---------------------------------------------------------------------------
# Tier (ii) — interpolation / solution accuracy at interior times
# ---------------------------------------------------------------------------


def test_interpolation_accuracy_fixed_step_tsit5_vs_analytic():
    """Tier (ii): interior, non-mesh times against the analytic linear-decay solution."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.017, 0.3183, 0.5, 0.77, 0.999])

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5", saveat=saveat)

    np.testing.assert_allclose(
        res.y_saved, _analytic(y0, saveat), rtol=REPLAY_ACC_TOL, atol=0)


def test_interpolation_accuracy_replay_vs_analytic():
    """Tier (ii) on the adaptive record-and-replay mesh (coarse steps — this is where the
    dense-output code actually has to work: at rtol=1e-6 this lane accepts ~5 steps of width
    up to 0.53, which is what made the original cubic-Hermite design miss this gate)."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.017, 0.3183, 0.5, 0.77, 0.999])

    res = gradsolve.solve(problem, y0, params, engine="warp_replay", saveat=saveat)

    np.testing.assert_allclose(
        res.y_saved, _analytic(y0, saveat), rtol=REPLAY_ACC_TOL, atol=0)


def test_interpolation_accuracy_fixed_step_imex_declared_tolerance():
    """Tier (ii) for the order-1 IMEX lane at its declared 1e-2 — its global error
    legitimately misses 1e-6, so gating it there would test a fiction. The tolerance is a
    statement about the integrator, not about the saving code (tier (i) covers that).
    """
    problem = LinearDecay(stiff=True)
    y0, params = _decay_batch()
    saveat = np.array([0.017, 0.5, 0.999])

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_imex", saveat=saveat)

    np.testing.assert_allclose(
        res.y_saved, _analytic(y0, saveat), rtol=IMEX_ACC_TOL, atol=0)
    # ...and it must be genuinely better than the tolerance is loose (else 1e-2 is hiding a
    # broken lane rather than a genuine order-1 error).
    assert np.max(np.abs(res.y_saved - _analytic(y0, saveat))) > 0.0


@requires_warp_nonstiff
def test_interpolation_accuracy_warp_replay_registered_field_vs_tight_reference():
    """Tier (ii) on the genuine Warp record + JAX replay, against a tight diffrax
    reference at the same interior times."""
    problem = Lorenz()
    n = 2
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    y0[:, 0] += 0.1 * np.arange(n)
    params = np.tile(np.array([28.0]), (n, 1))
    saveat = np.array([0.13, 0.5, 0.87])

    res = gradsolve.solve(problem, y0, params, engine="warp_replay", saveat=saveat,
                        rtol=1e-10, atol=1e-12)
    assert res.solver in ("warp_replay", "tsit5_replay")

    import diffrax as dfx
    term = dfx.ODETerm(problem.f_jax)

    def ref_one(y0i, pi):
        return dfx.diffeqsolve(
            term, dfx.Tsit5(), problem.t0, problem.t1, None, y0i, args=pi,
            stepsize_controller=dfx.PIDController(rtol=1e-12, atol=1e-14),
            saveat=dfx.SaveAt(ts=jnp.asarray(saveat)), max_steps=200_000).ys

    ref = np.asarray(jax.vmap(ref_one)(jnp.asarray(y0), jnp.asarray(params)))
    np.testing.assert_allclose(res.y_saved, ref, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Endpoint / domain / ordering contract
# ---------------------------------------------------------------------------


def test_endpoint_identity_t1_in_saveat_equals_y_final_exactly():
    """y_saved[:, -1] == y_final when t1 in saveat — an identity (a time at or past the final
    accumulated time resolves to the terminal state itself), so it is asserted bit-exactly,
    not to a tolerance."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.5, problem.t1])

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5", saveat=saveat)

    np.testing.assert_array_equal(res.y_saved[:, -1, :], res.y_final)


def test_t0_in_saveat_returns_y0_exactly():
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([problem.t0, 0.5])

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5", saveat=saveat)

    np.testing.assert_array_equal(res.y_saved[:, 0, :], y0)


def test_terminal_time_on_replay_lane_resolves_to_terminal_state():
    """The replayed mesh accumulates t0+sum(dts), which can land ~1e-16 below t1. A saveat
    entry of exactly t1 must still resolve to the terminal state rather than silently
    keeping its initial value)."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.5, problem.t1, problem.t1])  # duplicate terminal time

    res = gradsolve.solve(problem, y0, params, engine="warp_replay", saveat=saveat)

    np.testing.assert_array_equal(res.y_saved[:, -1, :], res.y_final)
    np.testing.assert_array_equal(res.y_saved[:, -2, :], res.y_final)


@pytest.mark.parametrize("bad", [
    np.array([0.5, 0.2]),          # unsorted
    np.array([-0.1, 0.5]),         # below t0
    np.array([0.5, 1.5]),          # above t1
])
def test_out_of_domain_or_unsorted_saveat_raises_valueerror(bad):
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=2)

    with pytest.raises(ValueError, match="saveat"):
        gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5", saveat=bad)


def test_ts_saved_is_returned_and_saveat_none_is_unchanged():
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.25, 0.75])

    res = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5", saveat=saveat)
    np.testing.assert_array_equal(res.ts_saved, saveat)
    assert isinstance(res.y_saved, np.ndarray)  # host NumPy, per the SolveResult contract

    plain = gradsolve.solve(problem, y0, params, engine="fixed_step_tsit5")
    assert plain.y_saved is None and plain.ts_saved is None
    np.testing.assert_array_equal(plain.y_final, res.y_final)  # saveat must not perturb it


# ---------------------------------------------------------------------------
# Routing — deterministic, each row states its EXPECTED actual engine
# ---------------------------------------------------------------------------


def test_auto_saveat_routes_nonstiff_unregistered_to_tsit5_replay():
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, saveat=np.array([0.5]))

    assert closure.route.actual == "tsit5_replay"


def test_solve_saveat_warp_replay_unregistered_records_the_reroute_reason():
    """solve(saveat=...) and grad_closure share _route_reverse, so they must also report the
    same reason for the same reroute. Naming warp_replay on a problem with no registered field
    reroutes to tsit5_replay; without the reason the route claimed the request was honoured."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.5])

    res = gradsolve.solve(problem, y0, params, saveat=saveat, engine="warp_replay")

    assert res.route.actual == "tsit5_replay"
    assert "no-registered-field" in res.route.reason


def test_auto_saveat_routes_stiff_unregistered_to_rodas5p_replay():
    problem = LinearDecay(stiff=True)
    y0, params = _decay_batch()

    closure = gradsolve.grad_closure(problem, y0, params, saveat=np.array([0.5]))

    assert closure.route.actual == "rodas5p_replay"


def test_solve_auto_saveat_stiff_unregistered_runs_the_rodas5p_dense_lane():
    """The forward saveat path shares _route_reverse with grad_closure, so an unregistered stiff
    problem lands on rodas5p_replay's dense lane (api._solve_saveat) — tolerance-honouring, at the
    replay tier's accuracy rather than the order-1 IMEX tier this replaces."""
    problem = LinearDecay(stiff=True)
    y0, params = _decay_batch()
    saveat = np.array([0.017, 0.5, 0.999])

    res = gradsolve.solve(problem, y0, params, saveat=saveat, rtol=1e-8, atol=1e-11)

    assert res.solver == "rodas5p_replay"
    assert res.route.actual == "rodas5p_replay"
    np.testing.assert_allclose(res.y_saved, _analytic(y0, saveat), rtol=REPLAY_ACC_TOL, atol=0)


@requires_warp_nonstiff
def test_auto_saveat_routes_nonstiff_registered_to_warp_replay():
    problem = Lorenz()
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    params = np.tile(np.array([28.0]), (2, 1))

    closure = gradsolve.grad_closure(problem, y0, params, saveat=np.array([0.5]))

    assert closure.route.actual == "warp_replay"


@requires_warp_stiff
def test_auto_saveat_routes_stiff_registered_to_rosenbrock_replay():
    problem = Robertson()
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    params = np.tile(np.array([0.04, 3.0e7, 1.0e4]), (2, 1))

    closure = gradsolve.grad_closure(problem, y0, params, saveat=np.array([50.0]),
                                   rtol=1e-6, atol=1e-9)

    assert closure.route.actual == "warp_rosenbrock"


@pytest.mark.parametrize("engine", ["diffrax", "cuda_tsit5", "warp_ode", "warp_rosenbrock",
                                    "fused_rosenbrock_backward"])
def test_solve_saveat_with_final_state_only_engine_raises(engine):
    """The fused kernels, the cuda lane and diffrax are final-state-only under this API
    (diffrax's own SaveAt is deliberately not wired). Explicitly naming one with saveat set
    must raise rather than silently ignore saveat or silently reroute."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=2)

    with pytest.raises(ValueError, match="saveat"):
        gradsolve.solve(problem, y0, params, engine=engine, saveat=np.array([0.5]))


@pytest.mark.parametrize("engine", ["diffrax", "cuda_tsit5", "fused_rosenbrock_backward"])
def test_grad_closure_saveat_with_incapable_engine_raises(engine):
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=2)

    with pytest.raises(ValueError, match="saveat"):
        gradsolve.grad_closure(problem, y0, params, engine=engine, saveat=np.array([0.5]))


# ---------------------------------------------------------------------------
# grad_closure(saveat=) — differentiable time-series loss
# ---------------------------------------------------------------------------


def test_grad_closure_saveat_returns_jax_array_and_differentiates_time_series_loss():
    """Time-series loss over the saved states, FD-checked on the same frozen mesh."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.25, 0.5, 0.9])

    closure = gradsolve.grad_closure(problem, y0, params, engine="warp_replay", saveat=saveat)

    out = closure(jnp.asarray(params))
    assert isinstance(out, jax.Array)
    assert out.shape == (y0.shape[0], len(saveat), problem.dim)

    target = jnp.asarray(_analytic(y0, saveat, k=0.6))

    def loss(p):
        return jnp.sum((closure(p) - target) ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(params)))
    assert g.shape == params.shape

    eps = 1e-6
    pp, pm = params.copy(), params.copy()
    pp[0, 0] += eps
    pm[0, 0] -= eps
    fd = (float(loss(jnp.asarray(pp))) - float(loss(jnp.asarray(pm)))) / (2 * eps)
    assert math.isclose(float(g[0, 0]), fd, rel_tol=1e-6)


def test_grad_closure_saveat_supports_wrt_y0():
    """saveat composes with the wrt= gradient — a time-series loss differentiated w.r.t. y0."""
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()
    saveat = np.array([0.25, 0.9])

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="warp_replay", wrt="y0", saveat=saveat)

    def loss(z):
        return jnp.sum(closure(z) ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(y0)))
    assert g.shape == y0.shape

    eps = 1e-6
    yp, ym = y0.copy(), y0.copy()
    yp[0, 0] += eps
    ym[0, 0] -= eps
    fd = (float(loss(jnp.asarray(yp))) - float(loss(jnp.asarray(ym)))) / (2 * eps)
    assert math.isclose(float(g[0, 0]), fd, rel_tol=1e-6)


def test_grad_closure_saveat_none_is_bit_compatible_with_final_state_closure():
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch()

    plain = gradsolve.grad_closure(problem, y0, params, engine="warp_replay")
    explicit_none = gradsolve.grad_closure(problem, y0, params, engine="warp_replay",
                                         saveat=None)

    pj = jnp.asarray(params)
    np.testing.assert_array_equal(np.asarray(plain(pj)), np.asarray(explicit_none(pj)))


# ---------------------------------------------------------------------------
# Memory — streaming, no O(n*S*dim) state history
# ---------------------------------------------------------------------------


def test_saveat_never_materializes_the_full_state_history():
    """The memory bound: the scan carries the active bracket plus the k
    outputs, so peak allocation is independent of the step count S.

    Asserted structurally against the lowered HLO rather than by wall-clock RSS: no buffer
    in the compiled closure may be large enough to hold an (S, dim) per-trajectory state
    history. S here is 10x the smaller run, so an O(S*dim) history would grow the largest
    buffer ~10x; a streaming implementation leaves it flat.
    """
    problem = LinearDecay(stiff=False)
    y0, params = _decay_batch(n=1)
    saveat = np.array([0.25, 0.5, 0.75])

    from gradsolve.solvers import fixed_step_tsit5

    def biggest_buffer(n_steps):
        lowered = jax.jit(
            lambda z, p: fixed_step_tsit5.solve_jax_saveat(
                problem, z, p, jnp.asarray(saveat), n_steps=n_steps)
        ).lower(jnp.asarray(y0), jnp.asarray(params))
        return lowered.compile().memory_analysis().temp_size_in_bytes

    small = biggest_buffer(1_000)
    large = biggest_buffer(10_000)
    assert large <= small + 4096, (
        f"temp allocation grew with the step count ({small} -> {large} bytes): the state "
        "history is being materialized instead of streamed")
