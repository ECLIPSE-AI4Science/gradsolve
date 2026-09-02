"""GPU/JAX recorder: same meshes as the host recorders, same failure semantics."""
from __future__ import annotations

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradsolve.solvers.method import (
    TSIT5_METHOD,
    ControllerState,
    controller_reject,
    controller_update,
    record_adaptive,
)
from gradsolve.solvers.record_jax import _record_one_traj, record_adaptive_jax
from gradsolve.solvers.rodas5p_replay import _record_one, record_rodas5p
from gradsolve.solvers.rodas5p_step import RODAS5P_METHOD, rodas5p_trial_step
from gradsolve.solvers.tsit5_replay import record_tsit5_jax, tsit5_adaptive
from gradsolve.solvers.vern7_replay import record_vern7
from gradsolve.solvers.vern7_step import VERN7_METHOD

# Every built-in method has beta2 == 0, so a synthetic proportional-integral method exercises
# the err_prev / accepted_any branch.
TSIT5_PI_METHOD = dataclasses.replace(TSIT5_METHOD, name="tsit5_pi_test", beta2=0.04)


@pytest.mark.parametrize("method", [TSIT5_METHOD, VERN7_METHOD, RODAS5P_METHOD, TSIT5_PI_METHOD])
def test_jnp_controller_matches_host_controller(method):
    from gradsolve.solvers.record_jax import (
        controller_reject_jnp,
        controller_update_jnp,
    )
    rng = np.random.default_rng(0)
    for _ in range(200):
        err = float(10.0 ** rng.uniform(-12, 2))
        h = float(10.0 ** rng.uniform(-6, 0))
        prev = float(10.0 ** rng.uniform(-3, 1))
        for accepted_any in (False, True):
            cs = ControllerState(err_prev=prev, accepted_any=accepted_any)
            dt_h, cs_h = controller_update(method, cs, err, h)
            dt_j, prev_j, any_j = controller_update_jnp(
                method, jnp.float64(prev), jnp.bool_(accepted_any), jnp.float64(err), jnp.float64(h))
            np.testing.assert_allclose(float(dt_j), dt_h, rtol=1e-15, atol=0)
            assert float(prev_j) == cs_h.err_prev and bool(any_j) == cs_h.accepted_any
            dt_hr, cs_hr = controller_reject(cs, err, h, method)
            dt_jr, prev_jr, any_jr = controller_reject_jnp(
                method, jnp.float64(prev), jnp.bool_(accepted_any), jnp.float64(err), jnp.float64(h))
            np.testing.assert_allclose(float(dt_jr), dt_hr, rtol=1e-15, atol=0)
            assert float(prev_jr) == cs_hr.err_prev and bool(any_jr) == cs_hr.accepted_any


class Lorenz:
    """Small self-contained problems (the Problem protocol: six members); the last parameter is
    spread across the ensemble so trajectories accept different numbers of steps."""
    name, dim, t0, t1, is_stiff = "user_lorenz", 3, 0.0, 1.0, False
    y0_ref, params_ref = (1.0, 0.0, 0.0), (10.0, 8.0 / 3.0, 21.0)

    def f_jax(self, t, y, p):
        sigma, beta, rho = p[0], p[1], p[2]
        return jnp.array([sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1],
                          y[0] * y[1] - beta * y[2]])


class VanDerPol:
    name, dim, t0, t1, is_stiff = "user_vdp", 2, 0.0, 5.0, False
    y0_ref, params_ref = (2.0, 0.0), (3.0,)

    def f_jax(self, t, y, p):
        return jnp.array([y[1], p[0] * ((1.0 - y[0] ** 2) * y[1] - y[0])])


class Robertson:
    name, dim, t0, t1, is_stiff = "user_robertson", 3, 0.0, 1.0e2, True
    y0_ref, params_ref = (1.0, 0.0, 0.0), (0.04, 3.0e7, 1.0e4)

    def f_jax(self, t, y, p):
        r1, r2, r3 = p[0] * y[0], p[1] * y[1] * y[1], p[2] * y[1] * y[2]
        return jnp.array([-r1 + r3, r1 - r3 - r2, r2])


def _batch(problem, n, seed=0):
    rng = np.random.default_rng(seed)
    y0 = np.tile(np.asarray(problem.y0_ref, dtype=np.float64), (n, 1))
    params = np.tile(np.asarray(problem.params_ref, dtype=np.float64), (n, 1))
    params[:, -1] *= 1.0 + 0.1 * rng.uniform(-1, 1, size=n)   # spread the last parameter
    return y0, params


def _eager_host_mesh(host_loop, prob, y0, params, rtol, atol):
    """Per-trajectory accepted mesh from the numpy host loop driven with a non-jitted RHS.

    The library recorders (record_tsit5_jax / record_vern7) jit the RHS while doing the stage
    sums in numpy; that RHS fusion is the bulk of the compiled-path drift. Feeding the same host
    loop an eager RHS removes it so the logic pins compare the loop decisions, not fused
    multiply-add. ``host_loop(f, y0_i, t0, t1, rtol, atol, dt0, max_steps)`` returns
    ``(y_final, dts, n_rej, status)``.
    """
    dt0 = (prob.t1 - prob.t0) / 100.0
    meshes = []
    for i in range(len(y0)):
        def f_eval(t, yv, p=params[i]):
            return np.asarray(prob.f_jax(t, yv, p), dtype=np.float64)
        _yf, dts_i, _nrej, status = host_loop(f_eval, y0[i], prob.t0, prob.t1, rtol, atol, dt0, 50000)
        assert status == 0
        meshes.append(dts_i)
    return meshes


def _eager_device_mesh(method, prob, y0, params, rtol, atol):
    """Per-trajectory device mesh, each run under jax.disable_jit() without vmap so the
    ``while_loop`` executes as an eager Python loop (no fused multiply-add). vmap (which
    record_adaptive_jax uses for the ensemble) lowers the loop to one batched XLA primitive
    that reintroduces FMA even under disable_jit, so the logic pin drives one trajectory at a
    time through the same per-trajectory builder record_adaptive_jax vmaps."""
    dt0 = (prob.t1 - prob.t0) / 100.0
    run = _record_one_traj(method, prob.f_jax, prob.t0, prob.t1, rtol, atol, dt0, 50000, 1024, -1.0)
    meshes = []
    with jax.disable_jit():
        for i in range(len(y0)):
            _y, dts_i, nacc_i, _nrej, status = run(jnp.asarray(y0[i]), jnp.asarray(params[i]))
            assert int(status) == 0
            meshes.append(np.asarray(dts_i)[: int(nacc_i)])
    return meshes


@pytest.mark.parametrize("problem_cls", [Lorenz, VanDerPol])
def test_jax_recorder_tsit5_logic_matches_host_eager(problem_cls):
    # Logic pin: run the per-trajectory while_loop eagerly (jit disabled, no vmap) so there is no
    # fused multiply-add, and compare to the numpy host loop driven with an eager RHS. With FMA
    # gone the meshes agree to ~1e-10 relative (Lorenz is bit-exact; one Van der Pol trajectory
    # amplifies a single XLA-vs-numpy last-bit difference at its stiff transition to ~1.7e-10) —
    # NOT the ~1e-8 of the compiled path. rtol 1e-8 sits far above that library-level floor and
    # far below anything a real logic bug (wrong accept threshold, controller, or buffer index,
    # which move dt by O(1) or change the step count) could hide under, so this pins the loop
    # LOGIC independent of arithmetic. Step counts are held exact.
    prob = problem_cls()
    y0, params = _batch(prob, n=4)
    dts_h = _eager_host_mesh(tsit5_adaptive, prob, y0, params, 1e-6, 1e-9)
    dts_j = _eager_device_mesh(TSIT5_METHOD, prob, y0, params, 1e-6, 1e-9)
    for i in range(len(y0)):
        assert len(dts_j[i]) == len(dts_h[i])
        np.testing.assert_allclose(dts_j[i], dts_h[i], rtol=1e-8, atol=0)


@pytest.mark.parametrize("problem_cls", [Lorenz, VanDerPol])
def test_jax_recorder_records_the_host_tsit5_mesh(problem_cls):
    # Compiled path: XLA fuses the trial-step sums into fused multiply-adds while the numpy host
    # does not, so the accepted step sizes drift by ~1e-8 relative even though the step counts and
    # final states are identical.
    prob = problem_cls()
    y0, params = _batch(prob, n=8)
    yf_h, dts_h, nacc_h, nrej_h = record_tsit5_jax(
        prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-6, atol=1e-9, return_rejected=True)
    yf_j, dts_j, nacc_j, nrej_j = record_adaptive_jax(
        TSIT5_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-6, atol=1e-9)
    assert nacc_j.tolist() == nacc_h.tolist()
    assert nrej_j.tolist() == nrej_h.tolist()
    assert dts_j.shape == dts_h.shape
    np.testing.assert_allclose(dts_j, dts_h, rtol=1e-6, atol=0)
    np.testing.assert_allclose(yf_j, yf_h, rtol=1e-10, atol=1e-12)


def test_jax_recorder_vern7_logic_matches_host_eager():
    # Logic pin (see the Tsit5 eager test): jit disabled, no vmap, eager host RHS ⇒ no fused
    # multiply-add ⇒ the mesh matches the numpy host to the XLA-vs-numpy last-bit floor, pinning
    # the loop logic. rtol 1e-8 (Vern7 Lorenz is bit-exact here).
    prob = Lorenz()
    y0, params = _batch(prob, n=4)
    dts_h = _eager_host_mesh(functools.partial(record_adaptive, VERN7_METHOD), prob, y0, params, 1e-7, 1e-10)
    dts_j = _eager_device_mesh(VERN7_METHOD, prob, y0, params, 1e-7, 1e-10)
    for i in range(len(y0)):
        assert len(dts_j[i]) == len(dts_h[i])
        np.testing.assert_allclose(dts_j[i], dts_h[i], rtol=1e-8, atol=0)


def test_jax_recorder_records_the_host_vern7_mesh():
    # Compiled path: fused multiply-add in the order-6 embedded estimate drifts the step sizes by
    # ~5e-7 relative; step counts and final states match the host exactly.
    prob = Lorenz()
    y0, params = _batch(prob, n=4)
    yf_h, dts_h, nacc_h = record_vern7(prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-7, atol=1e-10)
    yf_j, dts_j, nacc_j, _ = record_adaptive_jax(
        VERN7_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-7, atol=1e-10)
    assert nacc_j.tolist() == nacc_h.tolist()
    np.testing.assert_allclose(dts_j, dts_h, rtol=1e-5, atol=0)
    np.testing.assert_allclose(yf_j, yf_h, rtol=1e-10, atol=1e-12)


def test_jax_recorder_buffer_grows_until_the_mesh_fits():
    prob = Lorenz()
    y0, params = _batch(prob, n=2)
    ref = record_adaptive_jax(TSIT5_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1,
                              rtol=1e-8, atol=1e-11)
    cap0 = int(ref[2].max()) - 1          # one below the true count: exactly one doubling
    small = record_adaptive_jax(TSIT5_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1,
                                rtol=1e-8, atol=1e-11, cap0=cap0)
    assert cap0 >= 1
    np.testing.assert_array_equal(small[1], ref[1])


def test_jax_recorder_raises_on_exhausted_attempts():
    prob = Lorenz()
    y0, params = _batch(prob, n=2)
    with pytest.raises(RuntimeError, match="exhausted max_steps"):
        record_adaptive_jax(TSIT5_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1,
                            rtol=1e-8, atol=1e-11, max_steps=5)


def test_jax_recorder_empty_ensemble():
    prob = Lorenz()
    yf, dts, nacc, nrej = record_adaptive_jax(
        TSIT5_METHOD, prob.f_jax, np.zeros((0, prob.dim)), np.zeros((0, 3)), prob.t0, prob.t1,
        rtol=1e-6, atol=1e-9)
    assert yf.shape == (0, prob.dim) and dts.shape == (0, 0) and nacc.shape == (0,) and nrej.shape == (0,)


def _eager_rodas5p_host_mesh(prob, y0, params, rtol, atol):
    """Per-trajectory Rodas5P accepted mesh from the host _record_one, run eagerly (jit disabled)
    with the same jax RHS the device builder uses. Both sides then execute identical eager jax
    arithmetic (no XLA fused multiply-add on either), so the logic pin compares the loop decisions,
    not FMA. The explicit-method _eager_host_mesh feeds a numpy RHS, which the Rodas5P trial step
    (jax.jacfwd) cannot take, hence this Rodas5P-specific helper."""
    dt0 = (prob.t1 - prob.t0) / 100.0
    meshes = []
    with jax.disable_jit():
        for i in range(len(y0)):
            p_i = jnp.asarray(params[i])
            def step(t, y, dt, p_i=p_i):
                return rodas5p_trial_step(prob.f_jax, t, y, dt, p_i)
            _yf, dts_i, _nrej, status = _record_one(step, y0[i], prob.t0, prob.t1, rtol, atol, dt0, 50000)
            assert status == 0
            meshes.append(np.asarray(dts_i))
    return meshes


def test_jax_recorder_rodas5p_logic_matches_host_eager():
    # Logic pin (see the Tsit5 eager test): the per-trajectory while_loop and the host _record_one
    # both run eagerly under the same jax RHS ⇒ no fused multiply-add on either ⇒ the meshes agree
    # to ~machine precision, pinning the loop logic (accept threshold, controller, floor/non-finite
    # gating) independent of arithmetic. On a GPU, 2 of 38 step
    # sizes differed by 2.4e-8 relative at identical step counts (GPU arithmetic on one stage value),
    # so the pin sits at rtol 1e-7.
    prob = Robertson()
    y0, params = _batch(prob, n=4)
    dts_h = _eager_rodas5p_host_mesh(prob, y0, params, 1e-6, 1e-9)
    dts_j = _eager_device_mesh(RODAS5P_METHOD, prob, y0, params, 1e-6, 1e-9)
    for i in range(len(y0)):
        assert len(dts_j[i]) == len(dts_h[i])
        np.testing.assert_allclose(dts_j[i], dts_h[i], rtol=1e-7, atol=0)


def test_jax_recorder_records_the_host_rodas5p_mesh():
    # Compiled path: XLA fuses the Rodas5P trial-step sums into fused multiply-adds while the numpy
    # host loop does not, so accepted step sizes drift; step counts and final states match exactly.
    prob = Robertson()
    y0, params = _batch(prob, n=4)
    yf_h, dts_h, nacc_h, nrej_h = record_rodas5p(
        prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-6, atol=1e-9)
    yf_j, dts_j, nacc_j, nrej_j = record_adaptive_jax(
        RODAS5P_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-6, atol=1e-9)
    assert nacc_j.tolist() == nacc_h.tolist()
    assert nrej_j.tolist() == nrej_h.tolist()
    np.testing.assert_allclose(dts_j, dts_h, rtol=1e-6, atol=0)
    np.testing.assert_allclose(yf_j, yf_h, rtol=1e-9, atol=1e-14)


def test_jax_recorder_reports_the_underflow_floor():
    """A field that blows up in finite time drives the step size to the floor (status 2)."""
    class Blowup:
        dim, t0, t1, is_stiff, name = 1, 0.0, 2.0, True, "blowup"
        def f_jax(self, t, y, p):
            return p[0] * y * y          # y' = p*y^2 from y0=1 blows up at t=1

    prob = Blowup()
    with pytest.raises(RuntimeError, match="underflow floor"):
        record_adaptive_jax(RODAS5P_METHOD, prob.f_jax, np.ones((2, 1)), np.ones((2, 1)),
                            prob.t0, prob.t1, rtol=1e-6, atol=1e-9, max_steps=20000)


def test_choose_recorder_rule(monkeypatch):
    from gradsolve.solvers.record_jax import choose_recorder
    monkeypatch.delenv("GRADSOLVE_RECORDER", raising=False)
    assert choose_recorder(4, "host") == "host" and choose_recorder(4, "jax") == "jax"
    on_cpu = jax.default_backend() == "cpu"
    assert choose_recorder(4, "auto") == ("host" if on_cpu else "jax")
    assert choose_recorder(32, "auto") == "jax"
    monkeypatch.setenv("GRADSOLVE_RECORDER", "host")
    assert os.environ["GRADSOLVE_RECORDER"] == "host"
    assert choose_recorder(4096, "auto") == "host"
    # The env override also forces jax on a CPU.
    monkeypatch.setenv("GRADSOLVE_RECORDER", "jax")
    assert choose_recorder(4, "auto") == "jax"
    with pytest.raises(ValueError):
        choose_recorder(4, "gpu")


def test_public_recorders_accept_the_switch_and_agree():
    # The keyword-only switch reaches both paths and they agree: identical accepted-step counts,
    # step sizes to ~1e-6 relative (XLA fuses the trial-step sums into fused multiply-adds, the
    # numpy host loop does not — the same ~1e-8 drift the host-vs-device mesh tests document, here
    # slightly amplified on the shorter n=4 ensemble). Correct code cannot meet a much tighter
    # rtol because the two paths differ by that FMA drift, so this uses rtol=1e-6, the sibling
    # mesh test's value (test_jax_recorder_records_the_host_tsit5_mesh). The exact-count
    # assertion below carries the strict half of the agreement.
    prob = Lorenz()
    y0, params = _batch(prob, n=4)
    a = record_tsit5_jax(prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-6, atol=1e-9, recorder="host")
    b = record_tsit5_jax(prob.f_jax, y0, params, prob.t0, prob.t1, rtol=1e-6, atol=1e-9, recorder="jax")
    assert a[2].tolist() == b[2].tolist()
    np.testing.assert_allclose(a[1], b[1], rtol=1e-6, atol=0)


def test_jax_recorder_reuses_the_compiled_kernel_across_calls():
    # The device recorder's timing is only pure execution if repeated records of
    # the same (method, f, tols, horizon, cap, shapes) reuse one compiled jit instead of re-tracing.
    # Assert the cache hands back the identical jitted object on the second call, and that its
    # result is unchanged.
    from gradsolve.solvers.record_jax import _JIT_CACHE, _cached_run

    prob = Lorenz()
    y0, params = _batch(prob, n=4)
    _JIT_CACHE.clear()
    a = record_adaptive_jax(TSIT5_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1,
                            rtol=1e-6, atol=1e-9)
    after_first = dict(_JIT_CACHE)
    b = record_adaptive_jax(TSIT5_METHOD, prob.f_jax, y0, params, prob.t0, prob.t1,
                            rtol=1e-6, atol=1e-9)
    assert len(_JIT_CACHE) == len(after_first)          # no new kernel built on the second call
    run1 = _cached_run(TSIT5_METHOD, prob.f_jax, float(prob.t0), float(prob.t1),
                       1e-6, 1e-9, (prob.t1 - prob.t0) / 100.0, 50000, 1024, -1.0)
    run2 = _cached_run(TSIT5_METHOD, prob.f_jax, float(prob.t0), float(prob.t1),
                       1e-6, 1e-9, (prob.t1 - prob.t0) / 100.0, 50000, 1024, -1.0)
    assert run1 is run2                                  # same compiled object, not a rebuild
    np.testing.assert_array_equal(a[1], b[1])
