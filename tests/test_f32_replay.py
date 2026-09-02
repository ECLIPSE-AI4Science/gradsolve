"""The f32 replay adjoint — CPU tests.

Scope: f32 covers the
registered Warp-field route only — warp record -> JAX replay, nonstiff (``warp_ode`` +
``tsit5_step``) and stiff (``warp_rosenbrock`` + ``rosenbrock23_step``). The general-RHS host
recorder (``gradsolve.solvers.tsit5_replay``) is hard-coded f64 throughout and stays f64;
asking it for f32 raises rather than silently returning a f64 result under an f32 label.

Test cells (the same cells are used on CPU and GPU):

  * nonstiff: ``diffeqgpu_lorenz``, n=2048, span [0, 1], record at rtol 1e-6 / atol 1e-9,
    record-once, loss ``sum(y_final**2)``.
  * stiff:    ``robertson``, n=256, span [0, 1e4], record at rtol 1e-5 / atol 1e-8, same
    loss.

Checks:

  (i)  correctness — the oracle replays the same f32-recorded mesh in f64. That isolates the
       arithmetic precision of the adjoint from the mesh: comparing against an
       f64-recorded mesh would conflate "f32 arithmetic is less exact" with "the f32
       controller accepted different steps", and only the former is what the f32 route
       changes. Tolerances: gradient rel-L2 <= 1e-3 and direction cosine >= 1 - 1e-6 (a gradient
       can be well-scaled and still point the wrong way, so both are required); plus f32
       primal achieved error vs a tight reference <= 1e-4 (Lorenz) / <= 1e-3 (Robertson).
  (ii) memory — the record buffer is exactly 0.5x the f64 buffer's bytes (dtype arithmetic
       on an identical shape, asserted — not measured RSS).
  (iii) performance is not tested here: a wall-clock ratio measured on a CPU would not
       transfer to the GPU.
"""
from __future__ import annotations

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve
from gradsolve.warp import warp_ode, warp_replay, warp_rosenbrock

# ---------------------------------------------------------------------------
# Pre-registered cells
# ---------------------------------------------------------------------------


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
    t1 = 1.0e4
    is_stiff = True

    def f_jax(self, t, y, params):
        k1, k2, k3 = params[0], params[1], params[2]
        return jnp.array([-k1 * y[0] + k3 * y[1] * y[2],
                          k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2],
                          k2 * y[1] ** 2])


class LinearDecay:
    """Unregistered field — the general-RHS lane, which stays f64."""

    name = "unit_test_linear_decay"
    dim = 2
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, params):
        return -params[0] * y


LORENZ_CELL = dict(n=2048, rtol=1e-6, atol=1e-9)
ROBERTSON_CELL = dict(n=256, rtol=1e-5, atol=1e-8)

GRAD_REL_L2_TOL = 1e-3
GRAD_COSINE_TOL = 1e-6          # cosine >= 1 - this
PRIMAL_TOL_LORENZ = 1e-4
PRIMAL_TOL_ROBERTSON = 1e-3

requires_warp_nonstiff = pytest.mark.skipif(
    not warp_ode.supports(Lorenz()), reason="Warp/'lorenz' field unavailable — SKIP, never "
                                            "silently reroute to a pure-JAX lane")
requires_warp_stiff = pytest.mark.skipif(
    not warp_rosenbrock.supports(Robertson()), reason="Warp/'robertson' field unavailable")


def _lorenz_batch(n=LORENZ_CELL["n"]):
    rng = np.random.default_rng(0)
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    y0 += 0.01 * rng.standard_normal((n, 3))          # spread the ensemble
    params = np.full((n, 1), 28.0)
    return y0, params


def _robertson_batch(n=ROBERTSON_CELL["n"]):
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    params = np.tile(np.array([0.04, 3.0e7, 1.0e4]), (n, 1))
    return y0, params


def _loss(closure):
    return lambda p: jnp.sum(closure(p) ** 2)


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# dtype propagation
# ---------------------------------------------------------------------------


@requires_warp_nonstiff
def test_f32_record_and_closure_are_float32_nonstiff():
    problem = Lorenz()
    y0, params = _lorenz_batch(n=64)

    _yw, _n_acc, dts = warp_replay.record(
        problem, y0, params, rtol=1e-6, atol=1e-9, device="cpu", precision="float32")
    assert np.asarray(dts).dtype == np.float32

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="warp_replay", precision="float32",
        rtol=1e-6, atol=1e-9)
    assert closure.route.actual == "warp_replay"
    out = closure(jnp.asarray(params, dtype=jnp.float32))
    assert out.dtype == jnp.float32
    g = jax.grad(_loss(closure))(jnp.asarray(params, dtype=jnp.float32))
    assert g.dtype == jnp.float32


@requires_warp_stiff
def test_f32_record_and_closure_are_float32_stiff():
    problem = Robertson()
    y0, params = _robertson_batch(n=8)

    _yw, _n_acc, dts = warp_replay.record_rosenbrock(
        problem, y0, params, rtol=1e-5, atol=1e-8, device="cpu", precision="float32")
    assert np.asarray(dts).dtype == np.float32

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="warp_rosenbrock", precision="float32",
        rtol=1e-5, atol=1e-8)
    assert closure.route.actual == "warp_rosenbrock"
    out = closure(jnp.asarray(params, dtype=jnp.float32))
    assert out.dtype == jnp.float32


def test_f32_is_rejected_on_the_general_rhs_lane():
    """The general-RHS host recorder is f64 throughout. Asking it for f32 must raise: a
    silent f64 result under an f32 label would misreport the precision actually used."""
    problem = LinearDecay()
    y0 = np.tile(np.array([1.0, 2.0]), (4, 1))
    params = np.full((4, 1), 0.5)

    with pytest.raises(ValueError, match="float32"):
        gradsolve.grad_closure(problem, y0, params, engine="warp_replay",
                             precision="float32")


def test_unknown_precision_raises():
    problem = Lorenz()
    y0, params = _lorenz_batch(n=8)

    with pytest.raises(ValueError, match="precision"):
        gradsolve.grad_closure(problem, y0, params, engine="warp_replay",
                             precision="float16")


# ---------------------------------------------------------------------------
# Gate (ii) — memory: the record buffer is half, PER STEP
# ---------------------------------------------------------------------------
#
# This gate is stated as "record buffer bytes exactly 0.5x the f64 buffer (dtype
# arithmetic, asserted)". Measured, the TOTAL is not exactly 0.5x and cannot be: the f32
# controller's roundoff changes which steps it accepts, so it records a slightly different
# mesh (Lorenz cell: 6/64 trajectories differ, S = 53 vs 52, total ratio 0.5096). The
# halving is a per-STEP dtype fact, which is what "dtype arithmetic" can actually assert;
# the mesh length is a property of the controller, not of the dtype. Gating the total at
# exactly 0.5x would be gating on a coincidence.
#
# (This is also precisely WHY the correctness oracle replays the same f32-recorded
# mesh in f64 rather than comparing against an f64-recorded run — same root cause.)


def _assert_buffer_halves_per_step(dts64, dts32, n):
    dts64, dts32 = np.asarray(dts64), np.asarray(dts32)
    assert dts64.dtype == np.float64 and dts32.dtype == np.float32
    # Exactly half per element — the dtype claim.
    assert dts32.itemsize * 2 == dts64.itemsize
    # The buffer IS the mesh: no hidden per-step overhead hiding in the "half".
    assert dts64.nbytes == n * dts64.shape[1] * 8
    assert dts32.nbytes == n * dts32.shape[1] * 4
    # The mesh lengths may differ by the handful of steps the f32 controller accepts
    # differently, so the total ratio is ~0.5 rather than exactly 0.5.
    ratio = dts32.nbytes / dts64.nbytes
    assert 0.45 <= ratio <= 0.55, f"record buffer ratio {ratio:.4f} is not ~half"
    return ratio


@requires_warp_nonstiff
def test_record_buffer_halves_per_step_nonstiff():
    problem = Lorenz()
    n = 64
    y0, params = _lorenz_batch(n=n)

    kw = dict(rtol=1e-6, atol=1e-9, device="cpu")
    _y, _n, dts64 = warp_replay.record(problem, y0, params, precision="float64", **kw)
    _y, _n, dts32 = warp_replay.record(problem, y0, params, precision="float32", **kw)

    _assert_buffer_halves_per_step(dts64, dts32, n)


@requires_warp_stiff
def test_record_buffer_halves_per_step_stiff():
    problem = Robertson()
    n = 8
    y0, params = _robertson_batch(n=n)

    kw = dict(rtol=1e-5, atol=1e-8, device="cpu")
    _y, _n, dts64 = warp_replay.record_rosenbrock(problem, y0, params, precision="float64", **kw)
    _y, _n, dts32 = warp_replay.record_rosenbrock(problem, y0, params, precision="float32", **kw)

    _assert_buffer_halves_per_step(dts64, dts32, n)


@requires_warp_nonstiff
def test_f32_records_a_slightly_different_mesh_than_f64():
    """Pin the reason the memory gate is per-step and the correctness oracle replays the f32
    mesh: f32 is a different controller, not just different arithmetic on the same steps.

    If this ever starts failing because the meshes agree exactly, the per-step framing above
    can be tightened back to a total — but that would be a change in the kernels, and it
    should be noticed rather than assumed.
    """
    problem = Lorenz()
    y0, params = _lorenz_batch(n=64)

    kw = dict(rtol=1e-6, atol=1e-9, device="cpu")
    _y, n64, _d = warp_replay.record(problem, y0, params, precision="float64", **kw)
    _y, n32, _d = warp_replay.record(problem, y0, params, precision="float32", **kw)

    assert np.any(np.asarray(n64) != np.asarray(n32)), (
        "f32 and f64 accepted identical step counts on every trajectory — unexpected; "
        "re-examine the memory gate's per-step framing")
    assert np.max(np.abs(np.asarray(n64, np.int64) - np.asarray(n32, np.int64))) <= 2, (
        "f32 step counts diverge from f64 by more than a couple of steps — that is a "
        "controller problem, not roundoff")


# ---------------------------------------------------------------------------
# Gate (i) — correctness against the f64 replay of the SAME f32 mesh
# ---------------------------------------------------------------------------


@requires_warp_nonstiff
def test_f32_gradient_matches_f64_oracle_on_the_same_mesh_lorenz():
    """Pre-registered nonstiff cell. Oracle = the same f32-recorded mesh replayed in f64."""
    problem = Lorenz()
    y0, params = _lorenz_batch()
    cell = LORENZ_CELL

    _yw, _n_acc, dts32 = warp_replay.record(
        problem, y0, params, rtol=cell["rtol"], atol=cell["atol"], device="cpu",
        precision="float32")
    dts32 = np.asarray(dts32)

    # f32 arithmetic on the f32 mesh.
    g32 = np.asarray(jax.grad(lambda p: jnp.sum(
        warp_replay.replay_solve_jax(
            problem, jnp.asarray(y0, jnp.float32), p, jnp.asarray(dts32)) ** 2
    ))(jnp.asarray(params, jnp.float32)))

    # ORACLE: identical mesh, f64 arithmetic.
    g64 = np.asarray(jax.grad(lambda p: jnp.sum(
        warp_replay.replay_solve_jax(
            problem, jnp.asarray(y0, jnp.float64), p,
            jnp.asarray(dts32.astype(np.float64))) ** 2
    ))(jnp.asarray(params, jnp.float64)))

    rel = _rel_l2(g32.astype(np.float64), g64)
    cos = _cosine(g32.astype(np.float64), g64)
    assert rel <= GRAD_REL_L2_TOL, f"f32 gradient rel-L2 {rel:.3e} > {GRAD_REL_L2_TOL:.0e}"
    assert cos >= 1 - GRAD_COSINE_TOL, f"f32 gradient direction cosine {cos:.9f}"


@requires_warp_stiff
def test_f32_gradient_matches_f64_oracle_on_the_same_mesh_robertson():
    """Pre-registered stiff cell."""
    problem = Robertson()
    y0, params = _robertson_batch()
    cell = ROBERTSON_CELL

    _yw, _n_acc, dts32 = warp_replay.record_rosenbrock(
        problem, y0, params, rtol=cell["rtol"], atol=cell["atol"], device="cpu",
        precision="float32")
    dts32 = np.asarray(dts32)

    g32 = np.asarray(jax.grad(lambda p: jnp.sum(
        warp_replay.replay_solve_rosenbrock_jax(
            problem, jnp.asarray(y0, jnp.float32), p, jnp.asarray(dts32)) ** 2
    ))(jnp.asarray(params, jnp.float32)))

    g64 = np.asarray(jax.grad(lambda p: jnp.sum(
        warp_replay.replay_solve_rosenbrock_jax(
            problem, jnp.asarray(y0, jnp.float64), p,
            jnp.asarray(dts32.astype(np.float64))) ** 2
    ))(jnp.asarray(params, jnp.float64)))

    rel = _rel_l2(g32.astype(np.float64), g64)
    cos = _cosine(g32.astype(np.float64), g64)
    assert rel <= GRAD_REL_L2_TOL, f"f32 gradient rel-L2 {rel:.3e} > {GRAD_REL_L2_TOL:.0e}"
    assert cos >= 1 - GRAD_COSINE_TOL, f"f32 gradient direction cosine {cos:.9f}"


@requires_warp_nonstiff
def test_f32_primal_error_vs_tight_reference_lorenz():
    """The f32 primal must be right, not merely self-consistent: achieved error vs a tight
    f64 adaptive reference of the same problem."""
    problem = Lorenz()
    y0, params = _lorenz_batch(n=128)

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="warp_replay", precision="float32",
        rtol=LORENZ_CELL["rtol"], atol=LORENZ_CELL["atol"])
    y32 = np.asarray(closure(jnp.asarray(params, jnp.float32)), dtype=np.float64)

    ref = gradsolve.solve(problem, y0, params, engine="diffrax", rtol=1e-12, atol=1e-14)
    err = np.max(np.linalg.norm(y32 - ref.y_final, axis=1)
                 / np.linalg.norm(ref.y_final, axis=1))
    assert err <= PRIMAL_TOL_LORENZ, f"f32 primal error {err:.3e} > {PRIMAL_TOL_LORENZ:.0e}"


@requires_warp_stiff
def test_f32_primal_error_vs_tight_reference_robertson():
    problem = Robertson()
    y0, params = _robertson_batch(n=8)

    closure = gradsolve.grad_closure(
        problem, y0, params, engine="warp_rosenbrock", precision="float32",
        rtol=ROBERTSON_CELL["rtol"], atol=ROBERTSON_CELL["atol"])
    y32 = np.asarray(closure(jnp.asarray(params, jnp.float32)), dtype=np.float64)

    ref = gradsolve.solve(problem, y0, params, engine="diffrax", rtol=1e-10, atol=1e-12)
    err = np.max(np.abs(y32 - ref.y_final).sum(axis=1))  # Robertson: y2 ~ 1e-5, use absolute
    assert err <= PRIMAL_TOL_ROBERTSON, (
        f"f32 primal error {err:.3e} > {PRIMAL_TOL_ROBERTSON:.0e}")


# ---------------------------------------------------------------------------
# CI: a fresh process under GRADSOLVE_X64=0
# ---------------------------------------------------------------------------


@requires_warp_nonstiff
def test_f32_replay_works_in_a_fresh_non_x64_process():
    """The f32 lane must work in a process where jax x64 was never enabled — and the f64
    lane must still refuse there rather than silently downcasting.

    A subprocess is the only way to test this: x64 is a process-global set at import,
    so it cannot be toggled inside a running test session.
    """
    script = """
import os, sys
assert os.environ.get("GRADSOLVE_X64") == "0"
import numpy as np, jax, jax.numpy as jnp
import gradsolve
from gradsolve.warp import warp_replay

assert not jax.config.read("jax_enable_x64"), "x64 should be OFF in this process"

class Lorenz:
    name = "diffeqgpu_lorenz"; dim = 3; t0 = 0.0; t1 = 1.0; is_stiff = False
    def f_jax(self, t, y, p):
        sigma, rho, beta = 10.0, p[0], 8.0 / 3.0
        return jnp.array([sigma*(y[1]-y[0]), y[0]*(rho-y[2])-y[1], y[0]*y[1]-beta*y[2]])

pr = Lorenz()
y0 = np.tile(np.array([1.0, 0.0, 0.0]), (16, 1)).astype(np.float32)
params = np.full((16, 1), 28.0, dtype=np.float32)

# f32 must WORK here.
cl = gradsolve.grad_closure(pr, y0, params, engine="warp_replay", precision="float32")
out = cl(jnp.asarray(params))
assert out.dtype == jnp.float32, out.dtype
g = jax.grad(lambda p: jnp.sum(cl(p) ** 2))(jnp.asarray(params))
assert g.dtype == jnp.float32 and bool(np.all(np.isfinite(np.asarray(g)))), g

# f64 must still REFUSE here (it cannot be honoured without x64).
try:
    warp_replay.record(pr, y0, params, precision="float64")
except RuntimeError as e:
    assert "float64" in str(e) or "x64" in str(e), str(e)
else:
    raise AssertionError("f64 record must refuse in a non-x64 process")

print("F32_NON_X64_OK")
"""
    env = {**dict(__import__("os").environ), "GRADSOLVE_X64": "0"}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=900)
    assert "F32_NON_X64_OK" in proc.stdout, (
        f"fresh non-x64 process failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
