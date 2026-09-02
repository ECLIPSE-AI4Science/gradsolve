"""The opt-in ``saveat_dense`` lane — Rodas5P's continuous extension in place of a re-step.

The main check in this file is interior accuracy: the continuous extension must meet the
record tolerance at interior points, where generic cubic Hermite interpolation between step
endpoints does not (it is far less accurate than the endpoints themselves on adaptive
meshes; see ``dense.py``). Cubic Hermite is kept as a comparison arm that must fail.

Scoring is relative and against the record tolerance, deliberately not against the endpoint
error. This extension is order 4 on an order-5 step, so the interior/endpoint error ratio
grows like 1/h on a correct implementation (it rises as rtol tightens from 1e-6 to 1e-10,
tracking the shrinking maximum step). A constant-factor-of-endpoint gate would therefore fail
a correct method at tight tolerance. The tolerance the caller asked for is the contract that
matters.

Also gated here: agreement with the re-step path on an identical mesh, the terminal-save
bitwise identity, the theta safe-denominator gradient (whose value is identical either way,
so it needs its own test), and the host-side theta-in-[0,1] guard.

Imports only gradsolve (+ numpy/jax/scipy/pytest).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gradsolve  # noqa: F401
from gradsolve.solvers.dense import (
    scan_saveat_dense_one,
    scan_saveat_one,
    validate_dense_thetas,
    vmap_saveat,
    vmap_saveat_dense,
)
from gradsolve.solvers.rodas5p_replay import (
    make_rodas5p_frozen_mesh_kernel,
    record_rodas5p,
)
from gradsolve.solvers.rodas5p_step import (
    rodas5p_advance,
    rodas5p_dense_eval,
    rodas5p_stages_and_dense,
)

# ---------------------------------------------------------------------------------------
# Two fields: an easy one where the adaptive controller takes big steps (which is what
# defeated Hermite) and a stiff nonlinear one.

_LIN_Y0 = np.array([[1.0]])
_LIN_P = np.array([[0.0]])
_LIN_SPAN = (0.0, 3.0)

_NL_Y0 = np.array([[0.4, 0.6]])
_NL_P = np.array([[500.0, 1.0]])
_NL_SPAN = (0.0, 0.05)


def _lin_jnp(t, y, p):
    del t, p
    return -y


def _lin_np(t, y):
    del t
    return -y


def _nl_jnp(t, y, p):
    del t
    return jnp.array([-p[0] * (y[0] - y[1] ** 2), y[0] - y[1] - p[1] * y[1] ** 3])


def _nl_np(t, y, p=(500.0, 1.0)):
    del t
    return np.array([-p[0] * (y[0] - y[1] ** 2), y[0] - y[1] - p[1] * y[1] ** 3])


_FIELDS = {
    "linear_decay": (_lin_jnp, _lin_np, _LIN_Y0, _LIN_P, _LIN_SPAN),
    "nl_stiff": (_nl_jnp, _nl_np, _NL_Y0, _NL_P, _NL_SPAN),
}


@functools.lru_cache(maxsize=None)
def _mesh(field, rtol):
    """Record the adaptive mesh once per (field, rtol) — several gates walk the same one."""
    f_jnp, _f_np, y0, p, (t0, t1) = _FIELDS[field]
    _yf, dts, n_acc, _rej = record_rodas5p(f_jnp, y0, p, t0, t1, rtol=rtol, atol=rtol * 1e-3)
    return dts[0][: int(n_acc[0])].copy()


def _ref(f_np, t0, y0, t1):
    """Tight DOP853 reference at t1, integrated directly (no dense output of its own)."""
    return solve_ivp(f_np, (t0, t1), y0, method="DOP853", rtol=1e-13, atol=1e-16).y[:, -1]


def _cubic_hermite(y_l, f_l, y_r, f_r, dt, th):
    """The generic interpolant — the comparison arm that must fail."""
    h00 = 2 * th ** 3 - 3 * th ** 2 + 1
    h10 = th ** 3 - 2 * th ** 2 + th
    h01 = -2 * th ** 3 + 3 * th ** 2
    h11 = th ** 3 - th ** 2
    return h00 * y_l + h10 * dt * f_l + h01 * y_r + h11 * dt * f_r


# ---------------------------------------------------------------------------------------
# Interior accuracy against the record tolerance, with cubic Hermite as the comparison arm.

@pytest.mark.parametrize("field", ["linear_decay", "nl_stiff"])
@pytest.mark.parametrize("rtol", [1e-6, 1e-8, 1e-10])
def test_dense_interior_error_meets_the_record_tolerance_where_hermite_does_not(field, rtol):
    """The dense output must meet the requested tolerance at interior points, where generic
    cubic Hermite interpolation does not.

    Scored relative and against the record tolerance, deliberately not against the endpoint
    error: this extension is order 4 on an order-5 step, so the interior/endpoint error ratio
    grows like 1/h on a correct implementation (it rises as rtol tightens from 1e-6 to 1e-10,
    tracking the shrinking maximum step). A constant-factor-of-endpoint gate would therefore
    fail a correct method at tight tolerance. The tolerance the caller asked for is the
    contract that matters.
    """
    f_jnp, f_np, y0, p, (t0, _t1) = _FIELDS[field]
    steps = _mesh(field, rtol)
    p_j = jnp.asarray(p[0])
    y = jnp.asarray(y0[0])
    t = t0
    worst_dense = worst_hermite = worst_endpoint = 0.0

    for dt in steps:
        dt = float(dt)
        y_next, coeffs = rodas5p_stages_and_dense(f_jnp, t, y, dt, p_j)
        y_l, y_r = np.asarray(y), np.asarray(y_next)
        f_l = np.asarray(f_jnp(t, y, p_j))
        f_r = np.asarray(f_jnp(t + dt, y_next, p_j))
        # Amplitude-invariant: normalise by this step's own |y| scale, which is what the
        # controller's own error weight uses -- NOT elementwise, which a near-zero component
        # would turn into pure roundoff amplification.
        scale = max(np.abs(y_l).max(), np.abs(y_r).max())

        err = np.abs(y_r - _ref(f_np, t, y_l, t + dt)).max() / scale
        worst_endpoint = max(worst_endpoint, float(err))

        for th in (0.2, 0.4, 0.5, 0.6, 0.8):
            ref = _ref(f_np, t, y_l, t + th * dt)
            y_d = np.asarray(rodas5p_dense_eval(coeffs, th))
            y_h = _cubic_hermite(y_l, f_l, y_r, f_r, dt, th)
            worst_dense = max(worst_dense, float(np.abs(y_d - ref).max() / scale))
            worst_hermite = max(worst_hermite, float(np.abs(y_h - ref).max() / scale))

        t, y = t + dt, y_next

    # Reported, NEVER asserted: this ratio grows like 1/h on a correct order-4 extension.
    context = (f"{field} rtol={rtol:.0e} steps={len(steps)}: dense {worst_dense:.3e} "
               f"({rtol / worst_dense:.1f}x inside rtol), hermite {worst_hermite:.3e} "
               f"({worst_hermite / rtol:.1f}x outside), ratio {worst_hermite / worst_dense:.1f}x; "
               f"interior/endpoint {worst_dense / worst_endpoint:.1f}x (reported, not gated)")
    assert worst_dense <= 1.0 * rtol, f"dense output MISSED the record tolerance -- {context}"
    assert worst_hermite > 5.0 * rtol, f"cubic Hermite unexpectedly PASSED -- {context}"
    assert worst_hermite / worst_dense > 50.0, f"margin over Hermite collapsed -- {context}"


# ---------------------------------------------------------------------------------------
# Agreement with the path it replaces, on the SAME recorded mesh.

def test_dense_saveat_agrees_with_the_restep_path_on_an_identical_mesh():
    """The trade-off, stated explicitly: the polynomial is less accurate than a genuine
    re-step (about 5e-10 against Radau here, versus the re-step's 2e-10), but it stays well
    inside the record tolerance and it removes the k extra solver steps. Observed
    max|poly - restep| = 4.1e-10 on this configuration; gate 1e-8."""
    rtol = 1e-8
    steps = jnp.asarray(_mesh("nl_stiff", rtol))
    ts = jnp.asarray(np.linspace(0.0, 0.05, 11)[1:])
    y0 = jnp.asarray(_NL_Y0[0])
    p_j = jnp.asarray(_NL_P[0])

    _yf_p, ys_poly = scan_saveat_dense_one(
        _nl_jnp, p_j, 0.0, y0, steps, ts, rodas5p_stages_and_dense, rodas5p_dense_eval)
    _yf_r, ys_restep = scan_saveat_one(_nl_jnp, p_j, 0.0, y0, steps, ts, rodas5p_advance)

    ys_poly, ys_restep = np.asarray(ys_poly), np.asarray(ys_restep)
    agree = float(np.abs(ys_poly - ys_restep).max())
    assert agree <= 1e-8, f"poly vs re-step disagree by {agree:.3e} (expected ~4e-10)"

    ref = np.stack([_ref(_nl_np, 0.0, np.asarray(y0), float(t)) for t in ts])
    scale = float(np.abs(ref).max())
    err_poly = float(np.abs(ys_poly - ref).max()) / scale
    err_restep = float(np.abs(ys_restep - ref).max()) / scale
    assert err_poly <= 1.0 * rtol, (
        f"poly {err_poly:.3e} missed the record tolerance {rtol:.0e} "
        f"(re-step {err_restep:.3e}, ratio {err_poly / err_restep:.1f}x)")


def test_dense_saveat_terminal_save_is_bitwise_y_final():
    """Measured: with a naive theta, accumulated-mesh drift puts the terminal save at
    theta = 1 + O(ulp) and ys[-1] == y_final is false. The dense lane must carry dense.py's
    existing ``ts >= t_fin -> y_fin`` post-rule unchanged."""
    steps = jnp.asarray(_mesh("nl_stiff", 1e-8))
    ts = jnp.asarray(np.linspace(0.0, 0.05, 11)[1:])
    y0 = jnp.asarray(_NL_Y0[0])
    y_fin, ys = scan_saveat_dense_one(
        _nl_jnp, jnp.asarray(_NL_P[0]), 0.0, y0, steps, ts,
        rodas5p_stages_and_dense, rodas5p_dense_eval)
    np.testing.assert_array_equal(np.asarray(ys[-1]), np.asarray(y_fin))


def test_dense_saveat_on_exact_mesh_nodes_reproduces_the_node_states():
    """A save landing on a node reproduces the step output, so the dense lane cannot introduce
    a discontinuity at node boundaries -- the property behind the 'bracket to the step
    ending at ts' rule.

    Not bitwise, which is why the gate below is assert_allclose(atol=1e-14) and not
    array_equal: `b_i(1) == m_i` is bitwise, but the lane reaches theta by division,
    `(ts - t)/dt`, which rounds to 1 +/- 1-2 ulp at ~2/3 of interior nodes (measured max
    discrepancy 1.11e-16)."""
    steps_np = _mesh("nl_stiff", 1e-8)
    nodes = np.cumsum(steps_np)
    ts = jnp.asarray(nodes[2:6])
    y0 = jnp.asarray(_NL_Y0[0])
    p_j = jnp.asarray(_NL_P[0])
    _yf, ys = scan_saveat_dense_one(
        _nl_jnp, p_j, 0.0, y0, jnp.asarray(steps_np), ts,
        rodas5p_stages_and_dense, rodas5p_dense_eval)

    y, t = y0, 0.0
    walked = {}
    for i, dt in enumerate(steps_np):
        y = rodas5p_advance(_nl_jnp, t, y, float(dt), p_j)
        t = t + float(dt)
        walked[i] = np.asarray(y)
    for j in range(4):
        np.testing.assert_allclose(np.asarray(ys[j]), walked[2 + j], rtol=0, atol=1e-14)


def test_vmap_saveat_dense_matches_the_per_trajectory_scan():
    """The ensemble form mirrors vmap_saveat's in_axes handling; per-trajectory (n, S) meshes
    and a single shared (S,) grid must both work and agree with the one-trajectory scan."""
    steps = _mesh("nl_stiff", 1e-8)
    ts = jnp.asarray(np.linspace(0.0, 0.05, 6)[1:])
    y0 = jnp.asarray(np.array([[0.4, 0.6], [0.5, 0.5]]))
    params = jnp.asarray(np.array([[500.0, 1.0], [500.0, 1.0]]))

    shared = jnp.asarray(steps)
    yf_s, ys_s = vmap_saveat_dense(_nl_jnp, 0.0, y0, params, shared, ts,
                                   rodas5p_stages_and_dense, rodas5p_dense_eval)
    per_traj = jnp.asarray(np.stack([steps, steps]))
    yf_p, ys_p = vmap_saveat_dense(_nl_jnp, 0.0, y0, params, per_traj, ts,
                                   rodas5p_stages_and_dense, rodas5p_dense_eval)
    np.testing.assert_array_equal(np.asarray(ys_s), np.asarray(ys_p))
    np.testing.assert_array_equal(np.asarray(yf_s), np.asarray(yf_p))

    for i in range(2):
        yf_1, ys_1 = scan_saveat_dense_one(
            _nl_jnp, params[i], 0.0, y0[i], shared, ts,
            rodas5p_stages_and_dense, rodas5p_dense_eval)
        # The vmapped and the single-trajectory scan are the same arithmetic, but XLA may fuse
        # them differently (by 1-2 ulp, ~5e-16 relative);
        # a few-ulp tolerance keeps the check meaningful without pinning the fusion order.
        np.testing.assert_allclose(np.asarray(ys_1), np.asarray(ys_s[i]), rtol=1e-14, atol=0)
        np.testing.assert_allclose(np.asarray(yf_1), np.asarray(yf_s[i]), rtol=1e-14, atol=0)

    with pytest.raises(ValueError, match="dts must be"):
        vmap_saveat_dense(_nl_jnp, 0.0, y0, params, jnp.zeros((2, 2, 2)), ts,
                          rodas5p_stages_and_dense, rodas5p_dense_eval)


def test_restep_saveat_lane_is_untouched_by_the_dense_lane():
    """The dense lane is new code, and the existing re-step pair must stay
    byte-identical. Pinned here as well as by test_saveat.py, since both lanes now live in
    the same module."""
    steps = jnp.asarray(_mesh("nl_stiff", 1e-8))
    ts = jnp.asarray(np.linspace(0.0, 0.05, 6)[1:])
    y0 = jnp.asarray(np.array([[0.4, 0.6]]))
    params = jnp.asarray(np.array([[500.0, 1.0]]))
    yf_a, ys_a = vmap_saveat(_nl_jnp, 0.0, y0, params, steps, ts, rodas5p_advance)
    yf_b, ys_b = vmap_saveat(_nl_jnp, 0.0, y0, params, steps, ts, rodas5p_advance)
    np.testing.assert_array_equal(np.asarray(ys_a), np.asarray(ys_b))
    np.testing.assert_array_equal(np.asarray(yf_a), np.asarray(yf_b))


# ---------------------------------------------------------------------------------------
# The theta gradient trap: identical VALUE either way, so it needs its own test.

def test_theta_safe_denominator_keeps_the_padding_gradient_finite():
    """Measured: jnp.where(dt > 0, (ts-tl)/dt, 0.0) has value 0.0 and derivative NaN at
    dt == 0; the double-where form has value 0.0 and derivative 0.0. Replay meshes are
    zero-padded, so the naive form NaNs every gradient through a padded trajectory while
    looking correct in forward mode."""
    tl, dt = 0.0, 0.0

    def naive(ts):
        return jnp.where(dt > 0, (ts - tl) / dt, 0.0)

    def safe(ts):
        dt_s = jnp.where(dt > 0, dt, 1.0)
        return jnp.where(dt > 0, (ts - tl) / dt_s, 0.0)

    ts = jnp.asarray(0.3)
    assert float(naive(ts)) == 0.0 and float(safe(ts)) == 0.0        # identical in value
    assert np.isnan(float(jax.grad(naive)(ts))), "the naive form no longer NaNs -- re-derive"
    assert float(jax.grad(safe)(ts)) == 0.0

    # ...and the property that matters: a PADDED mesh must not NaN the gradient of the lane.
    steps_np = _mesh("nl_stiff", 1e-8)
    padded = jnp.asarray(np.concatenate([steps_np, np.zeros(5)]))
    ts_v = jnp.asarray(np.linspace(0.0, 0.05, 6)[1:])
    p_j = jnp.asarray(_NL_P[0])

    def loss(y0_j):
        _yf, ys = scan_saveat_dense_one(
            _nl_jnp, p_j, 0.0, y0_j, padded, ts_v,
            rodas5p_stages_and_dense, rodas5p_dense_eval)
        return jnp.sum(ys ** 2)

    g = np.asarray(jax.grad(loss)(jnp.asarray(_NL_Y0[0])))
    assert np.all(np.isfinite(g)), f"padded-mesh gradient is not finite: {g}"
    assert np.any(g != 0.0), f"padded-mesh gradient collapsed to zero: {g}"


def test_dense_saveat_padding_does_not_change_the_values():
    """Zero-padding a mesh is an exact identity in value too -- the dt == 0 steps claim no
    output time and every k_i is exactly zero."""
    steps_np = _mesh("nl_stiff", 1e-8)
    ts = jnp.asarray(np.linspace(0.0, 0.05, 6)[1:])
    y0 = jnp.asarray(_NL_Y0[0])
    p_j = jnp.asarray(_NL_P[0])
    args = (rodas5p_stages_and_dense, rodas5p_dense_eval)
    _a, ys_bare = scan_saveat_dense_one(_nl_jnp, p_j, 0.0, y0, jnp.asarray(steps_np), ts, *args)
    _b, ys_pad = scan_saveat_dense_one(
        _nl_jnp, p_j, 0.0, y0, jnp.asarray(np.concatenate([steps_np, np.zeros(7)])), ts, *args)
    np.testing.assert_array_equal(np.asarray(ys_bare), np.asarray(ys_pad))


# ---------------------------------------------------------------------------------------
# Host-side theta guard.

def test_validate_dense_thetas_accepts_a_real_mesh_including_the_terminal_save():
    """Terminal saves are exempt: fp drift in the accumulated mesh legitimately puts the last
    one at theta = 1 + O(ulp), and it takes the ts >= t_fin substitution before any theta is
    used. Asserted together with the drift actually being there, so the exemption is not
    silently testing nothing."""
    steps = _mesh("nl_stiff", 1e-8)
    ts = np.linspace(0.0, 0.05, 11)[1:]
    validate_dense_thetas(steps, ts, 0.0)                      # must not raise

    nodes = np.cumsum(steps)
    assert ts[-1] >= nodes[-1] or (ts[-1] - nodes[-2]) / steps[-1] <= 1.0, (
        "terminal save is neither past t_fin nor inside the last step -- re-derive the "
        "exemption; drift = " + repr(ts[-1] - nodes[-1]))


def test_validate_dense_thetas_rejects_a_double_bracketing_mesh():
    """Under the scan's own rule (dt > 0 & ts > t_l & ts <= t_r, with t_r = t_l + dt) theta is
    in (0, 1] by construction -- the bracket condition is the range check -- so the reachable
    failure is a mesh that brackets one time twice, which only a non-monotonic mesh can do.
    Last-write-wins would then silently keep one of the two. Here nodes run 0 -> 1 -> 0.5 ->
    1.5, and ts=0.8 is bracketed by steps 0 (theta 0.8) and 2 (theta 0.3)."""
    with pytest.raises(ValueError, match="not monotonic"):
        validate_dense_thetas(np.array([1.0, -0.5, 1.0]), np.array([0.8]), 0.0)


def test_validate_dense_thetas_rejects_an_out_of_range_supplied_bracket():
    """The case the range check exists for. A host-precomputed bracket map is not
    self-limiting the way the scan's mask is, so an off-by-one silently extrapolates the
    polynomial. Same mesh, same times, one bracket index moved by one step."""
    steps = _mesh("nl_stiff", 1e-8)
    nodes = np.cumsum(steps)                     # nodes[i] is the END of step i
    ts = np.array([nodes[3] + 0.3 * steps[4], nodes[5] + 0.5 * steps[6]])
    good = np.array([4, 6])                      # the steps that really bracket those times
    validate_dense_thetas(steps, ts, 0.0, bracket=good)        # must not raise
    with pytest.raises(ValueError, match="EXTRAPOLATES"):
        validate_dense_thetas(steps, ts, 0.0, bracket=good + 1)
    with pytest.raises(ValueError, match="EXTRAPOLATES"):
        validate_dense_thetas(steps, ts, 0.0, bracket=good - 1)


def test_validate_dense_thetas_rejects_a_supplied_bracket_on_a_padding_step():
    """A bracket index pointing at a dt == 0 padding step would divide by zero."""
    steps = np.concatenate([_mesh("nl_stiff", 1e-8), np.zeros(3)])
    nodes = np.cumsum(steps)
    ts = np.array([nodes[3] + 0.3 * steps[4]])
    with pytest.raises(ValueError, match="cannot bracket"):
        validate_dense_thetas(steps, ts, 0.0, bracket=np.array([len(steps) - 1]))
    with pytest.raises(ValueError, match="out of range"):
        validate_dense_thetas(steps, ts, 0.0, bracket=np.array([len(steps)]))


def test_validate_dense_thetas_accepts_the_ulp_slack_but_not_a_real_excursion():
    """The allowance is explicit and small: a theta one ulp past 1.0 passes, a theta a
    percent past it does not."""
    steps = np.array([1.0, 1.0])
    a_few_ulps_over = 1.0 + 8 * np.spacing(1.0)
    validate_dense_thetas(steps, np.array([a_few_ulps_over]), 0.0, bracket=np.array([0]))
    with pytest.raises(ValueError, match="EXTRAPOLATES"):
        validate_dense_thetas(steps, np.array([1.01]), 0.0, bracket=np.array([0]))


def test_validate_dense_thetas_is_a_noop_on_traced_input():
    """A jit'd mesh has no host value to check; the guard must not raise a tracer error."""
    ts = np.linspace(0.01, 0.04, 4)

    @jax.jit
    def go(dts):
        validate_dense_thetas(dts, ts, 0.0)
        return jnp.sum(dts)

    assert np.isfinite(float(go(jnp.asarray(_mesh("nl_stiff", 1e-8)))))


def test_validate_dense_thetas_exempts_times_at_or_before_t0():
    """Times at or before t0 resolve to y0 and are never interpolated."""
    validate_dense_thetas(np.array([1.0, 1.0]), np.array([-3.0, 0.0]), 0.0)


# =========================================================================================
# Wired through the frozen-mesh kernel, with the moving-time gates.

class _NLStiff:
    """The stiff Robertson field: dy/dt = [-p0*(y0 - y1**2), y0 - y1 - p1*y1**3]. The Jacobian depends on y and
    p, which is what exercises the dJ/dy path in reverse-through-jacfwd."""
    name = "rodas5p_nl_stiff"
    dim = 2
    t0 = 0.0
    t1 = 0.05
    is_stiff = True

    def f_jax(self, t, y, p):
        return _nl_jnp(t, y, p)


class _NonStiffDecay:
    """dy/dt = -p0*y with is_stiff=False, used only by the saveat_dense engine-rejection tests.
    They name engine="vern7_replay", which supports() only nonstiff problems: on a stiff problem
    the request would be rerouted to the general stiff lane (rodas5p_replay), which has the
    continuous extension, so the rejection under test would never fire."""
    name = "rodas5p_dense_nonstiff_decay"
    dim = 1
    t0 = 0.0
    t1 = 0.05
    is_stiff = False

    def f_jax(self, t, y, p):
        return -p[0] * y


_NS_Y0 = np.array([[1.0]])
_NS_P0 = np.array([[1.3]])

_KERN_Y0 = np.array([[0.4, 0.6]])
_KERN_P0 = np.array([[300.0, 1.2]])
#: Deliberately asymmetric, so no component of the gradient can cancel by symmetry.
_W = np.array([[[1.0, -0.4], [0.3, 0.9], [-0.7, 0.2], [0.5, -1.1], [0.8, 0.6]]])


def _dense_kernel(ts, **kw):
    return make_rodas5p_frozen_mesh_kernel(
        _NLStiff(), _KERN_Y0, _KERN_P0, rtol=1e-8, atol=1e-11, saveat=ts, **kw)


@functools.lru_cache(maxsize=None)
def _kern_mesh():
    """The mesh the kernel factory records for (_KERN_Y0, _KERN_P0) — not _mesh('nl_stiff'),
    which records at different parameters and so produces a different adaptive mesh."""
    _yf, dts, n_acc, _rej = record_rodas5p(
        _nl_jnp, _KERN_Y0, _KERN_P0, 0.0, 0.05, rtol=1e-8, atol=1e-11)
    return dts[0][: int(n_acc[0])].copy()


def _fd_grad(loss, x0, eps=1e-5):
    """Central FD, mirroring test_rodas5p_replay.py's helper."""
    g = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp = x0.copy()
        xp[idx] += eps
        xm = x0.copy()
        xm[idx] -= eps
        g[idx] = (float(loss(jnp.asarray(xp))) - float(loss(jnp.asarray(xm)))) / (2 * eps)
    return g


def test_dense_kernel_default_off_is_byte_identical_to_the_restep_kernel():
    """Dense defaults to False, and with it off the kernel must produce
    bitwise what it produced before the lane existed -- which is the re-step path, reached
    here by calling it directly on the same frozen mesh."""
    ts = np.linspace(0.0, 0.05, 6)[1:]
    yf_off, ys_off = _dense_kernel(ts)(jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0))
    yf_exp, ys_exp = _dense_kernel(ts, dense=False)(jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0))
    np.testing.assert_array_equal(np.asarray(ys_off), np.asarray(ys_exp))
    np.testing.assert_array_equal(np.asarray(yf_off), np.asarray(yf_exp))

    # ...and BITWISE the earlier code path itself, not merely self-consistent. That path is
    # vmap_saveat on the frozen mesh recorded for THIS kernel's (y0, params0), which is
    # verbatim what the builder used to return.
    steps = _kern_mesh()
    yf_pre, ys_pre = vmap_saveat(_nl_jnp, 0.0, jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0),
                                 jnp.asarray(steps[None, :]), jnp.asarray(ts), rodas5p_advance)
    np.testing.assert_array_equal(np.asarray(ys_off), np.asarray(ys_pre))
    np.testing.assert_array_equal(np.asarray(yf_off), np.asarray(yf_pre))

    # ...while dense=True really does reach the lane: same answer to 1e-8, NOT bitwise.
    yf_on, ys_on = _dense_kernel(ts, dense=True)(jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0))
    assert not np.array_equal(np.asarray(ys_on), np.asarray(ys_off)), (
        "dense=True produced BITWISE the re-step answer -- the flag is not reaching the lane")
    np.testing.assert_allclose(np.asarray(ys_on), np.asarray(ys_off), rtol=0, atol=1e-8)
    np.testing.assert_array_equal(np.asarray(yf_on), np.asarray(yf_off))   # y_final unaffected


def test_dense_kernel_rejects_dense_without_saveat():
    with pytest.raises(ValueError, match="dense=True is a saveat option"):
        make_rodas5p_frozen_mesh_kernel(_NLStiff(), _KERN_Y0, _KERN_P0, dense=True)


def _ts_loss(dense):
    """Scalar loss over the dense saveat output as a function of the traced save times.

    The kernel factories close over ``saveat`` (unchanged, by design), so the moving-time
    gates run against the lane itself -- which is the object whose theta the save time has to
    stay live in, and the shape the engine mirrors.
    """
    steps = jnp.asarray(_kern_mesh()[None, :])
    y0, p = jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0)
    w = jnp.asarray(_W)
    fn = (vmap_saveat_dense if dense else vmap_saveat)
    args = ((rodas5p_stages_and_dense, rodas5p_dense_eval) if dense else (rodas5p_advance,))

    def loss(ts_v):
        _yf, ys = fn(_nl_jnp, 0.0, y0, p, steps, ts_v, *args)
        return jnp.sum(w * ys)
    return loss


@pytest.mark.parametrize("j", [0, 2])
def test_moving_interior_save_fd(j):
    """d(loss)/d(ts[j]) for an interior save. AD-vs-FD at two relative epsilons, with the two
    FD estimates required to agree with each other first -- an unconverged FD reference
    certifies nothing.

    The save time is live in theta = (ts - t_left)/dt, and the extension is continuous across
    node boundaries (b_i(1) == m_i bitwise), so the derivative is well defined even where a
    perturbation moves ts into the neighbouring step."""
    loss = _ts_loss(dense=True)
    ts0 = np.linspace(0.0, 0.05, 6)[1:]
    ad = float(np.asarray(jax.grad(loss)(jnp.asarray(ts0)))[j])

    fds = []
    for rel in (1e-5, 1e-4):
        eps = rel * 0.05
        tp = ts0.copy()
        tp[j] += eps
        tm = ts0.copy()
        tm[j] -= eps
        fds.append((float(loss(jnp.asarray(tp))) - float(loss(jnp.asarray(tm)))) / (2 * eps))
    scale = max(abs(fds[0]), abs(fds[1]), 1e-30)
    assert abs(fds[0] - fds[1]) <= 1e-2 * scale, (
        f"the two FD estimates disagree ({fds[0]:.6e} vs {fds[1]:.6e}) -- unconverged "
        "reference, the AD comparison below would certify nothing")
    for fd in fds:
        assert np.allclose(ad, fd, rtol=1e-2, atol=1e-2 * abs(fd)), (
            f"d(loss)/d(ts[{j}]): AD {ad:.6e} vs FD {fd:.6e}")
    assert abs(ad) > 1e-6, f"gradient is ~0 ({ad:.3e}); the gate would pass vacuously"


def test_moving_terminal_save_fd():
    """The terminal-save gradient can silently collapse (-5.6e-4 against a true 5.936 with
    the value-pin on), exercised through the dense path specifically.

    The terminal save is placed strictly inside the mesh on purpose. A save at or past the
    final accumulated node is value-pinned to y_fin by the terminal-save post-rule, so its derivative is
    legitimately zero -- that boundary is pinned by its own test below, not smuggled in here
    where it would make this gate vacuous."""
    loss = _ts_loss(dense=True)
    t_fin = float(np.sum(_kern_mesh()))
    ts0 = np.linspace(0.0, 0.05, 6)[1:]
    ts0[-1] = t_fin - 1e-4
    assert ts0[-1] > ts0[-2], "terminal save must stay sorted"

    ad = float(np.asarray(jax.grad(loss)(jnp.asarray(ts0)))[-1])
    fds = []
    for rel in (1e-5, 1e-4):
        eps = rel * 0.05
        tp = ts0.copy()
        tp[-1] += eps
        tm = ts0.copy()
        tm[-1] -= eps
        fds.append((float(loss(jnp.asarray(tp))) - float(loss(jnp.asarray(tm)))) / (2 * eps))
    scale = max(abs(fds[0]), abs(fds[1]), 1e-30)
    assert abs(fds[0] - fds[1]) <= 1e-2 * scale, (
        f"the two FD estimates disagree ({fds[0]:.6e} vs {fds[1]:.6e}) -- unconverged reference")
    for fd in fds:
        assert np.allclose(ad, fd, rtol=1e-2, atol=1e-2 * abs(fd)), (
            f"d(loss)/d(ts[-1]): AD {ad:.6e} vs FD {fd:.6e} -- the terminal-save gradient "
            "collapsed, which is exactly the terminal-save failure this test guards against")
    assert abs(ad) > 1e-6, f"terminal gradient is ~0 ({ad:.3e}); gate would pass vacuously"


def test_save_at_or_past_t_fin_is_value_pinned_with_zero_derivative_by_design():
    """The boundary the gate above deliberately avoids, pinned rather than left implicit.
    The terminal-save post-rule substitutes y_fin for ts >= t_fin, which is what makes ys[-1] == y_final
    bitwise -- and the price is that such a save has no derivative w.r.t. its own time. A
    caller differentiating w.r.t. a terminal save time must keep it strictly inside the mesh."""
    loss = _ts_loss(dense=True)
    t_fin = float(np.sum(_kern_mesh()))
    ts0 = np.linspace(0.0, 0.05, 6)[1:]
    ts0[-1] = t_fin + 1e-6
    assert float(np.asarray(jax.grad(loss)(jnp.asarray(ts0)))[-1]) == 0.0


def test_dense_vs_restep_moving_save_gradients_agree():
    """A value-only comparison would miss a theta-tangent error entirely, so the two saveat
    paths are compared in gradient w.r.t. the save times as well."""
    ts0 = np.linspace(0.0, 0.05, 6)[1:]
    g_dense = np.asarray(jax.grad(_ts_loss(dense=True))(jnp.asarray(ts0)))
    g_restep = np.asarray(jax.grad(_ts_loss(dense=False))(jnp.asarray(ts0)))
    scale = float(np.abs(g_restep).max())
    np.testing.assert_allclose(g_dense, g_restep, rtol=2e-3, atol=2e-3 * scale)
    assert scale > 1e-6, "re-step reference gradient is ~0; the comparison would be vacuous"


def test_dense_kernel_reverse_matches_fd_params_y0_joint():
    """Reverse through the polynomial: jax.grad of a scalar loss over the dense saveat output,
    against the frozen-mesh FD gate, wrt params, y0 and jointly, on the stiff Robertson field
    whose Jacobian depends on both y and p. Same tolerances as the stiff Robertson frozen-mesh gate."""
    ts = np.linspace(0.0, 0.05, 6)[1:]
    kern = _dense_kernel(ts, dense=True)
    w = jnp.asarray(_W)

    def loss_p(p):
        _yf, ys = kern(jnp.asarray(_KERN_Y0), p)
        return jnp.sum(w * ys ** 2)

    def loss_z(z):
        _yf, ys = kern(z, jnp.asarray(_KERN_P0))
        return jnp.sum(w * ys ** 2)

    gp = np.asarray(jax.grad(loss_p)(jnp.asarray(_KERN_P0)))
    np.testing.assert_allclose(gp, _fd_grad(loss_p, _KERN_P0), rtol=2e-4, atol=1e-6)
    gz = np.asarray(jax.grad(loss_z)(jnp.asarray(_KERN_Y0)))
    np.testing.assert_allclose(gz, _fd_grad(loss_z, _KERN_Y0), rtol=2e-4, atol=1e-6)

    gzj, gpj = jax.grad(lambda z, p: jnp.sum(w * kern(z, p)[1] ** 2), argnums=(0, 1))(
        jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0))
    np.testing.assert_allclose(np.asarray(gpj), gp, rtol=2e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gzj), gz, rtol=2e-4, atol=1e-6)


def test_dense_vs_restep_parameter_gradients_agree():
    """dense-vs-re-step agreement in gradient w.r.t. params as well as w.r.t. the save times."""
    ts = np.linspace(0.0, 0.05, 6)[1:]
    w = jnp.asarray(_W)

    def g_for(dense):
        kern = _dense_kernel(ts, dense=dense)
        return np.asarray(jax.grad(
            lambda p: jnp.sum(w * kern(jnp.asarray(_KERN_Y0), p)[1] ** 2))(jnp.asarray(_KERN_P0)))

    gd, gr = g_for(True), g_for(False)
    np.testing.assert_allclose(gd, gr, rtol=2e-4, atol=2e-4 * float(np.abs(gr).max()))


# =========================================================================================
# The public API. choose_engine/DECISION_MAP are not touched (pinned by
# tests/test_dispatch.py); saveat_dense is an explicit per-lane opt-in, never a routing input.

def test_solve_saveat_dense_matches_the_restep_forward_solve():
    """solve(..., saveat_dense=True) reaches the lane on rodas5p_replay and agrees with the
    re-step forward solve well inside the record tolerance."""
    ts = np.linspace(0.0, 0.05, 6)[1:]
    kw = dict(engine="rodas5p_replay", rtol=1e-8, atol=1e-11, saveat=ts)
    res_off = gradsolve.solve(_NLStiff(), _KERN_Y0, _KERN_P0, **kw)
    res_on = gradsolve.solve(_NLStiff(), _KERN_Y0, _KERN_P0, saveat_dense=True, **kw)
    assert res_on.solver == "rodas5p_replay"
    assert res_on.y_saved.shape == res_off.y_saved.shape
    np.testing.assert_allclose(res_on.y_saved, res_off.y_saved, rtol=0, atol=1e-8)
    assert not np.array_equal(res_on.y_saved, res_off.y_saved), "saveat_dense did not reach the lane"
    np.testing.assert_array_equal(res_on.y_final, res_off.y_final)


def test_solve_saveat_dense_rejects_engines_without_a_continuous_extension():
    ts = np.linspace(0.0, 0.05, 6)[1:]
    # vern7_replay is dense-CAPABLE (it has a saveat lane) but has no published continuous
    # extension, which is exactly the case the rejection exists for.
    with pytest.raises(ValueError, match="only on the 'rodas5p_replay' lane"):
        gradsolve.solve(_NonStiffDecay(), _NS_Y0, _NS_P0, engine="vern7_replay",
                      saveat=ts, saveat_dense=True)


def test_solve_saveat_dense_requires_saveat():
    with pytest.raises(ValueError, match="requires saveat"):
        gradsolve.solve(_NLStiff(), _KERN_Y0, _KERN_P0, engine="rodas5p_replay", saveat_dense=True)


def test_dense_factory_passes_saveat_through_unconverted(monkeypatch):
    """The dense branch must hand `saveat` to validate_dense_thetas untouched.

    It used to np.asarray it first, which pre-empted that guard's documented tracer no-op and
    made this the only saveat factory in the library to reject a device/traced save vector.
    (The narrowing cannot be shown by building under jit: the recorder is a host loop, so no
    factory here can be traced at all.)
    """
    import gradsolve.solvers.dense as dense_mod
    seen = {}
    orig = dense_mod.validate_dense_thetas

    def spy(dts, ts, t0, *a, **kw):
        seen["ts"] = ts
        return orig(dts, ts, t0, *a, **kw)

    monkeypatch.setattr(dense_mod, "validate_dense_thetas", spy)
    ts = jnp.asarray(np.linspace(0.0, 0.05, 6)[1:])
    k = _dense_kernel(ts, dense=True)
    assert seen["ts"] is ts, "saveat was converted before the tracer-safe guard could see it"
    _yf, ys = k(jnp.asarray(_KERN_Y0), jnp.asarray(_KERN_P0))
    assert np.all(np.isfinite(np.asarray(ys)))


def test_grad_closure_saveat_dense_requires_saveat():
    """grad_closure must mirror solve()'s rule. Without this guard the params-only fast path
    returns a plain final-state closure and the flag is silently ignored."""
    with pytest.raises(ValueError, match="requires saveat"):
        gradsolve.grad_closure(_NLStiff(), _KERN_Y0, _KERN_P0, engine="rodas5p_replay",
                             saveat_dense=True)


def test_grad_closure_saveat_dense_rejects_other_engines_without_saveat():
    """The engine rejection must not be bypassed by the no-saveat shortcut either."""
    with pytest.raises(ValueError):
        gradsolve.grad_closure(_NLStiff(), _KERN_Y0, _KERN_P0, engine="vern7_replay",
                             saveat_dense=True)


def test_grad_closure_saveat_dense_is_differentiable_and_matches_restep():
    """The closure returns the (n, k, dim) series and differentiates through the polynomial."""
    ts = np.linspace(0.0, 0.05, 6)[1:]
    kw = dict(engine="rodas5p_replay", wrt="params", rtol=1e-8, atol=1e-11, saveat=ts)
    c_on = gradsolve.grad_closure(_NLStiff(), _KERN_Y0, _KERN_P0, saveat_dense=True, **kw)
    c_off = gradsolve.grad_closure(_NLStiff(), _KERN_Y0, _KERN_P0, **kw)
    assert c_on.route.actual == "rodas5p_replay"
    assert np.asarray(c_on(jnp.asarray(_KERN_P0))).shape == (1, 5, 2)

    def g(c):
        return np.asarray(jax.grad(lambda p: jnp.sum(c(p) ** 2))(jnp.asarray(_KERN_P0)))

    g_on, g_off = g(c_on), g(c_off)
    assert np.all(np.isfinite(g_on)) and np.abs(g_on).max() > 1e-6
    np.testing.assert_allclose(g_on, g_off, rtol=2e-4, atol=2e-4 * np.abs(g_off).max())


def test_grad_closure_saveat_dense_rejects_engines_without_a_continuous_extension():
    ts = np.linspace(0.0, 0.05, 6)[1:]
    with pytest.raises(ValueError, match="only on the 'rodas5p_replay' lane"):
        gradsolve.grad_closure(_NonStiffDecay(), _NS_Y0, _NS_P0, engine="vern7_replay",
                             wrt="params", saveat=ts, saveat_dense=True)
