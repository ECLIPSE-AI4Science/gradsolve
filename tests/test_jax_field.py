"""Tests for gradsolve.warp.jax_field — JAX RHS -> fused Warp @wp.func field.

Whole module is gated on Warp: the translator compiles to a @wp.func, so every test
needs Warp (matches how the other Warp-touching test modules skip warp-less CI).
"""
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

import gradsolve  # noqa: F401  (enables float64 on import)

wp = pytest.importorskip("warp")

from gradsolve.warp._warp_kernel import make_lorenz_field, make_vdp_field  # noqa: E402
from gradsolve.warp.jax_field import (  # noqa: E402
    UnsupportedRHS,
    jaxpr_to_warp_field,
)


def _eval_field_scalar_param(field, vecD, D, y, p_scalar):
    """Evaluate a hand-written field(t, y: vecD, p: scalar) on a batch via a probe kernel."""
    n = y.shape[0]

    @wp.kernel
    def probe(t: wp.array(dtype=wp.float64),
              yb: wp.array(dtype=vecD),
              pb: wp.array(dtype=wp.float64),
              out: wp.array(dtype=vecD)):
        j = wp.tid()
        out[j] = field(t[j], yb[j], pb[j])

    tb = wp.zeros(n, dtype=wp.float64, device="cpu")
    yb = wp.array(y, dtype=vecD, device="cpu")
    pb = wp.array(p_scalar, dtype=wp.float64, device="cpu")
    out = wp.zeros(n, dtype=vecD, device="cpu")
    wp.launch(probe, dim=n, inputs=[tb, yb, pb, out], device="cpu")
    return out.numpy()


def _eval_field_vec_param(field, vecD, vecP, t, y, p):
    """Evaluate a generated field(t, y: vecD, p: vecP) on a batch via a probe kernel."""
    n = y.shape[0]

    @wp.kernel
    def probe(tb: wp.array(dtype=wp.float64),
              yb: wp.array(dtype=vecD),
              pb: wp.array(dtype=vecP),
              out: wp.array(dtype=vecD)):
        j = wp.tid()
        out[j] = field(tb[j], yb[j], pb[j])

    tb = wp.array(t, dtype=wp.float64, device="cpu")
    yb = wp.array(y, dtype=vecD, device="cpu")
    pb = wp.array(p, dtype=vecP, device="cpu")
    out = wp.zeros(n, dtype=vecD, device="cpu")
    wp.launch(probe, dim=n, inputs=[tb, yb, pb, out], device="cpu")
    return out.numpy()


def _lorenz_jax(t, y, p):
    sigma = 10.0
    beta = 8.0 / 3.0
    rho = p[0]
    return jnp.array([
        sigma * (y[1] - y[0]),
        rho * y[0] - y[1] - y[0] * y[2],
        y[0] * y[1] - beta * y[2],
    ])


def _vdp_jax(t, y, p):
    x = y[0]
    v = y[1]
    mu = p[0]
    dx = v
    dv = mu * ((1.0 - x * x) * v - x)
    return jnp.array([dx, dv])


def test_generated_lorenz_field_matches_handwritten():
    rng = np.random.default_rng(0)
    y = rng.uniform(-20.0, 20.0, size=(100, 3))
    rho = rng.uniform(0.0, 40.0, size=(100, 1))
    t = rng.uniform(0.0, 1.0, size=100)

    field, vecD, vecP = jaxpr_to_warp_field(_lorenz_jax, 3, 1, wp_scalar=wp.float64)
    gen = _eval_field_vec_param(field, vecD, vecP, t, y, rho)

    hand_field, vec3 = make_lorenz_field(wp.float64)
    hand = _eval_field_scalar_param(hand_field, vec3, 3, y, rho[:, 0])

    np.testing.assert_allclose(gen, hand, rtol=1e-15, atol=0.0)


def test_generated_vdp_field_matches_handwritten():
    rng = np.random.default_rng(0)
    y = rng.uniform(-3.0, 3.0, size=(100, 2))
    mu = rng.uniform(0.0, 5.0, size=(100, 1))
    t = rng.uniform(0.0, 1.0, size=100)

    field, vecD, vecP = jaxpr_to_warp_field(_vdp_jax, 2, 1, wp_scalar=wp.float64)
    gen = _eval_field_vec_param(field, vecD, vecP, t, y, mu)

    hand_field, vec2 = make_vdp_field(wp.float64)
    hand = _eval_field_scalar_param(hand_field, vec2, 2, y, mu[:, 0])

    np.testing.assert_allclose(gen, hand, rtol=1e-15, atol=0.0)


def test_unsupported_primitive_raises():
    def cond_rhs(t, y, p):
        return lax.cond(t > 0.5, lambda yy: yy, lambda yy: -yy, y)

    with pytest.raises(UnsupportedRHS) as ei:
        jaxpr_to_warp_field(cond_rhs, 2, 1, wp_scalar=wp.float64)
    assert "cond" in str(ei.value)


# --- Jacobian for the stiff kernels -------------------------------------------------

from gradsolve.warp._warp_rosenbrock import make_hires, make_robertson  # noqa: E402
from gradsolve.warp.jax_field import jaxpr_to_warp_jacobian  # noqa: E402


def _eval_jac(jac, vecD, matDD, t, y, p):
    """Evaluate a jac(t, y: vecD, p: vecD) -> matDD on a batch via a probe kernel."""
    n = y.shape[0]

    @wp.kernel
    def probe(tb: wp.array(dtype=wp.float64),
              yb: wp.array(dtype=vecD),
              pb: wp.array(dtype=vecD),
              out: wp.array(dtype=matDD)):
        j = wp.tid()
        out[j] = jac(tb[j], yb[j], pb[j])

    tb = wp.array(t, dtype=wp.float64, device="cpu")
    yb = wp.array(y, dtype=vecD, device="cpu")
    pb = wp.array(p, dtype=vecD, device="cpu")
    out = wp.zeros(n, dtype=matDD, device="cpu")
    wp.launch(probe, dim=n, inputs=[tb, yb, pb, out], device="cpu")
    return out.numpy()  # (n, D, D)


def _robertson_jax(t, y, p):
    k1 = p[0]
    k2 = p[1]
    k3 = p[2]
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    return jnp.array([
        -k1 * y0 + k3 * y1 * y2,
        k1 * y0 - k3 * y1 * y2 - k2 * y1 * y1,
        k2 * y1 * y1,
    ])


def _hires_jax(t, y, p):
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    y3 = y[3]
    y4 = y[4]
    y5 = y[5]
    y6 = y[6]
    y7 = y[7]
    return jnp.array([
        -1.71 * y0 + 0.43 * y1 + 8.32 * y2 + 0.0007,
        1.71 * y0 - 8.75 * y1,
        -10.03 * y2 + 0.43 * y3 + 0.035 * y4,
        8.32 * y1 + 1.71 * y2 - 1.12 * y3,
        -1.745 * y4 + 0.43 * y5 + 0.43 * y6,
        -280.0 * y5 * y7 + 0.69 * y3 + 1.71 * y4 - 0.43 * y5 + 0.69 * y6,
        280.0 * y5 * y7 - 1.81 * y6,
        -280.0 * y5 * y7 + 1.81 * y6,
    ])


def test_generated_robertson_field_and_jac_match_handwritten():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.0, 1.0, size=(100, 3))
    k1 = rng.uniform(0.0, 1.0, size=100)
    k2 = rng.uniform(1e6, 1e8, size=100)
    k3 = rng.uniform(1e3, 1e5, size=100)
    p = np.stack([k1, k2, k3], axis=1)
    t = rng.uniform(0.0, 1.0, size=100)

    field, vecD, vecP = jaxpr_to_warp_field(_robertson_jax, 3, 3, wp_scalar=wp.float64)
    jac, vecDj, matDD, vecPj = jaxpr_to_warp_jacobian(
        _robertson_jax, 3, 3, wp_scalar=wp.float64)

    hand_field, hand_jac, vec3, mat33 = make_robertson(wp.float64)

    gen_f = _eval_field_vec_param(field, vecD, vecP, t, y, p)
    hand_f = _eval_field_vec_param(hand_field, vec3, vec3, t, y, p)
    np.testing.assert_allclose(gen_f, hand_f, rtol=1e-15, atol=0.0)

    gen_j = _eval_jac(jac, vecDj, matDD, t, y, p)
    hand_j = _eval_jac(hand_jac, vec3, mat33, t, y, p)
    np.testing.assert_allclose(gen_j, hand_j, rtol=1e-15, atol=0.0)


def test_generated_hires_field_and_jac_match_handwritten():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.0, 1.0, size=(100, 8))
    p = rng.uniform(0.0, 1.0, size=(100, 8))  # ignored by HIRES
    t = rng.uniform(0.0, 1.0, size=100)

    field, vecD, vecP = jaxpr_to_warp_field(_hires_jax, 8, 8, wp_scalar=wp.float64)
    jac, vecDj, matDD, vecPj = jaxpr_to_warp_jacobian(
        _hires_jax, 8, 8, wp_scalar=wp.float64)

    hand_field, hand_jac, vec8, mat88 = make_hires(wp.float64)

    gen_f = _eval_field_vec_param(field, vecD, vecP, t, y, p)
    hand_f = _eval_field_vec_param(hand_field, vec8, vec8, t, y, p)
    np.testing.assert_allclose(gen_f, hand_f, rtol=1e-15, atol=0.0)

    gen_j = _eval_jac(jac, vecDj, matDD, t, y, p)
    hand_j = _eval_jac(hand_jac, vec8, mat88, t, y, p)
    np.testing.assert_allclose(gen_j, hand_j, rtol=1e-15, atol=0.0)


def test_jacobian_requires_nparams_le_dim():
    def two_param(t, y, p):
        return jnp.array([p[0] * y[0] + p[1] * y[1]])

    with pytest.raises(UnsupportedRHS):
        jaxpr_to_warp_jacobian(two_param, 1, 2, wp_scalar=wp.float64)


# --- Version-robust coverage: right-hand sides built with ``jnp.stack``
# The tests above build each derivative with ``jnp.array([...])``. jax >= 0.10.2 lowers
# ``jnp.stack`` to its own single-output ``stack`` primitive (jax <= 0.10.1 lowered it to
# ``expand_dims`` + ``concatenate``), so the problems below build theirs with ``jnp.stack``
# to cover that primitive on any jax version.


class _StackProblem:
    def __init__(self, name, dim, n_params, is_stiff, f_jax):
        self.name, self.dim, self.n_params, self.is_stiff, self.f_jax = name, dim, n_params, is_stiff, f_jax


def _lorenz_stack(t, y, p):
    return jnp.stack([10.0 * (y[1] - y[0]),
                      p[0] * y[0] - y[1] - y[0] * y[2],
                      y[0] * y[1] - (8.0 / 3.0) * y[2]])


def _vdp_stack(t, y, p):
    return jnp.stack([y[1], p[0] * ((1.0 - y[0] * y[0]) * y[1] - y[0])])


def _robertson_stack(t, y, p):
    k1, k2, k3 = p[0], p[1], p[2]
    return jnp.stack([-k1 * y[0] + k2 * y[1] * y[2],
                      k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] ** 2,
                      k3 * y[1] ** 2])


_REFERENCE_PROBLEMS = [
    _StackProblem("lorenz", 3, 1, False, _lorenz_stack),
    _StackProblem("vdp", 2, 1, False, _vdp_stack),
    _StackProblem("robertson", 3, 3, True, _robertson_stack),
]


@pytest.mark.parametrize("prob", _REFERENCE_PROBLEMS, ids=lambda p: p.name)
def test_every_elementary_reference_problem_translates(prob):
    """Translate a ``jnp.stack``-built ``f_jax`` and check the generated field against
    ``f_jax`` on 100 random states to 1e-12; for stiff problems also check the generated
    Jacobian against ``jax.jacfwd``. A problem is skipped only when its RHS uses a primitive
    outside the translatable subset (the skip reason names the primitive)."""
    modname = prob.name
    dim = int(prob.dim)
    npar = int(prob.n_params)

    try:
        field, vecD, vecP = jaxpr_to_warp_field(prob.f_jax, dim, npar, wp_scalar=wp.float64)
    except UnsupportedRHS as ex:
        pytest.skip(f"{modname}.f_jax uses unsupported primitive {ex.primitive_name!r}")

    rng = np.random.default_rng(0)
    y = rng.uniform(-5.0, 5.0, size=(100, dim))
    p = rng.uniform(-2.0, 2.0, size=(100, max(npar, 1)))
    t = rng.uniform(0.0, 1.0, size=100)

    gen = _eval_field_vec_param(field, vecD, vecP, t, y, p)
    ref = np.array([
        np.asarray(prob.f_jax(float(t[i]), jnp.asarray(y[i]),
                              jnp.asarray(p[i, :npar] if npar > 0 else np.zeros(1))))
        for i in range(100)
    ])
    np.testing.assert_allclose(gen, ref, rtol=1e-12, atol=1e-12)

    if prob.is_stiff:
        jac, vecDj, matDD, vecPj = jaxpr_to_warp_jacobian(
            prob.f_jax, dim, npar, wp_scalar=wp.float64)
        pfull = np.zeros((100, dim))
        if npar > 0:
            pfull[:, :npar] = p[:, :npar]
        gen_j = _eval_jac(jac, vecDj, matDD, t, y, pfull)
        jf = jax.jacfwd(prob.f_jax, argnums=1)
        ref_j = np.array([
            np.asarray(jf(float(t[i]), jnp.asarray(y[i]),
                          jnp.asarray(p[i, :npar] if npar > 0 else np.zeros(1))))
            for i in range(100)
        ])
        np.testing.assert_allclose(gen_j, ref_j, rtol=1e-12, atol=1e-12)


# --- Registration + fused-recorder mesh parity --------------------------------------

from gradsolve import register_jax_field  # noqa: E402
from gradsolve.solvers.method import ROSENBROCK23_METHOD, TSIT5_METHOD  # noqa: E402
from gradsolve.solvers.record_jax import record_adaptive_jax  # noqa: E402
from gradsolve.warp import warp_ode, warp_rosenbrock  # noqa: E402

_T0, _T1, _RTOL, _ATOL, _MS = 0.0, 1.0, 1e-6, 1e-9, 4096


def _accepted_from_fused(dts_full, n_acc):
    """List of per-trajectory accepted-dt arrays from a fused [max_steps, n] record."""
    return [dts_full[: int(n_acc[j]), j] for j in range(len(n_acc))]


def _accepted_from_jax(dts_padded, n_acc):
    """List of per-trajectory accepted-dt arrays from a jax [n, S] padded record."""
    return [dts_padded[j, : int(n_acc[j])] for j in range(len(n_acc))]


def _max_rel_dt_gap(a_list, b_list):
    g = 0.0
    for a, b in zip(a_list, b_list):
        if a.size:
            g = max(g, float(np.max(np.abs(a - b) / np.abs(b))))
    return g


def test_registered_generated_lorenz_fused_mesh_matches_handwritten():
    """The fused recorder on a generated Lorenz field reproduces the built-in Lorenz
    field's mesh: accepted/rejected step counts identical, accepted dts to rtol<=1e-9
    (the general-field FMA floor documented in _warp_kernel.py). Here the generated
    field emits the same textual arithmetic as make_lorenz_field, so the gap is 0.0."""
    register_jax_field("user_gen_lorenz", _lorenz_jax, 3, 1, stiff=False)
    rng = np.random.default_rng(0)
    n = 16
    y0 = rng.uniform(-15.0, 15.0, size=(n, 3))
    rho = rng.uniform(20.0, 40.0, size=(n, 1))
    common = dict(t0=_T0, t1=_T1, rtol=_RTOL, atol=_ATOL, dt0=(_T1 - _T0) / 100.0,
                  max_steps=_MS, record=1, device="cpu")

    _gy, ga, gr, _gs, gdt = warp_ode._launch(
        y0, rho, field_key="user_gen_lorenz", dim=3, **common)
    _hy, ha, hr, _hs, hdt = warp_ode._launch(
        y0, rho[:, 0], field_key="lorenz", dim=3, **common)

    assert np.array_equal(ga, ha)  # accepted-step counts identical
    assert np.array_equal(gr, hr)  # rejected-step counts identical
    gap = _max_rel_dt_gap(_accepted_from_fused(gdt, ga), _accepted_from_fused(hdt, ha))
    assert gap <= 1e-9, gap


def test_registered_generated_field_matches_jax_recorder():
    """The fused recorder on a generated field matches the device JAX recorder's mesh:
    counts exact, dts to the per-method measured FMA floor (XLA fuses the trial-step
    sums, the Warp kernel FMA-contracts them, so the meshes agree only up to that floor).

    Measured floors on these batches (max relative accepted-dt gap):
      * Lorenz (Tsit5):      ~4.4e-9  -> pinned rtol 1e-8
      * Van der Pol (Tsit5): ~2.9e-8  -> pinned rtol 1e-7 (a stiffer nonstiff RHS
                                          amplifies the per-expression ULP more than Lorenz)
      * Robertson (Rosenbrock23): ~2.9e-9 -> pinned rtol 1e-8.
    The Robertson cross-check uses ROSENBROCK23_METHOD (order 2/3 — the method the fused
    stiff kernel actually runs), not Rodas5P (order 5): two different-order integrators
    cannot share a step mesh, so 'counts exact' is only meaningful against the matching
    method. Floors read from a one-time run of the two recorders, stated like
    test_record_jax.py's per-method floors (not reused from the Rodas5P/host-path 8e-9)."""
    # --- Lorenz (non-stiff, Tsit5) ---
    register_jax_field("user_gen_lorenz", _lorenz_jax, 3, 1, stiff=False)
    rng = np.random.default_rng(0)
    n = 16
    y0 = rng.uniform(-15.0, 15.0, size=(n, 3))
    rho = rng.uniform(20.0, 40.0, size=(n, 1))
    common = dict(t0=_T0, t1=_T1, rtol=_RTOL, atol=_ATOL, dt0=(_T1 - _T0) / 100.0,
                  max_steps=_MS, record=1, device="cpu")
    _gy, ga, gr, _gs, gdt = warp_ode._launch(
        y0, rho, field_key="user_gen_lorenz", dim=3, **common)
    _yf, dts_j, na_j, nr_j = record_adaptive_jax(
        TSIT5_METHOD, _lorenz_jax, y0, rho, _T0, _T1, rtol=_RTOL, atol=_ATOL)
    assert np.array_equal(ga, na_j)
    assert np.array_equal(gr, nr_j)
    gap = _max_rel_dt_gap(_accepted_from_fused(gdt, ga), _accepted_from_jax(dts_j, na_j))
    assert gap <= 1e-8, gap

    # --- Van der Pol (non-stiff, Tsit5) ---
    register_jax_field("user_gen_vdp", _vdp_jax, 2, 1, stiff=False)
    rng = np.random.default_rng(1)
    y0 = rng.uniform(-2.0, 2.0, size=(n, 2))
    mu = rng.uniform(0.5, 3.0, size=(n, 1))
    _gy, ga, gr, _gs, gdt = warp_ode._launch(
        y0, mu, field_key="user_gen_vdp", dim=2, **common)
    _yf, dts_j, na_j, nr_j = record_adaptive_jax(
        TSIT5_METHOD, _vdp_jax, y0, mu, _T0, _T1, rtol=_RTOL, atol=_ATOL)
    assert np.array_equal(ga, na_j)
    assert np.array_equal(gr, nr_j)
    gap = _max_rel_dt_gap(_accepted_from_fused(gdt, ga), _accepted_from_jax(dts_j, na_j))
    assert gap <= 1e-7, gap

    # --- Robertson (stiff, Rosenbrock23) ---
    register_jax_field("user_gen_robertson", _robertson_jax, 3, 3, stiff=True)
    rng = np.random.default_rng(2)
    ns = 8
    y0 = np.column_stack([np.ones(ns), np.zeros(ns), np.zeros(ns)]) \
        + rng.uniform(0.0, 0.01, size=(ns, 3))
    k1 = rng.uniform(0.02, 0.06, size=ns)
    k2 = rng.uniform(2e7, 4e7, size=ns)
    k3 = rng.uniform(8e3, 1.2e4, size=ns)
    pr = np.column_stack([k1, k2, k3])
    t1r = 1e2
    common_s = dict(t0=_T0, t1=t1r, rtol=_RTOL, atol=_ATOL, dt0=(t1r - _T0) / 100.0,
                    max_steps=_MS, record=1, device="cpu")
    _gy, ga, gr, _gs, gdt = warp_rosenbrock._launch(
        y0, pr, field_key="user_gen_robertson", dim=3, **common_s)
    _yf, dts_j, na_j, nr_j = record_adaptive_jax(
        ROSENBROCK23_METHOD, _robertson_jax, y0, pr, _T0, t1r, rtol=_RTOL, atol=_ATOL)
    assert np.array_equal(ga, na_j)
    assert np.array_equal(gr, nr_j)
    gap = _max_rel_dt_gap(_accepted_from_fused(gdt, ga), _accepted_from_jax(dts_j, na_j))
    assert gap <= 1e-8, gap


# --- Routing fused="auto"|True|False ------------------------------------------------
from dataclasses import dataclass  # noqa: E402


@dataclass
class _Prob:
    """Minimal library-Problem-protocol carrier (name/dim/t0/t1/is_stiff/f_jax), the
    tests/test_record_jax.py pattern — tests import only gradsolve."""
    name: str
    dim: int
    is_stiff: bool
    f_jax: object
    t0: float = 0.0
    t1: float = 1.0


def _cond_rhs(t, y, p):
    # lax.cond lowers to a `cond` jaxpr primitive, outside the translatable subset.
    k = p[0]
    return lax.cond(y[0] > 0.0, lambda: -k * y, lambda: k * y)


def _robertson_batch(n, seed):
    rng = np.random.default_rng(seed)
    y0 = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n)]) \
        + rng.uniform(0.0, 0.01, size=(n, 3))
    k1 = rng.uniform(0.02, 0.06, size=n)
    k2 = rng.uniform(2e7, 4e7, size=n)
    k3 = rng.uniform(8e3, 1.2e4, size=n)
    return y0, np.column_stack([k1, k2, k3])


def test_fused_auto_falls_back_on_unsupported_primitive():
    """fused='auto' translates the user RHS on first use; an untranslatable primitive
    (lax.cond) routes to the general path with a route.reason naming the primitive, and
    the result is the general path's (byte-identical to fused=False)."""
    prob = _Prob("user_cond_auto", 2, False, _cond_rhs)
    rng = np.random.default_rng(0)
    y0 = rng.uniform(-1.0, 1.0, size=(4, 2))
    params = rng.uniform(0.5, 1.5, size=(4, 1))

    res = gradsolve.solve(prob, y0, params, fused="auto", rtol=1e-8, atol=1e-11)

    assert res.route.actual not in ("warp_ode", "warp_rosenbrock", "cuda_tsit5")
    assert "cond" in res.route.reason and "fell back" in res.route.reason

    ref = gradsolve.solve(prob, y0, params, fused=False, rtol=1e-8, atol=1e-11)
    assert res.route.actual == ref.route.actual
    np.testing.assert_allclose(res.y_final, ref.y_final, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(res.accepted_steps, ref.accepted_steps)


def test_fused_true_reraises_unsupported():
    """fused=True demands the fused kernel; an untranslatable RHS re-raises UnsupportedRHS
    (naming the primitive) instead of silently falling back."""
    prob = _Prob("user_cond_true", 2, False, _cond_rhs)
    rng = np.random.default_rng(1)
    y0 = rng.uniform(-1.0, 1.0, size=(4, 2))
    params = rng.uniform(0.5, 1.5, size=(4, 1))

    with pytest.raises(UnsupportedRHS, match="cond"):
        gradsolve.solve(prob, y0, params, fused=True, rtol=1e-8, atol=1e-11)


def test_fused_false_skips_translation():
    """fused=False never calls the translator, so a translatable RHS is not registered and
    stays on the general path: grad_closure on a translatable nonstiff Lorenz routes to
    tsit5_replay (it would be warp_replay had the field been translated+registered)."""
    prob = _Prob("user_false_lorenz", 3, False, _lorenz_jax)
    rng = np.random.default_rng(2)
    y0 = rng.uniform(-15.0, 15.0, size=(4, 3))
    params = rng.uniform(20.0, 40.0, size=(4, 1))

    clo = gradsolve.grad_closure(prob, y0, params, fused=False, rtol=1e-6, atol=1e-9)

    assert clo.route.actual == "tsit5_replay"
    assert not warp_ode.supports(prob)  # translation was skipped -> no registered field


def test_registered_problem_routes_to_fused():
    """A Problem whose name matches a register_jax_field registration routes to the fused
    engine exactly as a built-in does: a stiff generated field -> warp_rosenbrock (solve),
    a nonstiff generated field -> warp_replay (grad_closure record path)."""
    # stiff -> warp_rosenbrock forward
    register_jax_field("user_route_robertson", _robertson_jax, 3, 3, stiff=True)
    prob_s = _Prob("user_route_robertson", 3, True, _robertson_jax, t1=1.0)
    y0s, ps = _robertson_batch(6, seed=3)
    res = gradsolve.solve(prob_s, y0s, ps, rtol=1e-6, atol=1e-9)
    assert res.route.actual == "warp_rosenbrock"

    # nonstiff -> warp_replay reverse (records the mesh on the fused forward)
    register_jax_field("user_route_lorenz", _lorenz_jax, 3, 1, stiff=False)
    prob_n = _Prob("user_route_lorenz", 3, False, _lorenz_jax)
    rng = np.random.default_rng(4)
    y0n = rng.uniform(-15.0, 15.0, size=(6, 3))
    pn = rng.uniform(20.0, 40.0, size=(6, 1))
    clo = gradsolve.grad_closure(prob_n, y0n, pn, rtol=1e-6, atol=1e-9)
    assert clo.route.actual == "warp_replay"


# --- Fresh-system round-trip + regression guard -------------------------------------
import jax  # noqa: E402


def _cubic_ring_jax(t, y, p):
    """Fresh dim=20 nonstiff system (not a built-in): a periodic diffusively-coupled
    cubic ring, ``dy_i/dt = a*(y_{i-1} - 2 y_i + y_{i+1}) - y_i^3 + b``, params (a, b).
    ``jnp.roll`` lowers to static-index gathers, inside the translatable subset."""
    a = p[0]
    b = p[1]
    lap = jnp.roll(y, -1) - 2.0 * y + jnp.roll(y, 1)
    return a * lap - y * y * y + b


@dataclass
class _RingProb:
    name: str
    dim: int
    is_stiff: bool
    f_jax: object
    t0: float = 0.0
    t1: float = 1.0


def test_fresh_user_system_round_trips():
    """A fresh user system (dim=20 cubic ring, not among the built-ins) round-trips
    register_jax_field -> fused engine -> matches the TSIT5 recorder.

    A dim=20 system is used here because without a CUDA build a
    low-NVAR (dim <= CUDA_NVAR_CEILING=16) nonstiff forward solve routes to cuda_tsit5,
    which is GPU-only and falls back to diffrax on CPU — never reaching a fused engine. A
    dim>16 nonstiff field routes to warp_ode (choose_engine), the genuine fused kernel
    that runs on Warp's CPU device, so the round-trip lands on a fused engine. The
    2-param vecP non-stiff path (the registration + fused-recorder mesh parity above) is
    exercised either way.
    y_final matches record_adaptive_jax(TSIT5_METHOD) to the Tsit5 recorder floor (1e-8);
    the measured agreement here is ~5e-14 (a simpler RHS than Lorenz)."""
    register_jax_field("fresh_cubic_ring", _cubic_ring_jax, 20, 2, stiff=False)
    prob = _RingProb("fresh_cubic_ring", 20, False, _cubic_ring_jax)
    rng = np.random.default_rng(0)
    n = 8
    y0 = rng.uniform(-1.0, 1.0, size=(n, 20))
    params = np.column_stack([rng.uniform(0.5, 1.5, size=n),
                              rng.uniform(-0.5, 0.5, size=n)])

    res = gradsolve.solve(prob, y0, params, fused="auto", rtol=_RTOL, atol=_ATOL)
    assert res.route.actual == "warp_ode"  # the fused non-stiff engine

    yf, _dts, _na, _nr = record_adaptive_jax(
        TSIT5_METHOD, _cubic_ring_jax, y0, params, _T0, _T1, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(res.y_final, yf, rtol=1e-8, atol=1e-10)


def test_grad_through_generated_field_matches_fd():
    """grad_closure on a registered generated Lorenz: the param-gradient agrees with a
    central finite difference to rel-L2 <= 1e-4 (mirrors the warp_replay FD checks; the
    reverse pass is the record-and-replay adjoint through the fused forward)."""
    register_jax_field("user_fd_lorenz", _lorenz_jax, 3, 1, stiff=False)
    prob = _RingProb("user_fd_lorenz", 3, False, _lorenz_jax)
    rng = np.random.default_rng(0)
    n = 4
    y0 = rng.uniform(-15.0, 15.0, size=(n, 3))
    params = rng.uniform(20.0, 40.0, size=(n, 1))

    clo = gradsolve.grad_closure(prob, y0, params, rtol=_RTOL, atol=_ATOL)
    assert clo.route.actual == "warp_replay"

    def loss(pr):
        return jnp.sum(clo(jnp.asarray(pr)) ** 2)

    g_ad = np.asarray(jax.grad(loss)(jnp.asarray(params)))
    g_fd = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        eps = 1e-6 * max(1.0, abs(float(params[idx])))
        pp = params.copy()
        pp[idx] += eps
        pm = params.copy()
        pm[idx] -= eps
        g_fd[idx] = (float(loss(pp)) - float(loss(pm))) / (2 * eps)

    rel = float(np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd))
    assert rel <= 1e-4, rel


def test_env_recorder_and_general_path_unaffected(monkeypatch):
    """Regression guard: the general (unregistered) path is unchanged by the fused switch.

    (a) fused='auto' on an unregistered problem is byte-identical to fused=False — 'auto'
        only checks fused-eligibility, it never adopts the fused engine on its own.
    (b) an unregistered general problem still solves and differentiates under both
        GRADSOLVE_RECORDER routings ('jax' and 'host'), the two recorder backends the
        general record-and-replay path can pick."""
    def _decay(t, y, p):
        return -p[0] * y

    prob = _RingProb("unreg_decay_regression", 2, False, _decay)
    rng = np.random.default_rng(0)
    y0 = rng.uniform(0.5, 2.0, size=(4, 2))
    params = rng.uniform(0.3, 1.5, size=(4, 1))

    a = gradsolve.solve(prob, y0, params, fused="auto", rtol=1e-8, atol=1e-11)
    b = gradsolve.solve(prob, y0, params, fused=False, rtol=1e-8, atol=1e-11)
    assert a.route.actual == b.route.actual
    np.testing.assert_array_equal(a.y_final, b.y_final)
    np.testing.assert_array_equal(a.accepted_steps, b.accepted_steps)

    for val in ("jax", "host"):
        monkeypatch.setenv("GRADSOLVE_RECORDER", val)
        clo = gradsolve.grad_closure(prob, y0, params, rtol=_RTOL, atol=_ATOL)

        def loss(pr, _c=clo):
            return jnp.sum(_c(jnp.asarray(pr)) ** 2)

        g = np.asarray(jax.grad(loss)(jnp.asarray(params)))
        assert np.all(np.isfinite(g)), val


# --- Fused reverse pass on a generated field ----------------------------------------
from gradsolve.warp import fused_backward  # noqa: E402


def test_fused_reverse_pass_on_generated_field_matches_fd():
    """The fused reverse pass already differentiates a generated field, so a hand-
    translated ``jaxpr_to_warp_vjp`` is not needed (it would be a speed refinement, not a
    capability).

    ``fused_rosenbrock_grad_closure`` runs ``wp.Tape`` through the fused Rosenbrock23
    kernel; that kernel calls the generated field and its analytic Jacobian, both plain
    ``@wp.func`` closures. Warp emits the reverse adjoint of any ``@wp.func``
    automatically (only the in-kernel Gaussian-elimination solve needs a custom
    ``@wp.func_grad``, already supplied), so a generated Robertson gets the in-kernel
    fused reverse pass with no hand-written VJP. Here the param-gradient of
    ``sum(y_final**2)`` agrees with central finite differences to rel-L2 <= 1e-4.
    A hand-emitted VJP would only replace Warp's automatic field adjoint with generated
    source, so it is not implemented."""
    register_jax_field("user_task7_robertson", _robertson_jax, 3, 3, stiff=True)
    prob = _Prob("user_task7_robertson", 3, True, _robertson_jax, t1=1.0)
    n = 4
    y0, params = _robertson_batch(n, seed=7)

    clo = fused_backward.fused_rosenbrock_grad_closure(
        prob, y0, params, rtol=1e-12, atol=1e-14)

    def loss(pr):
        return jnp.sum(clo(jnp.asarray(pr)) ** 2)

    g_ad = np.asarray(jax.grad(loss)(jnp.asarray(params)))
    g_fd = np.zeros_like(params)
    for idx in np.ndindex(params.shape):
        eps = 1e-3 * max(1.0, abs(float(params[idx])))
        pp = params.copy()
        pp[idx] += eps
        pm = params.copy()
        pm[idx] -= eps
        lp = float(np.sum(clo(jnp.asarray(pp)) ** 2))
        lm = float(np.sum(clo(jnp.asarray(pm)) ** 2))
        g_fd[idx] = (lp - lm) / (2.0 * eps)

    rel = float(np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd))
    assert rel <= 1e-4, rel
    assert np.all(np.isfinite(g_ad))
