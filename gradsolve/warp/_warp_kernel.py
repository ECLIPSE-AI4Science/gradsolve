"""General-field fused adaptive Tsit5 Warp kernel factory.

Module-internal kernel factory (leading underscore) — not public API; consumed by
``warp_ode.py``, not imported directly by users.

One kernel factory that, per ``(field_key, D, precision)``, builds an adaptive-Tsit5
kernel closing over (a) a concrete field ``@wp.func`` at the precision's scalar type and
(b) the fixed-length vector type ``wp.types.vector(length=D, dtype=wp_scalar)``.

WHY THIS MODULE EXISTS SEPARATELY FROM ``warp_ode.py`` (the structural decision):
  Warp 1.14 evaluates kernel/func annotations as LIVE objects
  (``inspect.get_annotations(eval_str=True, locals=closure_vars)``) and only resolves
  closure vars that the function body *references*. Under ``from __future__ import
  annotations`` (PEP-563) the closed-over ``vecD`` appears only inside (string)
  annotations, is never captured as a cell, and Warp raises ``NameError: vecD`` at
  codegen. ``warp_ode.py`` keeps its ``from __future__ import annotations`` (its
  ``Problem``/protocol type hints want it); this module deliberately does NOT import
  it, so the factory's annotations stay real objects. The two modules are a clean
  split along exactly this codegen boundary.

Parity contract (the public contract):
  * Accept/reject sequence and counts are bit-identical to the numpy reference for every
    field at every D (the core correctness claim — the FMA effect below never
    flips a single accept/reject decision).
  * Lorenz D=3 is bit-exact in value too: this factory at ``("lorenz", 3,
    "float64")`` emits typed-literal code (``wp.float64(...)``) that matches the numpy
    reference to 1e-10.
  * General fields: clang ``-ffp-contract=fast`` (default on Apple-Silicon ARM64 and
    NVCC on CUDA) contracts ``a*b + c*d`` into a single FMA with one rounding, while the
    un-fused numpy reference rounds the two products separately. This floors value-parity
    at ~1e-9 for D >~ 8 (controller-amplified ~1 ULP/expr). Warp exposes no public
    hook to inject ``-ffp-contract=off``, so the contract is: assert accept/reject
    sequence strictly for all problems; assert dt/y_final value-parity at atol 1e-9 for
    general fields (1e-10 for Lorenz D=3). Not a logic bug.

The tableau is imported from ``gradsolve.solvers.tsit5_step``, shared with the JAX solvers.
"""
import functools

import numpy as np
import warp as wp

from gradsolve.solvers.tsit5_step import (  # tableau shared with the JAX solvers
    _A21,
    _A31,
    _A32,
    _A41,
    _A42,
    _A43,
    _A51,
    _A52,
    _A53,
    _A54,
    _A61,
    _A62,
    _A63,
    _A64,
    _A65,
    _B1,
    _B2,
    _B3,
    _B4,
    _B5,
    _B6,
    _C,
    _E1,
    _E2,
    _E3,
    _E4,
    _E5,
    _E6,
    _E7,
)

# Not essential — the kernel's only loop is the dynamic-trip-count max_steps loop, never
# unrolled.
wp.config.max_unroll = 64

# _C is the tuple of nodes for stages 2..7; alias as module floats for kernel capture.
_C2, _C3, _C4, _C5, _C6 = _C[0], _C[1], _C[2], _C[3], _C[4]


# ---------------------------------------------------------------------------------
# Precision plumbing. ``_PRECISION_TYPES[precision] -> (np_dtype, wp_scalar)``; the
# fixed-length vector type is built per-(D, precision) inside the kernel factory.
# ---------------------------------------------------------------------------------
_PRECISION_TYPES = {
    "float64": (np.float64, wp.float64),
    "float32": (np.float32, wp.float32),
}


def _precision_types(precision):
    try:
        return _PRECISION_TYPES[precision]
    except KeyError:
        raise ValueError(
            f"precision must be 'float64' or 'float32', got {precision!r}"
        ) from None


# ---------------------------------------------------------------------------------
# Field @wp.func builders (one per problem). Each closes over the concrete vecD type
# and the precision scalar type. The textual arithmetic order
# matches the problem's f_np exactly (float64 association parity with the numpy
# reference, which calls the same numpy f). Each builder takes ``wp_scalar`` so f32/f64
# emit the right typed literals; ``vecD`` is built here and returned for the kernel.
# ---------------------------------------------------------------------------------


def make_lorenz_field(wp_scalar):
    """Lorenz, dim 3: ``sigma=10, beta=8/3``, ``p == rho``.

    Same textual arithmetic order as the built-in lorenz field (gradsolve registers it
    under the names ``lorenz`` and ``diffeqgpu_lorenz``), so the f64 instantiation
    matches the numpy reference to 1e-10.
    """
    vec3 = wp.types.vector(length=3, dtype=wp_scalar)

    @wp.func
    def field(t: wp_scalar, y: vec3, rho: wp_scalar):
        sigma = wp_scalar(10.0)
        beta = wp_scalar(8.0) / wp_scalar(3.0)
        return vec3(
            sigma * (y[1] - y[0]),
            rho * y[0] - y[1] - y[0] * y[2],
            y[0] * y[1] - beta * y[2],
        )

    return field, vec3


def make_linear_field(D):
    """Build a ``linear_<D>`` field builder: ``f(t, y, p) = 1.01 * y`` (p ignored).

    Matches the built-in linear-ladder field (gradsolve registers it under the name
    ``linear_ladder_<D>``) exactly.
    Returns a ``builder(wp_scalar) -> (field, vecD)`` closure so the registry entry is
    precision-agnostic (the precision is bound at kernel-factory time, not registration).
    """
    rate = 1.01

    def builder(wp_scalar):
        vecD = wp.types.vector(length=D, dtype=wp_scalar)

        @wp.func
        def field(t: wp_scalar, y: vecD, p: wp_scalar):
            out = vecD()
            for i in range(D):
                out[i] = wp_scalar(rate) * y[i]
            return out

        return field, vecD

    return builder


def make_vdp_field(wp_scalar):
    """Van der Pol, dim 2: ``dx = v; dv = mu*((1 - x*x)*v - x)``, ``p == mu``.

    Same arithmetic order as the built-in vdp field (gradsolve registers it under the
    name ``vdp``): ``mu * ((1.0 - x*x) * v - x)``.
    """
    vec2 = wp.types.vector(length=2, dtype=wp_scalar)

    @wp.func
    def field(t: wp_scalar, y: vec2, mu: wp_scalar):
        x = y[0]
        v = y[1]
        dx = v
        dv = mu * ((wp_scalar(1.0) - x * x) * v - x)
        return vec2(dx, dv)

    return field, vec2


def make_lorenz96_field(D):
    """Lorenz-96, dim D (nonlinear, nonstiff): ``dy_i = (y_{i+1} - y_{i-2})*y_{i-1} - y_i + F``,
    cyclic indices, ``p == F``. Matches the built-in lorenz96 field (gradsolve registers it
    under the name ``lorenz96_<D>``) exactly (roll(-1)/roll(2)/roll(1) ->
    y_{i+1}/y_{i-2}/y_{i-1}). builder(wp_scalar) -> (field, vecD)."""
    def builder(wp_scalar):
        vecD = wp.types.vector(length=D, dtype=wp_scalar)

        @wp.func
        def field(t: wp_scalar, y: vecD, F: wp_scalar):
            out = vecD()
            for i in range(D):
                ip1 = (i + 1) % D
                im1 = (i - 1 + D) % D
                im2 = (i - 2 + D) % D
                out[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + F
            return out

        return field, vecD

    return builder


# Registry: field_key -> (builder(wp_scalar) -> (field_@wp.func, vecD), dim).
# Seeded with the built-in fields; ``register_field`` extends it (idempotent).
_FIELD_REGISTRY = {
    "lorenz": (make_lorenz_field, 3),
    "vdp": (make_vdp_field, 2),
}
# Linear-ladder fields are keyed by dimension (``linear_<D>``); seed the common ones
# and synthesize any other ``linear_<D>`` lazily in ``_field_entry``.
for _d in (3, 8, 10, 32, 100):
    _FIELD_REGISTRY[f"linear_{_d}"] = (make_linear_field(_d), _d)

# Keys whose field takes a length-P vecP param (generated fields from
# ``jax_field.register_jax_field``) rather than the built-ins' scalar param. Only a
# flag — it does NOT carry the vecP length; ``warp_ode._launch`` reads that off
# ``params.shape[1]`` at launch time (the registry entry is only ``(builder, dim)``).
_VECP_FIELDS = set()


def register_field(key, builder, dim, *, vecp=False):
    """Register a ``builder(wp_scalar) -> (field_@wp.func, vecD)`` under ``key`` (dim
    ``dim``). Idempotent: a key already present is left untouched. ``vecp=True`` marks
    a generated field whose ``@wp.func`` takes a length-P ``vecP`` param (so
    ``warp_ode._launch`` passes a ``wp.array(dtype=vecP)`` instead of the scalar
    param) — the built-in scalar fields never set it."""
    _FIELD_REGISTRY.setdefault(key, (builder, int(dim)))
    if vecp:
        _VECP_FIELDS.add(key)


def _field_entry(field_key):
    """Resolve ``field_key`` to its registered ``(builder, dim)`` entry.

    Looks up ``_FIELD_REGISTRY`` first; on a miss, synthesizes and caches a fresh
    entry for two lazy families so callers never have to pre-seed every dimension
    they might ask for: ``lorenz96_<D>`` (``D >= 4``, via ``make_lorenz96_field``)
    and ``linear_<D>`` (``D > 0``, via ``make_linear_field``) — mirroring the
    handful of ``linear_<D>`` dims seeded eagerly at module load (see the loop
    above this function). A synthesized entry is written back into
    ``_FIELD_REGISTRY`` so a repeat lookup for the same key is a plain dict hit.

    Parameters
    ----------
    field_key : str
        Registered field token (e.g. ``"lorenz"``, ``"vdp"``) or a synthesizable
        ``"lorenz96_<D>"`` / ``"linear_<D>"`` token.

    Returns
    -------
    tuple
        ``(builder, dim)`` where ``builder(wp_scalar) -> (field_@wp.func, vecD)``
        and ``dim`` is the field's registered state dimension D.

    Raises
    ------
    KeyError
        If ``field_key`` is neither already registered nor a well-formed
        ``lorenz96_<D>`` / ``linear_<D>`` token (e.g. a non-integer or
        non-positive/too-small suffix).
    """
    if field_key in _FIELD_REGISTRY:
        return _FIELD_REGISTRY[field_key]
    if field_key.startswith("lorenz96_"):
        try:
            d96 = int(field_key.split("_", 1)[1])
        except ValueError:
            d96 = None
        if d96 is not None and d96 >= 4:
            entry = (make_lorenz96_field(d96), d96)
            _FIELD_REGISTRY[field_key] = entry
            return entry
    if field_key.startswith("linear_"):
        try:
            d = int(field_key.split("_", 1)[1])
        except ValueError:
            d = None
        if d is not None and d > 0:
            entry = (make_linear_field(d), d)
            _FIELD_REGISTRY[field_key] = entry
            return entry
    raise KeyError(
        f"unknown field_key {field_key!r}; registered: {sorted(_FIELD_REGISTRY)}")


def field_dim(field_key):
    """Registered state dimension D for ``field_key`` (synthesizes linear_<D>)."""
    return _field_entry(field_key)[1]


# ---------------------------------------------------------------------------------
# The general-f adaptive Tsit5 kernel factory. Cached on (field_key, D,
# precision) so each unique (RHS, dim, precision) compiles at most once per process.
# ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _make_kernel(field_key, D, precision, vecp_P=None):
    """Build (once, cached) the adaptive-Tsit5 kernel for ``(field_key, D, precision)``.

    Returns ``(kernel, vecD, wp_scalar, np_dtype)``. The controller body follows
    ``tsit5_adaptive`` in ``gradsolve/solvers/tsit5_replay.py`` (FSAL, WRMS, I-controller,
    record path) with two generalizations: the field call is the closed-over ``field_func``,
    and the WRMS norm is a ``for i in range(D)`` reduction with ``/ D``
    (same association as the reference's ``np.mean((e/sc)**2)``). Typed literals use the
    closed-over ``wp_scalar`` (``wp.float64`` for f64; ``wp.float32`` for f32).

    ``vecp_P`` selects the per-trajectory param type: ``None`` (the default, and the
    only value the built-ins ever pass) keeps the scalar param
    (``params: wp.array(dtype=wp_scalar)``, ``p = params[j]`` a scalar). An
    integer P builds a ``vecP = wp.types.vector(length=max(P,1))`` param
    (``params: wp.array(dtype=vecP)``) for a generated ``field(t, y, p: vecP)`` — the
    only line that differs. ``vecp_P`` is a fourth positional arg with default ``None``:
    the built-ins call ``_make_kernel(field_key, D, precision)`` (3 args), so their
    lru_cache key and kernel are unchanged.
    """
    np_dtype, wp_scalar = _precision_types(precision)
    builder, reg_dim = _field_entry(field_key)
    if int(D) != int(reg_dim):
        raise ValueError(
            f"field {field_key!r} has dim {reg_dim}, but D={D} was requested")
    field_func, vecD = builder(wp_scalar)
    # NOTE: generated-field kernel carries a vecP param array whose length is read
    # from params.shape[1] at launch; built-in scalar fields keep the scalar path.
    param_dtype = wp_scalar if vecp_P is None else \
        wp.types.vector(length=max(int(vecp_P), 1), dtype=wp_scalar)

    @wp.kernel
    def _tsit5_adaptive_general(
        y0: wp.array(dtype=vecD),
        params: wp.array(dtype=param_dtype),   # per-trajectory param p (scalar or vecP)
        conf: wp.array(dtype=wp_scalar),     # [t0, t1, rtol, atol, dt0] — FFI-safe scalars
        max_steps: int,
        record: int,                         # 1 -> write accepted-dt sequence
        y_out: wp.array(dtype=vecD),
        n_acc: wp.array(dtype=wp.int32),
        n_rej: wp.array(dtype=wp.int32),
        status: wp.array(dtype=wp.int32),
        dt_rec: wp.array2d(dtype=wp_scalar),  # [max_steps, n] when record==1 else [1,1]
    ):
        j = wp.tid()
        t = conf[0]
        t1 = conf[1]
        rtol = conf[2]
        atol = conf[3]
        dt = conf[4]
        y = y0[j]
        p = params[j]
        acc = int(0)
        rej = int(0)
        k1 = field_func(t, y, p)  # FSAL seed
        for _ in range(max_steps):
            if t >= t1:
                break
            h = wp.min(dt, t1 - t)
            k2 = field_func(t + wp_scalar(_C2) * h,
                            y + h * (wp_scalar(_A21) * k1), p)
            k3 = field_func(t + wp_scalar(_C3) * h,
                            y + h * (wp_scalar(_A31) * k1 + wp_scalar(_A32) * k2), p)
            k4 = field_func(t + wp_scalar(_C4) * h,
                            y + h * (wp_scalar(_A41) * k1 + wp_scalar(_A42) * k2
                                     + wp_scalar(_A43) * k3), p)
            k5 = field_func(t + wp_scalar(_C5) * h,
                            y + h * (wp_scalar(_A51) * k1 + wp_scalar(_A52) * k2
                                     + wp_scalar(_A53) * k3 + wp_scalar(_A54) * k4), p)
            k6 = field_func(t + wp_scalar(_C6) * h,
                            y + h * (wp_scalar(_A61) * k1 + wp_scalar(_A62) * k2
                                     + wp_scalar(_A63) * k3 + wp_scalar(_A64) * k4
                                     + wp_scalar(_A65) * k5), p)
            y5 = y + h * (wp_scalar(_B1) * k1 + wp_scalar(_B2) * k2 + wp_scalar(_B3) * k3
                          + wp_scalar(_B4) * k4 + wp_scalar(_B5) * k5 + wp_scalar(_B6) * k6)
            k7 = field_func(t + h, y5, p)  # FSAL stage (b7 == 0)
            e = h * (wp_scalar(_E1) * k1 + wp_scalar(_E2) * k2 + wp_scalar(_E3) * k3
                     + wp_scalar(_E4) * k4 + wp_scalar(_E5) * k5 + wp_scalar(_E6) * k6
                     + wp_scalar(_E7) * k7)
            # WRMS error: sum_i (e_i / sc_i)^2 then /D and sqrt — same association as
            # the oracle's np.mean((e/sc)**2). sc_i = atol + rtol*max(|y_i|, |y5_i|).
            ssum = wp_scalar(0.0)
            for i in range(D):
                sc = atol + rtol * wp.max(wp.abs(y[i]), wp.abs(y5[i]))
                comp = e[i] / sc
                ssum = ssum + comp * comp
            err = wp.sqrt(ssum / wp_scalar(D))
            if err <= wp_scalar(1.0):
                t = t + h
                y = y5
                k1 = k7  # FSAL: only on accept
                if record == 1:
                    dt_rec[acc, j] = h  # [step, traj] -> coalesced across threads
                acc = acc + 1
            else:
                rej = rej + 1
            # I-controller update after BOTH accept and reject.
            fac = wp_scalar(0.9) * wp.pow(wp.max(err, wp_scalar(1e-10)), wp_scalar(-0.2))
            dt = h * wp.clamp(fac, wp_scalar(0.2), wp_scalar(5.0))
        y_out[j] = y
        n_acc[j] = acc
        n_rej[j] = rej
        if t >= t1:
            status[j] = 0
        else:
            status[j] = 1

    return _tsit5_adaptive_general, vecD, wp_scalar, np_dtype
