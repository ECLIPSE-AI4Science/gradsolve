"""Stiff fused adaptive Rosenbrock23 (ode23s) Warp kernel factory.

Module-internal kernel factory (leading underscore) — not public API; consumed by
``warp_rosenbrock.py``, not imported directly by users.

The low-NVAR stiff counterpart to the explicit ``_warp_kernel.py`` (Tsit5) factory:
a linearly-implicit Rosenbrock23 pair (Shampine & Reichelt 1997, MATLAB ode23s),
specialized to autonomous systems (the d(f)/dt term vanishes — the registered stiff
problems are autonomous). One thread = one trajectory; the whole adaptive solve — per-step
analytic Jacobian build + LU/solve of W = (I - d*h*J) for the 3 Rosenbrock stages, WRMS
error, order-2 I-controller accept/reject — runs inside one Warp kernel launch.

Why a separate module (same boundary as ``_warp_kernel.py``): this module deliberately
does not ``from __future__ import annotations`` — Warp 1.14 codegen evaluates kernel/func
annotations as live objects and only resolves closure vars the body references; under
PEP-563 the closed-over ``vecD``/``matDD`` types appear only in string annotations,
are never captured as cells, and Warp raises ``NameError`` at codegen. ``warp_ode.py``
keeps its future-import (its protocol hints want it); the build is delegated across this
boundary.

Mechanism (the closure factory): a Python factory
``_make_rosenbrock_kernel(field_key, D, precision)`` (``functools.lru_cache``) builds the
``@wp.kernel`` closing over (a) a field ``@wp.func``, (b) an ANALYTIC Jacobian
``@wp.func`` returning the DxD ``matDD`` = ``wp.types.matrix(shape=(D, D))``, and (c) the
fixed-length state type ``vecD`` = ``wp.types.vector(length=D)``. Warp bakes the
closed-over function references and the concrete D at codegen.

W-solve: an in-kernel partial-pivot Gaussian-elimination solve over ``matDD`` with static
``range(D)`` loops (codegen-safe; ``wp.inverse(matDD)`` also works for small D, but the
explicit LU avoids forming the inverse and matches the reference's
``numpy.linalg.solve`` LU path). The three stage systems (k1, k2, k3) each run a fresh
elimination (the matrix is re-formed per RHS; for the GPU register budget at D<=12 this
is cheap and avoids a persistent LU store).

Parity contract (the public contract — mirrors the Tsit5 kernel, but LU-floored):
  * accept/reject sequence + counts are the strict claim — bit-identical to
    the numpy reference for the registered fields
    (the matrix solves never flip a single accept/reject decision);
  * accepted-dt sequence agrees to ~1e-10, y_final to ~1e-8 — the W-solves go through an
    LU factorization (numpy's ``solve`` vs the in-kernel Gaussian elimination + clang/NVCC
    ``-ffp-contract=fast`` FMA), so value parity is floored at the LU/FMA level (~1e-8),
    looser than the Tsit5 matrix-free 1e-10. Documented, not a logic bug.

Registered fields: ``robertson`` (D=3, analytic J), ``hires`` (D=8, analytic J), and
``linstiff_<key>`` constant-matrix fields (jac returns A) registered via
``register_linstiff``.

Controller (order-2 propagated solution => error exponent -1/3, not Tsit5's -1/5):
  err = sqrt(mean_i (err_vec_i / (atol + rtol*max(|y_i|,|ynew_i|)))^2)
  accept iff err <= 1; dt <- h * clamp(0.9 * max(err,1e-10)**(-1/3), 0.2, 5.0)  (both branches)
"""
import functools

import numpy as np
import warp as wp

# Not essential — the only dynamic-trip loop is the max_steps loop; the D-loops are static
# range(D), safely unrolled.
wp.config.max_unroll = 64

# Rosenbrock23 coefficients (Shampine & Reichelt 1997; autonomous specialization).
_D_COEF = 1.0 / (2.0 + np.sqrt(2.0))
_E32 = 6.0 + np.sqrt(2.0)

# Order-2 I-controller constants (the -1/3 exponent is the order-2 differentiator).
SAFETY = 0.9
ERR_FLOOR = 1e-10
FAC_MIN = 0.2
FAC_MAX = 5.0
ORDER_EXP = -1.0 / 3.0


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
# Field + analytic-Jacobian @wp.func builders (one pair per problem). Each closes over
# the concrete vecD/matDD types and the precision scalar. The textual arithmetic order
# of the field matches the problem's f_np exactly (value parity with the numpy reference).
# Each builder takes ``wp_scalar`` and returns ``(field, jac, vecD, matDD)``.
# ---------------------------------------------------------------------------------


def make_robertson(wp_scalar):
    """Robertson kinetics, D=3, params p=[k1, k2, k3].

    f0 = -k1*y0 + k3*y1*y2 ; f1 = k1*y0 - k3*y1*y2 - k2*y1*y1 ; f2 = k2*y1*y1.
    The field takes a 3-vector param ``p`` (k1, k2, k3); the analytic Jacobian is the
    closed-form df/dy. Same arithmetic order as the built-in robertson field (gradsolve
    registers it under the name ``robertson``) and the numpy reference Jacobian.
    """
    vec3 = wp.types.vector(length=3, dtype=wp_scalar)
    mat33 = wp.types.matrix(shape=(3, 3), dtype=wp_scalar)

    @wp.func
    def field(t: wp_scalar, y: vec3, p: vec3):
        k1 = p[0]
        k2 = p[1]
        k3 = p[2]
        y0 = y[0]
        y1 = y[1]
        y2 = y[2]
        d0 = -k1 * y0 + k3 * y1 * y2
        d1 = k1 * y0 - k3 * y1 * y2 - k2 * y1 * y1
        d2 = k2 * y1 * y1
        return vec3(d0, d1, d2)

    @wp.func
    def jac(t: wp_scalar, y: vec3, p: vec3):
        k1 = p[0]
        k2 = p[1]
        k3 = p[2]
        y1 = y[1]
        y2 = y[2]
        J = mat33()
        J[0, 0] = -k1
        J[0, 1] = k3 * y2
        J[0, 2] = k3 * y1
        J[1, 0] = k1
        J[1, 1] = -k3 * y2 - wp_scalar(2.0) * k2 * y1
        J[1, 2] = -k3 * y1
        J[2, 0] = wp_scalar(0.0)
        J[2, 1] = wp_scalar(2.0) * k2 * y1
        J[2, 2] = wp_scalar(0.0)
        return J

    return field, jac, vec3, mat33


def make_hires(wp_scalar):
    """HIRES-8 (Hairer & Wanner, vol. II, Sect. IV.10), D=8, autonomous (no free params).

    8-species stiff plant-photochemistry; the bimolecular ``280*y5*y7`` exchange in
    components 5-7 (0-based) is the stiff/nonlinear coupling. The field is AUTONOMOUS and
    has no per-trajectory parameters, so ``p`` (an 8-vector, for signature uniformity with
    the other registered fields — vecD == param type in ``_launch``) is accepted and
    ignored. Same arithmetic order as the built-in hires field (gradsolve registers it
    under the name ``hires``): same constants, same operation order. The analytic Jacobian
    below is the closed-form df/dy (the tests check it against ``jax.jacfwd`` of the field).
    """
    vec8 = wp.types.vector(length=8, dtype=wp_scalar)
    mat88 = wp.types.matrix(shape=(8, 8), dtype=wp_scalar)

    @wp.func
    def field(t: wp_scalar, y: vec8, p: vec8):
        y0 = y[0]
        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]
        y5 = y[5]
        y6 = y[6]
        y7 = y[7]
        d0 = -wp_scalar(1.71) * y0 + wp_scalar(0.43) * y1 + wp_scalar(8.32) * y2 + wp_scalar(0.0007)
        d1 = wp_scalar(1.71) * y0 - wp_scalar(8.75) * y1
        d2 = -wp_scalar(10.03) * y2 + wp_scalar(0.43) * y3 + wp_scalar(0.035) * y4
        d3 = wp_scalar(8.32) * y1 + wp_scalar(1.71) * y2 - wp_scalar(1.12) * y3
        d4 = -wp_scalar(1.745) * y4 + wp_scalar(0.43) * y5 + wp_scalar(0.43) * y6
        d5 = -wp_scalar(280.0) * y5 * y7 + wp_scalar(0.69) * y3 + wp_scalar(1.71) * y4 - wp_scalar(0.43) * y5 + wp_scalar(0.69) * y6
        d6 = wp_scalar(280.0) * y5 * y7 - wp_scalar(1.81) * y6
        d7 = -wp_scalar(280.0) * y5 * y7 + wp_scalar(1.81) * y6
        return vec8(d0, d1, d2, d3, d4, d5, d6, d7)

    @wp.func
    def jac(t: wp_scalar, y: vec8, p: vec8):
        y5 = y[5]
        y7 = y[7]
        J = mat88()
        # Row 0: d0 = -1.71*y0 + 0.43*y1 + 8.32*y2 + 0.0007
        J[0, 0] = -wp_scalar(1.71)
        J[0, 1] = wp_scalar(0.43)
        J[0, 2] = wp_scalar(8.32)
        # Row 1: d1 = 1.71*y0 - 8.75*y1
        J[1, 0] = wp_scalar(1.71)
        J[1, 1] = -wp_scalar(8.75)
        # Row 2: d2 = -10.03*y2 + 0.43*y3 + 0.035*y4
        J[2, 2] = -wp_scalar(10.03)
        J[2, 3] = wp_scalar(0.43)
        J[2, 4] = wp_scalar(0.035)
        # Row 3: d3 = 8.32*y1 + 1.71*y2 - 1.12*y3
        J[3, 1] = wp_scalar(8.32)
        J[3, 2] = wp_scalar(1.71)
        J[3, 3] = -wp_scalar(1.12)
        # Row 4: d4 = -1.745*y4 + 0.43*y5 + 0.43*y6
        J[4, 4] = -wp_scalar(1.745)
        J[4, 5] = wp_scalar(0.43)
        J[4, 6] = wp_scalar(0.43)
        # Row 5: d5 = -280*y5*y7 + 0.69*y3 + 1.71*y4 - 0.43*y5 + 0.69*y6
        J[5, 3] = wp_scalar(0.69)
        J[5, 4] = wp_scalar(1.71)
        J[5, 5] = -wp_scalar(280.0) * y7 - wp_scalar(0.43)
        J[5, 6] = wp_scalar(0.69)
        J[5, 7] = -wp_scalar(280.0) * y5
        # Row 6: d6 = 280*y5*y7 - 1.81*y6
        J[6, 5] = wp_scalar(280.0) * y7
        J[6, 6] = -wp_scalar(1.81)
        J[6, 7] = wp_scalar(280.0) * y5
        # Row 7: d7 = -280*y5*y7 + 1.81*y6
        J[7, 5] = -wp_scalar(280.0) * y7
        J[7, 6] = wp_scalar(1.81)
        J[7, 7] = -wp_scalar(280.0) * y5
        return J

    return field, jac, vec8, mat88


def make_linstiff(A):
    """Build a CONSTANT-Jacobian linear-stiff field builder: f(y) = A y, jac = A.

    ``A`` is a (D, D) numpy matrix. Returns a ``builder(wp_scalar) -> (field, jac,
    vecD, matDD)`` closure so the registry entry is precision-agnostic; the matrix
    entries are baked as typed literals at codegen. The param ``p`` (a D-vector) is
    accepted for signature uniformity and IGNORED by the field (constant operator).
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"linstiff A must be square (D, D); got {A.shape}")
    D = A.shape[0]

    def builder(wp_scalar):
        vecD = wp.types.vector(length=D, dtype=wp_scalar)
        matDD = wp.types.matrix(shape=(D, D), dtype=wp_scalar)
        # Bake A as a closed-over Warp CONSTANT matrix value (probed on Warp 1.14: a
        # ``wp.constant(matDD(np_array))`` is captured at codegen and ``A_const * y`` /
        # returning ``A_const`` compile cleanly — unlike a Python list, which Warp cannot
        # subscript in a dynamic @wp.func loop). The field is the exact matrix-vector
        # product, matching ``f(y) = A y`` and the analytic constant Jacobian.
        A_const = wp.constant(matDD(A.astype(np.float64)))

        @wp.func
        def field(t: wp_scalar, y: vecD, p: vecD):
            return A_const * y

        @wp.func
        def jac(t: wp_scalar, y: vecD, p: vecD):
            return A_const

        return field, jac, vecD, matDD

    return builder, D


# Registry: field_key -> (builder(wp_scalar) -> (field, jac, vecD, matDD), dim).
_FIELD_REGISTRY = {
    "robertson": (make_robertson, 3),
    "hires": (make_hires, 8),
}


def register_field(key, builder, dim):
    """Register a ``builder(wp_scalar) -> (field, jac, vecD, matDD)`` under ``key``
    (dim ``dim``). Idempotent: a key already present is left untouched."""
    _FIELD_REGISTRY.setdefault(key, (builder, int(dim)))


def register_linstiff(key, A):
    """Register a constant-matrix linear-stiff field ``f(y) = A y`` under ``key``.

    Convenience over ``register_field`` for the common test/benchmark case. Idempotent.
    NB: re-registering the SAME key with a DIFFERENT matrix is a no-op (the first matrix
    wins) — use a fresh key for a fresh matrix.
    """
    if key in _FIELD_REGISTRY:
        return
    builder, dim = make_linstiff(A)
    _FIELD_REGISTRY[key] = (builder, dim)


def _field_entry(field_key):
    """Resolve ``field_key`` to its registered ``(builder, dim)`` entry.

    Unlike the explicit-Tsit5 factory's ``_field_entry`` (``_warp_kernel.py``), this
    is a PLAIN dict lookup with no lazy synthesis: every stiff field (an analytic
    field + Jacobian pair) must be registered up front via ``register_field`` /
    ``register_linstiff`` (or seeded in ``_FIELD_REGISTRY`` at module load) before
    it can be resolved here.

    Parameters
    ----------
    field_key : str
        Registered stiff field token (e.g. ``"robertson"``, ``"hires"``, or a
        registered ``"linstiff_<key>"``).

    Returns
    -------
    tuple
        ``(builder, dim)`` where ``builder(wp_scalar) -> (field, jac, vecD, matDD)``
        and ``dim`` is the field's registered state dimension D.

    Raises
    ------
    KeyError
        If ``field_key`` is not in ``_FIELD_REGISTRY`` — the message lists the
        registered keys and points at ``register_linstiff``/``register_field``.
    """
    if field_key in _FIELD_REGISTRY:
        return _FIELD_REGISTRY[field_key]
    raise KeyError(
        f"unknown stiff field_key {field_key!r}; registered: {sorted(_FIELD_REGISTRY)} "
        "(use register_linstiff/register_field to add one).")


def field_dim(field_key):
    """Registered state dimension D for ``field_key``."""
    return _field_entry(field_key)[1]


# ---------------------------------------------------------------------------------
# The stiff Rosenbrock23 kernel factory. Cached on (field_key, D, precision).
# ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _make_rosenbrock_kernel(field_key, D, precision):
    """Build (once, cached) the adaptive Rosenbrock23 kernel for ``(field_key, D,
    precision)``. Returns ``(kernel, vecD, matDD, wp_scalar, np_dtype)``.

    In-kernel: build J via the closed-over analytic Jacobian @wp.func, form
    W = I - d*h*J, solve the 3 stage systems with an in-kernel partial-pivot Gaussian
    elimination (``_gauss_solve``, static range(D) loops), propagate ynew = y + h*k2,
    and run the order-2 I-controller (exponent -1/3) exactly as the numpy reference does.
    """
    np_dtype, wp_scalar = _precision_types(precision)
    builder, reg_dim = _field_entry(field_key)
    if int(D) != int(reg_dim):
        raise ValueError(
            f"stiff field {field_key!r} has dim {reg_dim}, but D={D} was requested")
    field_func, jac_func, vecD, matDD = builder(wp_scalar)

    # Rosenbrock constants as typed literals.
    dcoef = float(_D_COEF)
    e32 = float(_E32)

    @wp.func
    def _gauss_solve(A: matDD, b: vecD):
        # Partial-pivot Gaussian elimination solving A x = b for the DxD stage system.
        # Static range(D) loops -> codegen-safe (probed on Warp 1.14 CPU). A and b are
        # copied so the caller's W/RHS are untouched across the 3 stage solves.
        M = matDD(A)
        x = vecD(b)
        for col in range(D):
            piv = col
            big = wp.abs(M[col, col])
            for r in range(col + 1, D):
                v = wp.abs(M[r, col])
                if v > big:
                    big = v
                    piv = r
            if piv != col:
                for c in range(D):
                    tmp = M[col, c]
                    M[col, c] = M[piv, c]
                    M[piv, c] = tmp
                tb = x[col]
                x[col] = x[piv]
                x[piv] = tb
            for r in range(col + 1, D):
                factor = M[r, col] / M[col, col]
                for c in range(col, D):
                    M[r, c] = M[r, c] - factor * M[col, c]
                x[r] = x[r] - factor * x[col]
        out = vecD()
        for ii in range(D):
            i = D - 1 - ii
            s = x[i]
            for c in range(i + 1, D):
                s = s - M[i, c] * out[c]
            out[i] = s / M[i, i]
        return out

    @wp.kernel
    def _rosenbrock23_adaptive(
        y0: wp.array(dtype=vecD),
        params: wp.array(dtype=vecD),         # per-trajectory param vector (k1,k2,k3 / ignored)
        conf: wp.array(dtype=wp_scalar),      # [t0, t1, rtol, atol, dt0]
        max_steps: int,
        record: int,                          # 1 -> write accepted-dt sequence
        y_out: wp.array(dtype=vecD),
        n_acc: wp.array(dtype=wp.int32),
        n_rej: wp.array(dtype=wp.int32),
        status: wp.array(dtype=wp.int32),
        dt_rec: wp.array2d(dtype=wp_scalar),  # [max_steps, n] when record==1 else [1,1]
    ):
        jt = wp.tid()
        t = conf[0]
        t1 = conf[1]
        rtol = conf[2]
        atol = conf[3]
        dt = conf[4]
        y = y0[jt]
        p = params[jt]
        acc = int(0)
        rej = int(0)
        for _ in range(max_steps):
            if t >= t1:
                break
            h = wp.min(dt, t1 - t)
            J = jac_func(t, y, p)
            # W = I - d*h*J  (build the identity, then subtract).
            W = matDD()
            for i in range(D):
                for jc in range(D):
                    W[i, jc] = -wp_scalar(dcoef) * h * J[i, jc]
                W[i, i] = W[i, i] + wp_scalar(1.0)
            F0 = field_func(t, y, p)
            k1 = _gauss_solve(W, F0)
            F1 = field_func(t, y + wp_scalar(0.5) * h * k1, p)
            k2 = _gauss_solve(W, F1 - k1) + k1
            ynew = y + h * k2
            F2 = field_func(t, ynew, p)
            rhs3 = F2 - wp_scalar(e32) * (k2 - F1) - wp_scalar(2.0) * (k1 - F0)
            k3 = _gauss_solve(W, rhs3)
            err_vec = (h / wp_scalar(6.0)) * (k1 - wp_scalar(2.0) * k2 + k3)
            # WRMS error: sum_i (err_vec_i / sc_i)^2 then /D and sqrt — same association
            # as the oracle's np.mean((err_vec/sc)**2). sc_i = atol + rtol*max(|y|,|ynew|).
            ssum = wp_scalar(0.0)
            for i in range(D):
                sc = atol + rtol * wp.max(wp.abs(y[i]), wp.abs(ynew[i]))
                comp = err_vec[i] / sc
                ssum = ssum + comp * comp
            err = wp.sqrt(ssum / wp_scalar(D))
            if err <= wp_scalar(1.0):
                t = t + h
                y = ynew
                if record == 1:
                    dt_rec[acc, jt] = h
                acc = acc + 1
            else:
                rej = rej + 1
            # Order-2 I-controller (exponent -1/3) after BOTH accept and reject.
            fac = wp_scalar(SAFETY) * wp.pow(
                wp.max(err, wp_scalar(ERR_FLOOR)), wp_scalar(ORDER_EXP))
            dt = h * wp.clamp(fac, wp_scalar(FAC_MIN), wp_scalar(FAC_MAX))
        y_out[jt] = y
        n_acc[jt] = acc
        n_rej[jt] = rej
        if t >= t1:
            status[jt] = 0
        else:
            status[jt] = 1

    return _rosenbrock23_adaptive, vecD, matDD, wp_scalar, np_dtype
