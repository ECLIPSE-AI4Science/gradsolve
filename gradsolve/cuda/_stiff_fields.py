"""CUDA ``__device__`` field + analytic-Jacobian emitters for the
cuda_rosenbrock23 (stiff, forward-only) lane.

The stiff kernel needs both the vector field and its analytic Jacobian, and (unlike the
Tsit5 lane's ``real rho`` scalar) a per-trajectory param vector. So this lane uses a
deliberately different device ABI:

    field(real t, const real* s, const real* p, real* dy)          // dy = f(t, s, p)
    jac  (real t, const real* s, const real* p, real* J)           // J row-major DxD

Design choice 1: the field/jac take a param vector ``p`` (Robertson = [k1,k2,k3]; HIRES =
autonomous, ``p`` accepted and ignored; linstiff = constant operator, ``p`` ignored). The
Tsit5 ``real rho`` scalar ABI does not generalize to a multi-param stiff field, so this is a
separate ABI on purpose.

Every emitter is transcribed term for term from the Warp field builders in
``gradsolve/warp/_warp_rosenbrock.py`` (``make_robertson``, ``make_hires``, and the constant
matrix for linstiff) so float64 association lines up with the numpy reference; the kernel's
``real`` typedef is double|float. The Jacobian is zero-initialized then the nonzeros are
filled (the Warp code relies on ``mat33()``/``mat88()`` zero-init — mirrored here).
"""
from __future__ import annotations

import numpy as np


def _lit(v: float) -> str:
    """A typed float64 C literal that round-trips ``v`` exactly (``repr`` is shortest-exact)."""
    return f"(real)({float(v)!r})"


# ---------------------------------------------------------------------------------
# Robertson (D=3, params p=[k1,k2,k3]) — _warp_rosenbrock.make_robertson.
# f0 = -k1*y0 + k3*y1*y2 ; f1 = k1*y0 - k3*y1*y2 - k2*y1*y1 ; f2 = k2*y1*y1.
# ---------------------------------------------------------------------------------
_ROBERTSON_FIELD = """HD inline void field(real t, const real* s, const real* p, real* dy) {
  (void)t;
  real k1 = p[0];
  real k2 = p[1];
  real k3 = p[2];
  real y0 = s[0];
  real y1 = s[1];
  real y2 = s[2];
  dy[0] = -k1 * y0 + k3 * y1 * y2;
  dy[1] = k1 * y0 - k3 * y1 * y2 - k2 * y1 * y1;
  dy[2] = k2 * y1 * y1;
}
"""

_ROBERTSON_JAC = """HD inline void jac(real t, const real* s, const real* p, real* J) {
  (void)t; (void)s;
  real k1 = p[0];
  real k2 = p[1];
  real k3 = p[2];
  real y1 = s[1];
  real y2 = s[2];
  for (int i = 0; i < 9; ++i) J[i] = (real)0.0;
  J[0] = -k1;                              // J00
  J[1] = k3 * y2;                          // J01
  J[2] = k3 * y1;                          // J02
  J[3] = k1;                               // J10
  J[4] = -k3 * y2 - (real)2.0 * k2 * y1;   // J11
  J[5] = -k3 * y1;                         // J12
  J[7] = (real)2.0 * k2 * y1;              // J21  (J20 = J22 = 0)
}
"""


# ---------------------------------------------------------------------------------
# HIRES (D=8, autonomous, p ignored) — _warp_rosenbrock.make_hires.
# ---------------------------------------------------------------------------------
_HIRES_FIELD = """HD inline void field(real t, const real* s, const real* p, real* dy) {
  (void)t; (void)p;
  real y0 = s[0];
  real y1 = s[1];
  real y2 = s[2];
  real y3 = s[3];
  real y4 = s[4];
  real y5 = s[5];
  real y6 = s[6];
  real y7 = s[7];
  dy[0] = -(real)1.71 * y0 + (real)0.43 * y1 + (real)8.32 * y2 + (real)0.0007;
  dy[1] = (real)1.71 * y0 - (real)8.75 * y1;
  dy[2] = -(real)10.03 * y2 + (real)0.43 * y3 + (real)0.035 * y4;
  dy[3] = (real)8.32 * y1 + (real)1.71 * y2 - (real)1.12 * y3;
  dy[4] = -(real)1.745 * y4 + (real)0.43 * y5 + (real)0.43 * y6;
  dy[5] = -(real)280.0 * y5 * y7 + (real)0.69 * y3 + (real)1.71 * y4 - (real)0.43 * y5 + (real)0.69 * y6;
  dy[6] = (real)280.0 * y5 * y7 - (real)1.81 * y6;
  dy[7] = -(real)280.0 * y5 * y7 + (real)1.81 * y6;
}
"""

_HIRES_JAC = """HD inline void jac(real t, const real* s, const real* p, real* J) {
  (void)t; (void)p;
  real y5 = s[5];
  real y7 = s[7];
  for (int i = 0; i < 64; ++i) J[i] = (real)0.0;
  // Row 0: d0 = -1.71*y0 + 0.43*y1 + 8.32*y2 + 0.0007
  J[0*8+0] = -(real)1.71;
  J[0*8+1] = (real)0.43;
  J[0*8+2] = (real)8.32;
  // Row 1: d1 = 1.71*y0 - 8.75*y1
  J[1*8+0] = (real)1.71;
  J[1*8+1] = -(real)8.75;
  // Row 2: d2 = -10.03*y2 + 0.43*y3 + 0.035*y4
  J[2*8+2] = -(real)10.03;
  J[2*8+3] = (real)0.43;
  J[2*8+4] = (real)0.035;
  // Row 3: d3 = 8.32*y1 + 1.71*y2 - 1.12*y3
  J[3*8+1] = (real)8.32;
  J[3*8+2] = (real)1.71;
  J[3*8+3] = -(real)1.12;
  // Row 4: d4 = -1.745*y4 + 0.43*y5 + 0.43*y6
  J[4*8+4] = -(real)1.745;
  J[4*8+5] = (real)0.43;
  J[4*8+6] = (real)0.43;
  // Row 5: d5 = -280*y5*y7 + 0.69*y3 + 1.71*y4 - 0.43*y5 + 0.69*y6
  J[5*8+3] = (real)0.69;
  J[5*8+4] = (real)1.71;
  J[5*8+5] = -(real)280.0 * y7 - (real)0.43;
  J[5*8+6] = (real)0.69;
  J[5*8+7] = -(real)280.0 * y5;
  // Row 6: d6 = 280*y5*y7 - 1.81*y6
  J[6*8+5] = (real)280.0 * y7;
  J[6*8+6] = -(real)1.81;
  J[6*8+7] = (real)280.0 * y5;
  // Row 7: d7 = -280*y5*y7 + 1.81*y6
  J[7*8+5] = -(real)280.0 * y7;
  J[7*8+6] = (real)1.81;
  J[7*8+7] = -(real)280.0 * y5;
}
"""


def _linstiff_field(A: np.ndarray) -> str:
    """C++ ``field(...)`` for a CONSTANT linear-stiff operator ``f(y) = A y``.

    ``A`` is a (D, D) numpy matrix baked as typed literals: ``dy[i] = sum_j A[i,j]*s[j]``,
    left-to-right in j (the same association as ``y @ A.T`` on a row-batched ``y``).
    ``t`` and ``p`` are unused (autonomous, unparameterized) but kept for ABI uniformity.
    """
    A = np.asarray(A, dtype=np.float64)
    D = A.shape[0]
    lines = ["HD inline void field(real t, const real* s, const real* p, real* dy) {",
             "  (void)t; (void)p;"]
    for i in range(D):
        terms = " + ".join(f"{_lit(A[i, j])} * s[{j}]" for j in range(D))
        lines.append(f"  dy[{i}] = {terms};")
    lines.append("}\n")
    return "\n".join(lines)


def _linstiff_jac(A: np.ndarray) -> str:
    """C++ ``jac(...)`` for ``f(y) = A y`` — the constant Jacobian J == A (baked literals)."""
    A = np.asarray(A, dtype=np.float64)
    D = A.shape[0]
    lines = ["HD inline void jac(real t, const real* s, const real* p, real* J) {",
             "  (void)t; (void)s; (void)p;"]
    for i in range(D):
        for j in range(D):
            lines.append(f"  J[{i}*{D}+{j}] = {_lit(A[i, j])};")
    lines.append("}\n")
    return "\n".join(lines)


def emit_stiff_field(field_key: str, D: int, A: np.ndarray | None = None) -> str:
    """C++ ``field(real t, const real* s, const real* p, real* dy)`` for a stiff field key.

    Keys: ``"robertson"`` (D=3, p=[k1,k2,k3]), ``"hires"`` (D=8, autonomous), or
    ``"linstiff_<D>"`` (constant operator — the DxD matrix ``A`` must be supplied).
    """
    if field_key == "robertson":
        if int(D) != 3:
            raise ValueError(f"robertson field is D=3, got {D}")
        return _ROBERTSON_FIELD
    if field_key == "hires":
        if int(D) != 8:
            raise ValueError(f"hires field is D=8, got {D}")
        return _HIRES_FIELD
    if field_key.startswith("linstiff_"):
        if A is None:
            raise ValueError(f"{field_key!r} needs its constant matrix A (pass A=...)")
        A = np.asarray(A, dtype=np.float64)
        if A.shape != (int(D), int(D)):
            raise ValueError(f"linstiff A must be ({D},{D}); got {A.shape}")
        return _linstiff_field(A)
    raise ValueError(
        f"no CUDA stiff field emitter for {field_key!r} (robertson, hires, linstiff_<D>)")


def emit_stiff_jac(field_key: str, D: int, A: np.ndarray | None = None) -> str:
    """C++ ``jac(real t, const real* s, const real* p, real* J)`` (J row-major DxD)
    for a stiff field key — the analytic df/dy transcribed from the Warp source."""
    if field_key == "robertson":
        if int(D) != 3:
            raise ValueError(f"robertson jac is D=3, got {D}")
        return _ROBERTSON_JAC
    if field_key == "hires":
        if int(D) != 8:
            raise ValueError(f"hires jac is D=8, got {D}")
        return _HIRES_JAC
    if field_key.startswith("linstiff_"):
        if A is None:
            raise ValueError(f"{field_key!r} needs its constant matrix A (pass A=...)")
        A = np.asarray(A, dtype=np.float64)
        if A.shape != (int(D), int(D)):
            raise ValueError(f"linstiff A must be ({D},{D}); got {A.shape}")
        return _linstiff_jac(A)
    raise ValueError(
        f"no CUDA stiff jac emitter for {field_key!r} (robertson, hires, linstiff_<D>)")
