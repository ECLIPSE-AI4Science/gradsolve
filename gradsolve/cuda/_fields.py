"""CUDA ``__device__`` field emitters for the cuda_tsit5 lane.

Each emitter returns the C++ source of ``field(real t, const real* s, real rho, real* dy)``
transcribed term for term from the problem's ``f_np`` (one source of arithmetic order, so
float64 association lines up; the kernel's ``real`` typedef = double|float). Autonomous fields
ignore ``t``; ``rho`` is the per-trajectory scalar param (rho for lorenz, mu for vdp, ignored
for linear). Registered fields: lorenz/diffeqgpu_lorenz (D=3), vdp (D=2), linear_<D>.
"""
from __future__ import annotations

# diffeqgpu_lorenz: f0=10*(y1-y0); f1=rho*y0-y1-y0*y2; f2=y0*y1-(8/3)*y2
_LORENZ = """HD inline void field(real t, const real* s, real rho, real* dy) {
  (void)t;
  dy[0] = (real)10.0 * (s[1] - s[0]);
  dy[1] = rho * s[0] - s[1] - s[0] * s[2];
  dy[2] = s[0] * s[1] - ((real)8.0 / (real)3.0) * s[2];
}
"""

# vdp: dx=v; dv=mu*((1-x*x)*v - x)   (rho == mu; parenthesization exact)
_VDP = """HD inline void field(real t, const real* s, real rho, real* dy) {
  (void)t;
  dy[0] = s[1];
  dy[1] = rho * (((real)1.0 - s[0] * s[0]) * s[1] - s[0]);
}
"""


def _linear(D: int) -> str:
    """C++ ``field(...)`` body for the ``linear_<D>`` family (unrolled elementwise decay).

    Parameters
    ----------
    D : int
        State dimension; baked in as the unrolled loop bound (``#pragma unroll``).

    Returns
    -------
    str
        Source of ``field(real t, const real* s, real rho, real* dy)`` computing
        ``dy[i] = 1.01 * s[i]`` for ``i in range(D)``; ``t`` and ``rho`` are unused
        (autonomous, unparameterized system) but kept in the signature for ABI uniformity
        with ``_LORENZ``/``_VDP``.
    """
    # linear_ladder: dy/dt = 1.01*y elementwise (param ignored).
    return (
        "HD inline void field(real t, const real* s, real rho, real* dy) {\n"
        "  (void)t; (void)rho;\n"
        "  #pragma unroll\n"
        f"  for (int i = 0; i < {int(D)}; ++i) dy[i] = (real)1.01 * s[i];\n"
        "}\n"
    )


def emit_field(field_key: str, D: int) -> str:
    """Return the C++ ``field(...)`` body for a registered field key (from
    ``warp_ode._field_for``: "lorenz" | "vdp" | "linear_<D>")."""
    if field_key == "lorenz":
        if int(D) != 3:
            raise ValueError(f"lorenz field is D=3, got {D}")
        return _LORENZ
    if field_key == "vdp":
        if int(D) != 2:
            raise ValueError(f"vdp field is D=2, got {D}")
        return _VDP
    if field_key.startswith("linear_"):
        return _linear(D)
    raise ValueError(
        f"no CUDA field emitter for {field_key!r} (registered fields: lorenz, vdp, linear_<D>)")
