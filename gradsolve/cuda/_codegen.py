"""cuda_tsit5 codegen — assemble one ``.cu`` per (field_key, D, precision).

Fills ``kernels/tsit5_template.cu.in`` with the Tsit5 tableau (generated from
``gradsolve.solvers.tsit5_step`` — the same coefficients the JAX and Warp kernels use),
the per-field ``field(...)`` body (``_fields.emit_field``), the dimension literal, and
the scalar type. Pure source generation; the nvcc build → ``.so`` lives in ``_build.py``
``lru_cache``-keyed like ``_warp_kernel._make_kernel``.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from gradsolve.solvers import rosenbrock23_step as _rk
from gradsolve.solvers import tsit5_step as _ts

_KERNELS = Path(__file__).parent / "kernels"
_TEMPLATE = _KERNELS / "tsit5_template.cu.in"
_ROSEN_TEMPLATE = _KERNELS / "rosenbrock23_template.cu.in"
_PRECISION = {"float64": "double", "float32": "float"}
_FFI_DTYPE = {"float64": "ffi::F64", "float32": "ffi::F32"}

# Tableau coefficient names emitted as #defines (C2..C6 from _ts._C; the rest module attrs).
_A = ["A21", "A31", "A32", "A41", "A42", "A43", "A51", "A52", "A53", "A54",
      "A61", "A62", "A63", "A64", "A65"]
_B = ["B1", "B2", "B3", "B4", "B5", "B6"]
_E = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]


def _tableau_defines() -> str:
    """`#define` lines for the Tsit5 tableau, round-trip-exact from the single source."""
    lines = []
    for i, nm in enumerate(("C2", "C3", "C4", "C5", "C6")):
        lines.append(f"#define {nm} ((real){float(_ts._C[i])!r})")
    for nm in _A + _B + _E:
        lines.append(f"#define {nm} ((real){float(getattr(_ts, '_' + nm))!r})")
    return "\n".join(lines)


@functools.lru_cache(maxsize=None)
def emit_cu(field_key: str, D: int, precision: str = "float64") -> str:
    """The complete ``.cu`` source for (field_key, D, precision). Host/device dual (see the
    template): nvcc → CUDA kernel+launcher; plain c++ → a host main for CPU validation."""
    from gradsolve.cuda._fields import emit_field
    if precision not in _PRECISION:
        raise ValueError(f"precision {precision!r}; use float64|float32")
    src = _TEMPLATE.read_text()
    src = src.replace("@SCALAR@", _PRECISION[precision])
    src = src.replace("@FFI_DTYPE@", _FFI_DTYPE[precision])
    src = src.replace("@TABLEAU@", _tableau_defines())
    src = src.replace("@FIELD@", emit_field(field_key, int(D)))
    src = src.replace("@D@", str(int(D)))
    return src


# --- stiff Rosenbrock23 lane (cuda_rosenbrock23) --------------------------------------

def _rosenbrock_tableau_defines() -> str:
    """`#define DCOEF/E32` from ``gradsolve.solvers.rosenbrock23_step`` (the same numbers
    the pure-JAX replay step and the numpy reference use — never a magic literal)."""
    return (f"#define DCOEF ((real){float(_rk._D_COEF)!r})\n"
            f"#define E32 ((real){float(_rk._E32_COEF)!r})")


# Manual (not lru_cache) memo: linstiff carries its DxD operator ``A`` as a numpy array,
# which is unhashable; key the cache on A's bytes instead.
_ROSEN_CU_CACHE: dict[tuple, str] = {}


def emit_cu_rosenbrock(field_key: str, D: int, precision: str = "float64",
                       A: "np.ndarray | None" = None) -> str:
    """The complete stiff ``.cu`` source for (field_key, D, precision[, A]).

    Fills ``kernels/rosenbrock23_template.cu.in`` with the Rosenbrock coefficients
    (``_rosenbrock_tableau_defines``), the per-field RHS + analytic Jacobian
    (``_stiff_fields.emit_stiff_field`` / ``emit_stiff_jac``), the dimension and
    param-count literals, and the scalar/FFI dtypes. ``A`` (the DxD operator) is required
    for a ``linstiff_*`` field and ignored otherwise. ``P`` (NPARAM) is 3 for robertson
    (params [k1,k2,k3]) and 0 for the autonomous robertson-less fields (hires, linstiff)."""
    from gradsolve.cuda._stiff_fields import emit_stiff_field, emit_stiff_jac
    if precision not in _PRECISION:
        raise ValueError(f"precision {precision!r}; use float64|float32")
    P = 3 if field_key == "robertson" else 0
    a_key = None if A is None else np.asarray(A, dtype=np.float64).tobytes()
    key = (field_key, int(D), precision, a_key)
    cached = _ROSEN_CU_CACHE.get(key)
    if cached is not None:
        return cached
    src = _ROSEN_TEMPLATE.read_text()
    src = src.replace("@SCALAR@", _PRECISION[precision])
    src = src.replace("@FFI_DTYPE@", _FFI_DTYPE[precision])
    src = src.replace("@TABLEAU@", _rosenbrock_tableau_defines())
    src = src.replace("@FIELD@", emit_stiff_field(field_key, int(D), A))
    src = src.replace("@JAC@", emit_stiff_jac(field_key, int(D), A))
    src = src.replace("@D@", str(int(D)))
    src = src.replace("@P@", str(int(P)))
    _ROSEN_CU_CACHE[key] = src
    return src
