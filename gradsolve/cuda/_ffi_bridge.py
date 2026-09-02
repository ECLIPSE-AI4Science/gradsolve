"""jax.ffi glue for the cuda_tsit5 forward-only lane.

Registers the compiled ``Tsit5Fwd`` handler (``_build``) per (field_key, D, precision,
platform) and wraps it in ``jax.ffi.ffi_call``. Transposed-SoA on the wire (y0 [D,n], rho
[n] -> yf [D,n], nsteps [n] int32); scalar tols + max_steps as static numpy attrs. The op
carries no autodiff rule -> ``jax.grad`` through it raises = the forward-only contract.

CPU platform = the validation path (host-loop handler); CUDA = production.
"""
from __future__ import annotations

import ctypes
import functools

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.cuda import _build

_REGISTERED: set[str] = set()
_PLATFORM = {"cpu": "cpu", "cuda": "CUDA", "gpu": "CUDA"}
_DT = {"float64": jnp.float64, "float32": jnp.float32}


def _target_name(field_key: str, D: int, precision: str, platform: str) -> str:
    """Build the unique ``jax.ffi`` registration name for one build variant.

    Parameters
    ----------
    field_key : str
        Problem field key (e.g. ``"lorenz"``, ``"vdp"``, ``"linear_<D>"``).
    D : int
        State dimension.
    precision : str
        ``"float64"`` or ``"float32"``.
    platform : str
        Pre-mapping platform key as passed by callers (``"cpu"``, ``"cuda"``, or ``"gpu"``);
        note this is embedded verbatim, not normalized through ``_PLATFORM``.

    Returns
    -------
    str
        ``f"tsit5_fwd_{field_key}_{D}_{precision}_{platform}"`` — doubles as the
        ``_REGISTERED``-set membership key and the ``jax.ffi.register_ffi_target`` name, so
        two calls with the same arguments are guaranteed to hit the same registration.
    """
    return f"tsit5_fwd_{field_key}_{D}_{precision}_{platform}"


def _register(field_key: str, D: int, precision: str, platform: str) -> str:
    """Build + ctypes-load + register the Tsit5Fwd handler once. Returns the target name."""
    target = _target_name(field_key, D, precision, platform)
    if target in _REGISTERED:
        return target
    if platform == "cpu":
        so = _build.build_cpu_so(field_key, D, precision)
    else:
        so = _build.build_cuda_so(field_key, D, precision)
    lib = ctypes.cdll.LoadLibrary(so)
    jax.ffi.register_ffi_target(
        target, jax.ffi.pycapsule(lib.Tsit5Fwd), platform=_PLATFORM[platform], api_version=1)
    _REGISTERED.add(target)
    return target


@functools.lru_cache(maxsize=None)
def make_runner(field_key: str, D: int, precision: str, platform: str):
    """A jit-able ``run(y0_soa[D,n], rho[n], t1, rtol, atol, max_steps) -> (yf[D,n], nsteps[n])``."""
    target = _register(field_key, D, precision, platform)
    dt = _DT[precision]

    def run(y0_soa, rho, t1, rtol, atol, max_steps):
        n = int(y0_soa.shape[1])
        out = (jax.ShapeDtypeStruct((int(D), n), dt), jax.ShapeDtypeStruct((n,), jnp.int32))
        call = jax.ffi.ffi_call(target, out, vmap_method="sequential")
        return call(y0_soa, rho,
                    t1=np.float64(t1), rtol=np.float64(rtol), atol=np.float64(atol),
                    max_steps=np.int64(max_steps))
    return run


def solve_ffi(problem, y0, params, *, rtol, atol, device, max_steps=100_000):
    """Forward solve via the FFI target. ``y0[n,D]`` -> ``(yf[n,D], nsteps[n])``. t0=0 (the
    v1 fields all start at 0); integrates to ``problem.t1``."""
    from gradsolve.warp.warp_ode import _field_for, _param_of
    fk = _field_for(problem)
    if fk is None:
        raise ValueError(f"cuda_tsit5 has no field for {problem.name!r}")
    field_key, D = fk
    y0 = np.asarray(y0)
    precision = "float32" if y0.dtype == np.float32 else "float64"
    np_dt = np.float32 if precision == "float32" else np.float64
    platform = _PLATFORM.get(device, "cpu")
    platform = "cpu" if platform == "cpu" else "cuda"

    # _param_of is the GENERAL per-trajectory scalar dispatcher (rho for lorenz, mu for vdp,
    # ignored for linear); _rho_of is Lorenz-only.
    rho = np.asarray(_param_of(problem, params), dtype=np_dt)        # per-traj scalar [n]
    y0_soa = jnp.asarray(np.ascontiguousarray(y0.astype(np_dt).T))   # [n,D] -> [D,n]
    run = make_runner(field_key, int(D), precision, platform)
    yf_soa, nsteps = run(y0_soa, jnp.asarray(rho), float(problem.t1), float(rtol),
                         float(atol), int(max_steps))
    yf = np.ascontiguousarray(np.asarray(yf_soa).T)                  # [D,n] -> [n,D]
    return yf, np.asarray(nsteps)


# ---------------------------------------------------------------------------------------
# Stiff Rosenbrock23 lane (cuda_rosenbrock23) — a SEPARATE FFI target (symbol
# ``Rosenbrock23Fwd``) with the param-VECTOR ABI: y0[D,n] + params[P,n] SoA on the wire.
# ---------------------------------------------------------------------------------------

def _rosen_target_name(field_key: str, D: int, precision: str, platform: str) -> str:
    """``rosenbrock23_fwd_{field_key}_{D}_{precision}_{platform}`` — the registration name
    (also the ``_REGISTERED`` membership key), parallel to ``_target_name`` for Tsit5."""
    return f"rosenbrock23_fwd_{field_key}_{D}_{precision}_{platform}"


def _register_rosenbrock(field_key: str, D: int, precision: str, platform: str, A=None) -> str:
    """Build + ctypes-load + register the ``Rosenbrock23Fwd`` handler once. Returns the
    target name. ``A`` is the DxD operator for a ``linstiff_*`` field (ignored otherwise)."""
    target = _rosen_target_name(field_key, D, precision, platform)
    if target in _REGISTERED:
        return target
    if platform == "cpu":
        so = _build.build_cpu_so_rosenbrock(field_key, D, precision, A)
    else:
        so = _build.build_cuda_so_rosenbrock(field_key, D, precision, A)
    lib = ctypes.cdll.LoadLibrary(so)
    jax.ffi.register_ffi_target(
        target, jax.ffi.pycapsule(lib.Rosenbrock23Fwd), platform=_PLATFORM[platform],
        api_version=1)
    _REGISTERED.add(target)
    return target


def make_runner_rosenbrock(field_key: str, D: int, P: int, precision: str, platform: str,
                           A=None):
    """A jit-able ``run(y0_soa[D,n], p_soa[Pw,n], t1, rtol, atol, max_steps) ->
    (yf[D,n], nsteps[n])`` over the stiff kernel. ``P`` is the field's true param count
    (NPARAM: 3 for robertson, 0 for hires/linstiff); ``p_soa`` carries at least one row so
    zero-param fields still pass a well-formed (ignored) buffer. Not ``lru_cache``d — ``A``
    is an unhashable ndarray; the underlying build/register are memoized by name."""
    target = _register_rosenbrock(field_key, int(D), precision, platform, A)
    dt = _DT[precision]

    def run(y0_soa, p_soa, t1, rtol, atol, max_steps):
        n = int(y0_soa.shape[1])
        out = (jax.ShapeDtypeStruct((int(D), n), dt), jax.ShapeDtypeStruct((n,), jnp.int32))
        call = jax.ffi.ffi_call(target, out, vmap_method="sequential")
        return call(y0_soa, p_soa,
                    t1=np.float64(t1), rtol=np.float64(rtol), atol=np.float64(atol),
                    max_steps=np.int64(max_steps))
    return run


def _stiff_field_for(problem):
    """Map a stiff ``problem`` to ``(field_key, D, P)`` for the cuda_rosenbrock23 lane
    (warp-independent, unlike ``warp_rosenbrock._field_for`` which gates on Warp).

      * ``robertson`` -> ``("robertson", 3, 3)``  (params [k1,k2,k3])
      * ``hires``     -> ``("hires", 8, 0)``      (autonomous)
      * ``linstiff_<D>`` -> ``("linstiff_<D>", D, 0)``  (constant operator; A reconstructed
        from ``problem.f_jax`` in ``solve_ffi_rosenbrock``)

    Returns None for unsupported problems (so ``supports`` is a thin truthiness check)."""
    nm = problem.name
    if nm == "robertson":
        return ("robertson", 3, 3)
    if nm == "hires":
        return ("hires", 8, 0)
    if nm.startswith("linstiff_"):
        try:
            d = int(nm.rsplit("_", 1)[1])
        except ValueError:
            return None
        if d > 0:
            return (nm, d, 0)
    return None


def _reconstruct_linstiff_A(problem, D, params_row) -> np.ndarray:
    """Recover the constant operator ``A`` of a linear-stiff field ``f(y) = A y`` from the
    library ``Problem`` (which exposes only ``f_jax``, not the matrix): probe the D unit
    vectors. ``f(e_j) = A e_j`` = column j, so stacking ``f(e_j)`` row-wise gives A^T;
    subtracting ``f(0)`` makes this correct for an affine field too."""
    import jax.numpy as _jnp
    eye = np.eye(int(D), dtype=np.float64)
    p_probe = _jnp.asarray(np.repeat(np.asarray(params_row, np.float64)[None, :],
                                     int(D), axis=0))
    f0 = np.asarray(problem.f_jax(problem.t0, _jnp.zeros((int(D),)),
                                  _jnp.asarray(np.asarray(params_row, np.float64))))
    M = np.asarray(problem.f_jax(problem.t0, _jnp.asarray(eye), p_probe))  # M[j] = f(e_j)
    return np.ascontiguousarray((M - f0).T)                               # A = (M - f0)^T


def solve_ffi_rosenbrock(problem, y0, params, *, rtol, atol, device, max_steps=50_000):
    """Forward stiff solve via the Rosenbrock23 FFI target. ``y0[n,D]`` + ``params[n,P]`` ->
    ``(yf[n,D], nsteps[n])``. t0=0 (the stiff fields all start at 0); integrates to
    ``problem.t1``. Overflow rows (a trajectory whose final state came back non-finite —
    a blown-up / non-converged solve) are NaN'd, mirroring ``warp_rosenbrock``'s driver.

    Note: overflow == non-finite final state. The kernel ABI returns (yf, nsteps) with
    no status flag, so a genuine max-steps exhaustion that stayed finite is NOT flagged
    here (it returns its last reached state). If a status column is ever needed, add it to
    the kernel + handler and NaN on status!=0 like warp_rosenbrock. Not a concern for the
    well-posed CPU tests (all reach t1 with headroom)."""
    routed = _stiff_field_for(problem)
    if routed is None:
        raise ValueError(f"cuda_rosenbrock23 has no stiff field for {problem.name!r}")
    field_key, D, P = routed
    y0 = np.asarray(y0)
    precision = "float32" if y0.dtype == np.float32 else "float64"
    np_dt = np.float32 if precision == "float32" else np.float64
    platform = _PLATFORM.get(device, "cpu")
    platform = "cpu" if platform == "cpu" else "cuda"

    params_np = np.asarray(params, dtype=np_dt)
    if params_np.ndim == 1:
        params_np = params_np[:, None]
    n = int(y0.shape[0])

    A = None
    if field_key.startswith("linstiff_"):
        A = _reconstruct_linstiff_A(problem, D, params_np[0]).astype(np.float64)

    # Param SoA [Pw, n]. robertson: the [k1,k2,k3] columns transposed (P=3). Autonomous
    # fields (P=0): a single ignored placeholder row so the wire buffer is well-formed.
    if P > 0:
        p_soa = jnp.asarray(np.ascontiguousarray(params_np[:, :P].astype(np_dt).T))  # [P,n]
    else:
        p_soa = jnp.asarray(np.zeros((1, n), dtype=np_dt))
    y0_soa = jnp.asarray(np.ascontiguousarray(y0.astype(np_dt).T))                    # [D,n]

    run = make_runner_rosenbrock(field_key, int(D), int(P), precision, platform, A)
    yf_soa, nsteps = run(y0_soa, p_soa, float(problem.t1), float(rtol), float(atol),
                         int(max_steps))
    yf = np.ascontiguousarray(np.asarray(yf_soa).T)                                   # [D,n]
    nsteps = np.asarray(nsteps)
    bad = ~np.isfinite(yf).all(axis=1)
    if bad.any():
        yf[bad] = np.nan
    return yf, nsteps
