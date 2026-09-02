"""warp_rosenbrock backend — fused adaptive Rosenbrock23 (ode23s) Warp kernel.

The low-NVAR stiff sibling of ``warp_ode`` (explicit Tsit5): a linearly-implicit
Rosenbrock23 pair (Shampine & Reichelt 1997) with an in-kernel per-step analytic
Jacobian + LU/solve of W = (I - d*h*J). One thread = one trajectory; the whole adaptive
solve runs inside one Warp kernel launch. The executable semantic spec is the numpy
reference; the kernel reproduces it step for step
(same accept/reject sequence + counts, same accepted-dt sequence), checked by
``tests/test_adjoint_replay.py``.

Scope:
  - **Forward + record (float64).** The kernel is built by the closure factory
    ``gradsolve/warp/_warp_rosenbrock.py::_make_rosenbrock_kernel(field_key, D,
    precision)`` — one Rosenbrock23 body that, per registered (field, analytic-Jacobian)
    pair at small D, closes over the field/jac ``@wp.func``s, ``vecD = wp.types.vector(
    length=D)`` and ``matDD = wp.types.matrix(shape=(D, D))``. No ``wp.Tape``/backward —
    reverse mode is the record-and-replay adjoint
    (``warp_replay.make_rosenbrock_replay_closure``), which drives the ``wp.launch`` path.
  - **``wp.launch`` path only.** There is no jax_kernel FFI bridge for the stiff kernel
    (unlike the Lorenz-only bridge in ``warp_ode``); ``solve(device="cuda")`` drives
    ``wp.launch(device="cuda")`` directly.
  - **Registered fields:** ``robertson`` (D=3, analytic J, p=[k1,k2,k3]), ``hires``
    (D=8, analytic J, autonomous — p is an ignored placeholder), and
    ``linstiff_<key>`` constant-matrix linear-stiff fields registered via
    ``register_linstiff`` (jac returns A). ``supports()`` accepts ``robertson``,
    ``hires``, and any registered ``linstiff_*`` problem (low-NVAR stiff only).

  Parity contract (see ``_warp_rosenbrock.py`` docstring): accept/reject
    sequence + counts are bit-identical to the numpy reference for every registered field
    (the strict claim); accepted-dt agrees to ~1e-10 and y_final to ~1e-8
    — the W-solves go through an LU factorization (numpy ``solve`` vs in-kernel Gaussian
    elimination + ``-ffp-contract=fast`` FMA), so value parity is LU/FMA-floored at ~1e-8
    (looser than the matrix-free Tsit5 1e-10). Not a logic bug.

Register-limit guard: the fused kernel holds a DxD ``matDD`` (W and J) plus several
``vecD`` live per thread — the per-thread register/local-memory footprint is O(D^2),
heavier than the explicit kernel's O(D). ``_GPU_MAT_DIM_CEILING`` (conservative, 12)
raises a "register"-flagged RuntimeError for device in (cuda, gpu) above the limit so
the caller can record it as a register-limit failure (the fused stiff engine is a
low-NVAR engine). CPU has no such limit.

Import hygiene: the warp import is guarded (a warp-less env degrades cleanly —
``supports()`` returns False for everything; ``_launch`` raises RuntimeError if reached);
no ``wp.init()`` at import (Warp auto-initializes on first array/launch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gradsolve.base import SolveResult

if TYPE_CHECKING:  # gradsolve imports standalone; Problem is annotation-only
    from gradsolve.base import Problem

name = "warp_rosenbrock"

# GPU register/local-memory limit for the fused stiff kernel. The per-thread footprint
# is O(D^2) (the DxD W and J matrices + the Gaussian-elimination workspace), so the stiff
# engine spills the register/local budget at a lower dim than the explicit Tsit5 engine
# (_GPU_VEC_DIM_CEILING=64 there). Conservative (12) — the fused stiff engine's design
# point is low NVAR; high NVAR is served by the general engines (diffrax or the
# record-and-replay solvers).
# CPU has no such limit (stack-backed), so this guard fires only for device cuda/gpu.
_GPU_MAT_DIM_CEILING = 12

try:
    import warp as wp

    # The closure factory lives in a separate module without
    # ``from __future__ import annotations`` — Warp 1.14 codegen needs the kernel's
    # closed-over vecD/matDD annotations as live objects (see _warp_rosenbrock.py's
    # module docstring). This module keeps its own future-import (Problem/protocol
    # hints want it); the kernel build is delegated across that boundary.
    from gradsolve.warp import _warp_rosenbrock

    _WARP_AVAILABLE = True
except ImportError:  # warp-less env: degrade to the stub instead of breaking the package
    wp = None
    _warp_rosenbrock = None
    _WARP_AVAILABLE = False


# Re-export registration helpers so callers/tests can register a linstiff field via the
# BACKEND module (the canonical entry point), not the private factory.
def register_linstiff(key, A):
    """Register a constant-matrix linear-stiff field ``f(y) = A y`` under ``key``.

    Thin pass-through to ``_warp_rosenbrock.register_linstiff`` (idempotent). Raises if
    warp is unavailable (registration needs the factory)."""
    _require_warp()
    _warp_rosenbrock.register_linstiff(key, A)


def register_field(key, builder, dim):
    """Register a ``builder(wp_scalar) -> (field, jac, vecD, matDD)`` under ``key``.

    Thin pass-through to ``_warp_rosenbrock.register_field`` (idempotent)."""
    _require_warp()
    _warp_rosenbrock.register_field(key, builder, dim)


def _require_warp():
    """Guard clause: raise loudly if NVIDIA Warp failed to import.

    Called at the top of every entry point that actually needs the fused stiff
    kernel (``register_linstiff``, ``register_field``, ``_precision_spec``,
    ``_launch``) so a warp-less environment fails with a clear, actionable message
    at the call site instead of an ``AttributeError`` on ``wp``/``_warp_rosenbrock``
    (both ``None`` when the top-level ``import warp`` guard at module load fails).
    ``supports()`` stays side-effect-free (it checks ``_WARP_AVAILABLE`` directly,
    not this function), so a caller that never calls ``solve()`` is
    unaffected.

    Raises
    ------
    RuntimeError
        If ``_WARP_AVAILABLE`` is False (``import warp`` failed at module load) —
        the message points at ``pip install 'gradsolve[cuda12]'``.
    """
    if not _WARP_AVAILABLE:
        raise RuntimeError(
            "NVIDIA Warp is not installed: the fused adaptive Rosenbrock23 kernel needs it. "
            "Install the GPU extra with `pip install 'gradsolve[cuda12]'`."
        )


def _precision_spec(field_key, D, precision):
    """(np_dtype, wp_scalar, vecD, matDD, kernel) for ``(field_key, D, precision)``.

    Thin adapter over ``_warp_rosenbrock._make_rosenbrock_kernel`` (lru_cache-keyed, so
    each unique (field, dim, precision) compiles at most once)."""
    _require_warp()
    kernel, vecD, matDD, wp_scalar, np_dtype = (
        _warp_rosenbrock._make_rosenbrock_kernel(field_key, int(D), precision))
    return np_dtype, wp_scalar, vecD, matDD, kernel


def _validate_batch(y0_np, params_np, np_dtype, dim):
    """Coerce + shape-check the ensemble inputs -> (y0[n,dim], params[n,dim], n).

    The stiff field consumes a per-trajectory PARAM VECTOR of length ``dim`` (k1,k2,k3
    for Robertson; ignored for linstiff). Param batches narrower than ``dim`` are
    right-padded with zeros (constant fields ignore the param anyway); wider batches are
    truncated to the first ``dim`` columns (the field only reads what it needs)."""
    y0_np = np.ascontiguousarray(np.asarray(y0_np, dtype=np_dtype))
    if y0_np.ndim != 2 or y0_np.shape[1] != dim:
        raise ValueError(f"y0 must have shape (n, {dim}); got {y0_np.shape}")
    n = y0_np.shape[0]
    params_np = np.asarray(params_np, dtype=np_dtype)
    if params_np.ndim == 1:
        params_np = params_np.reshape(n, 1)
    if params_np.shape[0] != n:
        raise ValueError(f"y0 has {n} trajectories but params has {params_np.shape[0]}")
    pcols = params_np.shape[1]
    if pcols < dim:  # right-pad (constant fields ignore the param)
        pad = np.zeros((n, dim - pcols), dtype=np_dtype)
        params_np = np.concatenate([params_np, pad], axis=1)
    elif pcols > dim:  # the field reads only the first dim entries
        params_np = params_np[:, :dim]
    return y0_np, np.ascontiguousarray(params_np), n


def _launch(y0_np, params_np, *, t0, t1, rtol, atol, dt0, max_steps, record,
            device="cpu", precision="float64", field_key="robertson", dim=3):
    """Drive one fused Rosenbrock23 launch over an ensemble (``wp.launch`` path).

    Args:
        y0_np:      (n, dim) initial states, coerced to the run precision (strict).
        params_np:  (n,) or (n, P) per-trajectory params; mapped to a (n, dim) param
            vector per trajectory (the field reads the leading entries it needs).
        record:     1 -> record each trajectory's accepted-dt sequence; 0 -> skip.
        precision:  "float64" (default) or "float32".
        field_key:  registered stiff field token (default "robertson").
        dim:        state dimension D (must match the field's registered dim).

    Returns:
        ``(y_final[n, dim], n_acc[n], n_rej[n], status[n], dts)`` where ``dts`` is the
        full ``[max_steps, n]`` numpy record (zero-padded past n_acc[j] in column j) when
        ``record`` else None. status!=0 rows of y_final are NaN'd HERE in the driver
        (the kernel writes the last reached state), mirroring ``warp_ode._launch``.
    """
    _require_warp()
    np_dtype, wp_scalar, vecD, matDD, kernel = _precision_spec(field_key, dim, precision)
    y0_np, params_np, n = _validate_batch(y0_np, params_np, np_dtype, int(dim))

    y0_wp = wp.array(y0_np, dtype=vecD, device=device)
    params_wp = wp.array(params_np, dtype=vecD, device=device)
    conf = wp.array(np.array([t0, t1, rtol, atol, dt0], dtype=np_dtype),
                    dtype=wp_scalar, device=device)
    y_out = wp.zeros(n, dtype=vecD, device=device)
    n_acc = wp.zeros(n, dtype=wp.int32, device=device)
    n_rej = wp.zeros(n, dtype=wp.int32, device=device)
    status = wp.zeros(n, dtype=wp.int32, device=device)
    rec_shape = (int(max_steps), n) if record else (1, 1)
    # MUST be wp.zeros (not wp.empty): the kernel writes only rows < n_acc[j] of column
    # j, and the replay's zero-padding contract (dt=0 == identity step) relies on the
    # rest being 0.0 — same contract as warp_ode._launch.
    dt_rec = wp.zeros(rec_shape, dtype=wp_scalar, device=device)

    wp.launch(
        kernel,
        dim=n,
        inputs=[y0_wp, params_wp, conf, int(max_steps), int(1 if record else 0),
                y_out, n_acc, n_rej, status, dt_rec],
        device=device,
    )
    wp.synchronize_device(device)

    y_final = y_out.numpy().reshape(n, int(dim)).copy()
    acc = n_acc.numpy().astype(np.int64)
    rej = n_rej.numpy().astype(np.int64)
    st = status.numpy().astype(np.int64)
    y_final[st != 0] = np.nan  # overflow rows NaN'd by the DRIVER, not the kernel
    dts = dt_rec.numpy().copy() if record else None
    return y_final, acc, rej, st, dts


def _field_for(problem: Problem):
    """Map a supported ``problem`` to its ``(field_key, dim)`` in the stiff field registry.

    Routing:
      * ``robertson``      -> ``("robertson", 3)``  (p = [k1, k2, k3])
      * ``hires``          -> ``("hires", 8)``  (autonomous, p is a length-1 placeholder)
      * a problem whose name matches a PRE-REGISTERED ``linstiff_*`` field -> that field.
        Constant-matrix linear-stiff fields are opt-in: register one with
        ``register_linstiff(key, A)`` (or ``register_field``) BEFORE routing. The library
        includes no synthetic stiff matrices of its own, so an unregistered ``linstiff_*`` name
        is simply unsupported.

    Returns None for unsupported problems (so ``supports`` is a thin truthiness check)."""
    if not _WARP_AVAILABLE:
        return None
    nm = problem.name
    if nm == "robertson":
        return ("robertson", 3)
    if nm == "hires":
        return ("hires", 8)
    # Any field registered up front routes by name: a linstiff_* field
    # (register_linstiff) OR a generated (field, analytic-Jacobian) pair registered by
    # register_jax_field (jax_field, stiff=True). Unregistered names stay unsupported.
    if nm in _warp_rosenbrock._FIELD_REGISTRY:
        return (nm, _warp_rosenbrock.field_dim(nm))
    return None


def supports(problem: Problem) -> bool:
    """True for every LOW-NVAR STIFF problem the fused Rosenbrock23 kernel has a
    registered (field, analytic-Jacobian) pair for: ``robertson``, ``hires``, and any
    registered ``linstiff_*`` problem. Everything else (and a machine without Warp) returns
    False, so the caller sees an unsupported problem instead of an exception."""
    return _WARP_AVAILABLE and _field_for(problem) is not None


def _params_of(problem: Problem, params) -> np.ndarray:
    """Map a problem's ``params[n, P]`` to the per-trajectory param vector the stiff
    field expects (the leading entries the field reads).

    * ``robertson``: the full ``params[:, :3]`` = [k1, k2, k3] (the field reads all 3).
    * ``hires``: the field is autonomous and ignores its param; pass the length-1
      placeholder through (``_launch`` right-pads it to the dim-8 param vector the field
      accepts-and-ignores). Same handling as ``linstiff_*``.
    * ``linstiff_*``: the field ignores its param; pass ``params`` through (``_launch``
      pads/truncates to dim). A 1-col placeholder batch is fine.
    """
    if not supports(problem):
        raise ValueError(
            f"warp_rosenbrock does not support problem {problem.name!r}; supported: "
            "'robertson', 'hires', and any 'linstiff_*' field you register first via "
            "register_linstiff(key, A).")
    return np.asarray(params, dtype=np.float64)


def solve(
    problem: Problem,
    y0: np.ndarray,
    params: np.ndarray,
    *,
    rtol: float,
    atol: float,
    device: str,
) -> SolveResult:
    """Solve a LOW-NVAR STIFF ensemble with the fused adaptive Rosenbrock23 kernel.

    Field routing: ``_field_for(problem)`` selects the registered field + dim;
    ``_params_of`` maps the batch's ``params`` to the per-trajectory param vector. The
    kernel is built once per ``(field_key, dim, precision)``.

    Precision dispatch: float32 ``y0`` selects the f32 kernel (y_final comes back
    float32); anything else runs the f64 kernel. No silent upcast.

    Device router: ``"cpu"`` (or any warp device string) drives ``wp.launch`` via
    ``_launch``. For ``"cuda"``/``"gpu"``: above ``_GPU_MAT_DIM_CEILING`` raises a
    "register"-flagged RuntimeError (the fused stiff engine is a low-NVAR engine);
    otherwise drives ``wp.launch(device="cuda")``. There is no jax_kernel FFI bridge
    for the stiff kernel.

    Raises:
        RuntimeError: device cuda/gpu with dim > _GPU_MAT_DIM_CEILING ("register"-flagged).
        ValueError: if the problem is not supported.
    """
    routed = _field_for(problem)
    if routed is None:
        raise ValueError(
            f"warp_rosenbrock does not support problem {problem.name!r}; supported: "
            "'robertson', 'hires', and any 'linstiff_*' field you register first via "
            "register_linstiff(key, A).")
    field_key, dim = routed

    y0_np = np.asarray(y0)
    precision = "float32" if y0_np.dtype == np.float32 else "float64"
    params_np = _params_of(problem, params)

    dt0 = (problem.t1 - problem.t0) / 100.0
    max_steps = 50000  # stiff solves take many steps (Robertson ~3700 accepted at 1e-6)
    common = dict(
        t0=problem.t0, t1=problem.t1, rtol=rtol, atol=atol, dt0=dt0,
        max_steps=max_steps, record=0, precision=precision,
        field_key=field_key, dim=dim,
    )

    if device in ("cuda", "gpu"):
        if dim > _GPU_MAT_DIM_CEILING:
            # The fused stiff kernel holds a DxD W and J + Gaussian-elimination workspace
            # per thread (O(D^2) registers/local memory). Above this limit the launch
            # blows the GPU budget and can abort the CUDA process uncatchably (works on
            # CPU, which has no such limit). Refuse before launching with a
            # "register"-flagged error so the caller can record a register-limit
            # failure (the fused stiff engine is a low-NVAR engine; high NVAR belongs to
            # the general stiff engines). The limit is conservative.
            raise RuntimeError(
                f"warp fused stiff GPU register limit: dim={dim} > {_GPU_MAT_DIM_CEILING}. "
                "The fused adaptive Rosenbrock23 kernel is a low-NVAR GPU engine "
                "(per-thread DxD matrices); for high NVAR use the general stiff engines the "
                "auto-router selects (engine='diffrax' for a forward solve, "
                "engine='rodas5p_replay' for gradients) or the fixed-step scan "
                "engine='fixed_step_imex'. (Forward sweep would abort the CUDA process.)"
            )
        y_final, n_acc, n_rej, _status, _dts = _launch(
            y0_np, params_np, device="cuda", **common)
    else:
        y_final, n_acc, n_rej, _status, _dts = _launch(
            y0_np, params_np, device=device, **common)
    return SolveResult(
        y_final=y_final,
        accepted_steps=n_acc,
        rejected_steps=n_rej,
        solver="warp_rosenbrock23_adaptive",
    )
