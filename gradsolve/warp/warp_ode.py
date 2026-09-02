"""warp_ode backend — fused adaptive Tsit5 NVIDIA Warp kernel.

One thread = one trajectory: the whole adaptive solve — 6 effective Tsit5 stages with
FSAL, WRMS error norm, I-controller accept/reject — runs inside a single Warp kernel
launch (no per-step host round-trips). The executable semantic spec is
``tsit5_adaptive`` in ``gradsolve/solvers/tsit5_replay.py``; the kernel reproduces it step for step
(same accept/reject sequence, same accepted-dt sequence, same controller updates),
checked by ``tests/test_adjoint_replay.py``.

Scope:
  - **Forward only, float64 + float32.** The kernel is built by the closure factory
    in ``gradsolve/warp/_warp_kernel.py`` (``_make_kernel(field_key,
    D, precision)``) — one adaptive-Tsit5 body that, per registered field at arbitrary
    small dimension, closes over a concrete field ``@wp.func`` + ``wp.types.vector(
    length=D, dtype=wp_scalar)``. No
    ``wp.Tape``/backward pass here — reverse mode is the record-and-replay adjoint
    (f64-only), which drives the general ``wp.launch`` path.
  - **CUDA path = JAX FFI:** ``solve(device="cuda"/"gpu")`` routes through warp's
    ``jax_kernel`` FFI bridge (``_launch_ffi``) so the fused kernel is a JAX op
    (jit-composable). The bridge is CUDA-only — it is never lowered on CPU
    (``device="cpu"`` keeps using ``wp.launch`` via ``_launch``). **FFI is
    vec3/Lorenz only** (general-D FFI is not implemented): ``_launch_ffi``
    raises ``NotImplementedError`` for D != 3. The general-field path that gradient
    workflows actually use is the ``wp.launch`` (``_launch``) path, which is general-D.
  - **Registered fields:** ``lorenz`` (D=3, sigma=10/beta=8/3, p=rho),
    ``vdp`` (D=2, p=mu), and ``linear_<D>`` (any D, the
    linear-ladder ``1.01*y`` field). ``supports()`` accepts ``diffeqgpu_lorenz`` /
    ``lorenz`` (-> lorenz field), ``vdp`` (-> vdp field), and ``linear_ladder_<D>``
    (-> linear_<D> field). The textual arithmetic order of each field matches the
    problem's ``f_np`` exactly (float64 association parity with the oracle).

  Parity contract (see ``_warp_kernel.py``):
    accept/reject sequence + counts are bit-identical to the numpy reference for every
    field at every D; Lorenz D=3 is bit-exact in value too (the 1e-10 reference
    test); general fields agree in value (dts/y_final) only up to FMA
    contraction (~1e-9 for D >~ 8 — clang/NVCC ``-ffp-contract=fast`` vs the un-fused
    numpy reference, not a logic bug; Warp exposes no hook to disable it).
  - **Record path:** when ``record == 1`` each trajectory's accepted-dt sequence is
    written into ``dt_rec[max_steps, n]`` — step index = row, trajectory = column, so
    writes coalesce across a warp. Callers slice column ``j`` by ``n_acc[j]``; rows at
    and beyond ``n_acc[j]`` are zero-padding. When ``record == 0`` a dummy ``[1, 1]``
    array is passed and never written.
  - **Status:** ``status == 0`` iff the trajectory reached ``t1`` within ``max_steps``;
    otherwise ``status == 1`` and the *driver* (not the kernel) NaNs that ``y_final``
    row.

The Tsitouras (2011) tableau is imported from ``gradsolve/solvers/tsit5_step.py``,
shared with the JAX solvers.

Import hygiene: this module is imported eagerly by the backend registry, so the warp
import is guarded — a warp-less env degrades cleanly (``supports()`` returns False
for everything, so callers see an unsupported problem; ``_launch`` raises RuntimeError
if reached directly) — and there is **no** ``wp.init()`` at import time: Warp >= 1.0
auto-initializes on the first array/launch inside ``_launch``, keeping
``import gradsolve`` free of side effects.

Backend-protocol wiring: ``supports()``/``solve()`` are live. Reverse mode
is the record-and-replay adjoint in ``gradsolve/warp/warp_replay.py``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

from gradsolve.base import SolveResult

if TYPE_CHECKING:  # gradsolve imports standalone; Problem is annotation-only
    from gradsolve.base import Problem

name = "warp_ode"

# GPU register-vector limit for the general-D fused kernel (see solve()). The kernel's
# fixed-length ``wp.types.vector(length=dim)`` is register/local-memory-backed per thread;
# above this dim a CUDA launch can abort the process uncatchably. Conservative — the fused
# engine's design point is low NVAR; high NVAR is served by the general engines (diffrax or
# the record-and-replay solvers). CPU has no such limit (stack-backed), so this guard only
# fires for device cuda/gpu.
_GPU_VEC_DIM_CEILING = 64

try:
    import warp as wp

    # The closure factory lives in a separate module without
    # ``from __future__ import annotations`` — Warp 1.14 codegen needs the kernel's
    # closed-over ``vecD`` annotation as a live object, not a PEP-563 string (see
    # _warp_kernel.py's module docstring for the full diagnosis). warp_ode.py keeps
    # its own future-import (its Problem/protocol hints want it); the kernel build
    # is delegated across that boundary.
    from gradsolve.warp import _warp_kernel

    _WARP_AVAILABLE = True
except ImportError:  # warp-less env: degrade to the stub instead of breaking the package
    wp = None
    _warp_kernel = None
    _WARP_AVAILABLE = False


def _precision_spec(field_key, D, precision, vecp_P=None):
    """(np_dtype, wp_scalar, wp_vec, kernel) for ``(field_key, D, precision)``.

    Thin adapter over ``_warp_kernel._make_kernel`` (lru_cache-keyed, so each unique
    ``(field, dim, precision)`` compiles at most once) that returns the same 4-tuple
    shape the earlier ``_PRECISIONS[precision]`` lookup did, in the order ``_launch``
    consumes (np_dtype, wp_scalar, wp_vec, kernel). Raises ValueError on an unknown
    precision (propagated from ``_make_kernel``).

    ``vecp_P`` (default ``None``, the built-in scalar-param path) is passed through to
    ``_make_kernel`` unchanged so a generated ``field(t, y, p: vecP)`` gets a
    length-``vecp_P`` vecP param kernel; ``None`` keeps the 3-arg call and byte-identical
    built-in kernels."""
    _require_warp()
    if vecp_P is None:
        kernel, vecD, wp_scalar, np_dtype = _warp_kernel._make_kernel(
            field_key, int(D), precision)
    else:
        kernel, vecD, wp_scalar, np_dtype = _warp_kernel._make_kernel(
            field_key, int(D), precision, int(vecp_P))
    return np_dtype, wp_scalar, vecD, kernel


def _require_warp():
    """Guard clause: raise loudly if NVIDIA Warp failed to import.

    Called at the top of every entry point that actually needs the fused kernel
    (``_precision_spec``, ``_launch``, ``_ffi_kernel``, ``_launch_ffi``) so a
    warp-less environment fails with a clear, actionable message at the call site
    instead of an ``AttributeError`` on ``wp``/``_warp_kernel`` (both ``None`` when
    the top-level ``import warp`` guard at module load fails). ``supports()`` stays
    side-effect-free (it checks ``_WARP_AVAILABLE`` directly, not this function), so
    a caller that never calls ``solve()`` is unaffected.

    Raises
    ------
    RuntimeError
        If ``_WARP_AVAILABLE`` is False (``import warp`` failed at module load) —
        the message points at ``pip install warp-lang`` / ``pip install 'gradsolve[cuda12]'``.
    """
    if not _WARP_AVAILABLE:
        raise RuntimeError(
            "warp-lang not installed: the fused adaptive Tsit5 kernel needs "
            "`pip install warp-lang` (or `pip install 'gradsolve[cuda12]'`)."
        )


def _validate_batch(y0_np, rho_np, np_dtype, dim=3):
    """Coerce + shape-check the ensemble inputs in the run precision -> (y0, p, n).

    ``dim`` is the field's state dimension D (default 3 keeps the Lorenz/FFI call
    sites unchanged); ``y0`` must be ``(n, dim)`` and ``rho_np`` the per-trajectory
    scalar param sequence of length n."""
    y0_np = np.ascontiguousarray(np.asarray(y0_np, dtype=np_dtype))
    if y0_np.ndim != 2 or y0_np.shape[1] != dim:
        raise ValueError(f"y0 must have shape (n, {dim}); got {y0_np.shape}")
    n = y0_np.shape[0]
    rho_np = np.ascontiguousarray(np.asarray(rho_np, dtype=np_dtype).reshape(-1))
    if rho_np.shape[0] != n:
        raise ValueError(f"y0 has {n} trajectories but rho has {rho_np.shape[0]}")
    return y0_np, rho_np, n


def _record_handoff(dt_rec, n_acc, *, device, transport):
    """Trimmed ``[S, n]`` record handoff without the full-buffer host
    numpy round-trip (which otherwise dominates the driver cost at large n).

    ``S = max(n_acc)`` is computed on host from the (tiny, already-fetched) int
    array. ``dt_rec`` is row-major ``[max_steps, n]``, so the leading-rows slice
    ``dt_rec[:S]`` is contiguous and sliceable without a copy.

    Padding-correctness note (replay contract): rows >= n_acc[j] of column j
    must be exactly 0.0 (zero dt == identity step in the replay). The kernel
    only writes rows < n_acc[j], so this holds only because ``_launch``
    ``wp.zeros``-allocates the record buffer on every call (a device memset is
    cheap). Do not cache/reuse the buffer across launches without
    re-zeroing it.

    transport="dlpack" (primary): hand ``dt_rec[:S]`` to JAX zero-copy via the
      ``__dlpack__`` protocol (``jnp.from_dlpack``); the DLPack deleter keeps
      the warp buffer alive for the jax array's lifetime. Returns a jax.Array
      ``[S, n]`` living on dt_rec's device — CUDA stays on CUDA (no D2H at all);
      CPU bridges to CPU jax, which is what makes this path testable without a GPU.
    transport="pinned" (fallback): ``wp.copy`` only the trimmed ``[S, n]``
      slice into a host buffer (a small fraction of the full record buffer).
      Pinned allocation needs a CUDA context, so it is requested only when the
      source device is CUDA (plain host memory otherwise). Returns a numpy view
      of that buffer (the view's ``.base`` keeps the warp array alive).
    """
    if transport not in ("dlpack", "pinned"):
        raise ValueError(
            f"record_transport must be 'dlpack' or 'pinned', got {transport!r}")
    s = int(n_acc.max()) if n_acc.size else 0
    n = dt_rec.shape[1]
    if s == 0:  # degenerate (no accepted steps anywhere): nothing to hand off
        import jax.numpy as jnp

        np_dt = np.float64 if dt_rec.dtype == wp.float64 else np.float32
        empty = np.zeros((0, n), dtype=np_dt)
        return jnp.asarray(empty) if transport == "dlpack" else empty
    trimmed = dt_rec[:s]  # leading-rows slice of a row-major array: contiguous
    if transport == "dlpack":
        import jax.numpy as jnp

        return jnp.from_dlpack(trimmed)
    src_is_cuda = wp.get_device(device).is_cuda
    host_buf = wp.empty((s, n), dtype=dt_rec.dtype, device="cpu",
                        pinned=src_is_cuda)  # pinned alloc needs a CUDA context
    wp.copy(host_buf, trimmed)
    # D2H into pinned memory is ASYNCHRONOUS — synchronize before reading.
    wp.synchronize_device(device)
    return host_buf.numpy()


def _launch(y0_np, rho_np, *, t0, t1, rtol, atol, dt0, max_steps, record,
            device="cpu", precision="float64", record_mode="host",
            record_transport="dlpack", field_key="lorenz", dim=3):
    """Drive one fused-kernel launch over an ensemble (``wp.launch`` path).

    GENERAL-FIELD: ``field_key`` selects the registered field
    (``_warp_kernel._FIELD_REGISTRY``) and ``dim`` its state dimension D; the kernel
    is built once per ``(field_key, D, precision)`` by ``_precision_spec`` ->
    ``_warp_kernel._make_kernel`` (lru_cache). Defaults ``field_key="lorenz", dim=3``
    match the Lorenz call sites and the FFI path.

    Args:
        y0_np:  (n, dim) initial states, coerced to the run precision (strict —
            mis-shaped batches raise).
        rho_np: (n,) per-trajectory scalar param p the field expects (rho for
            lorenz, mu for vdp, ignored placeholder for linear; must match y0's n).
        record: 1 -> record each trajectory's accepted-dt sequence; 0 -> skip.
        precision: "float64" (default) or "float32" (vecDf/float32 kernel; conf and
            dt_rec also float32).
        record_mode: "host" (default: full-buffer device-to-host copy, then a numpy
            copy) or "device" (trimmed ``[S, n]`` handoff via ``_record_handoff`` — no
            full-buffer host round-trip).
        record_transport: device-mode transport, "dlpack" (zero-copy, primary)
            or "pinned" (trimmed pinned-host copy, fallback). Ignored in host
            mode.
        field_key: registered field token (default "lorenz").
        dim: state dimension D (default 3; must match the field's registered dim).

    Returns:
        (y_final[n, dim], n_acc[n], n_rej[n], status[n], dts)
        where y_final (and dts) carry the run precision's dtype; status!=0 rows
        of y_final are NaN'd here in the driver (kernel writes the last reached
        state). ``dts`` when ``record``:
          * record_mode="host":   full ``[max_steps, n]`` numpy array
            (zero-padded past n_acc[j] in column j) — unchanged contract;
          * record_mode="device": TRIMMED ``[S, n]`` with ``S = max(n_acc)`` —
            a jax.Array on dt_rec's device (dlpack) or a numpy pinned-buffer
            view (pinned);
        else None.
    """
    _require_warp()
    if record_mode not in ("host", "device"):
        raise ValueError(
            f"record_mode must be 'host' or 'device', got {record_mode!r}")
    if record_transport not in ("dlpack", "pinned"):
        raise ValueError(
            f"record_transport must be 'dlpack' or 'pinned', got {record_transport!r}")
    if field_key in _warp_kernel._VECP_FIELDS:
        # Generated field(t, y, p: vecP): the per-trajectory param is a length-P vector,
        # not a scalar. Read the vecP length off the caller's (n, P) params (the registry
        # entry carries no n_params) and build a wp.array(dtype=vecP), mirroring how the
        # stiff _launch builds its wp.array(params_np, dtype=vecD).
        # NOTE: vecP length is read from params.shape[1] at launch; built-in scalar
        # fields keep the scalar path below.
        params_2d = np.ascontiguousarray(np.asarray(rho_np))
        if params_2d.ndim != 2:
            raise ValueError(
                f"generated vecP field {field_key!r} needs a (n, P) param array; "
                f"got shape {params_2d.shape}")
        P = int(params_2d.shape[1])
        np_dtype, wp_scalar, wp_vec, kernel = _precision_spec(
            field_key, dim, precision, vecp_P=P)
        y0_np = np.ascontiguousarray(np.asarray(y0_np, dtype=np_dtype))
        if y0_np.ndim != 2 or y0_np.shape[1] != int(dim):
            raise ValueError(f"y0 must have shape (n, {dim}); got {y0_np.shape}")
        n = y0_np.shape[0]
        if params_2d.shape[0] != n:
            raise ValueError(
                f"y0 has {n} trajectories but params has {params_2d.shape[0]}")
        vecP = wp.types.vector(length=max(P, 1), dtype=wp_scalar)
        y0_wp = wp.array(y0_np, dtype=wp_vec, device=device)
        rho_wp = wp.array(params_2d.astype(np_dtype), dtype=vecP, device=device)
    else:
        np_dtype, wp_scalar, wp_vec, kernel = _precision_spec(field_key, dim, precision)
        y0_np, rho_np, n = _validate_batch(y0_np, rho_np, np_dtype, dim=int(dim))
        y0_wp = wp.array(y0_np, dtype=wp_vec, device=device)
        rho_wp = wp.array(rho_np, dtype=wp_scalar, device=device)
    conf = wp.array(np.array([t0, t1, rtol, atol, dt0], dtype=np_dtype),
                    dtype=wp_scalar, device=device)
    y_out = wp.zeros(n, dtype=wp_vec, device=device)
    n_acc = wp.zeros(n, dtype=wp.int32, device=device)
    n_rej = wp.zeros(n, dtype=wp.int32, device=device)
    status = wp.zeros(n, dtype=wp.int32, device=device)
    rec_shape = (int(max_steps), n) if record else (1, 1)
    # MUST be wp.zeros (not wp.empty) on every call: the kernel only writes rows
    # < n_acc[j] of column j, and the replay's zero-padding contract (zero dt ==
    # identity step) relies on the rest of the buffer being 0.0 — see
    # _record_handoff's padding-correctness note before "optimizing" this.
    dt_rec = wp.zeros(rec_shape, dtype=wp_scalar, device=device)

    wp.launch(
        kernel,
        dim=n,
        inputs=[y0_wp, rho_wp, conf, int(max_steps), int(1 if record else 0),
                y_out, n_acc, n_rej, status, dt_rec],
        device=device,
    )
    wp.synchronize_device(device)

    # n_acc/n_rej/status are tiny int32 arrays: host fetch in BOTH record modes.
    y_final = y_out.numpy().reshape(n, int(dim)).copy()
    acc = n_acc.numpy().astype(np.int64)
    rej = n_rej.numpy().astype(np.int64)
    st = status.numpy().astype(np.int64)
    y_final[st != 0] = np.nan  # overflow rows NaN'd by the DRIVER, not the kernel
    if not record:
        dts = None
    elif record_mode == "host":
        dts = dt_rec.numpy().copy()  # host mode: full [max_steps, n] buffer copied through host
    else:
        dts = _record_handoff(dt_rec, acc, device=device,
                              transport=record_transport)
    return y_final, acc, rej, st, dts


# ---------------------------------------------------------------------------------
# CUDA FFI path: the fused kernel as a JAX op via warp's jax_kernel bridge.
# Reference pattern: warp's jax_kernel FFI bridge (jit-hoisted cached runner;
# per-call launch_dims/output_dims). The bridge is documented
# CUDA-only: on a CPU-only build the wrapper CONSTRUCTS fine but lowering raises
# "NOT_FOUND: No FFI handler registered ... platform Host" — so _launch_ffi refuses
# up-front on a machine without CUDA and the device router never lowers FFI on CPU.
#
# No array2d fallback is needed — warp's FFI bridge maps vector-dtype arrays
# natively (wp.array(dtype=wp.vec3d) of shape (n,) <-> JAX (n, 3) float64;
# output_dims wants the WARP shape (n,), the trailing vec dim is appended by the
# bridge). The gpu-marked equivalence tests verify this on a GPU.
# ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _ffi_kernel(precision):
    """Build (once per precision) the JAX-callable FFI wrapper around the kernel.

    Construction needs warp but NOT CUDA (verified without a CUDA device); the jax_kernel import is
    deferred here so importing this module stays side-effect-free on installations
    without Warp or jax-ffi.

    FFI scope: the bridge is vec3/Lorenz only — it wraps the
    ``("lorenz", 3, precision)`` kernel. General-D FFI is not implemented
    (``_launch_ffi`` guards D != 3); the general-field path is the ``wp.launch``
    (``_launch``) path, which the replay gradient workflow uses.
    """
    _require_warp()
    try:
        from warp import jax_kernel  # top-level since warp 1.14
    except ImportError:  # pragma: no cover - older warp builds
        from warp.jax_experimental import jax_kernel  # deprecated shim, removed in 1.16
    kernel = _precision_spec("lorenz", 3, precision)[3]
    return jax_kernel(kernel, num_outputs=5, enable_backward=False)


@functools.lru_cache(maxsize=None)
def _ffi_runner(n, max_steps, record, precision):
    """jit-hoisted FFI runner, COMPILED ONCE per (n, max_steps, record, precision).

    Returns a ``jax.jit``-ed ``run(y0[n,3], rho[n], conf[5]) -> (y_out[n,3], n_acc[n],
    n_rej[n], status[n], dt_rec)``. Scalars travel in the ``conf`` array (the FFI
    bridge is array-oriented); ``max_steps``/``record`` pass as plain ints (static,
    closed over — never traced). Explicit ``launch_dims=n`` (one thread = one
    trajectory) and explicit ``output_dims`` in WARP shapes ((n,) for the vec3 y_out).
    """
    import jax

    wrapped = _ffi_kernel(precision)
    rec_rows, rec_cols = (int(max_steps), int(n)) if record else (1, 1)
    out_dims = {
        "y_out": (int(n),),     # vec3 array: JAX side is (n, 3)
        "n_acc": (int(n),),
        "n_rej": (int(n),),
        "status": (int(n),),
        "dt_rec": (rec_rows, rec_cols),
    }

    def run(y0, rho, conf):
        return wrapped(y0, rho, conf, int(max_steps), int(1 if record else 0),
                       launch_dims=int(n), output_dims=out_dims)

    return jax.jit(run)


def _launch_ffi(y0_np, rho_np, *, t0, t1, rtol, atol, dt0, max_steps, record,
                precision="float64", dim=3):
    """Drive one fused solve through the jax_kernel FFI bridge (CUDA-only, vec3/Lorenz).

    Same contract as ``_launch`` — identical argument meaning, identical return
    tuple (numpy outputs, status!=0 rows of y_final NaN'd by this driver) — so the
    gpu equivalence tests can compare the two paths element-for-element. Must NEVER
    be lowered on CPU: raises RuntimeError up-front when no CUDA device is visible.

    Scope: this path is vec3/Lorenz only. ``dim``
    must be 3 — general-D FFI is not implemented (the ``output_dims`` and the
    bridge's vector-dtype mapping would have to be parameterized per D). For general
    fields use ``device='cpu'`` (the ``wp.launch`` path via ``_launch``), which is
    general-D and is what the record-and-replay gradient workflow drives anyway.

    Precision note: the f64 path requires the calling process to have jax x64
    enabled (``gradsolve`` enables it at import); in a ``GRADSOLVE_X64=0`` /
    float32 process the f64 arrays would downcast to f32 and the
    bridge's dtype check raises loudly.

    record=1 note: no library caller reaches this today (``solve()`` is
    record=0 and ``warp_replay.record`` drives ``_launch``); the record=1 leg of
    the f64 gpu equivalence test covers this path anyway.
    """
    _require_warp()
    if int(dim) != 3:
        raise NotImplementedError(
            "warp_ode FFI path is vec3/Lorenz-only; general-D FFI is a gpu follow-on "
            "— use device='cpu' (wp.launch) for general fields, which the replay "
            f"gradient path uses anyway (got dim={dim})."
        )
    if not wp.is_cuda_available():
        raise RuntimeError(
            "warp_ode FFI path is CUDA-only: warp's jax_kernel registers its XLA "
            "FFI handler for the CUDA platform only, and lowering on CPU fails with "
            "'NOT_FOUND: No FFI handler registered ... platform Host' (a documented "
            "property of warp's jax_kernel bridge). Use device='cpu' "
            "(the wp.launch path) instead."
        )
    import jax
    import jax.numpy as jnp

    np_dtype = _precision_spec("lorenz", 3, precision)[0]
    y0_np, rho_np, n = _validate_batch(y0_np, rho_np, np_dtype, dim=3)
    conf_np = np.array([t0, t1, rtol, atol, dt0], dtype=np_dtype)

    run = _ffi_runner(int(n), int(max_steps), int(1 if record else 0), precision)
    outs = jax.block_until_ready(
        run(jnp.asarray(y0_np), jnp.asarray(rho_np), jnp.asarray(conf_np)))
    y_out, n_acc, n_rej, status, dt_rec = (np.asarray(o) for o in outs)

    y_final = np.array(y_out, dtype=np_dtype).reshape(n, 3)
    acc = n_acc.astype(np.int64)
    rej = n_rej.astype(np.int64)
    st = status.astype(np.int64)
    y_final[st != 0] = np.nan  # same driver NaN policy as _launch
    if record:
        dts = np.array(dt_rec, dtype=np_dtype)
        # XLA does NOT zero-initialize output buffers (unlike wp.zeros in _launch),
        # and the kernel only writes rows < n_acc[j] of column j — enforce the
        # documented zero-padding contract here in the driver.
        dts[np.arange(dts.shape[0])[:, None] >= acc[None, :]] = 0.0
    else:
        dts = None
    return y_final, acc, rej, st, dts


def _field_for(problem: Problem):
    """Map a supported ``problem`` to its ``(field_key, dim)`` in the field registry.

    Routing:
      * ``diffeqgpu_lorenz`` / ``lorenz`` -> ``("lorenz", 3)``  (rho the param)
      * ``vdp``                           -> ``("vdp", 2)``     (mu the param)
      * ``linear_ladder_<D>``             -> ``("linear_<D>", D)`` (param ignored)

    Returns None for unsupported problems (so ``supports`` is a thin truthiness check).
    """
    nm = problem.name
    if nm in ("diffeqgpu_lorenz", "lorenz"):
        return ("lorenz", 3)
    if nm == "vdp":
        return ("vdp", 2)
    if nm.startswith("lorenz96_"):
        try:
            d = int(nm.rsplit("_", 1)[1])
        except ValueError:
            return None
        if d >= 4:
            return (f"lorenz96_{d}", d)
    if nm.startswith("linear_ladder_"):
        try:
            d = int(nm.rsplit("_", 1)[1])
        except ValueError:
            return None
        if d > 0:
            return (f"linear_{d}", d)
    # A field registered by register_jax_field (jax_field) under this exact name routes to
    # the fused kernel exactly like a built-in. Plain dict membership (no lazy synthesis):
    # the generated field is written into _FIELD_REGISTRY at registration.
    entry = _warp_kernel._FIELD_REGISTRY.get(nm)
    if entry is not None:
        return (nm, entry[1])
    return None


def supports(problem: Problem) -> bool:
    """True for every problem the fused kernel has a registered field for.

    Scope: ``diffeqgpu_lorenz`` / ``lorenz`` (-> lorenz field, sigma=10/
    beta=8/3, rho per-trajectory), ``vdp`` (-> vdp field, mu per-trajectory), and
    ``linear_ladder_<D>`` for any D (-> linear_<D> field). Everything else returns
    False. Without Warp this is False for everything, so callers (solve and grad
    alike) see an unsupported problem instead of a RuntimeError.
    """
    return _WARP_AVAILABLE and _field_for(problem) is not None


def _param_of(problem: Problem, params) -> np.ndarray:
    """Map a problem's ``params[n, P]`` to the per-trajectory scalar the field expects.

    Single source of the param-mapping rules (shared by ``solve()`` and the
    record-and-replay adjoint in ``gradsolve/warp/warp_replay.py``):

    * Lorenz (``diffeqgpu_lorenz`` / ``lorenz``): delegates to ``_rho_of`` (which also
      enforces the sigma=10 / beta=8/3 guard on the 3-col ``lorenz`` batch).
    * ``vdp``: ``mu = params[:, 0]``.
    * ``linear_ladder_<D>``: the field ignores its scalar param — return zeros[n] as a
      placeholder of the right shape (kept for the uniform kernel signature).
    """
    # Gate on the field-mapping rule only: this helper is pure NumPy and is reused by the
    # CPU jax.ffi bridge (gradsolve/cuda/_ffi_bridge.py), which must work without Warp.
    if _field_for(problem) is None:
        raise ValueError(
            f"warp_ode does not support problem {problem.name!r}; supported: "
            "'diffeqgpu_lorenz', 'lorenz', 'vdp', 'linear_ladder_<D>'."
        )
    nm = problem.name
    # Warp-less environments load the kernel module as None; no generated vecP fields exist there.
    vecp_fields = getattr(_warp_kernel, "_VECP_FIELDS", ()) if _warp_kernel is not None else ()
    if nm in vecp_fields:
        # Generated vecP field: the field takes a length-P vector param, so hand _launch
        # the full (n, P) batch (it reads P off params.shape[1] and builds a
        # wp.array(dtype=vecP)); no scalar reduction happens for these fields.
        return np.asarray(params, dtype=np.float64)
    if nm in ("diffeqgpu_lorenz", "lorenz"):
        return _rho_of(problem, params)
    params_np = np.asarray(params, dtype=np.float64)
    if nm == "vdp":
        return params_np[:, 0]  # mu
    if nm.startswith("lorenz96_"):
        return params_np[:, 0]  # F
    # linear_ladder_<D>: param ignored by the field; placeholder of the right length.
    n = params_np.shape[0] if params_np.ndim >= 1 else int(params_np.size)
    return np.zeros(n, dtype=np.float64)


def _rho_of(problem: Problem, params) -> np.ndarray:
    """Map a problem's ``params[n, P]`` to the kernel's per-trajectory ``rho[n]``.

    Single source of the param-mapping rules (shared by ``solve()`` and the
    record-and-replay adjoint in ``gradsolve/warp/warp_replay.py``):

    * ``diffeqgpu_lorenz`` (n_params=1): ``rho = params[:, 0]``.
    * ``lorenz`` (n_params=3, columns [sigma, beta, rho]): ``rho = params[:, 2]``;
      sigma and beta are validated to equal 10 and 8/3 respectively — any
      hand-rolled batch that deviates raises ValueError.

    Lorenz-only (the sigma/beta guard is Lorenz-specific); general fields use the
    ``_param_of`` dispatcher, which delegates here only for the Lorenz problems.
    """
    if problem.name not in ("diffeqgpu_lorenz", "lorenz"):
        raise ValueError(
            f"_rho_of is Lorenz-only; got {problem.name!r}. Use _param_of for the "
            "general per-trajectory scalar."
        )
    params_np = np.asarray(params, dtype=np.float64)
    if problem.name == "diffeqgpu_lorenz":
        # n_params=1: params[:, 0] is rho
        return params_np[:, 0]
    # lorenz n_params=3: columns are [sigma, beta, rho]
    sigma_col = params_np[:, 0]
    beta_col = params_np[:, 1]
    rho_np = params_np[:, 2]
    if not np.allclose(sigma_col, 10.0):
        raise ValueError(
            f"warp_ode kernel hard-codes sigma=10, beta=8/3; got "
            f"sigma values in [{sigma_col.min():.4g}, {sigma_col.max():.4g}]"
        )
    if not np.allclose(beta_col, 8.0 / 3.0):
        raise ValueError(
            f"warp_ode kernel hard-codes sigma=10, beta=8/3; got "
            f"beta values in [{beta_col.min():.4g}, {beta_col.max():.4g}]"
        )
    return rho_np


def solve(
    problem: Problem,
    y0: np.ndarray,
    params: np.ndarray,
    *,
    rtol: float,
    atol: float,
    device: str,
) -> SolveResult:
    """Solve an ensemble with the fused adaptive Warp Tsit5 kernel.

    Field routing: ``_field_for(problem)`` selects the registered field +
    dim; ``_param_of`` maps the batch's ``params`` to the per-trajectory scalar (rho for
    Lorenz with the sigma=10/beta=8/3 guard, mu for vdp, ignored for linear). The kernel
    is built once per ``(field_key, dim, precision)``.

    Precision dispatch: float32
    ``y0`` selects the f32 kernel and ``y_final`` comes back float32; anything else
    runs the f64 kernel and comes back float64. No silent upcast.

    Device router: ``"cpu"`` (or any warp device string) drives ``wp.launch`` via
    ``_launch`` (general-D). For ``"cuda"``/``"gpu"``: vec3/Lorenz (dim==3) routes
    through the jax_kernel FFI path (``_launch_ffi``, JAX-composable); any other
    dimension routes through ``wp.launch(device="cuda")`` (``_launch``, general-D).
    The general-D FFI bridge (JAX-composable at arbitrary D) is not implemented.

    Raises:
        RuntimeError: if ``device`` is ``"cuda"``/``"gpu"`` (dim==3) but no CUDA device
            is visible (the FFI handler is registered for the CUDA platform only); or
            (dim!=3) if warp sees no CUDA device.
        ValueError: if the problem is not supported, or if ``lorenz`` params
            have sigma ≠ 10 or beta ≠ 8/3.
    """
    routed = _field_for(problem)
    if routed is None:
        raise ValueError(
            f"warp_ode does not support problem {problem.name!r}; supported: "
            "'diffeqgpu_lorenz', 'lorenz', 'vdp', 'linear_ladder_<D>'."
        )
    field_key, dim = routed

    y0_np = np.asarray(y0)
    precision = "float32" if y0_np.dtype == np.float32 else "float64"
    rho_np = _param_of(problem, params)  # validates in f64; _launch casts to run dtype

    dt0 = (problem.t1 - problem.t0) / 100.0
    max_steps = 4096
    common = dict(
        t0=problem.t0, t1=problem.t1, rtol=rtol, atol=atol, dt0=dt0,
        max_steps=max_steps, record=0, precision=precision,
    )

    if device in ("cuda", "gpu"):
        if dim == 3:
            # vec3/Lorenz: the jax_kernel FFI path (JAX-composable).
            y_final, n_acc, n_rej, _status, _dts = _launch_ffi(
                y0_np, rho_np, dim=dim, **common)
        elif dim > _GPU_VEC_DIM_CEILING:
            # The general-D fused kernel uses a fixed-length REGISTER vector type
            # ``wp.types.vector(length=dim)``; the adaptive solve holds ~12 of them live
            # (k1..k7, y, y5, e, ...). Beyond this limit the per-thread footprint blows
            # the GPU register/local-memory budget and the launch aborts the process
            # uncatchably on CUDA (it works on CPU, which has no such limit). Refuse
            # before launching with a "register"-flagged error so the caller can
            # record a register-limit failure (the fused kernel is a low-NVAR engine;
            # high NVAR belongs to the JAX scan/replay engines). The limit is conservative.
            raise RuntimeError(
                f"warp fused GPU register-vector limit: dim={dim} > {_GPU_VEC_DIM_CEILING}. "
                "The fused adaptive kernel is a low-NVAR GPU engine (register vectors); "
                "use engine='auto' or a JAX scan/replay engine (e.g. engine='tsit5_replay' "
                "or engine='fixed_step_tsit5') for high NVAR. (Forward sweep would abort the "
                "CUDA process.)"
            )
        else:
            # general-D fused forward on GPU via wp.launch (the general-D FFI bridge
            # is not implemented; a plain forward solve needs no JAX-composability).
            y_final, n_acc, n_rej, _status, _dts = _launch(
                y0_np, rho_np, device="cuda", field_key=field_key, dim=dim, **common)
    else:
        y_final, n_acc, n_rej, _status, _dts = _launch(
            y0_np, rho_np, device=device, field_key=field_key, dim=dim, **common)
    # Overflow rows (status != 0) are already NaN'd by _launch — do not mask further;
    # the caller handles nonfinite results downstream.
    return SolveResult(
        y_final=y_final,
        accepted_steps=n_acc,
        rejected_steps=n_rej,
        solver="warp_tsit5_adaptive",
    )
