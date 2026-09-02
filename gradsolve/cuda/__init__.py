"""gradsolve CUDA lane — hand-written CUDA forward-only engines via ``jax.ffi``.

Sibling of ``gradsolve/warp/`` (the fused/record-replay lane), kept separate so the warp
lane stays untouched. Currently: ``cuda_tsit5`` (forward-only fused adaptive Tsit5, the
fast forward-throughput lane). No import-time side effects — the engine module is
import-safe on a machine without nvcc/CUDA, where ``supports()`` returns False and the
router falls back to a scan engine.

Pipeline (codegen -> build -> ffi), one module per stage, keyed throughout on
``(field_key, D, precision[, platform])``:

1. ``_fields.py`` — ``emit_field(field_key, D)`` returns the C++ ``field(...)`` body for one
   registered problem (the RHS, transcribed term-for-term from the problem's ``f_np``).
2. ``_codegen.py`` — ``emit_cu(field_key, D, precision)`` fills
   ``kernels/tsit5_template.cu.in`` with that field body, the Tsit5 tableau (from
   ``gradsolve.solvers.tsit5_step``, shared with the JAX solvers), the scalar type, and the
   dimension literal, producing one complete ``.cu``/``.cpp`` source string.
3. ``_build.py`` — compiles that source into a shared lib (``lru_cache``-memoized):
   ``build_cpu_so`` (host c++/clang++, the CPU validation target) or ``build_cuda_so``
   (nvcc, the production GPU target); both cached under ``_cache_dir()``
   (``$GRADSOLVE_CUDA_CACHE`` / ``~/.cache/gradsolve_cuda``), keyed by a source hash so identical
   builds are never recompiled.
4. ``_ffi_bridge.py`` — ``ctypes``-loads the ``.so``, registers its ``Tsit5Fwd`` symbol as a
   ``jax.ffi`` target under ``_target_name(...)``, and wraps it in ``jax.ffi.ffi_call`` via
   ``make_runner`` / ``solve_ffi`` (transposed-SoA on the wire; scalar tols + max_steps as
   static numpy attrs; no autodiff rule registered, so ``jax.grad`` through it raises — the
   forward-only contract).
5. ``cuda_tsit5.py`` — the public engine surface (``supports``/``solve``) that the router
   (``gradsolve/dispatch.py``) calls; ``solve`` calls ``_ffi_bridge.solve_ffi``.
"""
from gradsolve.cuda import (
    cuda_rosenbrock23,  # noqa: F401  (stiff forward-only lane; disabled by default)
    cuda_tsit5,  # noqa: F401  (re-export for gradsolve.cuda.cuda_tsit5)
)
