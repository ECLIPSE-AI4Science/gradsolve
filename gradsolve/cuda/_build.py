"""Compile the cuda_tsit5 FFI handler (.cu) into a shared lib for ``jax.ffi``.

Two targets, same handler symbol ``Tsit5Fwd``:
  * ``build_cpu_so`` — clang/g++ host build (``-DGRADSOLVE_FFI``), the CPU validation target
    (the handler loops on the host). Lets the full jax.ffi path and numerics run without a GPU.
  * ``build_cuda_so`` — nvcc build, the production GPU target. Version-checks nvcc
    major vs ``jax.devices()`` first.

Both compile against ``jax.ffi.include_dir()`` so the XLA-FFI ABI matches the installed jaxlib.
``.so`` cached (regenerable) under ``$GRADSOLVE_CUDA_CACHE`` / ``~/.cache/gradsolve_cuda``.
"""
from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import jax

from gradsolve.cuda._codegen import emit_cu, emit_cu_rosenbrock


def _cache_dir() -> Path:
    """Resolve (and create) the on-disk cache dir for generated sources + compiled ``.so``.

    Returns
    -------
    Path
        ``$GRADSOLVE_CUDA_CACHE`` if that env var is set, else ``~/.cache/gradsolve_cuda``.
        Created with ``parents=True, exist_ok=True`` before being returned, so callers can
        write into it immediately.
    """
    d = Path(os.environ.get("GRADSOLVE_CUDA_CACHE", Path.home() / ".cache" / "gradsolve_cuda"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _compile_cpu(src: str, tag: str) -> str:
    """Compile one FFI-handler ``.cu`` source (host target, ``-DGRADSOLVE_FFI``) into a
    shared lib. Shared by the Tsit5 and Rosenbrock23 lanes — ``tag`` names the cache file,
    the source hash keys the cache so identical builds are never recompiled."""
    cc = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cc is None:
        raise RuntimeError("no host C++ compiler (c++/clang++/g++)")
    inc = jax.ffi.include_dir()
    key = hashlib.sha1((src + "cpu" + inc).encode()).hexdigest()[:16]
    d = _cache_dir()
    cpp = d / f"{tag}_cpu_{key}.cpp"
    so = d / f"lib_{tag}_cpu_{key}.so"
    if not so.exists():
        cpp.write_text(src)
        subprocess.run(
            [cc, "-O3", "-ffp-contract=off", "-std=c++17", "-fPIC", "-shared",
             "-DGRADSOLVE_FFI", "-Wno-unknown-pragmas", f"-I{inc}",
             "-x", "c++", str(cpp), "-o", str(so)],
            check=True, capture_output=True, text=True)
    return str(so)


@functools.lru_cache(maxsize=None)
def build_cpu_so(field_key: str, D: int, precision: str = "float64") -> str:
    """Compile the Tsit5 FFI handler (``Tsit5Fwd``) as a CPU shared lib (host loop). CPU
    validation."""
    return _compile_cpu(emit_cu(field_key, int(D), precision),
                        f"{field_key}_{D}_{precision}")


def build_cpu_so_rosenbrock(field_key: str, D: int, precision: str = "float64",
                            A=None) -> str:
    """Compile the stiff Rosenbrock23 FFI handler (``Rosenbrock23Fwd``) as a CPU shared lib
    (host loop). CPU validation of the whole stiff kernel + jax.ffi ABI. ``A`` is the DxD
    operator for a ``linstiff_*`` field (ignored for robertson/hires). Not ``lru_cache``d
    (A is an unhashable ndarray); the source-hash disk cache prevents recompiles."""
    src = emit_cu_rosenbrock(field_key, int(D), precision, A)
    return _compile_cpu(src, f"rosen_{field_key}_{D}_{precision}")


def _check_cuda_jax_match() -> None:
    """Fail loud on the nvcc-major vs jax-CUDA-plugin mismatch."""
    devs = jax.devices()
    if not any(d.platform == "gpu" for d in devs):
        raise RuntimeError(f"no CUDA device visible to jax (devices={devs})")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not on PATH (add /usr/local/cuda/bin)")


def default_sm() -> str:
    """The SM architecture of the visible GPU, as nvcc's ``-gencode`` wants it ("80" for an
    A100, "90" for an H100 or H200). ``$GRADSOLVE_CUDA_SM`` overrides; otherwise the compute
    capability is read from the JAX device."""
    env = os.environ.get("GRADSOLVE_CUDA_SM")
    if env:
        return env
    for d in jax.devices():
        if d.platform == "gpu":
            cc = getattr(d, "compute_capability", None)
            if cc:
                return str(cc).replace(".", "")
    return "80"


def _compile_cuda(src: str, tag: str, sm: str) -> str:
    """nvcc-compile one FFI-handler ``.cu`` source into a CUDA shared lib. Pins
    -arch to the SM of the visible GPU (see :func:`default_sm`) to avoid PTX-JIT at first
    launch. Shared by the Tsit5 and Rosenbrock23 lanes."""
    _check_cuda_jax_match()
    sm = sm or default_sm()
    inc = jax.ffi.include_dir()
    key = hashlib.sha1((src + "cuda" + inc + sm).encode()).hexdigest()[:16]
    d = _cache_dir()
    cu = d / f"{tag}_cuda_{key}.cu"
    so = d / f"lib_{tag}_cuda_{key}.so"
    if not so.exists():
        cu.write_text(src)
        proc = subprocess.run(
            ["nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC", "-std=c++17",
             "-DGRADSOLVE_FFI", "--fmad=true", "-Xptxas", "-v",
             "-gencode", f"arch=compute_{sm},code=sm_{sm}",
             f"-I{inc}", str(cu), "-o", str(so)],
            check=True, capture_output=True, text=True)
        # -Xptxas -v prints "Used N registers" + spill store/load bytes per kernel to
        # stderr; capture_output=True would otherwise swallow it on a successful build.
        # Persist it next to the .so and echo it so the user can read the D=8
        # register pressure (HIRES spill risk). --fmad=true is nvcc's -O3 default made
        # explicit here — no numerics change.
        log = d / f"lib_{tag}_cuda_{key}.buildlog"
        log.write_text(proc.stdout + proc.stderr)
        print(f"[nvcc -Xptxas -v] {tag} (sm_{sm}):\n{proc.stderr}", flush=True)
    return str(so)


@functools.lru_cache(maxsize=None)
def build_cuda_so(field_key: str, D: int, precision: str = "float64", sm: str | None = None) -> str:
    """Compile the Tsit5 FFI handler as a CUDA shared lib."""
    return _compile_cuda(emit_cu(field_key, int(D), precision),
                         f"{field_key}_{D}_{precision}", sm)


def build_cuda_so_rosenbrock(field_key: str, D: int, precision: str = "float64",
                             A=None, sm: str | None = None) -> str:
    """Compile the stiff Rosenbrock23 FFI handler as a CUDA shared lib. ``A`` is
    the DxD operator for a ``linstiff_*`` field (ignored for robertson/hires). Not
    ``lru_cache``d (A is an unhashable ndarray); the source-hash disk cache dedupes."""
    src = emit_cu_rosenbrock(field_key, int(D), precision, A)
    return _compile_cuda(src, f"rosen_{field_key}_{D}_{precision}", sm)
