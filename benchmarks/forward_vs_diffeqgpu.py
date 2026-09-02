"""Forward-only ensemble throughput: gradsolve's fused kernels against DiffEqGPU.jl's GPUTsit5.

One Lorenz ensemble (sigma = 10, beta = 8/3, rho swept over [0, 21], y0 = (1, 0, 0), t in [0, 1]),
solved forward only (no gradient), in double precision, on the same GPU:

* gradsolve ``engine="cuda_tsit5"``: the hand-written CUDA kernel (needs ``nvcc`` on PATH);
* gradsolve ``engine="warp_ode"``: the fused adaptive Tsit5 kernel in NVIDIA Warp;
* DiffEqGPU.jl ``GPUTsit5`` through ``vectorized_asolve`` (adaptive), run by the Julia script
  ``forward_vs_diffeqgpu/bench_diffeqgpu.jl`` when a ``julia`` executable is available.

For every ensemble size and tolerance the script records the wall time of one batched forward
solve, the time per trajectory, and the achieved accuracy: the median relative error of the final
state on an evenly spaced subsample of 4096 trajectories against a scipy DOP853 reference at
rtol 1e-11. Timing boundary: on a GPU the gradsolve arms are timed at the kernel launch, with
inputs and outputs resident on the device, through the engines' JAX FFI entry points; that is the
same boundary as DiffEqGPU's ``vectorized_asolve`` (BenchmarkTools' minimum of a CUDA-synchronised
call). The time through the public ``gradsolve.solve``, which adds the host-to-device copy of the
inputs, the copy of the results back and the Python overhead, is reported alongside as
``us_per_traj_api``. All gradsolve timings are one untimed warm-up, then the best of ``--repeats``
timed calls with ``block_until_ready``.

    python benchmarks/forward_vs_diffeqgpu.py --device cuda --n 2048 32768 524288 --julia auto
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time

import jax
import numpy as np
from reverse_mode_vs_diffrax import (  # noqa: E402
    Lorenz,
    make_batch,
    median_rel_err,
    reference,
)

import gradsolve  # enables float64 on import, before any array is created

HERE = os.path.dirname(os.path.abspath(__file__))
JULIA_PROJECT = os.path.join(HERE, "forward_vs_diffeqgpu")
ENGINES = {"gradsolve cuda_tsit5": "cuda_tsit5", "gradsolve warp_ode": "warp_ode"}
SUBSAMPLE = 4096


def subsample_indices(n):
    """The same evenly spaced subsample the Julia script writes (0-based)."""
    if n <= SUBSAMPLE:
        return np.arange(n)
    return np.round(np.linspace(1, n, SUBSAMPLE)).astype(int) - 1


def time_solve(prob, y0, params, *, engine, rtol, atol, device, repeats):
    """Best-of-``repeats`` wall time in ms of one forward ensemble solve through gradsolve.solve."""
    def once():
        res = gradsolve.solve(prob, y0, params, engine=engine, rtol=rtol, atol=atol, device=device)
        jax.block_until_ready(res.y_final)
        return res
    res = once()                                            # warm-up (compile), untimed
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        once()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best, res


def time_kernel(prob, y0, params, *, engine, rtol, atol, device, repeats):
    """Best-of-``repeats`` wall time in ms of the fused kernel alone, inputs and outputs on the
    device (CUDA only). Returns ``(ms, y_final)`` or ``None`` when the device is not CUDA."""
    if device not in ("cuda", "gpu"):
        return None
    import jax.numpy as jnp

    from gradsolve.warp.warp_ode import _field_for, _param_of

    field_key, dim = _field_for(prob)
    rho = jnp.asarray(_param_of(prob, params))
    if engine == "cuda_tsit5":
        from gradsolve.cuda._ffi_bridge import make_runner
        run = make_runner(field_key, dim, "float64", "cuda")
        y0_soa = jnp.asarray(np.ascontiguousarray(y0.T))
        call = lambda: run(y0_soa, rho, float(prob.t1), float(rtol), float(atol), 100_000)  # noqa: E731
        unpack = lambda outs: np.asarray(outs[0]).T  # noqa: E731
    else:
        from gradsolve.warp.warp_ode import _ffi_runner
        run = _ffi_runner(len(y0), 4096, 0, "float64")            # the same settings warp_ode.solve uses
        conf = jnp.asarray(np.array([prob.t0, prob.t1, rtol, atol, (prob.t1 - prob.t0) / 100.0]))
        y0_dev = jnp.asarray(y0)
        call = lambda: run(y0_dev, rho, conf)  # noqa: E731
        unpack = lambda outs: np.asarray(outs[0]).reshape(len(y0), 3)  # noqa: E731
    outs = jax.block_until_ready(call())                    # warm-up (compile), untimed
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(call())
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best, unpack(outs)


def run_gradsolve(ns, rtols, repeats, device, engines=ENGINES, log=print):
    prob = Lorenz()
    rows, refs = [], {}
    for n in ns:
        y0, params = make_batch(n)
        idx = subsample_indices(n)
        log(f"n = {n}: scipy reference on {len(idx)} trajectories ...")
        refs[n] = reference(prob, y0[idx], params[idx])
        for rtol in rtols:
            atol = rtol * 1e-3
            for label, engine in engines.items():
                try:
                    ms, res = time_solve(prob, y0, params, engine=engine, rtol=rtol, atol=atol, device=device, repeats=repeats)
                except Exception as exc:  # noqa: BLE001  (a lane that cannot build here)
                    log(f"  n={n} rtol={rtol:.0e} {label:22s} unavailable ({type(exc).__name__})")
                    continue
                if res.route.actual != engine:
                    log(f"  n={n} rtol={rtol:.0e} {label:22s} rerouted to {res.route.actual} ({res.route.reason}); skipped")
                    continue
                kernel = time_kernel(prob, y0, params, engine=engine, rtol=rtol, atol=atol, device=device, repeats=repeats)
                if kernel is None:
                    ms_k, yf, timing = ms, np.asarray(res.y_final), "gradsolve.solve"
                else:
                    ms_k, yf, timing = kernel[0], kernel[1], "kernel"
                err = median_rel_err(yf[idx], refs[n])
                rows.append(dict(n=n, rtol=rtol, arm=label, ms=ms_k, us_per_traj=ms_k * 1e3 / n, err=err,
                                 ms_api=ms, us_per_traj_api=ms * 1e3 / n, timing=timing))
                log(f"  n={n} rtol={rtol:.0e} {label:22s} {ms_k:9.3f} ms  {ms_k * 1e3 / n:.5f} us/traj ({timing}; "
                    f"through gradsolve.solve {ms * 1e3 / n:.5f})  err {err:.2e}")
    return rows, refs


def find_julia(julia):
    if julia in (None, "", "none"):
        return None
    if julia == "auto":
        return shutil.which("julia")
    return julia


def run_diffeqgpu(ns, rtols, refs, julia, out_dir, log=print):
    """Run the Julia driver once per tolerance and read back its timings and final states."""
    rows = []
    for rtol in rtols:
        cmd = [julia, f"--project={JULIA_PROJECT}", os.path.join(JULIA_PROJECT, "bench_diffeqgpu.jl"),
               out_dir, f"{rtol:g}"] + [str(n) for n in ns]
        log("  " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log(f"  DiffEqGPU.jl run failed (rtol={rtol:g}):\n{proc.stderr[-2000:]}")
            continue
        with open(os.path.join(out_dir, f"diffeqgpu_rtol{rtol:g}.csv"), newline="") as fh:
            timing = {int(r["n"]): float(r["min_ms"]) for r in csv.DictReader(fh)}
        for n in ns:
            if n not in timing:
                continue
            with open(os.path.join(out_dir, f"diffeqgpu_final_n{n}_rtol{rtol:g}.csv"), newline="") as fh:
                finals = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in csv.DictReader(fh)])
            err = median_rel_err(finals, refs[n])
            ms = timing[n]
            rows.append(dict(n=n, rtol=rtol, arm="DiffEqGPU.jl GPUTsit5", ms=ms, us_per_traj=ms * 1e3 / n, err=err,
                             ms_api=float("nan"), us_per_traj_api=float("nan"), timing="kernel"))
            log(f"  n={n} rtol={rtol:.0e} {'DiffEqGPU.jl GPUTsit5':22s} {ms:9.3f} ms  {ms * 1e3 / n:.5f} us/traj  err {err:.2e}")
    return rows


def run(ns, rtols, repeats, device, julia="auto", out_dir="benchmarks/results", log=print):
    rows, refs = run_gradsolve(ns, rtols, repeats, device, log=log)
    jl = find_julia(julia)
    if jl is None:
        log("julia not found: the DiffEqGPU.jl arm is skipped (pass --julia /path/to/julia)")
    else:
        os.makedirs(out_dir, exist_ok=True)
        rows += run_diffeqgpu(ns, rtols, refs, jl, out_dir, log=log)
    return rows


def matched_ratios(rows, baseline="DiffEqGPU.jl GPUTsit5"):
    """DiffEqGPU's time per trajectory interpolated (log-log) at each gradsolve arm's achieved
    error, divided by that arm's time per trajectory. NaN where the arm's error lies outside
    the range DiffEqGPU reached in the sweep (no measurement to interpolate against)."""
    out = []
    for r in rows:
        if r["arm"] == baseline:
            continue
        d = sorted((x for x in rows if x["arm"] == baseline and x["n"] == r["n"]), key=lambda x: x["err"])
        errs, us = np.log([x["err"] for x in d]), np.log([x["us_per_traj"] for x in d])
        if len(d) >= 2 and errs.min() <= math.log(r["err"]) <= errs.max():
            ratio = math.exp(float(np.interp(math.log(r["err"]), errs, us))) / r["us_per_traj"]
        else:
            ratio = float("nan")
        out.append(dict(n=r["n"], rtol=r["rtol"], arm=r["arm"], err=r["err"], us_per_traj=r["us_per_traj"], ratio_matched=ratio))
    return out


def plot(rows, path=None):
    import matplotlib.pyplot as plt

    ns = sorted({r["n"] for r in rows})
    rtols = sorted({r["rtol"] for r in rows})
    mid = rtols[len(rtols) // 2]
    arms = ["gradsolve cuda_tsit5", "gradsolve warp_ode", "DiffEqGPU.jl GPUTsit5"]
    markers = {"gradsolve cuda_tsit5": "*", "gradsolve warp_ode": "o", "DiffEqGPU.jl GPUTsit5": "D"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for arm in arms:
        pts = sorted((r["n"], r["us_per_traj"]) for r in rows if r["arm"] == arm and r["rtol"] == mid)
        if pts:
            ax1.loglog([n for n, _ in pts], [u for _, u in pts], marker=markers[arm], label=arm)
    ax1.set_xlabel("ensemble size n")
    ax1.set_ylabel("time per trajectory (us)")
    ax1.set_title(f"forward solve, rtol = {mid:g}")
    ax1.grid(True, which="major", alpha=0.4)
    ax1.legend()
    n_big = ns[-1]
    for arm in arms:
        pts = sorted((r["err"], r["us_per_traj"]) for r in rows if r["arm"] == arm and r["n"] == n_big)
        if pts:
            ax2.loglog([e for e, _ in pts], [u for _, u in pts], marker=markers[arm], label=arm)
    ax2.set_xlabel("achieved median relative error")
    ax2.set_ylabel("time per trajectory (us)")
    ax2.set_title(f"work-precision, n = {n_big}")
    ax2.grid(True, which="major", alpha=0.4)
    ax2.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[2048])
    ap.add_argument("--rtols", type=float, nargs="+", default=[1e-3, 1e-4, 1e-6, 1e-8])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu", help='"cpu" or "cuda"')
    ap.add_argument("--julia", default="auto", help='path to julia, "auto" (search PATH) or "none"')
    ap.add_argument("--out-dir", default="benchmarks/results")
    args = ap.parse_args()
    rows = run(args.n, args.rtols, args.repeats, args.device, args.julia, args.out_dir)
    if not rows:
        sys.exit("no arm produced a measurement")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "forward_vs_diffeqgpu.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if any(r["arm"] == "DiffEqGPU.jl GPUTsit5" for r in rows):
        print("\nDiffEqGPU.jl time per trajectory over gradsolve's, at matched achieved error:")
        for s in matched_ratios(rows):
            print(f"  n={s['n']:8d} rtol={s['rtol']:.0e} {s['arm']:22s} err {s['err']:.2e}  {s['us_per_traj']:.5f} us  ratio {s['ratio_matched']:.2f}x")
    plot(rows, os.path.join(args.out_dir, "forward_vs_diffeqgpu.png"))
    print(f"\nwrote {args.out_dir}/forward_vs_diffeqgpu.csv and .png")


if __name__ == "__main__":
    main()
