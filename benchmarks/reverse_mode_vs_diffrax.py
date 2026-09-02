"""Reverse-mode gradient cost: gradsolve against diffrax, at matched achieved accuracy.

One Lorenz ensemble (sigma = 10, beta = 8/3, rho swept over [0, 21], y0 = (1, 0, 0),
t in [0, 1]), the same configuration as the published GPU ODE benchmarks. For every
tolerance in a sweep, each arm computes one reverse-mode gradient of ``sum(y_final**2)``
with respect to the per-trajectory parameter, and the script records

* the achieved accuracy: the median relative error of ``y_final`` against a scipy DOP853
  reference solved at rtol 1e-11 (a reference that belongs to neither arm);
* the wall time of one gradient: ``jax.jit(jax.grad(...))`` for every arm alike, one untimed
  warm-up call, then the best of ``--repeats`` timed calls with ``block_until_ready``;
* for gradsolve, the one-off cost of recording the step mesh (``grad_closure``), which is
  paid once per parameter point and is not part of the per-gradient time.

Arms:
  gradsolve fused    engine="warp_replay":  fused Warp kernel records the mesh, JAX replays it
  gradsolve general  engine="tsit5_replay": pure-JAX record and replay of any user f_jax
  diffrax            Tsit5 + PIDController + RecursiveCheckpointAdjoint, configured so that
                     it is timed at its fastest setting rather than at a default: a probe solve
                     measures its step counts, the step budget is sized from them (four times
                     the accepted steps, at least 4096), and three checkpoint settings are
                     timed (checkpoints == budget sized to the attempts, 1.5x the attempts, and
                     the library default); the fastest is the one reported.

Speedups are read at matched accuracy: the diffrax cost is interpolated (log-log) onto the
error gradsolve achieved, never at the same requested tolerance.

    python benchmarks/reverse_mode_vs_diffrax.py                       # CPU, n = 128
    python benchmarks/reverse_mode_vs_diffrax.py --n 2048 32768 --device cuda --repeats 5
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import gradsolve  # enables float64 on import, before any array is created

SIGMA, BETA = 10.0, 8.0 / 3.0
REF_RTOL, REF_ATOL = 1e-11, 1e-12
MAX_STEPS_PROBE, MAX_STEPS_HEADROOM, MAX_STEPS_MIN, CHECKPOINT_HEADROOM = 100_000, 4, 4096, 1.5
ENGINES = {"gradsolve fused": "warp_replay", "gradsolve general": "tsit5_replay"}


class Lorenz:
    name = "diffeqgpu_lorenz"     # a registered field, so the fused engines can be selected
    dim = 3
    t0, t1 = 0.0, 1.0
    is_stiff = False

    def f_jax(self, t, y, p):
        x, yy, z = y[..., 0], y[..., 1], y[..., 2]
        rho = p[..., 0]
        return jnp.stack([SIGMA * (yy - x), rho * x - yy - x * z, x * yy - BETA * z], axis=-1)


def make_batch(n):
    rho = np.linspace(0.0, 21.0, n)
    return np.tile([1.0, 0.0, 0.0], (n, 1)), rho[:, None]


def reference(prob, y0, params):
    """Final states from scipy DOP853 at rtol 1e-11, one trajectory at a time."""
    from scipy.integrate import solve_ivp

    def f(t, y, rho):
        return [SIGMA * (y[1] - y[0]), rho * y[0] - y[1] - y[0] * y[2], y[0] * y[1] - BETA * y[2]]

    out = np.empty_like(y0)
    for i in range(len(y0)):
        sol = solve_ivp(f, (prob.t0, prob.t1), y0[i], args=(float(params[i, 0]),),
                        method="DOP853", rtol=REF_RTOL, atol=REF_ATOL)
        out[i] = sol.y[:, -1]
    return out


def median_rel_err(yf, ref):
    rel = np.abs(np.asarray(yf, dtype=np.float64) - ref) / np.maximum(np.abs(ref), 1e-30)
    return float(np.median(rel[np.isfinite(rel)])) if np.isfinite(rel).any() else float("inf")


def time_gradient(closure, params, repeats):
    """Best-of-``repeats`` wall time in ms of one jitted reverse-mode gradient."""
    p0 = jnp.asarray(params)
    grad = jax.jit(jax.grad(lambda p: jnp.sum(closure(p) ** 2)))
    jax.block_until_ready(grad(p0))                       # compile + warm-up, untimed
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(grad(p0))
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


# ----------------------------------------------------------------------------- diffrax
def _diffrax_solve_one(prob, *, rtol, atol, max_steps, checkpoints):
    import diffrax as dfx

    term = dfx.ODETerm(lambda t, y, p: prob.f_jax(t, y, p))
    solver, controller = dfx.Tsit5(), dfx.PIDController(rtol=rtol, atol=atol)
    adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=checkpoints)   # None: library default
    t0, t1 = jnp.asarray(prob.t0), jnp.asarray(prob.t1)

    def one(y0i, pi):
        return dfx.diffeqsolve(term, solver, t0, t1, None, y0i, args=pi, saveat=dfx.SaveAt(t1=True),
                               stepsize_controller=controller, adjoint=adjoint,
                               max_steps=int(max_steps), throw=False)
    return one


def diffrax_probe(prob, y0, params, *, rtol, atol, max_steps):
    """Untimed forward solve: (y_final, all successful, accepted steps, attempted steps)."""
    import diffrax as dfx

    one = _diffrax_solve_one(prob, rtol=rtol, atol=atol, max_steps=max_steps, checkpoints=None)

    def stats(a, b):
        sol = one(a, b)
        return sol.ys[-1], sol.result == dfx.RESULTS.successful, sol.stats["num_accepted_steps"], sol.stats["num_steps"]
    yf, ok, acc, tot = jax.jit(jax.vmap(stats))(jnp.asarray(y0), jnp.asarray(params))
    return np.asarray(yf), np.asarray(ok), np.asarray(acc), np.asarray(tot)


def diffrax_closure(prob, y0, *, rtol, atol, max_steps, checkpoints):
    one = _diffrax_solve_one(prob, rtol=rtol, atol=atol, max_steps=max_steps, checkpoints=checkpoints)
    y0j = jnp.asarray(y0)
    batch = jax.vmap(lambda a, b: one(a, b).ys[-1])
    return lambda p: batch(y0j, jnp.asarray(p))


def _gate(yf, ok, tot, max_steps):
    if not np.isfinite(yf).all():
        return "non-finite final states"
    if not ok.all():
        return f"{int((~ok).sum())} unsuccessful solves"
    if (tot >= max_steps).any():
        return f"{int((tot >= max_steps).sum())} trajectories reached max_steps={max_steps}"
    return ""


def fair_diffrax_arm(prob, y0, params, *, rtol, atol, repeats):
    """Return (closure, info) for diffrax timed at its fastest checkpoint setting."""
    yf, ok, acc, tot = diffrax_probe(prob, y0, params, rtol=rtol, atol=atol, max_steps=MAX_STEPS_PROBE)
    why = _gate(yf, ok, tot, MAX_STEPS_PROBE)
    if why:
        raise RuntimeError(f"diffrax probe at rtol={rtol:g} failed: {why}")
    attempts = int(tot.max())
    sized = max(MAX_STEPS_MIN, MAX_STEPS_HEADROOM * int(acc.max()))
    candidates = [("checkpoints == max_steps == attempts + 1", attempts + 1, attempts + 1),
                  ("ceil(1.5 x attempts) on the sized budget", math.ceil(CHECKPOINT_HEADROOM * attempts), sized),
                  ("library default (checkpoints=None)", None, sized)]
    timed = []
    for name, ck, ms in candidates:
        cl = diffrax_closure(prob, y0, rtol=rtol, atol=atol, max_steps=ms, checkpoints=ck)
        timed.append((time_gradient(cl, params, repeats), name, ck, ms, cl))
    best_ms, name, ck, ms, cl = min(timed, key=lambda t: t[0])
    yw, okw, _, totw = diffrax_probe(prob, y0, params, rtol=rtol, atol=atol, max_steps=ms)
    why = _gate(yw, okw, totw, ms)
    if why or not np.allclose(yw, yf, rtol=1e-6, atol=1e-12):
        raise RuntimeError(f"diffrax winner '{name}' failed its gate: {why or 'states differ from the probe'}")
    return cl, dict(ms=best_ms, y_final=yf, setting=name, checkpoints=ck, max_steps=ms,
                    attempts=attempts, accepted_max=int(acc.max()))


# ----------------------------------------------------------------------------- sweep
def run(ns, rtols, repeats, device, engines=ENGINES, log=print):
    prob = Lorenz()
    rows = []
    for n in ns:
        y0, params = make_batch(n)
        log(f"n = {n}: computing the scipy reference for {n} trajectories ...")
        ref = reference(prob, y0, params)
        for rtol in rtols:
            atol = rtol * 1e-3
            for label, engine in engines.items():
                t = time.perf_counter()
                cl = gradsolve.grad_closure(prob, y0, params, engine=engine, rtol=rtol, atol=atol, device=device)
                record_s = time.perf_counter() - t
                if cl.route.actual != engine:
                    log(f"  {label}: rerouted to {cl.route.actual} ({cl.route.reason}); arm skipped")
                    continue
                err = median_rel_err(cl(jnp.asarray(params)), ref)
                ms = time_gradient(cl, params, repeats)
                rows.append(dict(n=n, rtol=rtol, arm=label, engine=engine, err=err, ms=ms, record_s=record_s, setting=""))
                log(f"  n={n} rtol={rtol:.0e} {label:18s} {ms:9.3f} ms/gradient  err {err:.2e}  (record {record_s:.2f} s)")
            try:
                dcl, info = fair_diffrax_arm(prob, y0, params, rtol=rtol, atol=atol, repeats=repeats)
            except RuntimeError as exc:
                log(f"  n={n} rtol={rtol:.0e} diffrax: {exc}")
                continue
            err = median_rel_err(info["y_final"], ref)
            rows.append(dict(n=n, rtol=rtol, arm="diffrax", engine="diffrax", err=err, ms=info["ms"],
                             record_s=float("nan"), setting=info["setting"]))
            log(f"  n={n} rtol={rtol:.0e} {'diffrax':18s} {info['ms']:9.3f} ms/gradient  err {err:.2e}  "
                f"({info['setting']}, max_steps {info['max_steps']})")
    return rows


def matched_speedups(rows):
    """For every gradsolve row, diffrax's cost interpolated (log-log) at the same achieved
    error, divided by gradsolve's cost. NaN where the error lies outside diffrax's measured range."""
    out = []
    for r in rows:
        if r["arm"] == "diffrax":
            continue
        d = sorted((x for x in rows if x["arm"] == "diffrax" and x["n"] == r["n"]), key=lambda x: x["err"])
        errs, ms = np.log([x["err"] for x in d]), np.log([x["ms"] for x in d])
        if len(d) >= 2 and errs.min() <= math.log(r["err"]) <= errs.max():
            ratio = math.exp(float(np.interp(math.log(r["err"]), errs, ms))) / r["ms"]
        else:
            ratio = float("nan")
        out.append(dict(n=r["n"], rtol=r["rtol"], arm=r["arm"], err=r["err"], ms=r["ms"], speedup_matched=ratio))
    return out


def plot(rows, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted({r["n"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    n_big = ns[-1]
    for arm, marker in [("gradsolve fused", "o"), ("gradsolve general", "^"), ("diffrax", "s")]:
        pts = sorted((r["err"], r["ms"]) for r in rows if r["arm"] == arm and r["n"] == n_big)
        if pts:
            ax1.loglog([e for e, _ in pts], [m for _, m in pts], marker=marker, label=arm)
    ax1.set_xlabel("achieved median relative error")
    ax1.set_ylabel("ms per gradient")
    ax1.set_title(f"work-precision, n = {n_big}")
    ax1.grid(True, which="major", alpha=0.4)
    ax1.legend()
    sp = matched_speedups(rows)
    for arm, marker in [("gradsolve fused", "o"), ("gradsolve general", "^")]:
        xs, ys = [], []
        for n in ns:
            vals = [s["speedup_matched"] for s in sp if s["arm"] == arm and s["n"] == n and np.isfinite(s["speedup_matched"])]
            if vals:
                xs.append(n)
                ys.append(float(np.exp(np.mean(np.log(vals)))))
        if xs:
            ax2.semilogx(xs, ys, marker=marker, label=arm)
    ax2.axhline(1.0, color="0.5", ls="--", lw=1)
    ax2.set_xlabel("ensemble size n")
    ax2.set_ylabel("speedup over diffrax at matched accuracy")
    ax2.set_title("geometric mean over the tolerance sweep")
    ax2.grid(True, which="major", alpha=0.4)
    ax2.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[128])
    ap.add_argument("--rtols", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu", help='"cpu" or "cuda"')
    ap.add_argument("--out-dir", default="benchmarks/results")
    args = ap.parse_args()
    rows = run(args.n, args.rtols, args.repeats, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "reverse_mode_vs_diffrax.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmatched-accuracy speedup over diffrax (diffrax cost interpolated at gradsolve's achieved error):")
    for s in matched_speedups(rows):
        print(f"  n={s['n']:6d} rtol={s['rtol']:.0e} {s['arm']:18s} err {s['err']:.2e}  {s['ms']:8.3f} ms  speedup {s['speedup_matched']:.2f}x")
    plot(rows, os.path.join(args.out_dir, "reverse_mode_vs_diffrax.png"))
    print(f"\nwrote {args.out_dir}/reverse_mode_vs_diffrax.csv and .png")


if __name__ == "__main__":
    main()
