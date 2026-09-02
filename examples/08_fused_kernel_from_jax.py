"""Example 08: get the fused kernel for your own JAX RHS via register_jax_field.

Examples 04-05 show the two ends of the spectrum: a Problem whose name matches one of
gradsolve's built-in fused fields (04, Lorenz) drives the genuine fused Warp kernel, while
an unregistered name (05, Lotka-Volterra) runs on the general-purpose engines. This
example closes the gap: you write ``f_jax(t, y, p)`` once, and

    gradsolve.register_jax_field(name, f_jax, dim, n_params, stiff=False)

translates its jaxpr into the same fused Warp field the built-ins use (and, for a stiff
system, the analytic Jacobian too). A Problem whose ``name`` matches then routes to the
fused engines exactly as Lorenz/Van der Pol/Robertson do — no C, no Warp code, no kernel
by hand. The translator covers a fixed subset of JAX primitives (arithmetic, the usual
elementwise transcendentals, static-index reshapes/gathers, small dot_general); an RHS
outside it raises ``UnsupportedRHS``, and ``fused="auto"`` then falls back to the general
path with the reason recorded on ``SolveResult.route`` (see ``fused=`` in docs/api.md).

The system here is a periodic diffusively-coupled cubic ring (dim=20, params (a, b)):

    dy_i/dt = a * (y_{i-1} - 2 y_i + y_{i+1}) - y_i^3 + b

deliberately not one of gradsolve's built-ins. dim=20 (> the low-dim cuda lane's reach) sends
the forward solve to the fused ``warp_ode`` kernel; the gradient records that fused forward
and replays it (``warp_replay``) — both run on Warp's CPU device here. The script checks
correctness with a finite difference; it does not measure GPU speed.

Run with:
    python examples/08_fused_kernel_from_jax.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve


class DiffusiveRing:
    """A fresh nonstiff system gradsolve has never heard of, 2 free parameters (a, b)."""
    name = "user_diffusive_ring"
    dim = 20
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, p):
        a, b = p[..., 0], p[..., 1]
        lap = jnp.roll(y, -1, axis=-1) - 2.0 * y + jnp.roll(y, 1, axis=-1)
        return a[..., None] * lap - y * y * y + b[..., None]


def main():
    prob = DiffusiveRing()

    # 1) Register the RHS. This translates f_jax's jaxpr into the fused Warp field; it is
    #    idempotent and does the codegen lazily (when the kernel is first built).
    gradsolve.register_jax_field(prob.name, DiffusiveRing().f_jax, prob.dim, n_params=2,
                               stiff=False)

    n = 8
    rng = np.random.default_rng(0)
    y0 = rng.uniform(-1.0, 1.0, size=(n, prob.dim))
    params = np.column_stack([rng.uniform(0.5, 1.5, size=n),   # a
                              rng.uniform(-0.5, 0.5, size=n)])  # b

    # 2) Forward ensemble solve. fused="auto" (the default) routes a registered field to the
    #    fused engine; the dim=20 nonstiff forward lands on warp_ode (the fused adaptive
    #    Tsit5 kernel), running on Warp's CPU device here.
    res = gradsolve.solve(prob, y0, params, fused="auto", device="cpu")
    print(f"forward: route.actual={res.route.actual!r}  y_final.shape={res.y_final.shape}")
    assert res.route.actual == "warp_ode", res.route
    assert bool(np.all(np.isfinite(res.y_final)))

    # 3) Reverse-mode gradient THROUGH the fused kernel. grad_closure records the fused
    #    forward's step mesh once and replays it (warp_replay) — an exact discrete adjoint.
    clo = gradsolve.grad_closure(prob, y0, params, fused="auto", device="cpu")
    print(f"reverse: route.actual={clo.route.actual!r}")
    assert clo.route.actual == "warp_replay", clo.route

    params_j = jnp.asarray(params)
    loss = lambda q: jnp.sum(clo(q) ** 2)
    g = jax.grad(loss)(params_j)
    assert g.shape == params_j.shape and bool(jnp.all(jnp.isfinite(g)))

    # 4) Finite-difference spot-check on one trajectory's a (column 0).
    i, j, eps = 0, 0, 1e-6
    pp = np.asarray(params, dtype=np.float64).copy()
    pp[i, j] += eps
    lp = float(loss(jnp.asarray(pp)))
    pp[i, j] -= 2 * eps
    lm = float(loss(jnp.asarray(pp)))
    fd = (lp - lm) / (2 * eps)
    rel = abs(float(g[i, j]) - fd) / (abs(fd) + 1e-12)
    print(f"grad[{i},{j}]={float(g[i, j]):.6g}  finite-diff={fd:.6g}  rel_err={rel:.2e}")
    assert rel < 1e-4, f"gradient disagrees with FD (rel {rel:.2e})"

    print("OK — fused kernel generated from a user f_jax: forward warp_ode + reverse warp_replay")


if __name__ == "__main__":
    main()
