"""Example 02: reverse-mode gradient through an ensemble solve of differential equations.

Builds on 01_quickstart.py's Van der Pol ensemble: instead of only running it forward,
``gradsolve.grad_closure`` returns a ``jax.grad``-able closure. Engine choice:
``engine="auto"`` with ``need_grad`` implied (grad_closure always resolves the *gradient*
routing target, e.g. ``warp_replay`` on GPU for nonstiff low-dimensional problems). This
particular problem's name isn't one of gradsolve's registered fused fields (see 04 for that
case), so it runs through the general-purpose reverse-differentiation path here, which is
equally correct; only the engine differs.

``wrt=`` picks what you differentiate:

* ``wrt="params"`` (the default, shown first) -> closure ``params -> y_final``.
* ``wrt="y0"`` (shown second) -> closure ``y0 -> y_final``, for fitting an initial
  condition rather than a rate constant. ``wrt=("y0", "params")`` gives ``(y0, params) ->
  y_final`` when you want both at once.

The mesh is recorded once at the ``(y0, params)`` handed to ``grad_closure`` and frozen for
the closure's lifetime, so a y0 gradient carries the same frozen-controller caveat as a
parameter gradient (see docs/api.md). Every closure also carries ``.route`` — the engine
actually used, and why — printed below.

Run with:
    python examples/02_reverse_grad.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve


class VanDerPol:
    """Same system as 01_quickstart.py — the Van der Pol oscillator, mu the parameter."""
    name = "user_vdp_quickstart"
    dim = 2
    t0 = 0.0
    t1 = 6.0
    is_stiff = False

    def f_jax(self, t, y, p):
        x, v = y[..., 0], y[..., 1]
        mu = p[..., 0]
        return jnp.stack([v, mu * (1.0 - x * x) * v - x], axis=-1)


def main():
    prob = VanDerPol()
    n = 10
    y0 = np.tile(np.array([2.0, 0.0]), (n, 1))
    params = np.linspace(0.5, 3.0, n)[:, None]

    closure = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cpu")

    print(f"route:           {closure.route.actual} (asked for {closure.route.requested!r}"
          f", reason: {closure.route.reason})")

    loss = lambda q: jnp.sum(closure(q) ** 2)
    params_j = jnp.asarray(params)
    g = jax.grad(loss)(params_j)

    print(f"gradient shape:  {g.shape}")
    print(f"all finite:      {bool(jnp.all(jnp.isfinite(g)))}")

    assert g.shape == params_j.shape, f"shape mismatch: {g.shape} != {params_j.shape}"
    assert bool(jnp.all(jnp.isfinite(g))), "gradient contains non-finite values"

    # Central finite-difference spot-check on trajectory 0's mu.
    eps = 1e-5
    i = (0, 0)
    qp = params_j.at[i].add(eps)
    qm = params_j.at[i].add(-eps)
    fd = (loss(qp) - loss(qm)) / (2 * eps)
    rel = abs(float(g[i]) - float(fd)) / (abs(float(fd)) + 1e-12)
    print(f"jax.grad g[0,0]: {float(g[i]):.6f}")
    print(f"finite-diff fd:  {float(fd):.6f}")
    print(f"rel error:       {rel:.2e}")
    assert rel < 1e-2, f"gradient disagrees with FD (rel {rel:.2e})"

    # --- wrt="y0": same solve, differentiated w.r.t. the INITIAL STATE instead. --------
    # The mesh is recorded at the same (y0, params); only what stays free changes.
    y0_closure = gradsolve.grad_closure(prob, y0, params, wrt="y0", engine="auto", device="cpu")
    y0_j = jnp.asarray(y0)
    loss_y0 = lambda z: jnp.sum(y0_closure(z) ** 2)
    g_y0 = jax.grad(loss_y0)(y0_j)

    print(f"d loss/d y0 shape: {g_y0.shape}")
    assert g_y0.shape == y0_j.shape, f"shape mismatch: {g_y0.shape} != {y0_j.shape}"
    assert bool(jnp.all(jnp.isfinite(g_y0))), "y0 gradient contains non-finite values"

    # Same frozen mesh on both sides of the difference — the closure replays, never re-records.
    fd_y0 = (loss_y0(y0_j.at[i].add(eps)) - loss_y0(y0_j.at[i].add(-eps))) / (2 * eps)
    rel_y0 = abs(float(g_y0[i]) - float(fd_y0)) / (abs(float(fd_y0)) + 1e-12)
    print(f"jax.grad g_y0[0,0]: {float(g_y0[i]):.6f}")
    print(f"finite-diff fd:     {float(fd_y0):.6f}")
    print(f"rel error:          {rel_y0:.2e}")
    assert rel_y0 < 1e-2, f"y0 gradient disagrees with FD (rel {rel_y0:.2e})"
    print("OK")


if __name__ == "__main__":
    main()
