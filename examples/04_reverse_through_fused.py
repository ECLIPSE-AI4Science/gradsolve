"""Example 04: reverse-mode gradients through the routed fused/record-replay ensemble path.

diffrax is reverse-differentiable too: its ``RecursiveCheckpointAdjoint`` gives an exact
reverse-mode gradient, so the distinction here is not a missing capability but the
reverse-pass cost and the execution mechanism. gradsolve routes the gradient through
warp_replay, a record-and-replay discrete adjoint that records the accepted-step mesh on
the forward fused Warp kernel and replays it in reverse. That yields an exact reverse-mode
gradient at a fraction of a checkpointed adjoint's per-gradient cost.

Getting the fused kernel itself (not the general-purpose fallback every other example in
this directory exercises) needs a Problem whose ``name`` matches one of gradsolve's own
registered fused fields — ``"lorenz"``/``"diffeqgpu_lorenz"``, ``"vdp"``, or
``"linear_ladder_<D>"`` (the registry lives in ``gradsolve/warp/_warp_kernel.py``). The
Lorenz system below is defined inline right here — no import beyond gradsolve — but
deliberately named ``"diffeqgpu_lorenz"`` so ``gradsolve.grad_closure(engine="auto")``
finds that field and drives the record-and-replay adjoint over the fused kernel. (Compare
with 00/02/05, where the Problem's name is deliberately unregistered and the general
fallback runs instead — both are correct; only the engine differs.)

This example runs on Warp's CPU backend and checks correctness with a finite-difference
spot-check; it does not measure GPU speed.

Run with:
    python examples/04_reverse_through_fused.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve


class Lorenz:
    """The Lorenz system (sigma=10, beta=8/3; rho is the parameter).

    Named ``"diffeqgpu_lorenz"`` on purpose: that name matches gradsolve's own registered
    fused-kernel field (a single-parameter Lorenz, ``rho = p[..., 0]``), so the
    fused/record-replay engines actually engage instead of falling back.
    """
    name = "diffeqgpu_lorenz"
    dim = 3
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, p):
        x, yv, z = y[..., 0], y[..., 1], y[..., 2]
        rho = p[..., 0]
        return jnp.stack([10.0 * (yv - x), rho * x - yv - x * z, x * yv - (8.0 / 3.0) * z], axis=-1)


def main():
    prob = Lorenz()
    n = 8
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    params = np.linspace(0.0, 21.0, n)[:, None]  # sweep rho across the ensemble

    # auto -> warp_replay on nonstiff+low-dim+grad; Warp's CPU backend here checks correctness.
    closure = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cpu")

    params_j = jnp.asarray(params)  # jnp array for .at indexing + jax.grad
    loss = lambda q: jnp.sum(closure(q) ** 2)
    g = jax.grad(loss)(params_j)

    # Finite-difference spot-check on coordinate (0, 0).
    eps = 1e-4
    i = (0, 0)
    qp = params_j.at[i].add(eps)
    qm = params_j.at[i].add(-eps)
    fd = (loss(qp) - loss(qm)) / (2 * eps)
    rel = abs(float(g[i]) - float(fd)) / (abs(float(fd)) + 1e-12)

    print(f"jax.grad g[0,0]: {float(g[i]):.6f}")
    print(f"finite-diff fd:  {float(fd):.6f}")
    print(f"rel error:       {rel:.2e}")

    assert g.shape == params_j.shape, f"gradient shape mismatch: {g.shape} != {params_j.shape}"
    assert bool(jnp.all(jnp.isfinite(g))), "gradient contains non-finite values"
    assert rel < 1e-2, f"FD spot-check failed: rel={rel:.2e} >= 1e-2"
    print("OK — reverse-mode through the fused-ensemble path is correct")


if __name__ == "__main__":
    main()
