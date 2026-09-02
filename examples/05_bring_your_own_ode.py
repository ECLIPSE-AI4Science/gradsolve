"""Example 05: bring your own ODE — the full Problem contract, walked through.

gradsolve never asks you to register a model. Any Python object exposing five things is a
Problem it can solve and differentiate:

    name       str    -- identifier (only matters if it happens to match one of gradsolve's
                          own registered fused fields, e.g. "lorenz"/"vdp" -- see 04. Any
                          other string, like the one below, is simply "unregistered" and
                          runs through the general-purpose engines instead. Both are
                          correct; only the engine differs.)
    dim        int     -- state dimension.
    t0, t1     float   -- integration interval.
    is_stiff   bool     -- selects explicit vs L-stable/implicit engines (see 06_stiff.py).
    f_jax(t, y, params) -- JAX-traceable RHS for one trajectory. y is (dim,), params is
                          (P,), returns dy/dt of shape (dim,). gradsolve vmaps it over the
                          whole ensemble for you -- you never write the batch loop yourself.

That's the entire contract (gradsolve/base.py:Problem is the formal, structural version).
This example uses the classic Lotka-Volterra predator-prey model -- nonstiff, 4 free
parameters, a heterogeneous ensemble -- to show the pattern is not special-cased to
single-parameter toy systems: any JAX-traceable f_jax with any parameter layout works.

Run with:
    python examples/05_bring_your_own_ode.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve


class LotkaVolterra:
    """Predator-prey dynamics, 4 free parameters (alpha, beta, delta, gamma):

        dx/dt = alpha*x - beta*x*y      (prey)
        dy/dt = delta*x*y - gamma*y     (predator)

    Deliberately named something gradsolve has never heard of -- this is the "unregistered
    model" case a user's own model starts in.
    """
    name = "user_lotka_volterra"
    dim = 2
    t0 = 0.0
    t1 = 10.0
    is_stiff = False

    def f_jax(self, t, y, p):
        x, yy = y[..., 0], y[..., 1]
        alpha, beta, delta, gamma = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        dx = alpha * x - beta * x * yy
        dy = delta * x * yy - gamma * yy
        return jnp.stack([dx, dy], axis=-1)


def main():
    prob = LotkaVolterra()
    n = 12
    y0 = np.tile(np.array([10.0, 5.0]), (n, 1))
    # A heterogeneous ensemble: each trajectory's 4 params jittered +/-10% around a base
    # point (a deterministic RNG -- fixed seed, reproducible across runs).
    rng = np.random.default_rng(0)
    base = np.array([1.1, 0.4, 0.1, 0.4])
    params = base[None, :] * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=(n, 4)))

    # 1) Forward ensemble solve, auto-routed. Since "user_lotka_volterra" matches no
    #    registered fused field, this is served by the general engines (diffrax forward; the
    #    Tsit5 record-and-replay for the gradient) -- still correct, just not the specialized
    #    kernel path (see 04 for that case, and gradsolve/dispatch.py for the full routing policy).
    #    The record-and-replay records the step mesh once on the device by default on a GPU
    #    (recorder="auto"); here it is a small CPU ensemble, so it uses the host loop that
    #    recorder="host" also selects. Either way the replayed gradient is the same.
    res = gradsolve.solve(prob, y0, params, engine="auto", device="cpu")
    print(f"forward: engine={res.solver}  y_final.shape={res.y_final.shape}")
    assert res.y_final.shape == y0.shape
    assert bool(np.all(np.isfinite(res.y_final))), "y_final contains non-finite values"

    # 2) Reverse-mode gradient THROUGH the solve w.r.t. all 4 parameters at once.
    closure = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cpu")
    params_j = jnp.asarray(params)
    loss = lambda q: jnp.sum(closure(q) ** 2)
    g = jax.grad(loss)(params_j)
    print(f"reverse: grad.shape={g.shape}  all finite={bool(jnp.all(jnp.isfinite(g)))}")
    assert g.shape == params_j.shape
    assert bool(jnp.all(jnp.isfinite(g)))

    # 3) Finite-difference spot-check on one trajectory's beta (column 1).
    i, j, eps = 0, 1, 1e-5
    pp = np.asarray(params, dtype=np.float64).copy()
    pp[i, j] += eps
    lp = float(loss(jnp.asarray(pp)))
    pp[i, j] -= 2 * eps
    lm = float(loss(jnp.asarray(pp)))
    fd = (lp - lm) / (2 * eps)
    rel = abs(float(g[i, j]) - fd) / (abs(fd) + 1e-12)
    print(f"grad[{i},{j}]={float(g[i, j]):.6g}  finite-diff={fd:.6g}  rel_err={rel:.2e}")
    assert rel < 1e-2, f"gradient disagrees with FD (rel {rel:.2e})"

    print("OK — bring-your-own ODE: forward + reverse-mode gradient over 4 parameters")


if __name__ == "__main__":
    main()
