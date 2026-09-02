"""Example 01: the shortest path to a forward solve of an ensemble of differential equations.

Define a Problem (a small duck-typed contract — see the ``VanDerPol`` class below),
hand it plus a batch of initial states/parameters to ``gradsolve.solve``, done.
``engine="auto"`` routes the workload to the engine the routing table selects for its
(dim, stiff, need_grad) corner — see ``examples/03_engine_routing.py`` for what that
table looks like and ``gradsolve/dispatch.py`` for the policy itself.

Run with:
    python examples/01_quickstart.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import gradsolve


class VanDerPol:
    """The Van der Pol oscillator: a nonstiff limit-cycle system, mu is the parameter.

    x'  = v
    v'  = mu*(1 - x^2)*v - x
    """
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
    y0 = np.tile(np.array([2.0, 0.0]), (n, 1))       # same start, an ensemble over mu
    params = np.linspace(0.5, 3.0, n)[:, None]

    res = gradsolve.solve(prob, y0, params, engine="auto", device="cpu")

    print(f"engine:        {res.solver}")
    print(f"y_final shape: {res.y_final.shape}")
    print(f"y_final[0]:    {res.y_final[0]}")

    assert res.y_final.shape == y0.shape, f"shape mismatch: {res.y_final.shape} != {y0.shape}"
    assert bool(np.all(np.isfinite(res.y_final))), "y_final contains non-finite values"
    print("OK")


if __name__ == "__main__":
    main()
