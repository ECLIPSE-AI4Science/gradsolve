"""Example 00: your own right-hand side through gradsolve, forward solve plus reverse-mode gradient.

gradsolve only needs a small duck-typed Problem: attributes ``name``, ``dim``, ``t0``, ``t1``,
``is_stiff`` and a JAX right-hand side ``f_jax(t, y, params)`` for a single trajectory — ``y``
has shape ``(dim,)`` and ``params`` has shape ``(P,)`` — which gradsolve vmaps over the whole
ensemble for you. That is the whole contract; everything below runs with gradsolve installed on
its own, no other file in this repo required.

See also: 05_bring_your_own_ode.py (the same idea, walked through step by step for a
different nonstiff model) and 06_stiff.py (the ``is_stiff=True`` side of the contract).

    python examples/00_standalone.py    # prints OK

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve


class Lorenz:
    """The Lorenz system as a minimal user Problem (sigma=10, beta=8/3; rho is the parameter).

    The name is deliberately not one of gradsolve's built-in fused-kernel field names (``lorenz``,
    ``vdp``, ``linear_ladder_*``): a user's own RHS is handled by the general-purpose engines
    that differentiate through ``f_jax`` directly, so it needs no registered fused kernel field.
    """
    name = "user_lorenz"
    dim = 3
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, p):
        rho = p[0]
        return jnp.stack([10.0 * (y[1] - y[0]), rho * y[0] - y[1] - y[0] * y[2], y[0] * y[1] - (8.0 / 3.0) * y[2]])


def main():
    problem = Lorenz()
    n = 16
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    params = np.linspace(20.0, 30.0, n)[:, None]          # a per-trajectory ensemble over rho

    # 1) Forward ensemble solve, auto-routed.
    result = gradsolve.solve(problem, y0, params, engine="auto", device="cpu")
    assert result.y_final.shape == (n, problem.dim)
    print(f"forward: engine={result.solver}  y_final.shape={result.y_final.shape}")

    # 2) Reverse-mode gradient THROUGH the solve: d/d(rho) of a scalar loss on the final state.
    final_states = gradsolve.grad_closure(problem, y0, params, engine="auto", device="cpu")
    loss = lambda p: jnp.sum(final_states(p) ** 2)
    gradient = jax.grad(loss)(jnp.asarray(params))
    assert gradient.shape == params.shape and bool(jnp.all(jnp.isfinite(gradient)))
    print(f"reverse: grad.shape={gradient.shape}  all finite={bool(jnp.all(jnp.isfinite(gradient)))}")

    # 3) Check the gradient against a central finite difference on one trajectory.
    i, eps = 0, 1e-4
    pp = np.asarray(params, dtype=np.float64).copy()
    pp[i, 0] += eps; lp = float(loss(jnp.asarray(pp)))
    pp[i, 0] -= 2 * eps; lm = float(loss(jnp.asarray(pp)))
    fd = (lp - lm) / (2 * eps)
    rel = abs(float(gradient[i, 0]) - fd) / (abs(fd) + 1e-12)
    print(f"grad[{i}]={float(gradient[i,0]):.4g}  finite-diff={fd:.4g}  rel_err={rel:.2e}")
    assert rel < 1e-2, f"gradient disagrees with FD (rel {rel:.2e})"
    print("OK — standalone forward + reverse-mode gradient through gradsolve")


if __name__ == "__main__":
    main()
