"""Example 07: fit a parameter to a time series, not just a final state (``saveat``).

01-02 differentiate a loss on the final state. Real fits usually have observations spread
over time, which needs dense output: ``saveat=<times>`` makes the closure return the states
at those times, ``(n, k, dim)``, and a time-series loss differentiates straight through it.

Here: a Lorenz ensemble is observed at 12 times, and rho is recovered by gradient descent
from a deliberately wrong starting guess.

What is worth knowing about the mechanism:

* The saved states are not interpolated. The lane records which step brackets each requested
  time and then takes one Tsit5 step to exactly that time, so a saved state is as accurate
  as the endpoint; the last check in this script confirms that the state saved at ``t1``
  equals ``y_final``.
* The step mesh is recorded once, at the (y0, params) handed to ``grad_closure``, and frozen
  for the closure's lifetime. The gradient is the exact discrete adjoint of that replayed
  frozen-step integration — valid near the recording point.
* Dense output lives on the JAX scan/replay lanes; the fused kernels and the cuda lane stay
  final-state-only. ``.route`` (printed below) says which lane actually ran.

Why this fit re-records: rho 24 -> 28 is not a small perturbation, and one mesh recorded at
rho=24 is not valid there — descending on it alone stalls near the starting guess. That is
the frozen-controller caveat described in the docs, not a bug. So the fit is an outer loop
that re-records at the current parameters and an inner loop of cheap gradient steps on
that frozen mesh, which is the usual workflow for a record-and-replay adjoint. Recording is
the expensive part; amortizing it over ~25 gradient steps is what makes the fit affordable.

Runs on CPU in a few seconds.

Run with:
    python examples/07_saveat_timeseries_fit.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import gradsolve


class Lorenz:
    """Classic Lorenz; params[0] = rho (sigma, beta fixed)."""

    name = "diffeqgpu_lorenz"
    dim = 3
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, p):
        sigma, rho, beta = 10.0, p[0], 8.0 / 3.0
        return jnp.array([
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ])


RHO_TRUE = 28.0
RHO_GUESS = 24.0


def main():
    prob = Lorenz()
    n = 8
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    y0[:, 0] += 0.05 * np.arange(n)          # a spread of initial conditions
    saveat = np.linspace(0.1, 1.0, 12)       # 12 observation times

    # Synthetic observations at the TRUE rho.
    truth = gradsolve.solve(prob, y0, np.full((n, 1), RHO_TRUE), saveat=saveat)
    obs = jnp.asarray(truth.y_saved)
    print(f"observations:   y_saved {truth.y_saved.shape} at {len(saveat)} times "
          f"(lane: {truth.solver})")

    # Fit rho from a wrong guess: re-record at the current parameters (outer), then take
    # cheap gradient steps on that frozen mesh (inner).
    p = jnp.asarray(np.full((n, 1), RHO_GUESS))
    lr, n_outer, n_inner = 0.05, 12, 25

    for rnd in range(n_outer):
        closure = gradsolve.grad_closure(prob, y0, np.asarray(p), saveat=saveat, device="cpu")
        if rnd == 0:
            print(f"route:          {closure.route.actual} (asked for "
                  f"{closure.route.requested!r}, reason: {closure.route.reason})")

        def loss(q, closure=closure):
            return jnp.mean((closure(q) - obs) ** 2)

        grad_fn = jax.jit(jax.grad(loss))
        if rnd == 0:
            print(f"start:          rho={float(p[0, 0]):.4f}  loss={float(loss(p)):.6e}")
        for _ in range(n_inner):
            p = p - lr * grad_fn(p)
        if rnd % 4 == 3:
            print(f"  record {rnd + 1:2d}:    rho={float(p[0, 0]):.4f}  "
                  f"loss={float(loss(p)):.6e}")

    rho_fit = float(p[0, 0])
    print(f"recovered:      rho={rho_fit:.4f}   (true {RHO_TRUE}, guess {RHO_GUESS})")

    # The fit must land near the truth, not merely drift toward it.
    assert abs(rho_fit - RHO_TRUE) < 0.1, (
        f"time-series fit did not recover rho={RHO_TRUE}: got {rho_fit}")

    # Dense output is consistent with the final state when t1 is among the saved times.
    np.testing.assert_array_equal(truth.y_saved[:, -1, :], truth.y_final)
    print("OK")


if __name__ == "__main__":
    main()
