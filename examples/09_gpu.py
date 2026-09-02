"""Example 09: the same two calls, on a GPU.

Examples 00 to 08 pass ``device="cpu"`` so they run anywhere. This one asks JAX for a
GPU and, when there is one, runs the forward solve and the reverse-mode gradient of a
Lorenz ensemble on it. The Problem is named ``"diffeqgpu_lorenz"`` (as in 04) so the
registered fused field is found: ``engine="auto"`` sends the forward solve to the fused
``cuda_tsit5`` kernel (compiled with ``nvcc`` on first use) or to ``warp_ode``, and the
gradient to ``warp_replay``, the record-and-replay adjoint over the fused Warp kernel.
Each call's ``.route`` says which engine actually ran.

Without a GPU the script says so and exits, so it is harmless in a CPU-only job.
Timing goes through the public API with NumPy arrays in and out, so it includes the
host-device copies; the second call is timed, after the first has compiled.

Run with:
    pip install "gradsolve[cuda12]"
    python examples/09_gpu.py
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import gradsolve


class Lorenz:
    """The Lorenz system (sigma=10, beta=8/3; rho is the per-trajectory parameter)."""
    name = "diffeqgpu_lorenz"  # matches gradsolve's registered fused field, see example 04
    dim = 3
    t0 = 0.0
    t1 = 1.0
    is_stiff = False

    def f_jax(self, t, y, p):
        x, yv, z = y[..., 0], y[..., 1], y[..., 2]
        rho = p[..., 0]
        return jnp.stack([10.0 * (yv - x), rho * x - yv - x * z, x * yv - (8.0 / 3.0) * z], axis=-1)


def timed(fn):
    """Run ``fn`` twice and return (result, seconds of the second call)."""
    fn()  # first call: compile / build the kernel
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main():
    backend = jax.default_backend()
    if backend != "gpu":
        print(f"JAX sees no GPU (jax.default_backend() == {backend!r}); nothing to run here. "
              "Examples 00 to 08 are the CPU tour.")
        return
    print(f"device: {jax.devices()[0].device_kind}")

    prob = Lorenz()
    n = 131_072
    y0 = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    params = np.linspace(20.0, 30.0, n)[:, None]  # rho sweep, one value per trajectory

    result, dt = timed(lambda: gradsolve.solve(prob, y0, params, engine="auto", device="cuda"))
    print(f"forward:   engine={result.route.actual:<12s} n={n}  {dt * 1e3:8.1f} ms")
    assert result.y_final.shape == (n, prob.dim)
    assert np.all(np.isfinite(result.y_final))

    final_states = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cuda")
    params_j = jnp.asarray(params)
    def loss(p):
        return jnp.sum(final_states(p) ** 2)

    gradient, dt = timed(lambda: jax.block_until_ready(jax.grad(loss)(params_j)))
    print(f"gradient:  engine={final_states.route.actual:<12s} n={n}  {dt * 1e3:8.1f} ms")
    assert gradient.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))

    print("OK — forward solve and reverse-mode gradient ran on the GPU")


if __name__ == "__main__":
    main()
