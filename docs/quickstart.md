# gradsolve quickstart

`gradsolve` solves large ensembles of differential equations in JAX and can
return a reverse-mode gradient through the solve. This page gets you from zero
to a forward solve and a `jax.grad` on a laptop CPU in about a minute.

## Install

`gradsolve` targets Python ≥3.11 and needs only `jax` and `numpy` at runtime:

```bash
pip install gradsolve
```

Alternatively, install from a source checkout (a fresh conda environment is a
convenient place for it):

```bash
git clone https://github.com/ECLIPSE-AI4Science/gradsolve
cd gradsolve
conda create -n gradsolve python=3.12
conda activate gradsolve
pip install -e .                       # exposes `import gradsolve`
```

Nothing else is required for this quickstart. Optional extras add engines
gradsolve can route to but doesn't require:

```bash
pip install "gradsolve[diffrax]"   # adaptive Kvaerno5/Tsit5 fallback engine
pip install "gradsolve[warp]"      # fused NVIDIA-Warp kernels (CPU or NVIDIA GPU)
```

From a source checkout the same extras are `pip install -e ".[diffrax]"` and
`pip install -e ".[warp]"`. Everything in this quickstart runs on the plain CPU
install (no `diffrax`, no `warp-lang` needed).

### GPU install

`cuda_tsit5` needs an NVIDIA GPU and `nvcc`; the Warp kernels need the `warp`
extra and also run on Warp's CPU backend, but are only fast on a GPU. Install
the CUDA extra and make sure `nvcc` is on your `PATH` — the
`cuda_tsit5` kernel is compiled with it on first use:

```bash
pip install "gradsolve[cuda12]"    # jax[cuda12] + warp-lang
```

None of the GPU engines are needed for the CPU walkthrough below: without a GPU,
`gradsolve` takes the general path — diffrax for a forward solve when installed,
otherwise the record-and-replay engines.

### float64 on import

Importing `gradsolve` enables JAX's float64 mode (`jax_enable_x64`) as a
side effect — the library assumes double precision. To opt out, set
`GRADSOLVE_X64=0` in the environment *before* importing gradsolve.

## Forward + reverse, in one script

`gradsolve` needs only a small duck-typed problem: `name`, `dim`, `t0`, `t1`,
`is_stiff`, and a JAX `f_jax(t, y, params)` mapping a single trajectory's state
`y` of shape `(dim,)` and its params `p` of shape `(P,)` to the derivative —
gradsolve vmaps `f_jax` over the ensemble for you. Nothing else is required — no
registration, no subclassing.

```python
import gradsolve, jax, jax.numpy as jnp, numpy as np

class Lorenz:                          # a problem: any object with these six members
    name = "user_lorenz"               # a label; a registered name (built-in or via register_jax_field) selects a fused kernel
    dim = 3                            # number of state components
    t0, t1 = 0.0, 1.0                  # time span
    is_stiff = False                   # False: explicit engines; True: implicit engines for stiff systems

    def f_jax(self, t, y, p):          # the right-hand side dy/dt of one trajectory: y has shape (dim,), p its parameters
        rho = p[0]                     # sigma = 10 and beta = 8/3 are fixed; rho is the parameter
        return jnp.stack([10.0 * (y[1] - y[0]), rho * y[0] - y[1] - y[0] * y[2], y[0] * y[1] - (8.0 / 3.0) * y[2]])

problem = Lorenz()
y0 = np.tile([1.0, 0.0, 0.0], (16, 1))           # 16 trajectories, shape (n, dim)
params = np.linspace(20, 30, 16)[:, None]        # one rho per trajectory, shape (n, P)

result = gradsolve.solve(problem, y0, params, device="cpu")     # the router picks the engine
print(result.solver, result.y_final.shape)       # diffrax (16, 3)   (tsit5_replay without the diffrax extra)

final_states = gradsolve.grad_closure(problem, y0, params, device="cpu")   # records the accepted step sizes once
loss = lambda p: jnp.sum(final_states(p) ** 2)
gradient = jax.grad(loss)(jnp.asarray(params))
print(gradient.shape, bool(jnp.all(jnp.isfinite(gradient))))
```

Run it with `python your_script.py`. Expected output (with the `diffrax` extra
installed; see below if it isn't):

```
diffrax (16, 3)
(16, 1) True
```

`result.solver` names the engine `gradsolve.solve` actually routed to — here
`diffrax`, the general engine that forward solves use when it is installed,
since a user's own right-hand side has no registered fused field. `gradient` is the
gradient of the loss with respect to each trajectory's own `rho`, one value per
trajectory (shape `(16, 1)`), and every entry is finite.

## What just happened

`gradsolve.solve` / `gradsolve.grad_closure` inspect three properties of the
workload — state dimension, stiffness, and whether a gradient is needed — and
pick one of a handful of engines (`gradsolve.dispatch.choose_engine`). This
3-variable Lorenz system is nonstiff and low-dimensional, but a fused kernel
exists only for registered fields (the built-in problems, or your own after
`gradsolve.register_jax_field`), so both calls above take the general path:
the forward solve uses `diffrax` (or `tsit5_replay`, the general
record-and-replay engine, if `diffrax` isn't installed), and the gradient
closure always uses `tsit5_replay`, which records the mesh (the accepted step
sizes) once and replays it as a `jax.grad`-able `lax.scan`.
`final_states.route` (`Route(requested='auto', actual='tsit5_replay',
reason='no-registered-field')`) says exactly why. The full routing table —
including the fused GPU engines and the record-and-replay adjoint — is in the
main [`README.md`](https://github.com/ECLIPSE-AI4Science/gradsolve#how-it-works) and in the
[Guide](guide.md).

## More runnable examples

Two more scripts in `examples/` print `OK` when run:

```bash
python examples/00_standalone.py     # your own right-hand side: forward + reverse, gradient checked by finite differences
python examples/03_engine_routing.py # inspect the routing table for every (dim, stiff, need_grad) cell
```

`00_standalone.py` is the same Lorenz problem as above, extended with a
central finite-difference check on the gradient.

The full set of tutorials in `examples/` (`00`–`09`) builds on the same two entry
points against a range of registered problems and the fused reverse path; see
[`examples/README.md`](https://github.com/ECLIPSE-AI4Science/gradsolve/tree/main/examples) for the full index.

## Next steps

- Full routing table and performance summary: [`README.md`](https://github.com/ECLIPSE-AI4Science/gradsolve#readme).
- Engine routing in depth, stiff vs. nonstiff, and when the record-and-replay adjoint applies: [Guide](guide.md).
- Public functions and the `Problem` protocol: [API reference](api.md).
