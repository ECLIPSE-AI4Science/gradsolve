# gradsolve

`gradsolve` is a JAX library for solving large ensembles of differential
equations on GPUs. It integrates the whole ensemble in one call and can return a
reverse-mode gradient through that solve, so the same library serves both
forward simulation and gradient-based fitting. It handles ordinary
differential equations, stiff and nonstiff. Which engine runs a solve is
decided by one pure function of the state dimension, the stiffness, and
whether a gradient is needed:
[`gradsolve.dispatch.choose_engine`](https://github.com/ECLIPSE-AI4Science/gradsolve/blob/main/gradsolve/dispatch.py).

## The record-and-replay adjoint

`gradsolve` records the accepted step sizes on the forward pass and replays
them as a fixed-length `lax.scan` on the backward pass. The gradient is the
discrete adjoint of the accepted steps with the step sizes held fixed. When
this applies, and what happens when it doesn't, is described in the
[Guide](guide.md).

## Pages

| Page | What's there |
|---|---|
| [Quickstart](quickstart.md) | Install, your first forward solve, your first gradient, runnable examples |
| [Guide](guide.md) | The `Problem` contract, engine routing in depth, stiff vs. nonstiff, when the record-and-replay adjoint applies |
| [API reference](api.md) | `gradsolve.solve`, `gradsolve.grad_closure`, the `Problem` protocol, the engine registry |
| [Examples](https://github.com/ECLIPSE-AI4Science/gradsolve/tree/main/examples) | The tutorial scripts and the getting-started notebook, from a standalone right-hand side to fused kernels |
| [Benchmarks](https://github.com/ECLIPSE-AI4Science/gradsolve/tree/main/benchmarks) | Time gradsolve against diffrax and DiffEqGPU.jl on your own hardware |
| [README](https://github.com/ECLIPSE-AI4Science/gradsolve#readme) | Single-page tour: install, quickstart, routing table, performance summary, citation |
| [Contributing](https://github.com/ECLIPSE-AI4Science/gradsolve/blob/main/CONTRIBUTING.md) | Development setup, the checks that must pass, pull request expectations |
| [Changelog](https://github.com/ECLIPSE-AI4Science/gradsolve/blob/main/CHANGELOG.md) | Release history |
