# Changelog

All notable changes to gradsolve (Python package `gradsolve`) are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/). Newest first; dates are ISO (YYYY-MM-DD).

This file records user-facing library changes per release. Releases before 0.2.1 predate the
public repository; their entries are kept for reference.

## [0.2.1] - 2026-09-02

First public release.

### Added
- Device-resident mesh recorder for any user `f_jax` (`recorder="auto"|"jax"|"host"`): the
  one-time record of the accepted step mesh runs as a batched `lax.while_loop` on whatever
  device holds the data (a GPU by default), instead of a per-trajectory host loop. Host and
  device records agree on step counts exactly and on step sizes to about 1e-8.
- `gradsolve.register_jax_field(name, f_jax, dim, n_params, stiff=False)`: translates a user's
  JAX right-hand side (and its Jacobian, when stiff) into the same fused NVIDIA-Warp field the
  built-in problems use, so user problems can route to the fused GPU engines. `solve` /
  `grad_closure` take `fused="auto"|True|False`.
- Experimental hand-written CUDA Rosenbrock23 forward kernel for stiff ensembles
  (opt-in, off by default).
- CI checks that `gradsolve.__version__`, the `CITATION.cff` version and the release tag agree.
- Project logo (`docs/assets/logo/`: marks, wordmark lockups, favicons); the README opens with the
  light/dark lockup and shows four of the figures (`docs/assets/figures/`).
- `benchmarks/reverse_mode_vs_diffrax.{py,ipynb}`: time one reverse-mode gradient for gradsolve
  and for diffrax (at its fastest checkpoint setting) on your own hardware, at matched achieved
  accuracy.
- `benchmarks/forward_vs_diffeqgpu.{py,ipynb}` with a Julia driver: forward-only throughput of the
  fused kernels against DiffEqGPU.jl's `GPUTsit5` on the same GPU, with achieved accuracy.

### Changed
- Unregistered problems now auto-route to the general record-and-replay engines of their
  stiffness class (`tsit5_replay` nonstiff, `rodas5p_replay` stiff) for gradients and
  `saveat`, and to diffrax for forward-only solves when that optional extra is installed.
  These honour `rtol`/`atol`, which the fixed-step scans they replace did not, so accuracy at
  the same requested tolerance improves by several orders of magnitude. The fixed-step scans
  remain reachable by explicit `engine=`.
- Explicit `engine="diffrax"` without diffrax installed now reroutes to the replay lane
  (`route.reason == "diffrax-not-installed"`) instead of raising at call time.
- Forward results now carry `route` metadata (`SolveResult.route`), as gradient closures
  already did.

### Fixed
- The hand-written CUDA lanes compile for the GPU actually present (compute capability read
  from the JAX device, `GRADSOLVE_CUDA_SM` overrides) instead of a fixed A100 target, so they
  run on H100/H200-class cards.
- A user's unregistered nonstiff RHS was previously served by a 10,000-step fixed-step scan
  that accepted but ignored `rtol`/`atol`; a user's stiff RHS by an order-1 IMEX scan. Both
  now route to adaptive record-and-replay engines (see Changed).

## [0.2.0] - 2026-07-27

### Added
- `vern7_replay`: seventh-order nonstiff record-and-replay engine.
- `rodas5p_replay`: fifth-order stiff record-and-replay engine, with opt-in dense output
  (continuous extension).
- Gradients with respect to the initial state: `wrt=('y0','params')` on `grad_closure`.
- Dense output / `saveat` on the pure-JAX replay engines.
- Scoped float32 replay adjoint for registered Warp fields (`precision="float32"` under
  `GRADSOLVE_X64=0`).
- Method-abstraction layer so the replay engines share one record-and-replay loop.

### Fixed
- Warp-less registration path; assorted stiff-engine correctness fixes.

## [0.1.0] - 2026-07-16

Initial release.

### Added
- Forward and reverse-mode differentiable GPU ensemble ODE solving in JAX.
- Record-and-replay adjoint: an exact reverse-mode gradient through a fast ensemble solve,
  taken by replaying the recorded step mesh.
- Routed `solve()` / `grad_closure()` API that dispatches an ensemble to the appropriate
  engine for the problem (dimension, stiffness, gradient need).
- Solver engines: fixed-step and adaptive Tsit5, IMEX and adaptive IMEX, Rosenbrock23 for
  stiff systems, a hand-written fused CUDA Tsit5 forward lane, fused NVIDIA-Warp adaptive
  Tsit5 / Rosenbrock kernels, and a universal adaptive diffrax fallback.
- Duck-typed `Problem` and `Backend` protocols; float64 enabled on import (`GRADSOLVE_X64`).
