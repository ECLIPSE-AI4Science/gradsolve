# Benchmarks

Two notebooks, each with a command-line twin, measure gradsolve against another library on
whatever hardware runs them. Both need only `pip install 'gradsolve[diffrax,reproduce]'`; the
second additionally needs Julia for its DiffEqGPU.jl arm.

## Reverse-mode gradient against diffrax

`reverse_mode_vs_diffrax.ipynb` (and `reverse_mode_vs_diffrax.py`) times one
reverse-mode gradient through a Lorenz ensemble solve for gradsolve and for diffrax on the same
hardware, at the same achieved accuracy. It needs only `gradsolve` with the `diffrax` extra, scipy,
pandas and matplotlib:

```bash
pip install 'gradsolve[diffrax,reproduce]'
python benchmarks/reverse_mode_vs_diffrax.py                                  # CPU, 128 trajectories
python benchmarks/reverse_mode_vs_diffrax.py --n 2048 32768 --device cuda --repeats 5   # GPU
```

The same knobs reach the notebook through environment variables set before the kernel starts:
`GRADSOLVE_BENCH_NS` (ensemble sizes), `GRADSOLVE_BENCH_RTOLS`, `GRADSOLVE_BENCH_REPEATS`,
`GRADSOLVE_BENCH_DEVICE`. The committed notebook outputs come from one NVIDIA H200.

What it reports, per ensemble size and tolerance: the achieved accuracy of each arm (median relative
error of the final state against a scipy DOP853 reference at rtol 1e-11), the wall time of one
gradient (`jax.jit(jax.grad(...))`, one untimed warm-up, best of several timed calls), the one-off
cost of gradsolve's mesh recording, and the speedup over diffrax at matched accuracy, read by
interpolating diffrax's cost onto the error gradsolve achieved.

How diffrax is configured, so that it is timed at its fastest setting rather than at a default: an
untimed probe solve measures its step counts, the step budget is sized from them (four times the
accepted steps, at least 4096), and three checkpoint settings of `RecursiveCheckpointAdjoint` are
timed (checkpoints equal to a budget sized to the attempts, 1.5 times the attempts, and the library
default). The fastest is the one reported.

Results are written to `benchmarks/results/` (not tracked).

## Forward-only throughput against DiffEqGPU.jl

`forward_vs_diffeqgpu.ipynb` (and `forward_vs_diffeqgpu.py`) times a forward-only solve of the same
Lorenz ensemble for gradsolve's two fused kernels (`engine="cuda_tsit5"`, which needs `nvcc` on the
PATH, and `engine="warp_ode"`) and for DiffEqGPU.jl's fused `GPUTsit5` kernel, on the same GPU, at
several tolerances, with the achieved accuracy of every arm against the same scipy reference on a
subsample of trajectories. The DiffEqGPU.jl arm runs through the Julia script
`forward_vs_diffeqgpu/bench_diffeqgpu.jl`; set it up once with

```bash
julia --project=benchmarks/forward_vs_diffeqgpu -e 'using Pkg; Pkg.instantiate()'
python benchmarks/forward_vs_diffeqgpu.py --device cuda --n 2048 32768 524288 --julia auto
```

Without Julia (`--julia none`, or when no `julia` is on the PATH) the notebook reports the gradsolve
arms only. On a GPU the gradsolve arms are timed at the kernel launch with device-resident inputs and
outputs, the same boundary as DiffEqGPU.jl's `vectorized_asolve`; the time through the public
`gradsolve.solve`, which adds the host-device copies, is reported alongside.
The notebook's environment variables: `GRADSOLVE_BENCH_NS`, `GRADSOLVE_BENCH_RTOLS`,
`GRADSOLVE_BENCH_REPEATS`, `GRADSOLVE_BENCH_DEVICE`, `GRADSOLVE_JULIA`.
