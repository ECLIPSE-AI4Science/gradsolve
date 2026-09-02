# tests/ — the gradsolve library suite

The test suite for the `gradsolve` package. Every test here imports **only**
`gradsolve` (+ numpy/jax/scipy/pytest) and runs on **CPU** — correctness, reverse-mode
autodiff validation, dispatcher routing, and the record-and-replay adjoint, all without a
GPU. `pip install -e '.[test]'` followed by `pytest -q` is the whole contributor loop.

## Run it

```bash
pip install -e '.[test]'   # gradsolve + pytest + scipy + ruff
pytest -q                  # library tests on CPU (gpu-marked tests auto-deselected)
```

Optional marker selections:

```bash
pytest -q -m slow          # only the slow tests (heavier sweeps / integration)
pytest -q -m "not slow"    # fast inner loop
pytest -q -m gpu           # GPU-only tests — run only on a CUDA machine
```

`addopts = -m 'not gpu'` in `pyproject.toml` means a bare `pytest -q` already excludes any
GPU-marked test, so you do **not** need `-m "not gpu"` by hand on a CPU-only machine. `testpaths =
["tests"]` scopes collection to this directory.

## Markers (defined in `pyproject.toml`)

| Marker | Meaning |
|---|---|
| `gpu` | Requires a CUDA GPU. Auto-deselected on CPU via `addopts`; select with `-m gpu` on a machine with a CUDA GPU. |
| `slow` | Heavier tests (larger sweeps / integration). Deselect with `-m "not slow"` for a fast inner loop. |

The whole library suite passes on CPU; the markers exist so heavier or GPU-only cases can
be tagged without changing the default invocation.

## Layout

The main groups are:

- **`test_problem_protocol.py`** — the `gradsolve.base.Problem` Protocol + `SolveResult` /
  `Backend` contract: a duck-typed class (no inheritance) satisfies `solve()` /
  `grad_closure()` in full.
- **`test_solvers_fixed_step.py`**, **`test_solvers_adaptive_imex.py`**, **`test_steps.py`**,
  **`test_tsit5_error_weights.py`** — the core solver tests: fixed-step explicit / Tsit5 / IMEX
  forward accuracy, the adaptive-IMEX CFL/accuracy step sizing + reverse-diff replay, and
  the per-stage Tsit5 / Rosenbrock23 step arithmetic.
- **`test_tsit5_replay.py`**, **`test_adjoint_replay.py`** — the record-and-replay adjoint:
  frozen-mesh recording shapes/padding, and reverse-mode gradients (nonstiff Tsit5 + stiff
  Rosenbrock23) matched against central finite differences. `test_adjoint_replay.py` uses
  `pytest.importorskip("warp")`, so a warp-less env skips it cleanly.
- **`test_dispatch.py`** — `gradsolve/dispatch.py`: `choose_engine` / `choose_remat` over the
  full `(dim, stiff, need_grad)` grid, and `DECISION_MAP` internal consistency.
- **`test_api.py`** — the routed public API (`solve` / `grad_closure`): auto-routing, engine
  override, fallback when the requested engine is unsupported, `SolveResult` stamping, value +
  gradient correctness.
- **`test_standalone_import.py`** — regression that gradsolve imports and runs *standalone*: a
  fresh subprocess blocks any `examples` package in `sys.modules`, then still solves +
  differentiates an inline Problem against a closed form.
- **`test_no_examples_import.py`** — a second standalone gate: with `examples` blocked,
  routes a **stiff** problem through gradsolve and asserts success + that no `examples.*`
  module ended up in `sys.modules`.
- **`test_fjax_shape_contract.py`** — asserts the `Problem.f_jax` contract is
  **per-trajectory**: gradsolve vmaps over the ensemble and calls the RHS with `y` shape
  `(dim,)` and `params` shape `(P,)`.
- **`test_warp_import_guard.py`** — with `warp` blocked in a subprocess, asserts the warp
  modules still import, report their fused lane unavailable, and routing falls back.
- **`test_cuda_ffi_cpu.py`** — CPU smoke for the `jax.ffi` bridge: builds the host-C++ FFI
  target (`gradsolve/cuda/_build.py::build_cpu_so`, no nvcc) and invokes it; skips if no host
  C++ compiler is on `PATH`.

### `_oracles/`

Independent reference integrators, written from the published methods, used by the
replay/adjoint tests as ground truth. They are **numpy-only + gradsolve-only** (no external
dependency): an adaptive Tsit5 (`tier0_fused_adaptive_oracle.py`) and an adaptive
Rosenbrock23 with an analytic Jacobian (`gradsolve_g2_spike_oracle.py`). Kept in-tree so the
reverse-mode gradient claims are checked against a second implementation, not just
self-consistency.
