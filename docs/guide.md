# gradsolve: how the pieces fit together

`gradsolve` solves ensembles of differential equations in JAX and can return a reverse-mode
gradient through the solve. This page explains the *mechanics* behind its two entry points: the
duck-typed `Problem` you hand it, the difference between `solve()` and `grad_closure()`, how
routing picks an engine, what stiffness changes, when the record-and-replay adjoint is (and
isn't) the thing running under the hood, and the `remat` memory/speed knob.

For the full API surface see [`docs/api.md`](api.md); for a one-minute start see
[`docs/quickstart.md`](quickstart.md); for the project overview see the
[README](https://github.com/ECLIPSE-AI4Science/gradsolve#readme). Nothing on this page needs a GPU to execute.

---

## 1. The `Problem` protocol

`gradsolve` doesn't ask you to subclass anything. `gradsolve/base.py` defines a structural
(`Protocol`) contract, and the library reads exactly six members off whatever object you pass:

```python
class Problem(Protocol):
    name: str
    dim: int
    t0: float
    t1: float

    @property
    def is_stiff(self) -> bool: ...

    def f_jax(self, t: float, y: Any, params: Any) -> Any: ...
```

- `dim` — the state dimension (this is what the register-limit routing decision below keys
  on).
- `t0`, `t1` — the integration horizon.
- `is_stiff` — a bool that steers explicit vs. implicit engines (Section 4).
- `f_jax(t, y, params)` — the JAX-traceable right-hand side for one trajectory: a state `y` of
  shape `(dim,)` and its parameters `params` of shape `(P,)` in, `dy/dt` out; gradsolve vmaps it
  over the ensemble.

A richer problem object may carry more members; gradsolve ignores them. A plain class with
those six members is enough — [`docs/quickstart.md`](quickstart.md) has the canonical Lorenz
example (`name = "user_lorenz"`, `dim = 3`, nonstiff, sixteen trajectories with one parameter
each) and is the `problem`, `y0`, `params` that the snippets below reuse.

`"user_lorenz"` is deliberately not one of gradsolve's registered field names (nonstiff:
`lorenz`, `vdp`, `lorenz96_<d>`, `linear_ladder_<D>`; stiff: `robertson`, `hires`; plus any
field you register with `register_jax_field` or `register_linstiff`). A user's own right-hand
side therefore gets no fused kernel, only the general-purpose engines that differentiate
through `f_jax` directly (Section 5).

---

## 2. `solve()` vs. `grad_closure()`

Both live in `gradsolve/api.py` and share the same keyword arguments (`engine`, `fused`,
`rtol`, `atol`, `device`, `batch_n`, `accuracy_target`, `saveat`, `saveat_dense`; see
[`docs/api.md`](api.md)):

| | `gradsolve.solve(...)` | `gradsolve.grad_closure(...)` |
|---|---|---|
| Returns | a `SolveResult` (`y_final`, step counts, `solver`, `route`; plus `y_saved`/`ts_saved` when `saveat` is given) — a value, computed once | a closure, by default `params -> y_final` (`wrt` selects `params`, `y0` or both), meant to be traced under `jax.grad`/`jax.jit`/`jax.vmap` |
| Routes via | `choose_engine(..., need_grad=False)` | `choose_engine(..., need_grad=True)` |
| Extra kwargs | — | `wrt`, `precision`, `remat` (`remat` in Section 6; `wrt` and `precision` in [`docs/api.md`](api.md)) |
| Unsupported engine | falls back (diffrax if installed, else `rodas5p_replay` if stiff / `tsit5_replay`); `result.route` says why | falls back to the same replay lanes (Section 5) |
| Unknown `engine=` name | raises `ValueError` listing the registry keys | same, plus accepts the non-registry name `"warp_replay"` |
| Engine has no reverse path | n/a | raises `ValueError(f"engine {name!r} has no reverse closure implemented")` — e.g. `engine="cuda_tsit5"` |

The asymmetry that matters in practice: for every record-and-replay engine, `grad_closure`
records the accepted step sequence eagerly at call time, at the concrete `params` you pass in
(a Warp kernel launch for a registered field, a JAX `while_loop` or host loop otherwise),
before handing you back a pure-JAX closure. `solve()` never does this — it just runs the
chosen engine forward.
That's why `grad_closure` takes longer to construct than `solve` takes to run, and why the
closure it returns is only valid *near* the `params` you built it with (Section 5).

Two input-handling facts worth knowing up front:

- **`solve()` promotes float32 inputs; `grad_closure()` does not cast.** `solve()` on any
  general engine (`diffrax`, `tsit5_replay`, `fixed_step_tsit5`, ...) accepts float32
  `y0`/`params` and promotes them to float64 without raising or warning. `grad_closure()` on
  the general path does not cast its arguments: without `saveat` a float32 `params` gives a
  float32 gradient, and with `saveat` set float32 `y0`/`params` raise a `lax.scan` carry-type
  `TypeError`, so pass float64. The only genuine float32 path is the registered-field replay
  (`precision="float32"` on `warp_replay`/`warp_rosenbrock`).
- **`saveat` (dense output) exists on the JAX scan/replay lanes only.** The fused Warp
  kernels, `cuda_tsit5`, `cuda_rosenbrock23`, and `diffrax` return a final state only and
  raise `ValueError` if you name them explicitly with `saveat` set in `solve()`; in
  `grad_closure()` the names `warp_ode`/`warp_rosenbrock` refer to the replay lanes and
  accept `saveat`. `engine="auto"` routes to a dense-capable lane in both.

You can always sidestep routing with an explicit `engine=`. `engine="diffrax"` is the
universal fallback (`jax.vmap(diffrax.diffeqsolve(...))`, `Kvaerno5` if stiff else `Tsit5`,
`RecursiveCheckpointAdjoint` for the reverse pass):

```python
result = gradsolve.solve(problem, y0, params, engine="diffrax", device="cpu")
print(result.solver, result.y_final.shape)    # -> diffrax (16, 3)  (problem/y0/params as in the quickstart)

final_states = gradsolve.grad_closure(problem, y0, params, engine="diffrax", device="cpu")
loss = lambda p: jnp.sum(final_states(p) ** 2)
gradient = jax.grad(loss)(jnp.asarray(params))
print(bool(jnp.all(jnp.isfinite(gradient))))   # -> True
```

---

## 3. Engine routing: `choose_engine` / `DECISION_MAP`

`gradsolve.dispatch.choose_engine(dim, stiff, need_grad, ...)` is a **pure function** — no side
effects, never raises — mapping the workload's three salient axes to the engine that is faster
for that corner:

```python
from gradsolve.dispatch import choose_engine

for dim, stiff, grad in [(3, False, False), (3, False, True),
                          (3, True, False), (3, True, True),
                          (1000, False, False), (1000, True, True)]:
    print(dim, stiff, grad, "->", choose_engine(dim=dim, stiff=stiff, need_grad=grad))
```
```
3     False  False  ->  cuda_tsit5
3     False  True   ->  warp_replay
3     True   False  ->  warp_rosenbrock
3     True   True   ->  warp_rosenbrock
1000  False  False  ->  diffrax
1000  True   True   ->  rodas5p_replay
```

The auditable version of this table is `gradsolve.dispatch.DECISION_MAP` — a list of dict rows,
one per `(dim-side, stiff, need_grad)` case, each carrying an `evidence` string giving the
rationale for that row's routing. A test in the suite asserts `choose_engine` agrees with
`DECISION_MAP` on every row, so the table can't drift from the code. Three module-level
constants gate whether a fused engine is actually reachable, each overridable per-call
(`stiff_fused_enabled=`, `cuda_tsit5_enabled=`, `cuda_rosenbrock23_enabled=`) for testing
without mutating module state:

- `STIFF_FUSED_ENABLED` — gates `warp_rosenbrock` (`True` because the fused engine is both
  faster and more accurate than the fixed-step IMEX scan engine (`fixed_step_imex`) on
  Robertson).
- `CUDA_TSIT5_ENABLED` — gates `cuda_tsit5` (`True` because the CUDA kernel stays
  register-resident up to `dim=64` with no spill).
- `CUDA_ROSENBROCK23_ENABLED` — gates `cuda_rosenbrock23`, a hand-written CUDA forward-only
  stiff Rosenbrock23 kernel for `dim ≤ CUDA_ROSEN_NVAR_CEILING` (16). Currently `False`, so
  it is reachable only by explicit `engine="cuda_rosenbrock23"`.

Two constants set the dimension crossovers. `NVAR_CEILING = 64` is the register limit: above
it the fused kernels' per-thread register vector exceeds the GPU register budget, so
`choose_engine` routes to `diffrax` (forward) / `tsit5_replay` or `rodas5p_replay` (gradient)
instead; the fixed-step `lax.scan` engines (`fixed_step_tsit5`, `fixed_step_imex`) still exist
but accept and ignore `rtol`/`atol`, and are reachable only by explicit `engine=`.
`CUDA_NVAR_CEILING = 16` bounds the sub-band of the
low-dimension forward case where the hand-written CUDA `cuda_tsit5` kernel is the faster
choice; `16 < dim ≤ 64` stays on `warp_ode`.

If the chosen engine's `supports(problem)` is `False` — no `nvcc`/Warp available, or the
problem has no registered field — both `solve` and `grad_closure` fall back rather than raise:
gradients (and `saveat`) go to the record-and-replay engine of the problem's class
(`rodas5p_replay` if stiff, `tsit5_replay` otherwise); a forward-only solve goes to diffrax
when the extra is installed, else that same replay engine (`api._fallback`). That's why the
quickstart's Lorenz example (an unregistered field, run on a machine without CUDA `nvcc`)
lands on `diffrax` rather than `cuda_tsit5`, even though `choose_engine(3, False, False)` says
`cuda_tsit5`.

---

## 4. Stiff vs. nonstiff

`is_stiff` is the second routing axis, and it changes the *family* of engine, not just which
one: explicit Runge-Kutta (Tsit5) for nonstiff, implicit/linearly-implicit for stiff.

| Regime | Forward engines | Why |
|---|---|---|
| nonstiff | `cuda_tsit5`, `warp_ode`, `fixed_step_tsit5` | explicit 7-stage Tsit5 (Tsitouras 2011); `fixed_step_tsit5.supports(problem)` is literally `not problem.is_stiff` |
| stiff | `warp_rosenbrock`, `cuda_rosenbrock23`, `fixed_step_imex` | per-step linear solve against the Jacobian — L-stable, survives problems explicit steppers can't |
| either | `diffrax` | picks `Kvaerno5` (ESDIRK) if `problem.is_stiff` else `Tsit5`, at `solve`/`grad_closure` time |

`fixed_step_imex.supports()` actually returns `True` unconditionally (its linearly-implicit
Euler-like step, `(I - h J) dy = h f(t, y)`, survives both regimes) — it's `choose_engine`'s
decision to prefer other engines everywhere that keeps it off the auto-routed path entirely:
cheaper explicit engines on nonstiff problems, and — at **high dimension + stiff**, where the
fused kernel is out of register budget — `diffrax` (forward) / `rodas5p_replay` (gradient)
instead. `fixed_step_imex` stays reachable by explicit `engine=` on any stiff problem at any
dimension.

Robertson (`dim=3`, stiff) as an inline duck-typed problem. Because this is an *unregistered*
field on a machine with no Warp/`nvcc`, `choose_engine` still names the fused `warp_rosenbrock`
it would route to on a GPU, but `solve` falls back to `diffrax` (or `rodas5p_replay`, the
general stiff record-and-replay engine, if the `diffrax` extra isn't installed):

```python
import gradsolve, jax.numpy as jnp, numpy as np
from gradsolve.dispatch import choose_engine, choose_remat

class Robertson:                  # classic 3-species stiff kinetics
    name = "user_robertson"; dim = 3; t0 = 0.0; t1 = 1.0; is_stiff = True
    def f_jax(self, t, y, p):
        k1, k2, k3 = p[0], p[1], p[2]
        return jnp.stack([-k1 * y[0] + k2 * y[1] * y[2],
                          k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] ** 2,
                          k3 * y[1] ** 2])

problem = Robertson()
y0 = np.tile([1.0, 0.0, 0.0], (4, 1))
params = np.tile([0.04, 1e4, 3e7], (4, 1))

print(choose_engine(dim=problem.dim, stiff=problem.is_stiff, need_grad=False))  # -> warp_rosenbrock
print(choose_engine(dim=problem.dim, stiff=problem.is_stiff, need_grad=True))   # -> warp_rosenbrock
print(choose_remat(dim=problem.dim, stiff=problem.is_stiff))                    # -> False (dim=3 < 16)

result = gradsolve.solve(problem, y0, params, device="cpu")
print(result.solver, result.y_final.shape)   # -> diffrax (4, 3)  (unregistered field -> fallback; rodas5p_replay without diffrax)
```

Why stiff routing distinguishes between engines rather than always using `fixed_step_imex`:
the `fixed_step_imex` scan (a fixed 10,000 first-order steps, explicit `engine=` only) accepts
`rtol`/`atol` but ignores them, so it over-resolves a problem like Robertson. The *fused*
`warp_rosenbrock` record-and-replay reverse (Section 5) is an adaptive second-order replay that
honours the tolerances, which is why `choose_engine` prefers it at low-dimension stiff instead
of leaving `fixed_step_imex` to serve both forward and grad there. For tight tolerances,
request the fifth-order `rodas5p_replay` (Steinebach 2023) explicitly with
`engine="rodas5p_replay"`; `choose_engine` does not branch on accuracy and selects
`rodas5p_replay` automatically only for stiff gradients above `dim=64`, while `api._fallback`
selects it for gradients when the problem has no registered stiff field, and for forward
solves only when diffrax is not installed.

---

## 5. When the record-and-replay adjoint applies

This is `gradsolve`'s core mechanism (`gradsolve/warp/warp_replay.py`), and it's what
`warp_replay` and `warp_rosenbrock`'s reverse path both are: the fused Warp kernel runs the
forward integration once, at concrete `params`, and records the sequence of accepted step
sizes (`dt`) it took. The gradient is then computed by re-running that *exact same* per-step
arithmetic (`tsit5_step` / `rosenbrock23_step`) as a fixed-length `jax.lax.scan` over the
recorded `dt`s — a scan is natively reverse-mode differentiable, so no custom adjoint or
kernel backward pass is needed. Trajectories that finish early are zero-padded; a `dt=0` step
is exactly an identity map (`y_next = y`, `dy_next/dy = I`, `dy_next/dp = 0`), so both values
and gradients flow correctly through the padded tail.

**The frozen-controller caveat, stated in the module docstring and worth repeating:** this
differentiates the *realized* trajectory with the step-size controller **frozen** — `dt` is
data, not a differentiable function of the parameters. The gradient is the exact discrete
adjoint of the replayed, fixed-step-sequence integration, *not* of the adaptive solver as a
parameter-dependent algorithm. `grad_closure` records once at the `params` you call it with;
the closure it returns is valid in a neighborhood of that point. If you move far enough that
the controller would have taken a different step sequence, call `grad_closure` again to
re-record.

Where it applies — three variants, all in `gradsolve/warp/warp_replay.py` unless noted:

- **`warp_replay`** (nonstiff) — needs NVIDIA Warp and a registered nonstiff field: `lorenz`,
  `vdp`, `lorenz96_<d>`, `linear_ladder_<D>`, or one added with `register_jax_field`
  (`warp_ode._field_for`). This is what `choose_engine` returns for low-dimension + nonstiff +
  `need_grad=True`.
- **`warp_rosenbrock`'s reverse** (stiff) — same mechanism, `rosenbrock23_step` in place of
  `tsit5_step`; needs a registered stiff field: `robertson`, `hires`, or one added with
  `register_linstiff` or `register_jax_field(..., stiff=True)` (`warp_rosenbrock._field_for`).
- **The general-RHS pure-JAX fallback** (`gradsolve/solvers/tsit5_replay.py`,
  `make_tsit5_frozen_mesh_closure`) — applies to **any nonstiff `f_jax`**, registered field or
  not; instead of a Warp kernel it records the mesh with a plain NumPy host loop for small CPU
  ensembles (fewer than 32 trajectories) and otherwise with a batched `jax.lax.while_loop` on
  the device (`gradsolve/solvers/record_jax.py`); `recorder="host"` or `"jax"` forces either.
  This is what actually runs under the quickstart's Lorenz example (n=16 on a CPU, so
  the host loop): that problem carries no registered Warp field, so `warp_ode.supports()`
  returns `False`, the router never enters the Warp branch, and the returned route reads
  `actual="tsit5_replay"`, `reason="no-registered-field"` — which is why that closure works on
  a machine with no Warp/CUDA at all. (A field that matches by name but has a different param
  layout is a separate path: there the Warp build is attempted, its
  `IndexError`/`KeyError`/`ValueError`/`TypeError` is caught, and the reason reads
  `warp-build-failed:<ExceptionName>`.)

One more named engine exists **outside** this mechanism, reachable only by an explicit
`engine="fused_rosenbrock_backward"` override (never an `"auto"`/`choose_engine` target): a
*genuine* single-kernel fused Warp backward (`gradsolve/warp/fused_backward.py`, via
`jax.custom_vjp`), rather than record-and-replay. It is opt-in because it is sensitive to
conditioning and no faster than the record-and-replay path.

**What doesn't get record-and-replay at all:** `cuda_tsit5` is forward-only by construction —
it has no `_api_reverse`, so `grad_closure(engine="cuda_tsit5")` raises `ValueError` once
`supports(problem)` is `True`; on a problem it doesn't
support it reroutes instead, same as any other engine (Section 2). And at high dimension
(`dim > 64`) there is no fused kernel to record from in the first place — the auto-routed
default there is `tsit5_replay` / `rodas5p_replay`, which do record and replay; an explicit
`engine="fixed_step_tsit5"` / `"fixed_step_imex"` request instead gets their own native
`lax.scan` reverse-diff — no recording step, no frozen-controller caveat, because there the
entire fixed-step schedule is already parameter-independent by construction.

The saving comes from differentiating a fixed-length replay of recorded steps rather than an
adaptive loop. When the right-hand side is nearly free to evaluate (for example
`linear_ladder`), fixed per-gradient costs dominate and the advantage over diffrax disappears.

---

## 6. The `remat` policy

`grad_closure(..., remat=None)` (the default) calls `gradsolve.dispatch.choose_remat(dim, stiff,
batch_n=None) -> bool` to decide whether the record-and-replay scan wraps each step in
`jax.checkpoint` — recompute the forward step during the backward sweep instead of storing it.
This is **grad-transparent** (same value, same gradient) and trades compute for memory. The
warp record-and-replay paths (`warp_replay`, `warp_rosenbrock`'s replay) and `rodas5p_replay`
consume it; the plain scan engines (`fixed_step_tsit5`, `fixed_step_imex`) and `tsit5_replay`
accept and ignore the kwarg; remat is not implemented for those engines.

```python
from gradsolve.dispatch import choose_remat

print(choose_remat(dim=3, stiff=True))    # -> False
print(choose_remat(dim=32, stiff=True))   # -> True
```

The policy (per `gradsolve/dispatch.py`):

- **Nonstiff (Tsit5 replay): always remat.** At low dimension it is an outright *speed*
  advantage as well as a large memory saving. At mid/high dimension the speed is unchanged
  but memory still drops sharply — so it's never a net loss, and it raises the batch size you
  can reach before running out of memory.
- **Stiff (Rosenbrock23 replay): remat only at `dim ≥ STIFF_REMAT_DIM` (16).** Below that
  threshold the per-step tape (`jacfwd` over a `D×D` Jacobian plus a `D³` linear-solve
  residual) fits in memory anyway, so remat there is a pure speed *cost* with no benefit
  — off. At/above 16 the tape grows too large without it (at `dim=64` the replay runs out of
  memory without remat), so the modest speed cost is what makes those problems fit at all.

`choose_remat` also accepts `batch_n` (a larger batch makes remat more valuable), but the
policy currently branches on `dim` and `stiff` only; `batch_n` does not yet change the
decision.

---

## 7. Fused kernels from your own JAX RHS

The fused engines (`warp_ode`, `warp_replay`, `warp_rosenbrock`) run only on a *registered*
analytic field; the library provides Lorenz, Van der Pol, Lorenz-96, the linear ladder,
Robertson and HIRES as built-in fields. Any other
`f_jax` takes the general record-and-replay path (Section 5), which is correct but not the fused
kernel. `gradsolve.register_jax_field` lets you get the fused kernel for your own RHS: it walks
`jax.make_jaxpr(f_jax)` (and `jax.jacfwd(f_jax)` for a stiff Jacobian) and emits the same
`@wp.func` field the built-ins use, so a `Problem` whose `name` matches then routes to the
fused engines exactly as a built-in does.

```python
import gradsolve, jax.numpy as jnp, numpy as np

class DiffusiveRing:                       # a fresh nonstiff system, params (a, b)
    name = "user_diffusive_ring"; dim = 20; t0 = 0.0; t1 = 1.0; is_stiff = False
    def f_jax(self, t, y, p):
        a, b = p[..., 0], p[..., 1]
        lap = jnp.roll(y, -1, axis=-1) - 2.0 * y + jnp.roll(y, 1, axis=-1)
        return a[..., None] * lap - y * y * y + b[..., None]

problem = DiffusiveRing()
gradsolve.register_jax_field(problem.name, problem.f_jax, problem.dim, n_params=2, stiff=False)

y0 = np.random.default_rng(0).uniform(-1, 1, size=(8, 20))
params = np.column_stack([np.linspace(0.5, 1.5, 8), np.linspace(-0.5, 0.5, 8)])

result = gradsolve.solve(problem, y0, params)             # -> route.actual == "warp_ode"
final_states = gradsolve.grad_closure(problem, y0, params)  # -> route.actual == "warp_replay"
```

This example needs `pip install "gradsolve[warp]"`; it runs on Warp's CPU device, so no GPU is
required.

`solve`/`grad_closure` take `fused="auto" | True | False` (default `"auto"`). On `"auto"`, a
registered field routes to the fused engine and an unregistered problem takes the general
path unchanged; the RHS is translated only to *check* whether it could be fused (adoption
stays opt-in via `register_jax_field`/`fused=True`). `fused=True` demands the fused kernel
and re-raises if the RHS is untranslatable; `fused=False` skips translation entirely. The
translator covers a fixed primitive subset (arithmetic, comparisons, the elementwise
transcendentals, and static-index reshapes, slices and concatenations); anything outside it —
`jax.lax.cond`, say — raises `UnsupportedRHS`, and `fused="auto"` then records the reason on
`SolveResult.route.reason` and falls back to the general path. See
[`docs/api.md`](api.md#gradsolveregister_jax_field) for the full primitive list and
`examples/08_fused_kernel_from_jax.py` for the runnable version.

---

## Where to look next

- [`docs/api.md`](api.md) — the full public surface: every `SolveResult` field, every
  `ENGINE_REGISTRY` entry.
- [`docs/quickstart.md`](quickstart.md) — the shortest path from zero to a forward solve and a
  gradient.
- [`README.md`](https://github.com/ECLIPSE-AI4Science/gradsolve#readme) — the project overview.
- `gradsolve/base.py`, `gradsolve/api.py`, `gradsolve/dispatch.py`, `gradsolve/warp/warp_replay.py`,
  `gradsolve/solvers/tsit5_replay.py` — the source this guide describes.
