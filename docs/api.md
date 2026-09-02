# gradsolve API reference

gradsolve is a library of differentiable ensemble solvers for differential equations on
GPUs, written in JAX. This page documents its public interface: the two entry points
(`solve`, `grad_closure`), the `Problem` protocol your model must satisfy, the `SolveResult`
it returns, and the engine registry the routing layer dispatches to. For the project
overview and installation instructions see the [README](https://github.com/ECLIPSE-AI4Science/gradsolve#readme) and the
[quickstart](quickstart.md).

```python
import gradsolve
```

Importing `gradsolve` enables `jax_enable_x64` (accuracy work assumes float64) unless
`GRADSOLVE_X64=0` is set in the environment before import.

> With `GRADSOLVE_X64=0`, `tsit5_replay`, `vern7_replay`, `rodas5p_replay` and the fixed-step
> scans run in float32 without raising, even though `precision="float64"` is the only value they
> accept and the result is still labelled float64. Only the Warp engines raise in this case. Keep
> x64 on (the default) for accuracy work.

---

## `gradsolve.solve`

```python
def solve(
    problem,
    y0,
    params,
    *,
    saveat=None,                       # dense/time-grid output; see the saveat section below
    saveat_dense: bool = False,
    engine: str = "auto",
    fused: str | bool = "auto",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    device: str = "cpu",
    batch_n: int | None = None,
    accuracy_target: float | None = None,
) -> SolveResult
```

Runs one batched, forward ensemble ODE solve and returns a `SolveResult`. `problem` is any
object satisfying the `Problem` protocol below; `y0` is the initial state, shape `(n, dim)`;
`params` is the per-trajectory parameter array, shape `(n, P)`.

With `engine="auto"` (the default), `solve` resolves an engine name via
`gradsolve.dispatch.choose_engine(dim=problem.dim, stiff=problem.is_stiff, need_grad=False, ...)`
and dispatches to it. Pass a canonical registry name (e.g. `engine="fixed_step_tsit5"`,
`"warp_ode"`, `"diffrax"`) to force a specific engine; an unrecognized name raises
`ValueError` listing the known keys. If the resolved engine's `supports(problem)` is
`False` — e.g. `warp_ode`/`cuda_tsit5` require NVIDIA Warp/`nvcc`, absent on a machine
without an NVIDIA GPU — `solve` falls back to diffrax (when the `diffrax` extra is installed) for a
forward-only solve, and otherwise to the general record-and-replay engine of the problem's
class (`rodas5p_replay` if stiff, `tsit5_replay` otherwise) rather than raising; `result.route`
records the reroute and its reason. `rtol`/`atol` are honoured by every auto-routed engine;
the fixed-step scans (explicit `engine=` only) accept them nominally. `batch_n` and
`accuracy_target` are accepted routing hints not yet used by
the routing policy. The returned `SolveResult.solver` is stamped with the routing
name that ran (a registry key, or `"warp_replay"` on the `saveat` path), not the engine's
internal descriptive string.

With `problem`, `y0` and `params` defined as in the [quickstart](quickstart.md) (the
`user_lorenz` model, 16 trajectories):

```python
import gradsolve   # problem, y0, params defined as in the quickstart
result = gradsolve.solve(problem, y0, params, device="cpu")
print(result.solver, result.y_final.shape)   # diffrax (16, 3): user_lorenz has no registered fused field, so the general engine runs (tsit5_replay without the diffrax extra)
```

## `gradsolve.grad_closure`

```python
def grad_closure(
    problem,
    y0,
    params,
    *,
    wrt="params",                      # 'params' | 'y0' | ('y0', 'params')
    saveat=None,                       # dense/time-grid output; see the saveat section below
    saveat_dense: bool = False,
    precision: str = "float64",        # see the precision section below
    engine: str = "auto",
    fused: str | bool = "auto",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    device: str = "cpu",
    batch_n: int | None = None,
    accuracy_target: float | None = None,
    remat: bool | None = None,
)
```

Returns a `jax.grad`-able closure over the initial state, the parameters, or both. By
default (`wrt="params"`) that closure is `params -> y_final[n, dim]`. Calling `jax.grad` on a
scalar function of the closure's output gives exact reverse-mode gradients with respect to
`params`. With `engine="auto"`, routing runs the same `choose_engine` as `solve` but with
`need_grad=True`, which can additionally resolve to `"warp_replay"` (the record-and-replay
adjoint: a fused forward records the accepted step mesh once, and the gradient is a
fixed-length `lax.scan` replay over it — reverse-mode-differentiable by construction, no
custom kernel backward pass). For a right-hand side with no registered Warp field,
`grad_closure` falls back to the general record-and-replay engine for the problem's stiffness
class (`gradsolve.solvers.tsit5_replay` for non-stiff, `gradsolve.solvers.rodas5p_replay` for
stiff), so it works on any `f_jax`, not only the library's own registered problems (Lorenz,
Van der Pol, the linear ladder). The mesh for that general engine is recorded by a batched
`jax.lax.while_loop` on the device (`gradsolve.solvers.record_jax`) on a GPU or for ensembles
of 32 or more trajectories, and by a host loop otherwise; the choice is automatic, and setting
`GRADSOLVE_RECORDER=host` or `GRADSOLVE_RECORDER=jax` in the environment forces the host or
device recorder. Both recorders produce the same accepted step mesh up to floating-point
rounding.

Every record-and-replay engine (`warp_replay`, `warp_rosenbrock`, and the general-RHS
fallback) shares one property, stated in each module's docstring: the step-size
controller is frozen at the recording call. The closure differentiates the *realized*
trajectory — dt is data, not a differentiable function of the parameters — so the gradient
is the exact discrete adjoint of that replayed, fixed-step integration, valid in a
neighborhood of the `params` passed to `grad_closure`. Re-record (call `grad_closure`
again) if you move far from that point.

```python
import gradsolve, jax, jax.numpy as jnp

final_states = gradsolve.grad_closure(problem, y0, params, device="cpu")
loss = lambda p: jnp.sum(final_states(p) ** 2)
gradient = jax.grad(loss)(jnp.asarray(params))  # gradient w.r.t. params, per trajectory
```

### `wrt`: gradients w.r.t. the initial state

`wrt` selects which inputs the returned closure takes and is differentiable in. Return
shapes mirror the corresponding input shapes.

| `wrt` | closure signature | closed over |
|---|---|---|
| `"params"` (default) | `params -> y_final[n, dim]` | `y0` |
| `"y0"` | `y0 -> y_final[n, dim]` | `params` |
| `("y0", "params")` | `(y0, params) -> y_final[n, dim]` | — |

The mesh is recorded once, at the `(y0, params)` passed to `grad_closure`, and frozen for
the closure's lifetime. A y0 gradient therefore carries exactly the same frozen-controller
caveat as a parameter gradient — perturbing `y0` replays the recorded mesh rather than
re-running the step-size controller — and is valid in a neighborhood of the recording
point.

```python
# Fit an initial condition instead of a rate constant.
final_states = gradsolve.grad_closure(problem, y0, params, wrt="y0")
loss = lambda z: jnp.sum((final_states(z) - y_target) ** 2)
gradient_y0 = jax.grad(loss)(jnp.asarray(y0))  # shape == y0.shape
```

One exclusion: `fused_rosenbrock_backward`'s `custom_vjp` differentiates parameters by
construction, so it is params-only. Requesting a y0 gradient from it does not fail and does
not silently ignore the request — it reroutes to the stiff replay engine and records why on
the closure's `.route` (below).

### `saveat`: dense output (states at requested times)

`saveat=<sorted times within [t0, t1]>` makes the closure return the states at those times,
`(n, k, dim)`, instead of the final state — so a time-series loss differentiates through the
solve. `solve(..., saveat=ts)` correspondingly fills `SolveResult.y_saved (n, k, dim)` and
`SolveResult.ts_saved (k,)` (host NumPy). `saveat=None` (default) returns the final state
only.

```python
ts = np.linspace(0.1, 1.0, 12)
final_states = gradsolve.grad_closure(problem, y0, params, saveat=ts)
loss = lambda p: jnp.mean((final_states(p) - observations) ** 2)
gradient = jax.grad(loss)(jnp.asarray(params))
```

**Saved states are integrated, not interpolated, by default.** The engine records which step
brackets each requested time, then takes one genuine step of its own method to exactly that
time. So a saved state has the same accuracy as the final state; no interpolant is involved,
which matters because an adaptive controller takes wide steps where the problem is easy.

`saveat_dense=True` (requires `saveat`) evaluates the method's own continuous extension at
each save time instead of taking an extra step to it, which saves one solver step per
requested time at some cost in accuracy; it is implemented on `rodas5p_replay` only and
raises `ValueError` on any other engine.

Dense output adds memory independent of the step count: beyond the mesh and tape, the scan
carries only the active bracket and the `k` outputs, about `k*n*dim*8` bytes. The full
`(S, dim)` state history is never stored.

`saveat` composes with `wrt` — a time-series loss differentiated w.r.t. `y0` is
`grad_closure(..., wrt="y0", saveat=ts)`. The mesh is still recorded once, at the passed
`(y0, params)`; a fit that travels far from that point should re-record (see
`examples/07_saveat_timeseries_fit.py`, which does exactly that).

Scope — dense output exists on the **JAX scan and replay engines only**:

| engine | `saveat` |
|---|---|
| `tsit5_replay`, `warp_replay` (nonstiff record-and-replay) | yes |
| `rodas5p_replay` (stiff record-and-replay; also the only engine accepting `saveat_dense`) | yes |
| `vern7_replay` (nonstiff, explicit `engine=` only) | yes |
| `fixed_step_tsit5` (10k-step scan) | yes |
| `warp_rosenbrock` replay (stiff, via `grad_closure`) | yes |
| `fixed_step_imex` (order-1) | yes |
| fused Warp kernels, `cuda_tsit5`, `diffrax` | **no** — `ValueError` |

`fixed_step_imex` is first order, so saved states carry its lower accuracy. The fused kernels
and the `cuda_tsit5` engine hold the integration in registers and emit a final state; diffrax
has its own `SaveAt` which this API deliberately does not wire. Naming one of them explicitly
with `saveat` set raises `ValueError` rather than silently ignoring the request.
`engine="auto"` routes to a dense-capable engine. Unsorted or out-of-domain times raise
`ValueError`; `y_saved[:, -1] == y_final` exactly when `t1` is in `saveat`.

### `precision`: float32 record and replay

`precision="float32"` records the step mesh with the f32 Warp kernel and replays in f32.
The closure casts its inputs to the run precision, so the arithmetic is the one you asked
for rather than whatever your dtypes happened to promote to.

With a registered field and Warp installed — here the built-in Lorenz field, reached by giving
the quickstart's one-parameter `Lorenz` class the name `"diffeqgpu_lorenz"` (`"lorenz"` is the
same field with a three-column `[sigma, beta, rho]` parameter layout; user fields register via
`register_jax_field`):

```python
class Lorenz:                     # as in the quickstart, but named for the registered field
    name = "diffeqgpu_lorenz"; dim = 3; t0 = 0.0; t1 = 1.0; is_stiff = False
    def f_jax(self, t, y, p):
        rho = p[0]
        return jnp.stack([10.0 * (y[1] - y[0]), rho * y[0] - y[1] - y[0] * y[2], y[0] * y[1] - (8.0 / 3.0) * y[2]])

final_states = gradsolve.grad_closure(Lorenz(), y0, params, engine="warp_replay", precision="float32")
loss = lambda p: jnp.sum(final_states(p) ** 2)
gradient = jax.grad(loss)(jnp.asarray(params, jnp.float32))  # float32
```

**Scope — the registered Warp field route only** (`warp_replay` nonstiff, `warp_rosenbrock`
stiff). The general-RHS recorders (`tsit5_replay`, `vern7_replay`, `rodas5p_replay`) are
float64 throughout, as are the fixed scans; none of them can honour an f32 *request*, so they
raise `ValueError` rather than return a float64 result under an f32 label.

On the general-RHS and fixed-step engines pass float64 `y0` and `params`; float32 inputs are
used as given and are not cast (the run-precision cast belongs to the Warp float32 route). In
an x64 process a plain `solve` promotes them through its float64 arithmetic, a closure returns
a gradient at the dtype given, and `saveat` on a record-and-replay engine raises a `lax.scan`
carry-type `TypeError`.

In float32 the gradients agree with a float64 replay of the **same f32-recorded mesh** to a
relative L2 error of at most 1e-3 with direction cosine at least 1 - 1e-6 (the bound the test
suite enforces); replaying the same mesh isolates arithmetic precision from the choice of
mesh.

The record buffer halves **per step** (4 bytes vs 8). The *total* is about half rather than
exactly half, because f32 is a different controller, not just different arithmetic: its
roundoff changes which steps the error estimate accepts, so it records a slightly different
mesh. That is also why the comparison above replays the f32 mesh rather than comparing
against an f64-recorded run.

f64 needs an x64 process. On the Warp route, `precision="float64"` raises in a non-x64
process rather than replaying in float32 under a float64 label; the general-RHS recorders and
fixed scans do not check (see the note at the top of this page). Ask for f32 explicitly and it
works in any process.

### `.route`: which engine actually ran

Every closure `grad_closure` returns carries a `.route` with three fields — `requested`
(the `engine=` you passed), `actual` (the engine that built the closure), and `reason`
(`None` when the request was honoured; otherwise why it was not). Read it instead of
assuming: an engine is rerouted when it does not `supports()` the problem, when the problem
matches no registered Warp field, or when the request exceeds what the engine can do.

```python
final_states = gradsolve.grad_closure(problem, y0, params, wrt="y0", engine="fused_rosenbrock_backward")
final_states.route.requested   # 'fused_rosenbrock_backward'
final_states.route.actual      # 'tsit5_replay'  — the general-RHS replay engine
final_states.route.reason      # 'y0-unsupported; engine-does-not-support-problem'
# (problem here is the nonstiff, unregistered user_lorenz from the quickstart, so the params-only
#  fused backward first reroutes off the y0 request and then off the nonstiff problem. On a
#  registered stiff field with Warp available, actual reads 'warp_rosenbrock' and reason
#  reads 'y0-unsupported'.)
```

`actual` is a registry key except for `"warp_replay"`, the Warp record-and-replay path, which
has no registry entry. A nonstiff problem with no registered field falls back to the pure-JAX
recorder and reports `actual="tsit5_replay"` with `reason="no-registered-field"`, which is how
you tell it apart from a genuine Warp record-and-replay.

Passing an unregistered `engine=` name (not a key in the registry and not `"warp_replay"`)
raises `ValueError`; an engine with no reverse path implemented (e.g. `engine="cuda_tsit5"`,
which is forward-only by design) raises `ValueError("engine ... has no reverse closure
implemented")` when that engine supports the problem (otherwise the request is rerouted and
`.route` records why).

### `remat`: checkpointing the replay scan

`remat` (default `None` → auto via `gradsolve.dispatch.choose_remat(dim, stiff)`) controls
whether the replay scan is wrapped in `jax.checkpoint`. It affects the warp
record-and-replay paths and `rodas5p_replay` (the plain scan engines and `tsit5_replay`
accept and ignore it); the default policy
is: always remat for the non-stiff Tsit5 replay (faster at low dimension, always less
memory), remat only at or above `dispatch.STIFF_REMAT_DIM`
(16) for the stiff Rosenbrock replay (a modest speed cost that buys the memory headroom
needed to avoid OOM at higher dimension).

---

## `gradsolve.register_jax_field`

```python
def register_jax_field(name, f_jax, dim, n_params, *, stiff=False) -> None
```

Translates a user right-hand side `f_jax(t, y, p)` into the same fused Warp field the
built-in problems (Lorenz, Van der Pol; Robertson, HIRES) use, and registers it under
`name`. A `Problem` whose `name` matches then routes to the fused engines (`warp_ode` /
`warp_replay` nonstiff, `warp_rosenbrock` stiff) exactly as a built-in does — no
hand-written kernel. `register_jax_field` walks `jax.make_jaxpr(f_jax)` (and, for
`stiff=True`, `jax.jacfwd(f_jax)` for the analytic Jacobian) and emits a `@wp.func`; the
codegen is lazy (it runs when the fused kernel is first built for a given precision) and
idempotent (a name already registered is left untouched). `import gradsolve` stays warp-less —
the Warp import happens only when you call this.

The translator covers a fixed subset of JAX primitives: arithmetic (`add/sub/mul/div/neg/
pow/integer_pow`), comparisons (`lt/le/gt/ge/eq/ne`), `abs/sign/max/min`, the elementwise
transcendentals (`exp/log/sin/cos/tan/tanh/sqrt`), `convert_element_type`, and static-shape
ops (`reshape/squeeze/expand_dims/broadcast_in_dim/transpose/slice/split/concatenate/stack/
iota`), with `jit`-wrapped sub-functions inlined. An RHS using anything else (e.g.
`jax.lax.cond`) raises
`gradsolve.warp.jax_field.UnsupportedRHS`, naming the offending primitive. For a stiff field
`n_params` must be `<= dim` (the stiff kernel hands a length-`dim` param vector).

### `fused`: the codegen switch on `solve` / `grad_closure`

`solve` and `grad_closure` take `fused="auto" | True | False` (default `"auto"`):

- **`"auto"`** — a registered generated field (or a built-in) routes to the fused engine as
  usual; an unregistered problem takes the general path, identical to `fused=False`.
  `"auto"` *checks* fused-eligibility (it translates the RHS to see whether it could be
  fused, priming the translator cache) but does not auto-adopt the fused engine — adoption
  is opt-in, via `register_jax_field` or `fused=True`. An `UnsupportedRHS` during that check
  is swallowed and the reason is appended to `SolveResult.route.reason`
  (`"fused-unsupported:<primitive>; fell back to the general path"`).
- **`True`** — demand the fused kernel: translate and register the RHS on this call so the
  problem routes to the fused engine, and let `UnsupportedRHS` propagate (no silent
  fallback).
- **`False`** — never call the translator; the general path runs even for a translatable
  RHS.

`n_params` for this translation is read from the batched `params` the caller already
passes (`params` is `(n, P)`), not from the `Problem` — the library `Problem` protocol has
no `n_params`.

---

## The `Problem` protocol

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

Structural (duck-typed), defined in `gradsolve/base.py`. `gradsolve` reads only these six
members off a problem object — `dim` is the state dimension (drives the register-limit
routing decision), `t0`/`t1` the integration horizon, `is_stiff` selects explicit vs.
implicit engines, and `f_jax(t, y, params)` is the JAX-traceable right-hand side mapping one
trajectory's state `y` of shape `(dim,)` and its parameters `params` of shape `(P,)` to `dy/dt`
(gradsolve vmaps it over the ensemble). Any object exposing these — a plain class, as in the
quickstart, or a richer dataclass that additionally carries reference solutions, batch
constructors or a right-hand side for another framework — satisfies the protocol with no
inheritance required. Everything beyond the six members is the caller's concern, not the
library's.

## `SolveResult`

```python
@dataclass
class SolveResult:
    y_final: np.ndarray                 # (n, dim)
    accepted_steps: np.ndarray = ...    # (n,), int64
    rejected_steps: np.ndarray = ...    # (n,), int64
    solver: str = ""
    y_saved: np.ndarray | None = None   # (n, k, dim); None unless saveat was passed
    ts_saved: np.ndarray | None = None  # (k,), the echoed output times
    route: Route | None = None          # the Route record, set by gradsolve.solve
```

Defined in `gradsolve/base.py`. `y_final` is the final state per trajectory. `accepted_steps`
/ `rejected_steps` carry the per-trajectory step counts needed for warp-divergence metrics
(the step-count dispersion across a batch): adaptive engines (diffrax, the Warp kernels)
report the true measured counts; fixed-step scan engines (`fixed_step_tsit5`,
`fixed_step_imex`) report the constant configured step count repeated `n` times for
`accepted_steps` and zeros for `rejected_steps` — by contract, never left empty. The
length-0 default (`np.empty(0)`) is reserved for a backend whose API genuinely exposes no
step counts. `solver` is the canonical engine key that `solve` resolved to (see above).
`y_saved` holds the states at the requested times and `ts_saved` echoes those times; both are
`None` unless `saveat` was passed (see the `saveat` section). `route` is the `Route` record
(`requested`, `actual`, `reason`) described under `.route` above; `gradsolve.solve` sets it,
and it is `None` on a result built by an engine's own `solve`.

---

## The engine registry

`gradsolve.api.ENGINE_REGISTRY` maps each canonical engine name to an `EngineSpec(name,
supports, solve, reverse)`. `supports(problem) -> bool` and `solve(problem, y0, params, *,
rtol, atol, device) -> SolveResult` are required; `reverse` is the optional closure factory
`grad_closure` calls (`None` means the engine is forward-only). The registered keys:

| Engine | Module | Stiff? | Reverse? | What it is |
|---|---|---|---|---|
| `fixed_step_tsit5` | `gradsolve/solvers/fixed_step_tsit5.py` | non-stiff only | yes (native `lax.scan`) | Classical 7-stage Tsit5 (Tsitouras 2011), run as a *fixed* number of steps (`DEFAULT_N_STEPS = 10_000`) inside one `jax.lax.scan`, `vmap`'d over the batch. Every trajectory takes the same number of steps, so there is no warp divergence, and it is reverse-mode differentiable by construction, but it accepts `rtol`/`atol` without honouring them — explicit `engine=` only; `choose_engine`'s auto route for `dim > 64` is `diffrax` (forward) / `tsit5_replay` (gradient), which do honour them. |
| `fixed_step_imex` | `gradsolve/solvers/fixed_step_imex.py` | stiff and non-stiff | yes (native `lax.scan`) | Fixed-step, order-1 linearly-implicit (Rosenbrock-)Euler: per step solves `(I - h J) dy = h f(t, y)` with `J = df/dy`. A first-order stand-in for an IMEX scheme (accuracy from step count, not order). L-stable, so it survives stiff problems where dense-Jacobian JAX solvers OOM at high dimension, but likewise accepts `rtol`/`atol` without honouring them — explicit `engine=` only; `choose_engine`'s auto route for stiff + high-dimension is `diffrax` (forward) / `rodas5p_replay` (gradient). |
| `diffrax` | `gradsolve/solvers/diffrax_fallback.py` | routes automatically (`Kvaerno5` if stiff, else `Tsit5`) | yes (`RecursiveCheckpointAdjoint`) | Universal catch-all wrapping `jax.vmap(diffrax.diffeqsolve(...))`. The forward-only general engine `choose_engine` names for `dim > 64` and `api._fallback` uses for any forward request the fused kernels cannot serve (when installed), and the reference reverse-mode engine (adaptive, `RecursiveCheckpointAdjoint`), reachable for gradients by explicit `engine="diffrax"` only. |
| `tsit5_replay` | `gradsolve/solvers/tsit5_replay.py` | non-stiff | yes (record-and-replay `lax.scan`) | The reference record-and-replay reverse-mode engine for an arbitrary non-stiff right-hand side. It records the accepted adaptive Tsit5 step mesh once, at the current parameters (numpy host loop on a small CPU ensemble, or a batched JAX `while_loop` on a GPU or for `n >= 32`; see the recorder note above), then replays it as a fixed-length `jax.lax.scan` that is reverse-differentiable through the user's own `f_jax` — no custom kernel, any RHS. `choose_engine`'s general non-stiff route for the gradient at `dim > 64`, and the forward general engine when `diffrax` is not installed. float64 only. |
| `rodas5p_replay` | `gradsolve/solvers/rodas5p_replay.py` | stiff | yes (record-and-replay `lax.scan`, including the per-step Jacobian) | The stiff high-order sibling of `tsit5_replay`. It records the accepted adaptive Rodas5P mesh once (jitted host loop or batched JAX `while_loop`, chosen as for `tsit5_replay`; the Rosenbrock step needs a traceable Jacobian, so the host recorder is jitted rather than plain numpy), then replays it as a fixed-length `lax.scan`, reverse-differentiable including through the per-step automatic-differentiation Jacobian. `choose_engine`'s general stiff route for the gradient at high dimension, and the general stiff forward engine when `diffrax` is absent. float64 only. |
| `vern7_replay` | `gradsolve/solvers/vern7_replay.py` | non-stiff | yes (record-and-replay `lax.scan`) | The high-order sibling of `tsit5_replay`: it records the accepted adaptive Vern7 mesh and replays it as a fixed-length `lax.scan`. Override-only (reachable by `engine="vern7_replay"`), never an `"auto"`/`choose_engine` target. float64 only. |
| `warp_ode` | `gradsolve/warp/warp_ode.py` | non-stiff | via `warp_replay` (not directly) | Fused adaptive Tsit5 as one NVIDIA Warp CUDA kernel launch, one thread per trajectory (forward + record only, float32/float64). Needs `nvcc`/Warp and a registered analytic field (`lorenz`, `vdp`, `linear_ladder_<D>`, `lorenz96_<d>`); `supports()` is `False` without them, in which case callers fall back to `diffrax`/`tsit5_replay`. Registered fields cap the per-thread state vector at `dim ≤ 64` (`dispatch.NVAR_CEILING`) — above that, register spill makes the fused kernel slower than the general path. |
| `warp_rosenbrock` | `gradsolve/warp/warp_rosenbrock.py` | stiff | via record-and-replay (`_reverse_for`) | The stiff sibling of `warp_ode`: a fused, linearly-implicit Rosenbrock23 kernel with an in-kernel analytic Jacobian and LU solve per stage, one thread per trajectory. Registered fields: `robertson`, `hires`, and any registered `linstiff_*` constant-matrix field. Low-NVAR design point (register footprint is `O(dim²)`, tighter than `warp_ode`'s `O(dim)`). |
| `cuda_tsit5` | `gradsolve/cuda/cuda_tsit5.py` | non-stiff | **no** (forward-only by design) | A hand-written CUDA Tsit5 kernel exposed via `jax.ffi`, holding the whole integration in registers: the fast forward-only engine. No reverse path exists (keeping the state in registers is what makes it fast, and that layout does not admit an adjoint), so `grad_closure(engine="cuda_tsit5")` raises `ValueError` once `supports(problem)` is `True` (a registered field on a machine with CUDA). On a problem it doesn't support (no CUDA, or an unregistered field), it reroutes instead; see `.route`. |
| `cuda_rosenbrock23` | `gradsolve/cuda/cuda_rosenbrock23.py` | stiff | **no** (forward-only by design) | The stiff analogue of `cuda_tsit5`: a hand-written CUDA forward-only fused adaptive Rosenbrock23 kernel exposed via `jax.ffi`, one thread per trajectory with an in-kernel analytic Jacobian and a single per-step LU factorisation reused across the three stage solves. Override-only and **disabled by default** (`dispatch.CUDA_ROSENBROCK23_ENABLED = False`), so the stiff, low-dimensional, forward case still routes to `warp_rosenbrock`; it is reachable only by name (`engine="cuda_rosenbrock23"`). No reverse path exists, so `grad_closure(engine="cuda_rosenbrock23")` raises; the reverse stiff engine stays `rodas5p_replay`/`warp_rosenbrock`. |
| `fused_rosenbrock_backward` | `gradsolve/warp/fused_backward.py` | stiff | yes (genuine fused-kernel `wp.Tape` backward) | Override-only (never an `"auto"`/`choose_engine` target): forward delegates to `warp_rosenbrock.solve`; the reverse pass is the *genuine* single-kernel fused Warp Rosenbrock backward (`jax.custom_vjp`-style, via `wp.Tape`), rather than the record-and-replay scan. Kept as an explicit opt-in because it is sensitive to conditioning and roughly on par with the record-and-replay path. |

`warp_replay` appears in the tables above and in `choose_engine`'s return values but is
**not** a key in `ENGINE_REGISTRY` — by design. It has no independent forward solve; it is
the record-and-replay reverse routing target whose forward is `warp_ode` and whose helpers
(`make_replay_closure`, `make_rosenbrock_replay_closure`, `replay_solve_jax`) live in
`gradsolve/warp/warp_replay.py`. `grad_closure` special-cases it directly.

### How routing decides

```python
def choose_engine(
    dim: int,
    stiff: bool,
    need_grad: bool,
    *,
    batch_n: int | None = None,
    accuracy_target: float | None = None,
    stiff_fused_enabled: bool | None = None,
    cuda_tsit5_enabled: bool | None = None,
    cuda_rosenbrock23_enabled: bool | None = None,
) -> str
```

`gradsolve.dispatch.choose_engine` is a pure function (no side effects, never raises) mapping
the workload's three salient axes — state dimension, stiffness, and whether a gradient is
needed — to the engine that is faster for that case. Both `solve` and
`grad_closure` call it under `engine="auto"`. The auditable list of rows it implements is
`gradsolve.dispatch.DECISION_MAP`, one dict per `(dim, stiff, need_grad)` case with an
`evidence` string giving the rationale for that routing; a test asserts `choose_engine` agrees
with `DECISION_MAP` on every row. Three module constants control whether a fused GPU engine
is actually routed to: `STIFF_FUSED_ENABLED` (gates `warp_rosenbrock`, `True` by default),
`CUDA_TSIT5_ENABLED` (gates `cuda_tsit5`, `True`) and `CUDA_ROSENBROCK23_ENABLED` (gates
`cuda_rosenbrock23`, `False`); each is overridable per call via `stiff_fused_enabled=`,
`cuda_tsit5_enabled=` and `cuda_rosenbrock23_enabled=` for testing without mutating module
state.

```python
from gradsolve.dispatch import choose_engine

choose_engine(dim=3, stiff=False, need_grad=False)   # -> "cuda_tsit5"  (dim <= 16, forward-only)
choose_engine(dim=3, stiff=False, need_grad=True)    # -> "warp_replay" (needs a gradient)
choose_engine(dim=1000, stiff=True, need_grad=True)  # -> "rodas5p_replay" (dim > 64)
```

A second pure function, `choose_remat(dim, stiff, *, batch_n=None) -> bool`, decides
whether `grad_closure` wraps the replay scan in `jax.checkpoint` (see the `remat`
discussion above); `grad_closure`'s default `remat=None` calls it automatically.

---

## Runnable examples

`examples/00_standalone.py` and `examples/03_engine_routing.py` depend only on `gradsolve`
and print `OK` on a clean CPU checkout (install as in the
[install section](quickstart.md#install)):

```bash
python examples/00_standalone.py   # your own RHS, no extra dependencies: forward + reverse (FD-checked)
python examples/03_engine_routing.py  # inspect choose_engine's routing table across (dim, stiff, grad)
```

The full set of tutorials (`examples/00`–`09`) build on the same two entry points against a range
of registered problems and the fused reverse path; `examples/README.md` is the full index.
