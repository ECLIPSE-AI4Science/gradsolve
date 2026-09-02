"""Routed public API: solve(engine='auto'|<name>) + a jax.grad-able reverse closure.

Thin layer over the engine roster (scan spines + warp kernels + record-replay adjoints).
The ENGINE_REGISTRY maps each canonical engine name to an EngineSpec (supports, solve,
optional reverse). solve() resolves the engine name (override -> as-is; 'auto' ->
choose_engine()), dispatches the forward solve, falls back to the general
record-and-replay engine of the problem's stiffness class (diffrax for a forward-only
solve, when installed) when supports(problem) is False, and stamps SolveResult.solver with
the registry key so callers get a stable name rather than the engine's internal
descriptive string.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from gradsolve.base import Route, SolveResult
from gradsolve.cuda import (
    cuda_rosenbrock23,  # forward-only hand-CUDA stiff Rosenbrock23 lane (no _api_reverse)
    cuda_tsit5,  # forward-only hand-CUDA Tsit5 lane (no _api_reverse)
)
from gradsolve.dispatch import NVAR_CEILING, choose_engine, choose_remat
from gradsolve.solvers import (
    diffrax_fallback as _diffrax_fallback_mod,  # name="diffrax"
)
from gradsolve.solvers import (
    fixed_step_imex as _fixed_step_imex_mod,  # name="fixed_step_imex"
)
from gradsolve.solvers import fixed_step_tsit5
from gradsolve.solvers import (
    rodas5p_replay as _rodas5p_replay_mod,  # name="rodas5p_replay"
)
from gradsolve.solvers import tsit5_replay as _tsit5_replay_mod  # name="tsit5_replay"
from gradsolve.solvers import (
    vern7_replay as _vern7_replay_mod,  # name="vern7_replay" (opt-in)
)
from gradsolve.solvers.dense import validate_saveat
from gradsolve.warp import fused_backward as _fused_backward
from gradsolve.warp import warp_ode, warp_rosenbrock
from gradsolve.warp import warp_replay as _warp_replay

# NOTE on the imex import: gradsolve/solvers/__init__.py exports `imex` (the thin facade
# in gradsolve/solvers/imex.py) with name="imex_euler".  The underlying implementation
# module is gradsolve/solvers/fixed_step_imex.py with name="fixed_step_imex" — that is the
# registry key we want.  We import the underlying module directly.


@dataclass(frozen=True)
class EngineSpec:
    """One entry in ``ENGINE_REGISTRY``: an engine's forward solve plus optional reverse.

    Attributes
    ----------
    name : str
        Canonical engine key — the value stamped into ``SolveResult.solver`` by
        ``solve()``.
    supports : Callable
        ``(problem) -> bool``; whether this engine can handle the given problem.
    solve : Callable
        ``(problem, y0, params, *, rtol, atol, device) -> SolveResult``; the
        forward solve.
    reverse : Callable, optional
        ``(problem, y0, params, *, rtol, atol, device) -> closure``, where the
        returned closure is ``params -> y_final`` and jax.grad-able. ``None`` for
        forward-only engines (e.g. ``cuda_tsit5``) — ``grad_closure`` raises if
        asked to route there.
    """
    name: str
    supports: Callable           # (problem) -> bool
    solve: Callable              # (problem, y0, params, *, rtol, atol, device) -> SolveResult
    reverse: Optional[Callable] = None  # (problem, y0, params, *, rtol, atol, device) -> closure


def _spec(mod) -> EngineSpec:
    return EngineSpec(
        name=mod.name,
        supports=mod.supports,
        solve=mod.solve,
        reverse=getattr(mod, "_api_reverse", None),
    )


ENGINE_REGISTRY: dict[str, EngineSpec] = {
    "fixed_step_tsit5": _spec(fixed_step_tsit5),
    # General-RHS nonstiff record-and-replay (the reference implementation) — the routed
    # target for any nonstiff f_jax the fused kernels cannot serve (gradients; forward when diffrax
    # is absent). Also nameable: engine="tsit5_replay".
    "tsit5_replay": _spec(_tsit5_replay_mod),
    # Opt-in high-order nonstiff record-and-replay engine. General-RHS: supports any
    # nonstiff f_jax. NOT an auto-routing target — choose_engine/DECISION_MAP never return it;
    # reachable only by explicit engine="vern7_replay".
    "vern7_replay": _spec(_vern7_replay_mod),
    # General-RHS STIFF record-and-replay (jax.jacfwd Jacobian). The routed target for stiff
    # gradients the fused kernel cannot serve (dim > NVAR_CEILING, or no registered field);
    # forward when diffrax is absent.
    "rodas5p_replay": _spec(_rodas5p_replay_mod),
    "fixed_step_imex": _spec(_fixed_step_imex_mod),
    "diffrax": _spec(_diffrax_fallback_mod),
    "warp_ode": _spec(warp_ode),
    "warp_rosenbrock": _spec(warp_rosenbrock),
    # Forward-ONLY hand-CUDA Tsit5 lane (the fast forward-throughput lane). No _api_reverse
    # -> _spec gives reverse=None -> grad_closure(engine="cuda_tsit5") raises (the contract).
    # Routed by choose_engine only when dispatch.CUDA_TSIT5_ENABLED is set.
    "cuda_tsit5": _spec(cuda_tsit5),
    # Forward-only hand-CUDA stiff Rosenbrock23 lane (the stiff analogue of cuda_tsit5). No
    # _api_reverse -> reverse=None -> grad_closure(engine="cuda_rosenbrock23") raises. Disabled
    # by default; enable with ``dispatch.CUDA_ROSENBROCK23_ENABLED = True`` or by naming the
    # engine explicitly (``engine="cuda_rosenbrock23"``).
    "cuda_rosenbrock23": _spec(cuda_rosenbrock23),
    # Named, OVERRIDE-ONLY fused-backward reverse engine. NOT a
    # choose_engine/'auto' target — it is reachable only by explicit engine= override
    # (the fused stiff backward is conditioning-sensitive and no faster than the
    # record-and-replay path). Its forward delegates to warp_rosenbrock.solve; its
    # reverse is the genuine single-kernel fused Warp backward (jax.custom_vjp).
    "fused_rosenbrock_backward": EngineSpec(
        name="fused_rosenbrock_backward",
        supports=warp_rosenbrock.supports,
        solve=warp_rosenbrock.solve,
        reverse=_fused_backward.fused_rosenbrock_grad_closure,
    ),
}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _resolve_engine(problem, engine, *, batch_n, accuracy_target, need_grad) -> str:
    if engine != "auto":
        if engine not in ENGINE_REGISTRY:
            raise ValueError(
                f"unknown engine {engine!r}; known: {sorted(ENGINE_REGISTRY)}"
            )
        return engine
    return choose_engine(
        dim=problem.dim,
        stiff=problem.is_stiff,
        need_grad=need_grad,
        batch_n=batch_n,
        accuracy_target=accuracy_target,
    )


def _fallback(problem, *, need_grad: bool) -> str:
    """The engine for a request no fused kernel can serve (unregistered field, no Warp/CUDA).

    Gradients — and ``saveat``, which lives on the replay lanes — take the general
    record-and-replay engine of the problem's stiffness class; a forward-only solve takes
    diffrax when it is installed (an optional extra), else that same replay engine. All of
    these honour ``rtol``/``atol``. The fixed-step scans stay reachable by explicit
    ``engine=`` only.
    """
    replay = "rodas5p_replay" if problem.is_stiff else "tsit5_replay"
    if need_grad or not ENGINE_REGISTRY["diffrax"].supports(problem):
        return replay
    return "diffrax"


def _ensure_fused_field(problem, params, fused, reasons: list) -> None:
    """Decide whether the user's RHS can drive a fused Warp kernel, per the ``fused`` policy.

    ``fused`` selects the policy:
      * ``False`` — do nothing (skip translation). An already-registered field still routes
        to the fused engine; an unregistered problem takes the general path, byte-identical
        to before this switch existed.
      * ``"auto"`` (default) — translate the RHS to *check* fused-eligibility, but do NOT
        auto-adopt the fused engine: on success routing is byte-identical to the general
        path (adoption is opt-in via :func:`gradsolve.register_jax_field` or ``fused=True``);
        on ``UnsupportedRHS`` a fallback reason naming the primitive is recorded on the
        route. This keeps the default routing unchanged while telling the caller, via
        ``route.reason``, when an RHS cannot be fused.
      * ``True`` — opt in: translate AND register the field so the problem routes to the
        fused engine; an ``UnsupportedRHS`` re-raises instead of falling back.

    ``n_params`` is read from the batched ``params`` (shape ``(n, P)``) — the library
    ``Problem`` protocol (``gradsolve/base.py``) exposes only ``name/dim/t0/t1/is_stiff/f_jax``,
    NOT ``n_params`` — exactly as the stiff ``_launch`` reads ``P`` off ``params``. A problem
    already served by a fused engine (a built-in or a prior ``register_jax_field``) needs no
    translation, and an installation without Warp has no fused path, so both short-circuit.
    """
    if fused is False:
        return
    if fused is not True and fused != "auto":
        raise ValueError(f"fused must be True, False, or 'auto'; got {fused!r}")
    if not warp_ode._WARP_AVAILABLE:
        return  # no fused kernels warp-less; auto/True just proceed to the general path
    if warp_ode.supports(problem) or warp_rosenbrock.supports(problem):
        return  # already routable to a fused engine (built-in or previously registered)
    if fused == "auto" and problem.dim > NVAR_CEILING:
        # High-NVAR problems never route to a fused cell (choose_engine sends them to the
        # general engines), so the auto diagnostic would translate a field nothing can use.
        # An explicit fused=True still translates below (it honours the caller's intent).
        return

    import warp as wp  # safe: guarded by the _WARP_AVAILABLE check above

    from gradsolve import (
        register_jax_field,  # thin lazy wrapper (keeps import gradsolve warp-less)
    )
    from gradsolve.warp.jax_field import (
        UnsupportedRHS,
        jaxpr_to_warp_field,
        jaxpr_to_warp_jacobian,
    )

    n_params = int(params.shape[1])  # params is (n, P); see the n_params note above
    # Translate eagerly (float64; the primitive subset is precision-independent) to decide
    # fused-eligibility up front — the register_jax_field builder translates LAZILY (at
    # kernel-build time), so without this an UnsupportedRHS would only surface deep inside
    # routing. The eager call primes jax_field's cache, so a later build (or a fused=True
    # registration) for the run precision reuses it (same args -> same key). Args mirror the
    # builders exactly for that cache reuse: the stiff field takes a length-dim vecP
    # (n_params=dim); the Jacobian takes n_params.
    try:
        if problem.is_stiff:
            jaxpr_to_warp_field(problem.f_jax, problem.dim, problem.dim, wp_scalar=wp.float64)
            jaxpr_to_warp_jacobian(problem.f_jax, problem.dim, n_params, wp_scalar=wp.float64)
        else:
            jaxpr_to_warp_field(problem.f_jax, problem.dim, n_params, wp_scalar=wp.float64)
    except UnsupportedRHS as exc:
        if fused is True:
            raise
        reasons.append(
            f"fused-unsupported:{exc.primitive_name}; fell back to the general path")
        return
    if fused is True:
        # Opt-in: register so the problem actually routes to the fused engine. 'auto' only
        # diagnosed eligibility above and leaves routing byte-identical to the general path.
        register_jax_field(
            problem.name, problem.f_jax, problem.dim, n_params, stiff=problem.is_stiff)


# --- saveat capability ------------------------------------------------------
# Dense output is implemented on the JAX scan/replay lanes ONLY (gradsolve/solvers/dense.py).
# The fused Warp kernels and the hand-CUDA lane hold the whole integration in registers and
# emit a final state; diffrax has its own SaveAt which this API deliberately does not wire.
# Naming one of those engines explicitly WITH saveat set raises rather than silently
# ignoring saveat or silently rerouting.
#
# The two lists differ because the same engine NAME already means different things in the
# two entry points (this predates saveat): in solve(), "warp_ode"/"warp_rosenbrock" are the
# fused forward kernels; in grad_closure() they name the record-and-replay reverse lanes,
# which ARE dense-capable.
_SAVEAT_INCAPABLE_SOLVE = frozenset({
    "diffrax", "cuda_tsit5", "cuda_rosenbrock23", "warp_ode", "warp_rosenbrock",
    "fused_rosenbrock_backward",
})
_SAVEAT_INCAPABLE_GRAD = frozenset({
    "diffrax", "cuda_tsit5", "cuda_rosenbrock23", "fused_rosenbrock_backward",
})


# --- f32 capability -------------------------------------------------------
# f32 is scoped to the REGISTERED WARP FIELD route (warp record -> JAX replay), nonstiff and
# stiff. The general-RHS host recorder (gradsolve/solvers/tsit5_replay.py) is hard-coded f64
# throughout, and the fixed scans/diffrax are driven at whatever dtype the caller supplies —
# none of them can honour an f32 request, so they raise rather than return f64 under an f32
# label.
_F32_CAPABLE = frozenset({"warp_ode", "warp_replay", "warp_rosenbrock"})


def _reject_incapable_precision(name: str, precision: str) -> None:
    if precision == "float64":
        return
    if precision not in ("float64", "float32"):
        raise ValueError(
            f"unknown precision {precision!r}; known: ['float64', 'float32']")
    if name not in _F32_CAPABLE:
        raise ValueError(
            f"precision='float32' is not supported on lane {name!r}: the f32 adjoint is "
            "scoped to the registered Warp field route (warp record -> JAX replay), "
            "nonstiff and stiff. The general-RHS recorder is float64 throughout. Use a "
            "registered field, or precision='float64'."
        )


def _reject_incapable_saveat(engine: str, incapable: frozenset) -> None:
    if engine in incapable:
        raise ValueError(
            f"engine {engine!r} does not support saveat: dense output is implemented on the "
            f"JAX scan/replay lanes only (the fused Warp kernels and the cuda lane are "
            f"final-state by design, and diffrax's own SaveAt is not wired into this API). "
            f"Drop saveat, or use engine='auto' to route to a dense-capable lane."
        )


# ---------------------------------------------------------------------------
# Public API: solve()
# ---------------------------------------------------------------------------

def solve(
    problem,
    y0,
    params,
    *,
    saveat=None,
    saveat_dense: bool = False,
    engine: str = "auto",
    fused: "str | bool" = "auto",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    device: str = "cpu",
    batch_n: Optional[int] = None,
    accuracy_target: Optional[float] = None,
) -> SolveResult:
    """Route and run a batched ensemble ODE solve.

    Resolves ``engine`` (``'auto'`` -> :func:`gradsolve.dispatch.choose_engine`, else
    used as-is), dispatches to that engine's forward solve, and falls back
    (:func:`_fallback`) to diffrax — or, for gradients and when diffrax is absent, the
    general record-and-replay engine — if the resolved engine does not ``supports()`` the
    problem.

    Parameters
    ----------
    problem : Problem
        A Problem instance (``.dim``, ``.is_stiff``, ``.f_jax``, etc.).
    y0 : np.ndarray
        Initial state array, shape ``(n, dim)``.
    params : np.ndarray
        Per-trajectory parameter array, shape ``(n, P)``.
    saveat : array_like, optional
        Sorted output times within ``[t0, t1]``. When given, the result carries
        ``y_saved (n, k, dim)`` + ``ts_saved (k,)`` (host NumPy) alongside ``y_final``, and
        ``engine='auto'`` routes to a dense-capable JAX scan/replay lane. Naming a
        final-state-only engine (the fused Warp kernels, ``cuda_tsit5``, ``diffrax``)
        explicitly with ``saveat`` set raises ``ValueError``. ``None`` (default) returns the
        final state only.
    saveat_dense : bool, default False
        Opt in to the method's own continuous extension instead of bracket-and-re-step, which
        deletes the one extra solver step per requested time. Implemented only on
        ``rodas5p_replay`` (order 4), because it needs a published interpolant for
        the method; requesting it on any other engine raises. Requires ``saveat``. The
        polynomial is less accurate than a genuine re-step at matched mesh -- both well
        inside the record tolerance -- so the default is ``False``.
    engine : str, default 'auto'
        ``'auto'`` (route via ``choose_engine``) or a canonical engine name
        (``'tsit5_replay'``, ``'rodas5p_replay'``, ``'diffrax'``, ``'warp_ode'``,
        ``'warp_rosenbrock'``, ``'fixed_step_tsit5'``, ...). Unknown names raise
        ``ValueError``.
    fused : {'auto', True, False}, default 'auto'
        Whether the problem's ``f_jax`` may drive a fused Warp kernel. ``'auto'`` routes an
        already-registered field (a built-in, or a prior ``register_jax_field``) to the
        fused engine and otherwise takes the general path unchanged — it translates the RHS
        only to check fused-eligibility, recording a reason on ``SolveResult.route`` (naming
        the offending primitive) when the RHS is outside the translatable subset, without
        adopting the fused engine on its own. ``True`` opts in: it translates AND registers
        the field so the problem routes to the fused engine, re-raising ``UnsupportedRHS``
        for an untranslatable RHS. ``False`` skips translation entirely. Without Warp installed
        there is no fused path, so ``'auto'``/``True`` simply proceed to the general path there.
    rtol, atol : float
        Solver tolerances, honoured by every auto-routed engine; the fixed-step scans
        (explicit ``engine=`` only) accept them nominally.
    device : str, default 'cpu'
        ``'cpu'`` or ``'cuda'``.
    batch_n : int, optional
        Accepted routing hint; the current policy does not branch on it.
    accuracy_target : float, optional
        Accepted routing hint; the current policy does not branch on it.

    Returns
    -------
    SolveResult
        ``.solver`` is set to the canonical engine key actually used (the
        registry name, not the engine's internal descriptive string). ``.route`` records the
        requested engine, the engine that ran, and why they differ
        (``'engine-does-not-support-problem'``, ``'diffrax-not-installed'``).
    """
    if saveat is not None:
        return _solve_saveat(
            problem, y0, params, saveat=saveat, engine=engine, rtol=rtol, atol=atol,
            device=device, batch_n=batch_n, accuracy_target=accuracy_target,
            saveat_dense=saveat_dense)
    if saveat_dense:
        raise ValueError("saveat_dense=True requires saveat=...; it is a saveat option.")

    # fused='auto'/True: translate the user RHS into a fused field before routing so a
    # matching problem reaches the fused engine (fused=False / warp-less: no-op). An
    # untranslatable RHS under 'auto' records its reason here and stays on the general path.
    reasons: list[str] = []
    _ensure_fused_field(problem, params, fused, reasons)

    name = _resolve_engine(
        problem, engine, batch_n=batch_n, accuracy_target=accuracy_target, need_grad=False
    )
    # If the chosen engine doesn't support this problem (e.g. warp absent on CPU,
    # or warp_rosenbrock called on a non-stiff problem), fall back (_fallback).
    spec = ENGINE_REGISTRY.get(name)
    if spec is None or not spec.supports(problem):
        if name != "diffrax":
            reasons.append("engine-does-not-support-problem")
        name = _fallback(problem, need_grad=False)
        if name != "diffrax":
            reasons.append("diffrax-not-installed")
        spec = ENGINE_REGISTRY[name]
    res = spec.solve(problem, y0, params, rtol=rtol, atol=atol, device=device)
    # Stamp the registry key so callers get a stable name independent of the engine's
    # internal descriptive string (e.g. "tsit5[10000]" -> "fixed_step_tsit5").
    res.solver = name
    return _stamp_route(res, engine, name, reasons)


def _route_reverse(problem, engine, *, batch_n, accuracy_target):
    """Resolve the reverse/dense engine name for a request, with the reasons it moved.

    Shared by ``grad_closure`` and the ``saveat`` forward path so the two cannot drift into
    disagreeing about where a request goes. Covers the name-resolution stage only; the
    registered-field fallbacks are decided at BUILD time (they depend on whether the record
    actually succeeds) by the caller.
    """
    reasons: list[str] = []
    name = engine
    if engine == "auto":
        # need_grad=True is what masks the fused/cuda rows out of choose_engine — the same
        # masking dense output needs, since those lanes are final-state-only.
        name = choose_engine(
            dim=problem.dim, stiff=problem.is_stiff, need_grad=True,
            batch_n=batch_n, accuracy_target=accuracy_target)
    if name in ENGINE_REGISTRY and not ENGINE_REGISTRY[name].supports(problem):
        name = _fallback(problem, need_grad=True)
        reasons.append("engine-does-not-support-problem")
    return name, reasons


def _reject_incapable_saveat_dense(name: str) -> None:
    """saveat_dense needs a PUBLISHED continuous extension for the method, so it is a
    per-lane capability, not a generic one. Only rodas5p_replay has one (order 4); a
    generic interpolant is not offered as a substitute, because its interior error can
    exceed the record tolerance on adaptive meshes (see gradsolve/solvers/dense.py)."""
    if name != "rodas5p_replay":
        raise ValueError(
            f"saveat_dense=True is implemented only on the 'rodas5p_replay' lane, which has a "
            f"published continuous extension; engine {name!r} has none, and a generic "
            "interpolant is not offered as a substitute (see gradsolve/solvers/dense.py). "
            "Use engine='rodas5p_replay', or drop saveat_dense for the re-step path.")


def _solve_saveat(problem, y0, params, *, saveat, engine, rtol, atol, device, batch_n,
                  accuracy_target, saveat_dense=False) -> SolveResult:
    """Forward solve with dense output — ``solve(..., saveat=ts)``.

    Runs the dense-capable JAX scan/replay lane eagerly and packages ``y_saved``/``ts_saved``
    into the ``SolveResult``. Deliberately calls each lane directly rather than through the
    kernel factories: the record step is what knows the per-trajectory accepted-step counts,
    and ``SolveResult``'s contract reserves the empty default for backends that genuinely
    cannot report them — so routing through a kernel that discards ``n_acc`` would force a
    mislabelled result.
    """
    import numpy as _np

    from gradsolve.solvers.tsit5_replay import record_tsit5_jax

    _reject_incapable_saveat(engine, _SAVEAT_INCAPABLE_SOLVE)
    if engine != "auto" and engine not in ENGINE_REGISTRY and engine != "warp_replay":
        raise ValueError(
            f"unknown engine {engine!r}; known: {sorted(ENGINE_REGISTRY)} + 'warp_replay'")
    ts = validate_saveat(saveat, problem.t0, problem.t1)

    name, reasons = _route_reverse(
        problem, engine, batch_n=batch_n, accuracy_target=accuracy_target)
    if saveat_dense:
        _reject_incapable_saveat_dense(name)

    def _done(y_final, ys, n_acc, lane, rejected=None):
        return _stamp_route(_dense_result(y_final, ys, ts, n_acc, lane, rejected=rejected),
                            engine, lane, reasons)

    if name in ("warp_ode", "warp_replay"):
        if warp_ode.supports(problem):
            try:
                _yw, n_acc, dts = _warp_replay.record(
                    problem, y0, params, rtol=rtol, atol=atol, device=device)
                y_final, ys = _warp_replay.replay_solve_saveat_jax(
                    problem, y0, params, dts, ts)
                return _done(y_final, ys, n_acc, "warp_replay")
            except (IndexError, KeyError, ValueError, TypeError) as exc:
                # Same catch, same reason vocabulary as grad_closure's warp_replay branch:
                # name what was OBSERVED, not the assumed param-schema mismatch.
                reasons.append(f"warp-build-failed:{type(exc).__name__}")
        else:
            reasons.append("no-registered-field")
        name = "tsit5_replay"

    if name == "warp_rosenbrock":  # supports() already verified by _route_reverse
        _yw, n_acc, dts = _warp_replay.record_rosenbrock(
            problem, y0, params, rtol=rtol, atol=atol, device=device)
        y_final, ys = _warp_replay.replay_solve_rosenbrock_saveat_jax(
            problem, y0, params, dts, ts)
        return _done(y_final, ys, n_acc, "warp_rosenbrock")

    if name == "tsit5_replay":
        _yf, dts, n_acc = record_tsit5_jax(
            problem.f_jax, y0, params, problem.t0, problem.t1, rtol=rtol, atol=atol)
        y_final, ys = _warp_replay.replay_solve_saveat_jax(problem, y0, params, dts, ts)
        return _done(y_final, ys, n_acc, "tsit5_replay")

    if name == "vern7_replay":
        import jax.numpy as jnp

        from gradsolve.solvers.dense import vmap_saveat
        from gradsolve.solvers.vern7_replay import record_vern7
        from gradsolve.solvers.vern7_step import vern7_advance
        _yf, dts, n_acc = record_vern7(
            problem.f_jax, y0, params, problem.t0, problem.t1, rtol=rtol, atol=atol)
        y_final, ys = vmap_saveat(
            problem.f_jax, problem.t0, y0, params, jnp.asarray(dts), ts, vern7_advance)
        return _done(y_final, ys, n_acc, "vern7_replay")

    if name == "rodas5p_replay":
        import jax.numpy as jnp

        from gradsolve.solvers.dense import vmap_saveat
        from gradsolve.solvers.rodas5p_replay import record_rodas5p
        from gradsolve.solvers.rodas5p_step import rodas5p_advance
        _yf, dts, n_acc, n_rej = record_rodas5p(
            problem.f_jax, y0, params, problem.t0, problem.t1, rtol=rtol, atol=atol)
        if saveat_dense:
            from gradsolve.solvers.dense import validate_dense_thetas, vmap_saveat_dense
            from gradsolve.solvers.rodas5p_step import (
                rodas5p_dense_eval,
                rodas5p_stages_and_dense,
            )
            validate_dense_thetas(dts, ts, problem.t0)
            y_final, ys = vmap_saveat_dense(
                problem.f_jax, problem.t0, y0, params, jnp.asarray(dts), ts,
                rodas5p_stages_and_dense, rodas5p_dense_eval)
        else:
            y_final, ys = vmap_saveat(
                problem.f_jax, problem.t0, y0, params, jnp.asarray(dts), ts, rodas5p_advance)
        return _done(y_final, ys, n_acc, "rodas5p_replay", rejected=n_rej)

    mod = _fixed_step_imex_mod if name == "fixed_step_imex" else fixed_step_tsit5
    y_final, ys = mod.solve_jax_saveat(problem, y0, params, ts)
    # Same structural-constant contract these lanes' own solve() reports.
    n_acc = _np.full(_np.asarray(y0).shape[0], mod.DEFAULT_N_STEPS, dtype=_np.int64)
    return _done(y_final, ys, n_acc, mod.name)


def _dense_result(y_final, ys, ts, accepted, solver, rejected=None) -> SolveResult:
    """Package a dense lane's outputs as a host-NumPy SolveResult.

    ``rejected`` (optional): per-trajectory rejected-step counts when the lane's recorder
    reports them (rodas5p_replay); the other dense lanes keep the zeros default, so this is
    backward-compatible.
    """
    import numpy as _np

    return SolveResult(
        y_final=_np.asarray(y_final),
        accepted_steps=_np.asarray(accepted, dtype=_np.int64),
        rejected_steps=(_np.zeros(_np.asarray(y_final).shape[0], dtype=_np.int64)
                        if rejected is None else _np.asarray(rejected, dtype=_np.int64)),
        solver=solver,
        y_saved=_np.asarray(ys),
        ts_saved=_np.asarray(ts),
    )


# ---------------------------------------------------------------------------
# Reverse-path helpers
# ---------------------------------------------------------------------------

def _reverse_for(name):
    """Return a closure factory for the given engine name.

    The factory signature is:
        (problem, y0, params, *, rtol, atol, device) -> (params -> y_final)

    Notes on rtol/atol:
    - warp_ode / warp_replay / warp_rosenbrock: passed straight through to the
      record step, which controls mesh density.
    - scan engines (fixed_step_tsit5, fixed_step_imex): rtol/atol are nominal
      (accuracy governed by n_steps); ignored here — the closure uses the default
      n_steps=10_000 which is already the configured DEFAULT_N_STEPS.

    ``warp_replay`` is ``choose_engine``'s return for nonstiff+low-dim+need_grad;
    its forward solve is warp_ode and its reverse is make_replay_closure.
    """
    if name in ("warp_ode", "warp_replay"):
        # warp_replay is a reverse-only alias; its forward = warp_ode, its closure =
        # make_replay_closure.  warp_ode's explicit reverse is also make_replay_closure.
        # remat: jax.checkpoint the replay scan (choose_remat policy) — grad-transparent.
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            _warp_replay.make_replay_closure(
                problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat)

    if name == "warp_rosenbrock":
        # make_rosenbrock_replay_closure already accepts rtol/atol and forwards them
        # to record_rosenbrock, which controls the mesh density.
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            _warp_replay.make_rosenbrock_replay_closure(
                problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat)

    if name == "fused_rosenbrock_backward":
        # Override-only named engine: the GENUINE single-kernel fused Warp backward
        # (jax.custom_vjp around the fused kernel). Records the frozen stable mesh at
        # the closure's rtol/atol (stiff replay needs a FINE mesh — robertson ~1e-12).
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            _fused_backward.fused_rosenbrock_grad_closure(
                problem, y0, params, rtol=rtol, atol=atol, device=device)

    if name == "vern7_replay":
        # Default params-only closure: record ONCE at params0, return p -> y_final.
        from gradsolve.solvers.vern7_replay import make_vern7_frozen_mesh_closure
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            make_vern7_frozen_mesh_closure(problem, y0, params, rtol=rtol, atol=atol)

    if name == "rodas5p_replay":
        # Default params-only closure: record ONCE at params0, return p -> y_final. remat is
        # threaded into the final-state replay scan (grad-transparent).
        from gradsolve.solvers.rodas5p_replay import make_rodas5p_frozen_mesh_closure
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            make_rodas5p_frozen_mesh_closure(problem, y0, params, rtol=rtol, atol=atol,
                                             remat=remat)

    if name == "fixed_step_tsit5":
        # solve_jax is reverse-diff by construction (lax.scan); rtol/atol nominal.
        # remat is accepted and ignored by the scan engines.
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            (lambda q: fixed_step_tsit5.solve_jax(problem, y0, q))

    if name == "fixed_step_imex":
        # solve_jax is reverse-diff by construction (lax.scan); rtol/atol nominal.
        # remat is accepted and ignored by the scan engines.
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            (lambda q: _fixed_step_imex_mod.solve_jax(problem, y0, q))

    if name == "diffrax":
        return lambda problem, y0, params, *, rtol, atol, device, remat=False: \
            _diffrax_fallback_mod._api_reverse(
                problem, y0, params, rtol=rtol, atol=atol, device=device)

    return None


def _reverse_kernel_for(name):
    """Return a KERNEL factory for the given engine name — the two-argument form.

    The factory signature mirrors :func:`_reverse_for`::

        (problem, y0, params, *, rtol, atol, device, remat) -> ((y0, params) -> y_final)

    where the returned kernel leaves BOTH the initial state and the parameters free, so
    ``grad_closure`` can build the ``wrt="y0"`` and ``wrt=("y0","params")`` closures from a
    single recorded mesh. The mesh (where a lane has one) is still recorded ONCE, at the
    ``(y0, params)`` passed here — differentiating w.r.t. ``y0`` inherits exactly the same
    frozen-controller semantics as w.r.t. ``params``.

    ``wrt="params"`` deliberately does NOT come through here: it stays on
    :func:`_reverse_for` so the default closure is bit-compatible with the previous API.

    Returns ``None`` for engines with no reverse kernel: ``cuda_tsit5`` (forward-only) and
    ``fused_rosenbrock_backward`` (its ``custom_vjp`` is params-only by construction — a y0
    request is rerouted by ``grad_closure`` before it reaches here).
    """
    if name in ("warp_ode", "warp_replay"):
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None, \
            precision="float64": \
            _warp_replay.make_replay_kernel(
                problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat,
                saveat=saveat, precision=precision)

    if name == "warp_rosenbrock":
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None, \
            precision="float64": \
            _warp_replay.make_rosenbrock_replay_kernel(
                problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat,
                saveat=saveat, precision=precision)

    if name == "tsit5_replay":
        # The general-RHS pure-JAX record-and-replay (no registered Warp field needed).
        from gradsolve.solvers.tsit5_replay import make_tsit5_frozen_mesh_kernel
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None: \
            make_tsit5_frozen_mesh_kernel(
                problem, y0, params, rtol=rtol, atol=atol, saveat=saveat)

    if name == "vern7_replay":
        from gradsolve.solvers.vern7_replay import make_vern7_frozen_mesh_kernel
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None: \
            make_vern7_frozen_mesh_kernel(
                problem, y0, params, rtol=rtol, atol=atol, saveat=saveat)

    if name == "rodas5p_replay":
        # remat threads into the final-state replay only; with saveat the dense path goes through
        # the shared dense.vmap_saveat, which takes no remat argument (documented no-op there).
        from gradsolve.solvers.rodas5p_replay import make_rodas5p_frozen_mesh_kernel
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None, \
            dense=False: \
            make_rodas5p_frozen_mesh_kernel(
                problem, y0, params, rtol=rtol, atol=atol, saveat=saveat, remat=remat,
                dense=dense)

    if name == "fixed_step_tsit5":
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None: \
            _scan_kernel(fixed_step_tsit5, problem, saveat)

    if name == "fixed_step_imex":
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None: \
            _scan_kernel(_fixed_step_imex_mod, problem, saveat)

    if name == "diffrax":
        # saveat is rejected before we get here (_SAVEAT_INCAPABLE_*): diffrax's own SaveAt
        # is deliberately not wired into this API.
        return lambda problem, y0, params, *, rtol, atol, device, remat=False, saveat=None: \
            _diffrax_fallback_mod._api_reverse_kernel(
                problem, y0, params, rtol=rtol, atol=atol, device=device)

    return None


def _scan_kernel(mod, problem, saveat):
    """Two-argument kernel for a fixed-step scan lane (no mesh to record)."""
    if saveat is None:
        return lambda z, p: mod.solve_jax(problem, z, p)
    return lambda z, p: mod.solve_jax_saveat(problem, z, p, saveat)


_VALID_WRT = "'params', 'y0', or ('y0', 'params')"


def _normalize_wrt(wrt) -> tuple:
    """Validate ``wrt`` and normalize it to a canonical tuple."""
    if wrt == "params":
        return ("params",)
    if wrt == "y0":
        return ("y0",)
    if isinstance(wrt, (tuple, list, set, frozenset)) and set(wrt) == {"y0", "params"}:
        return ("y0", "params")
    raise ValueError(f"invalid wrt={wrt!r}; expected {_VALID_WRT}")


def _build_reverse(name, problem, y0, params, wrt_t, *, rtol, atol, device, remat,
                   saveat=None, precision="float64", saveat_dense=False):
    """Build the reverse closure for engine ``name`` under the requested ``wrt``."""
    _reject_incapable_precision(name, precision)
    if wrt_t == ("params",) and saveat is None and precision == "float64":
        # The exact default path — keeps the default closure bit-compatible and keeps the
        # engine dispatch observable through _reverse_for (tests spy on it).
        if name == "tsit5_replay":
            from gradsolve.solvers.tsit5_replay import make_tsit5_frozen_mesh_closure
            return make_tsit5_frozen_mesh_closure(problem, y0, params, rtol=rtol, atol=atol)
        rev = _reverse_for(name)
        if rev is None:
            raise ValueError(f"engine {name!r} has no reverse closure implemented")
        return rev(problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat)

    kernel = _build_kernel(
        name, problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat,
        saveat=saveat, precision=precision, saveat_dense=saveat_dense)
    if saveat is not None:
        # The dense kernels return (y_final, ys); the closure's contract is the (n, k, dim)
        # time series alone.
        base, kernel = kernel, (lambda z, p: base(z, p)[1])
    if wrt_t == ("y0",):
        return lambda z: kernel(z, params)  # params closed over at the recording point
    if wrt_t == ("params",):
        return lambda p: kernel(y0, p)
    return lambda z, p: kernel(z, p)


def _build_kernel(name, problem, y0, params, *, rtol, atol, device, remat, saveat=None,
                  precision="float64", saveat_dense=False):
    """Instantiate engine ``name``'s two-argument kernel (records the mesh once)."""
    factory = _reverse_kernel_for(name)
    if factory is None:
        raise ValueError(f"engine {name!r} has no reverse closure implemented")
    kw = {}
    if saveat is not None:
        kw["saveat"] = saveat
    if saveat_dense:
        _reject_incapable_saveat_dense(name)
        kw["dense"] = True
    if precision != "float64":
        kw["precision"] = precision
    return factory(problem, y0, params, rtol=rtol, atol=atol, device=device, remat=remat,
                   **kw)


def _stamp_route(closure, requested, actual, reasons):
    """Attach the observable :class:`Route` and return the object.

    Used on both kinds of result: a built ``grad_closure`` closure (a function object, which
    takes the attribute) and a ``SolveResult`` (whose ``route`` field defaults to ``None``).
    """
    closure.route = Route(
        requested=requested, actual=actual,
        reason="; ".join(reasons) if reasons else None,
    )
    return closure


def grad_closure(
    problem,
    y0,
    params,
    *,
    wrt="params",
    saveat=None,
    saveat_dense: bool = False,
    precision: str = "float64",
    engine: str = "auto",
    fused: "str | bool" = "auto",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    device: str = "cpu",
    batch_n: Optional[int] = None,
    accuracy_target: Optional[float] = None,
    remat: Optional[bool] = None,
):
    """Return a jax.grad-able closure over the initial state, the parameters, or both.

    The closure records the frozen step mesh at the given ``(y0, params)`` (warp/replay
    engines) or wraps a ``lax.scan`` (scan engines). Calling ``jax.grad`` on a scalar
    function of the closure's output gives reverse-mode gradients w.r.t. whichever inputs
    ``wrt`` selects.

    The recorded mesh is frozen for the closure's lifetime: it is recorded once, here, at
    the passed ``(y0, params)``, and perturbing either afterwards replays that same mesh
    rather than re-running the step-size controller. The same holds for the y0 gradient: it
    is the exact discrete adjoint of the replayed frozen-step integration, valid in a
    neighbourhood of the recording point.

    On the general-RHS and fixed-scan lanes the closure's arguments are used at the dtype
    given and are not cast: pass float64. (The run-precision cast described under
    ``precision`` belongs to the registered Warp f32 route.)

    Parameters
    ----------
    problem : Problem
        A Problem instance (``.dim``, ``.is_stiff``, ``.f_jax``, etc.).
    y0 : np.ndarray
        Initial state array, shape ``(n, dim)``.
    params : np.ndarray
        Per-trajectory parameter array, shape ``(n, P)``.
    wrt : {'params', 'y0', ('y0', 'params')}, default 'params'
        Which inputs the returned closure takes and is differentiable in. Return shapes
        mirror the corresponding input shapes.

        * ``'params'`` (default) — ``params -> y_final``, with ``y0`` closed over.
        * ``'y0'`` — ``y0 -> y_final``, with ``params`` closed over.
        * ``('y0', 'params')`` — ``(y0, params) -> y_final``.

        ``fused_rosenbrock_backward`` is params-only (its ``custom_vjp`` differentiates
        parameters by construction); requesting a y0 gradient from it reroutes to the stiff
        replay lane and records ``reason='y0-unsupported'`` on the returned closure's
        ``.route`` — never a silent substitution.
    saveat : array_like, optional
        Sorted output times within ``[t0, t1]``. When given, the closure returns the dense
        time series ``(n, k, dim)`` instead of the final state ``(n, dim)``, so time-series
        losses differentiate. The states are streamed off the same frozen mesh (each
        requested time is reached by re-stepping from the step that brackets it — see
        ``gradsolve.solvers.dense``), so the gradient carries exactly the same
        frozen-controller semantics as a final-state gradient. ``None`` (default) returns the
        final state ``(n, dim)``.

        Dense output exists on the JAX scan/replay lanes only. ``engine='auto'`` routes to a
        dense-capable lane; naming ``diffrax``, ``cuda_tsit5`` or ``fused_rosenbrock_backward``
        explicitly with ``saveat`` set raises ``ValueError`` rather than silently ignoring it.
        Unsorted or out-of-domain times raise ``ValueError``.
    saveat_dense : bool, default False
        Opt in to the method's own continuous extension instead of the per-save re-step,
        deleting one solver step per requested time. ``rodas5p_replay`` only (order 4);
        any other engine raises, because a generic interpolant is not offered as a
        substitute. Requires ``saveat``.

        The save times stay live in ``theta``, so the interpolation itself carries their
        tangent -- but not through this entry point: ``saveat`` is a construction-time
        constant here (it is validated eagerly, outside any trace, so neither ``solve`` nor
        ``grad_closure`` can be traced w.r.t. it).
        A caller who needs the save-time derivative drives
        ``gradsolve.solvers.dense.vmap_saveat_dense`` directly with an already-recorded mesh.
        One further boundary applies there: a save at or past the final accumulated mesh node
        is value-pinned to ``y_final`` (so ``ys[-1] == y_final`` stays bitwise) and therefore
        has exactly zero derivative w.r.t. its own time. Keep such a save strictly inside the
        mesh if you need that gradient.
    precision : {'float64', 'float32'}, default 'float64'
        Arithmetic precision of the record and the replay. ``'float32'`` records the mesh
        with the f32 Warp kernel — exactly half the record buffer's bytes — and replays in
        f32; the closure casts its inputs to the run precision, so the arithmetic is the one
        requested rather than whatever the caller's dtypes promoted to.

        Scope: f32 is supported on the registered Warp field route only (``warp_replay``
        nonstiff, ``warp_rosenbrock`` stiff). The general-RHS recorder is float64 throughout;
        asking any other lane for f32 raises ``ValueError`` rather than returning a f64
        result under an f32 label. On the registered cells, f32 gradients agree
        with the f64 replay of the same f32-recorded mesh to rel-L2 <= 1e-3 with direction
        cosine >= 1 - 1e-6.
    engine : str, default 'auto'
        ``'auto'`` resolves via ``choose_engine(..., need_grad=True)``, which may
        return ``'warp_replay'`` (nonstiff+low-dim), ``'warp_rosenbrock'``
        (stiff+low-dim), ``'tsit5_replay'`` (nonstiff+high-dim), or
        ``'rodas5p_replay'`` (stiff+high-dim); a problem with no registered field takes the
        replay lane of its class. A named override is used as-is (must be an
        ``ENGINE_REGISTRY`` key or ``'warp_replay'``).
    fused : {'auto', True, False}, default 'auto'
        Whether the problem's ``f_jax`` may record its mesh on a fused Warp forward
        (``warp_replay`` / ``warp_rosenbrock``) instead of the general host recorder.
        ``'auto'`` routes an already-registered field to the fused reverse path and otherwise
        takes the general replay lane unchanged — it translates the RHS only to check
        fused-eligibility, recording a reason on ``.route`` (naming the primitive) for an RHS
        outside the translatable subset. ``True`` opts in: translate AND register the field
        (re-raising ``UnsupportedRHS`` for an untranslatable RHS). ``False`` skips
        translation. Already-registered problems, and installations without Warp, are
        unaffected.
    rtol, atol : float
        Warp engines (``warp_ode``, ``warp_replay``, ``warp_rosenbrock``): control
        the mesh recorded at ``params``; perturbing ``params`` after recording does
        not re-run the controller (frozen-controller discrete adjoint — see the
        ``warp_replay`` module). Replay lanes (``tsit5_replay``, ``rodas5p_replay``,
        ``vern7_replay``): same frozen-mesh semantics via the host recorder. Scan engines
        (explicit ``engine=`` only): nominal.
    device : str, default 'cpu'
        ``'cpu'`` or ``'cuda'``.
    batch_n : int, optional
        Accepted routing hint; the current policy does not branch on it.
    accuracy_target : float, optional
        Accepted routing hint; the current policy does not branch on it.
    remat : bool, optional
        ``None`` (default) auto-selects via ``choose_remat(dim, stiff)`` —
        ``jax.checkpoint`` the warp replay scan (grad-transparent; trades compute
        for memory). Pass True/False to force. On a GPU it is faster at low
        dim (nonstiff), roughly neutral at mid/high dim, and modestly slower but
        with substantially less memory on stiff problems (so larger batches fit).

    Returns
    -------
    Callable
        The closure selected by ``wrt`` (see above), jax.grad-able in its argument(s), and
        carrying a :class:`Route` as ``.route`` — the requested engine, the engine that
        actually ran, and the reason they differ. Read ``.route`` rather than assuming the
        request was honoured: an engine is rerouted when it does not ``supports()`` the
        problem, when the problem matches no registered Warp field, or when the request
        exceeds what the engine can do.

    Raises
    ------
    ValueError
        If ``wrt`` is not one of the accepted forms; if ``engine`` is not ``'auto'``, an
        ``ENGINE_REGISTRY`` key, or ``'warp_replay'``; or if the resolved engine has no
        reverse closure implemented.
    """
    wrt_t = _normalize_wrt(wrt)
    wants_y0 = "y0" in wrt_t

    # Special case: 'warp_replay' is returned by choose_engine but is NOT a key in
    # ENGINE_REGISTRY (it has no forward solve; only a reverse path).  We accept it
    # directly in grad_closure without going through the registry.
    if engine != "auto" and engine not in ENGINE_REGISTRY and engine != "warp_replay":
        raise ValueError(
            f"unknown engine {engine!r}; known: {sorted(ENGINE_REGISTRY)} + 'warp_replay'"
        )

    if saveat is not None:
        _reject_incapable_saveat(engine, _SAVEAT_INCAPABLE_GRAD)
        saveat = validate_saveat(saveat, problem.t0, problem.t1)
    elif saveat_dense:
        # Mirror solve()'s rule. Without this the params-only fast path below returns a plain
        # final-state closure, silently ignoring the flag AND skipping the engine rejection.
        raise ValueError("saveat_dense=True requires saveat=...; it is a saveat option.")

    requested = engine
    reasons: list[str] = []

    # fused='auto'/True: translate the user RHS into a fused field before routing so a
    # matching nonstiff/stiff problem reaches warp_replay / warp_rosenbrock (fused=False /
    # warp-less: no-op). An untranslatable RHS under 'auto' records its reason and stays on
    # the general replay lane.
    _ensure_fused_field(problem, params, fused, reasons)

    name = engine
    if engine == "auto":
        name = choose_engine(
            dim=problem.dim,
            stiff=problem.is_stiff,
            need_grad=True,
            batch_n=batch_n,
            accuracy_target=accuracy_target,
        )

    # The fused backward's custom_vjp is params-only by construction. A y0 request reroutes
    # to the stiff replay lane LOUDLY (recorded in .route.reason), never silently.
    if name == "fused_rosenbrock_backward" and wants_y0:
        name = "warp_rosenbrock"
        reasons.append("y0-unsupported")

    # If a non-replay named engine doesn't support the problem, fall back.
    # warp_replay has no entry in ENGINE_REGISTRY, so we check only registry engines.
    if name in ENGINE_REGISTRY and not ENGINE_REGISTRY[name].supports(problem):
        name = _fallback(problem, need_grad=True)
        reasons.append("engine-does-not-support-problem")

    # remat policy: None -> auto via choose_remat (consumed by the warp and rodas5p replay
    # paths; the scan engines accept and ignore it). Grad-transparent; trades compute for memory.
    if remat is None:
        remat = choose_remat(problem.dim, problem.is_stiff, batch_n=batch_n)

    # The fused reverse engines need a codegen'd analytic field per problem (with a specific
    # param layout). For an arbitrary user RHS (or a machine without Warp) fall back to a general-RHS
    # reverse engine that differentiates through problem.f_jax directly: the pure-JAX Tsit5
    # record-and-replay for nonstiff problems (the reference implementation), the
    # Rodas5P record-and-replay for stiff ones. This lets grad_closure work on any f_jax, not
    # only registered fields.
    if name == "warp_replay":
        if warp_ode.supports(problem):
            try:
                built = _build_reverse(
                    "warp_replay", problem, y0, params, wrt_t,
                    rtol=rtol, atol=atol, device=device, remat=remat, saveat=saveat,
                    precision=precision, saveat_dense=saveat_dense)
            except (IndexError, KeyError, ValueError, TypeError) as exc:
                # EXPECTED cause: supports() matched a field by NAME but the user's param
                # schema differs from the registered field's — fall through to the
                # general-RHS pure-JAX replay.
                #
                # The reason names what was OBSERVED, not that assumed cause: this catch is
                # broad enough to swallow an ordinary bug in the warp build (it silently ate
                # a TypeError from a kwarg mismatch and rerouted, which is how
                # this text got written), and .route.reason is a contract — it must not
                # assert a diagnosis it has not established.
                reasons.append(f"warp-build-failed:{type(exc).__name__}")
            else:
                return _stamp_route(built, requested, "warp_replay", reasons)
        else:
            reasons.append("no-registered-field")
        name = "tsit5_replay"

    built = _build_reverse(
        name, problem, y0, params, wrt_t, rtol=rtol, atol=atol, device=device, remat=remat,
        saveat=saveat, precision=precision, saveat_dense=saveat_dense)
    return _stamp_route(built, requested, name, reasons)
