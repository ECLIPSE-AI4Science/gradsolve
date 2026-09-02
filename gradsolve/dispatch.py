"""The engine="auto" dispatcher and its decision map.

``choose_engine(dim, stiff, need_grad) -> str`` is a pure function that maps an
ensemble-ODE workload's three salient axes — state dimension (NVAR), stiffness class,
and whether reverse-mode gradients are needed — to the engine that benchmarks show is
fastest in that corner. ``DECISION_MAP`` is the auditable list of rows backing the same
policy, and a test asserts ``choose_engine`` agrees with it on every row.

What ``choose_engine`` returns — engine identifiers, not backend keys
--------------------------------------------------------------------
The strings ``choose_engine`` can return — ``"cuda_tsit5"``, ``"warp_ode"``,
``"warp_replay"``, ``"warp_rosenbrock"``, ``"diffrax"``, ``"tsit5_replay"``,
``"rodas5p_replay"`` — are routing targets consumed by the gradient-closure factory and the
engine roster, **not** a guarantee that each is a directly ``solve``-able Backend module.
All but one are registered engines; the exception is ``"warp_replay"``, the
record-and-replay adjoint routing target: the fused ``warp_ode`` forward records accepted
dts, and the gradient is a fixed-step ``lax.scan`` replay. Its helpers live in
``gradsolve.warp.warp_replay`` (``make_replay_closure`` /
``make_rosenbrock_replay_closure`` / ``solve_jax``), driven by the gradient-closure
factory, but that module does not expose the ``name``/``supports``/``solve`` Backend
protocol and is not a registered backend. That is by design: ``warp_replay`` is a routing
target, not a ``solve``-able Backend module.

The decision map
----------------
The thresholds below are the routing rule; the qualitative reason behind each cell is
recorded next to it.

Registered generated fields. The fused cells below ("warp_ode", "warp_replay",
"warp_rosenbrock") route by whether the problem has a registered Warp field. That set is
not limited to the hand-written built-ins: ``gradsolve.register_jax_field`` (and
``fused=True`` on ``solve``/``grad_closure``, which translates a user ``f_jax`` and
registers it) adds a generated field to the same registries, so a matching ``Problem``
routes to these fused cells exactly as a built-in does. The default ``fused="auto"`` does
not change this routing on its own: it routes an already-registered field and, for an
unregistered problem, only checks fused-eligibility (recording a reason on
``SolveResult.route`` when the RHS is outside the translatable subset) while leaving the
general-engine routing here unchanged. Either way: no new engine name, no new cell.

  * low-NVAR (dim <= NVAR_CEILING) + non-stiff + forward-only  -> "warp_ode"
        Fused adaptive Tsit5, one CUDA thread per trajectory: the whole integration stays
        in registers with no per-step host round trip, which is what makes it faster than
        the pure-JAX baselines at low dimension. With ``CUDA_TSIT5_ENABLED`` the
        ``dim <= CUDA_NVAR_CEILING`` sub-band goes to the hand-CUDA ``cuda_tsit5`` kernel
        instead (see the flag below).

  * low-NVAR + non-stiff + need_grad                           -> "warp_replay"
        Record-and-replay adjoint (fused forward records accepted dts; gradient is a
        fixed-step lax.scan replay in pure JAX, reverse-differentiable by construction).
        The device-resident record keeps the total gradient cost well below a full
        adaptive reverse solve, and the gradient agrees with finite differences (relative
        error below 1e-5). Note: it differentiates the realized trajectory with the step
        controller frozen (dt = data).

  * high-NVAR (dim > NVAR_CEILING) [non-stiff] -> "diffrax" (forward) / "tsit5_replay" (grad)
        Above NVAR_CEILING the fused kernel's per-thread register vector spills and the
        kernel is no longer competitive. The request is served by general-RHS engines that
        honour rtol/atol: diffrax for a forward-only solve, the Tsit5 record-and-replay for
        gradients (and for forward when diffrax is absent). The fixed 10 000-step scan is
        explicit engine= only: it accepts rtol/atol but ignores them.

  * stiff + low-NVAR [forward or need_grad] -> "warp_rosenbrock"
        Fused linearly-implicit Rosenbrock23 (in-kernel analytic Jacobian + LU per stage).
        Forward: an adaptive second-order L-stable method in a single fused launch is both
        much faster and more accurate on stiff problems than the fixed-step first-order
        IMEX scan. Gradient: its reverse path is the record-and-replay Rosenbrock adjoint,
        which at matched accuracy costs less per gradient than an adaptive implicit reverse
        solve through diffrax (Kvaerno5); the fixed-step IMEX scan is slower than both at
        matched accuracy, so it is never auto-routed. See the routing flag below for the
        coverage caveat. A stiff problem without a registered Rosenbrock field takes
        diffrax (forward) / rodas5p_replay (gradient).

  * stiff + high-NVAR -> "diffrax" (forward) / "rodas5p_replay" (grad)
        The fused kernel is out of register budget. rodas5p_replay is the general-RHS
        Rodas5P record-and-replay (jax.jacfwd Jacobian, exact frozen-mesh adjoint, honours
        rtol/atol) and costs less per gradient than an adaptive implicit reverse solve at
        matched accuracy. The order-1 fixed IMEX scan (10 000 steps) is explicit engine=
        only: on a stiff problem it is both slower and far less accurate than the adaptive
        engines at matched tolerance.

Routing flag — ``STIFF_FUSED_ENABLED``
--------------------------------------
``warp_rosenbrock`` (the stiff-fused engine) is enabled only where it is measurably faster
than the best pure-JAX stiff baseline; the threshold is a speedup of at least 3x. On Robertson
(float64) the fused kernel clears that margin by a wide factor while also being more
accurate (a few hundred adaptive second-order steps against 10 000 fixed first-order
steps), so the criterion holds under any matched-accuracy reading. ``STIFF_FUSED_ENABLED``
is therefore ``True``: the stiff + low-NVAR + forward-only cell routes to
``warp_rosenbrock``. Coverage caveat (identical in kind to ``warp_ode``'s registered-field
coverage): the kernel needs an analytic-Jacobian ``@wp.func`` per problem and supports
robertson, hires, and any ``linstiff_*`` field the caller registers via
``register_linstiff``; a stiff problem without a registered field is not
``supports()``-ed, and the caller falls back to diffrax / rodas5p_replay.
``choose_engine`` also takes an override parameter ``stiff_fused_enabled`` (default = the
module flag) so either routing is testable without mutating module state.
"""
from __future__ import annotations

from typing import List, TypedDict

# --- Contract constants (the decision map's thresholds and flags) --------------------

#: High-NVAR threshold. dim <= NVAR_CEILING is "low NVAR" (the fused kernel's design
#: point); dim > NVAR_CEILING is "high NVAR" (served by the general engines: diffrax and
#: the record-and-replay solvers).
#: Must equal ``warp_ode._GPU_VEC_DIM_CEILING`` — the same register-vector crossover the
#: fused GPU kernel itself enforces (a test pins them equal).
NVAR_CEILING = 64

#: cuda_tsit5's upper dim within the low-NVAR forward cell. The hand-CUDA kernel stays
#: register-resident (no spill) up to D=64 but is fastest only at low D: its throughput
#: degrades sharply by D=32, where per-thread register pressure collapses occupancy. So the
#: hand-CUDA lane is routed only where it is fastest (dim <= this bound) and warp_ode keeps
#: the rest of the low-NVAR band (this < dim <= NVAR_CEILING).
CUDA_NVAR_CEILING = 16

#: cuda_rosenbrock23's upper dim within the low-NVAR stiff forward cell — the stiff
#: analogue of ``CUDA_NVAR_CEILING``. The hand-CUDA stiff kernel holds a DxD W + analytic
#: Jacobian + LU workspace per thread (O(D^2) registers), so like the fused Warp stiff kernel
#: it is a low-NVAR engine; the bound of 16 is conservative. Above it, warp_rosenbrock keeps
#: the low-NVAR stiff band.
CUDA_ROSEN_NVAR_CEILING = 16

#: remat policy threshold: dim at/above which the stiff replay's per-step tape
#: (jacfwd D*D + linsolve D^3 residuals) is large enough that jax.checkpoint pays off.
#: At dim 8 the tape fits comfortably and remat only costs speed; at dim 32 remat cuts
#: reverse memory by two orders of magnitude and at dim 64 the replay does not fit without
#: it. The onset lies between, hence 16.
STIFF_REMAT_DIM = 16

#: Routing flag: warp_rosenbrock (stiff-fused) is enabled only where it is measurably
#: faster than the best pure-JAX stiff baseline (the threshold is a speedup of at least 3x).
#: On Robertson (float64) the fused adaptive Rosenbrock23 kernel clears that margin by a
#: wide factor and is more accurate than the fixed-step first-order IMEX scan, so the flag
#: is True and the stiff+low-NVAR+forward cell routes to warp_rosenbrock. Coverage caveat
#: (same as warp_ode): warp_rosenbrock has registered analytic-Jacobian fields for
#: robertson, hires and linstiff_* only; other stiff problems need a registered field.
#: Callers verify supports() and fall back to
#: diffrax / rodas5p_replay, exactly as for warp_ode's field coverage.
STIFF_FUSED_ENABLED = True

#: Routing flag: ``cuda_tsit5`` (the hand-CUDA forward-only Tsit5 lane) is the validated
#: fast-forward kernel: register-resident with no spill up to D=64, GPU results at machine
#: precision against warp_ode, and several times faster than warp_ode at D=3 on the same
#: Tsit5 algorithm. While False, the nonstiff + low-NVAR + forward-only cell keeps routing
#: to ``warp_ode``; True routes the ``dim <= CUDA_NVAR_CEILING`` sub-band of that cell to
#: ``cuda_tsit5`` and warp_ode keeps the rest of the low-NVAR band. ``choose_engine`` takes
#: an override ``cuda_tsit5_enabled`` (default = this flag) so both routings stay testable
#: without mutating module state — exactly mirroring ``STIFF_FUSED_ENABLED``. The reverse
#: cell is unaffected (need_grad still -> ``warp_replay``); ``warp_ode`` stays for the
#: record/reverse path regardless.
CUDA_TSIT5_ENABLED = True

#: Routing flag: ``cuda_rosenbrock23`` (the hand-CUDA forward-only stiff Rosenbrock23 lane)
#: is the stiff analogue of ``cuda_tsit5``; it is disabled by default. While False, the
#: stiff + low-NVAR + forward cell keeps routing to ``warp_rosenbrock``; enable it with
#: ``dispatch.CUDA_ROSENBROCK23_ENABLED = True`` or by naming the engine explicitly
#: (``engine="cuda_rosenbrock23"``).
#: ``choose_engine`` takes an override ``cuda_rosenbrock23_enabled`` (default = this flag)
#: so both routings stay testable without mutating module state — mirroring
#: ``CUDA_TSIT5_ENABLED``. Once enabled, the dim <= CUDA_ROSEN_NVAR_CEILING sub-band routes
#: here. The reverse stiff cell is unaffected (need_grad stays on warp_rosenbrock /
#: rodas5p_replay).
CUDA_ROSENBROCK23_ENABLED = False


# --- Engine names (the dispatcher's codomain) ----------------------------------------
_WARP_ODE = "warp_ode"
_CUDA_TSIT5 = "cuda_tsit5"  # GPU-gated forward-only lane; reachable only when CUDA_TSIT5_ENABLED
_WARP_REPLAY = "warp_replay"
_WARP_ROSENBROCK = "warp_rosenbrock"  # GPU-gated; only reachable when STIFF_FUSED_ENABLED
_CUDA_ROSENBROCK23 = "cuda_rosenbrock23"  # GPU-gated stiff forward lane; only when CUDA_ROSENBROCK23_ENABLED
_DIFFRAX = "diffrax"                # forward-only general engine (optional extra; api._fallback skips it when absent)
_TSIT5_REPLAY = "tsit5_replay"      # general-RHS nonstiff record-and-replay (gradients; forward without diffrax)
_RODAS5P_REPLAY = "rodas5p_replay"  # general-RHS stiff record-and-replay (gradients; forward without diffrax)


def choose_engine(
    dim: int,
    stiff: bool,
    need_grad: bool,
    *,
    batch_n: int | None = None,            # accepted runtime key; not yet a branch point
    accuracy_target: float | None = None,  # accepted runtime key; not yet a branch point
    stiff_fused_enabled: bool | None = None,
    cuda_tsit5_enabled: bool | None = None,
    cuda_rosenbrock23_enabled: bool | None = None,
) -> str:
    """Map a workload (dim, stiff, need_grad) to the engine name for that corner (pure).

    Args:
        dim: state dimension (NVAR). ``dim <= NVAR_CEILING`` is "low NVAR".
        stiff: True if the problem is stiff (drives implicit vs explicit).
        need_grad: True if reverse-mode gradients (w.r.t. params) are required.
        batch_n: number of parallel trajectories. Accepted runtime key; the policy
            currently branches on dim/stiff/need_grad only, and batch thresholds are not
            yet populated. Divergence is deliberately not a key (frozen-mesh engines are
            divergence-immune).
        accuracy_target: desired per-step accuracy. Accepted runtime key; not yet a
            branch point, and accuracy thresholds are not yet populated. ``batch_n`` and
            ``accuracy_target`` are accepted now so the call signature stays stable when
            they become branch points.
        stiff_fused_enabled: override for the module ``STIFF_FUSED_ENABLED`` routing
            flag (default None -> use the module flag). When True, the stiff + low-NVAR +
            forward-only cell routes to ``warp_rosenbrock``; when False it routes to the
            general stiff engines (``diffrax`` forward / ``rodas5p_replay`` gradient).
        cuda_rosenbrock23_enabled: override for the module ``CUDA_ROSENBROCK23_ENABLED``
            routing flag (default None -> use the module flag). When True, the stiff +
            forward + ``dim <= CUDA_ROSEN_NVAR_CEILING`` sub-cell routes to
            ``cuda_rosenbrock23``; when False (the default) that cell stays on
            ``warp_rosenbrock``. The reverse stiff cell is unaffected either way.

    Returns:
        One of: ``"cuda_tsit5"``, ``"cuda_rosenbrock23"``, ``"warp_ode"``, ``"warp_replay"``,
        ``"warp_rosenbrock"``, ``"diffrax"``, ``"tsit5_replay"``, ``"rodas5p_replay"``. Total
        over the whole
        (dim, stiff, need_grad) space — never raises, no side effects. ``"diffrax"`` is a
        forward-only target; ``api._fallback`` replaces it with the replay lane of the same
        class when diffrax is not installed.
    """
    if stiff_fused_enabled is None:
        stiff_fused_enabled = STIFF_FUSED_ENABLED
    if cuda_tsit5_enabled is None:
        cuda_tsit5_enabled = CUDA_TSIT5_ENABLED
    if cuda_rosenbrock23_enabled is None:
        cuda_rosenbrock23_enabled = CUDA_ROSENBROCK23_ENABLED
    low_nvar = dim <= NVAR_CEILING

    if stiff:
        # STIFF: at low NVAR the fused Rosenbrock23 kernel owns forward AND gradient (its grad is
        # the record->replay rosenbrock adjoint, cheaper per gradient than an adaptive implicit
        # reverse solve through diffrax-Kvaerno5 at matched accuracy).
        # Gated behind the routing flag; a problem without a registered analytic-Jacobian field
        # falls through api._fallback to the same general engines as the high-NVAR cell.
        if low_nvar and stiff_fused_enabled:
            # Forward-only sub-band: the hand-CUDA stiff kernel owns dim <=
            # CUDA_ROSEN_NVAR_CEILING once enabled (mirrors cuda_tsit5's forward sub-band).
            # Disabled by default (CUDA_ROSENBROCK23_ENABLED=False), so this is a no-op until
            # the flag is enabled; warp_rosenbrock keeps the rest of the low-NVAR stiff band
            # and the whole cell today. The reverse cell is untouched.
            if (not need_grad and cuda_rosenbrock23_enabled
                    and dim <= CUDA_ROSEN_NVAR_CEILING):
                return _CUDA_ROSENBROCK23
            return _WARP_ROSENBROCK
        # General stiff engines: the Rodas5P record-and-replay for gradients (exact frozen-mesh
        # adjoint, honours rtol/atol); diffrax Kvaerno5 for a forward-only solve (api._fallback
        # substitutes rodas5p_replay when diffrax is not installed).
        return _RODAS5P_REPLAY if need_grad else _DIFFRAX

    # NON-stiff.
    if not low_nvar:
        # High NVAR: the fused register-vector kernel loses its budget. General engines: the Tsit5
        # record-and-replay for gradients; diffrax Tsit5 forward (tsit5_replay when absent).
        return _TSIT5_REPLAY if need_grad else _DIFFRAX

    # Low NVAR, non-stiff: the fused engine's design point.
    if need_grad:
        return _WARP_REPLAY  # record-and-replay adjoint (frozen-controller discrete adjoint)
    # Forward-only: cuda_tsit5 owns the sub-cell where it is fastest (dim <= CUDA_NVAR_CEILING;
    # register-resident with no spill, several times faster than warp_ode at low D); warp_ode
    # keeps the rest of the low-NVAR band (CUDA_NVAR_CEILING < dim <= NVAR_CEILING). Reverse
    # cell untouched.
    if cuda_tsit5_enabled and dim <= CUDA_NVAR_CEILING:
        return _CUDA_TSIT5
    return _WARP_ODE


def choose_remat(dim: int, stiff: bool, *, batch_n: int | None = None) -> bool:
    """Whether to ``jax.checkpoint`` the record-and-replay reverse scan (pure policy).

    Consumed by ``grad_closure`` and threaded into ``warp_replay`` /
    ``warp_rosenbrock`` replay closures (the scan-replay reverse paths). remat is
    gradient-transparent (recompute the forward step in the backward sweep instead of
    storing it) — it trades compute for memory.

    Policy, from GPU measurements at fixed batch size:
      * Non-stiff (Tsit5 replay): always remat. At low dim it is a speed gain as well as a
        memory saving; at mid/high dim the speed is roughly neutral but reverse memory
        still drops sharply, so it is never a net loss and larger batches fit.
      * Stiff (Rosenbrock23 replay, per-step jacfwd+linsolve): remat only at
        ``dim >= STIFF_REMAT_DIM``. There the tape grows very large (at dim 32 remat cuts
        reverse memory by two orders of magnitude; at dim 64 the replay does not fit
        without it), so the modest speed cost buys the headroom that makes the cell run.
        Below the threshold it fits anyway, so remat is a pure speed loss and stays off.

    ``batch_n`` is accepted (larger batch raises tape memory -> remat more valuable) but
    does not affect the policy, which branches on dim/stiff only. The warp replay paths and
    ``rodas5p_replay`` use the result; the scan engines (fixed_step_tsit5/imex) ignore
    ``remat``.
    """
    if not stiff:
        return True
    return dim >= STIFF_REMAT_DIM


# --- DECISION_MAP: the auditable rows choose_engine must agree with -------------------

class DecisionRow(TypedDict):
    """One row of the decision map.

    ``DECISION_MAP`` is a ``List[DecisionRow]`` — the auditable table a test asserts
    ``choose_engine`` agrees with on every row.

    Attributes
    ----------
    nvar : str
        ``"low"`` or ``"high"`` — which side of ``NVAR_CEILING`` this row covers.
    stiff : bool
        Whether this row's RHS is stiff.
    need_grad : bool
        Whether this row's workload needs reverse-mode gradients.
    dim : int
        A representative dimension on this row's side of ``NVAR_CEILING`` (used
        by the test that pins ``choose_engine`` to this table).
    engine : str
        The routing in effect — ``choose_engine``'s return value under the current
        module-level routing flags (e.g. ``STIFF_FUSED_ENABLED``,
        ``CUDA_TSIT5_ENABLED``).
    gated_engine : str or None
        The routing on the other side of this row's routing flag (what it
        routed to, or would route to, before/without the gate flip); ``None``
        where the row has no gated alternative.
    evidence : str
        The design rationale for this row's routing.
    """
    nvar: str          # "low" | "high"
    stiff: bool
    need_grad: bool
    dim: int           # representative dim on this side of NVAR_CEILING
    engine: str        # routing under the default flags
    gated_engine: str | None  # routing once a GPU gate enables the fused alternative
    evidence: str      # the rationale for this row's routing


DECISION_MAP: List[DecisionRow] = [
    {
        "nvar": "low", "stiff": False, "need_grad": False,
        "dim": 3, "engine": _CUDA_TSIT5, "gated_engine": _WARP_ODE,
        "evidence": "The hand-CUDA cuda_tsit5 kernel serves the forward-only nonstiff cell up to "
                    "dim <= CUDA_NVAR_CEILING because it stays register-resident there and is "
                    "several times faster than warp_ode on the same Tsit5 algorithm, while a "
                    "gradient request goes to warp_replay and a build without nvcc/CUDA falls "
                    "back to warp_ode.",
    },
    {
        "nvar": "low", "stiff": False, "need_grad": False,
        "dim": 32, "engine": _WARP_ODE, "gated_engine": None,
        "evidence": "warp_ode, the fused adaptive Tsit5 kernel, keeps the upper low-NVAR forward "
                    "band (CUDA_NVAR_CEILING < dim <= NVAR_CEILING) because cuda_tsit5's "
                    "throughput degrades sharply by dim 32 as per-thread register pressure "
                    "collapses occupancy.",
    },
    {
        "nvar": "low", "stiff": False, "need_grad": True,
        "dim": 3, "engine": _WARP_REPLAY, "gated_engine": None,
        "evidence": "The record-and-replay adjoint differentiates through a fused forward with "
                    "the step controller frozen, which keeps the total gradient cost well below "
                    "a full adaptive reverse solve while the gradient agrees with finite "
                    "differences.",
    },
    {
        "nvar": "high", "stiff": False, "need_grad": False,
        "dim": 1000, "engine": _DIFFRAX, "gated_engine": _TSIT5_REPLAY,
        "evidence": "Above NVAR_CEILING the fused kernel's per-thread register vector spills and "
                    "no fused kernel is competitive, so a forward-only request goes to diffrax "
                    "(adaptive Tsit5, honours rtol/atol) when that optional extra is installed "
                    "and otherwise to tsit5_replay, the same adaptive Tsit5 with a host recorder "
                    "and JAX replay; the fixed 10 000-step scan is explicit engine= only because "
                    "it ignores rtol/atol.",
    },
    {
        "nvar": "high", "stiff": False, "need_grad": True,
        "dim": 1000, "engine": _TSIT5_REPLAY, "gated_engine": None,
        "evidence": "The general-RHS Tsit5 record-and-replay (the reference implementation) is an "
                    "exact frozen-mesh adjoint through the problem's f_jax that honours rtol/atol "
                    "and is reverse-differentiable by construction, so it serves every nonstiff "
                    "gradient the fused kernels cannot.",
    },
    {
        "nvar": "low", "stiff": True, "need_grad": False,
        "dim": 3, "engine": _WARP_ROSENBROCK, "gated_engine": _DIFFRAX,
        "evidence": "The fused Rosenbrock23 kernel (adaptive, second-order, L-stable, one launch) "
                    "is both far faster and more accurate on stiff problems than the fixed-step "
                    "first-order IMEX scan, so STIFF_FUSED_ENABLED is True; a stiff problem "
                    "without a registered Rosenbrock field (robertson, hires, linstiff) takes "
                    "diffrax Kvaerno5 when installed and otherwise rodas5p_replay, and the "
                    "hand-CUDA cuda_rosenbrock23 lane (dim <= CUDA_ROSEN_NVAR_CEILING) is "
                    "disabled by default.",
    },
    {
        "nvar": "low", "stiff": True, "need_grad": True,
        "dim": 3, "engine": _WARP_ROSENBROCK, "gated_engine": _RODAS5P_REPLAY,
        "evidence": "The warp_rosenbrock record-and-replay adjoint costs less per gradient at "
                    "matched accuracy than an adaptive implicit reverse solve through diffrax "
                    "Kvaerno5, so it owns the stiff low-NVAR gradient cell; a stiff problem "
                    "without a registered Rosenbrock field takes rodas5p_replay (general-RHS "
                    "Rodas5P record-and-replay, exact frozen-mesh adjoint, honours rtol/atol) and "
                    "never the order-1 fixed scan, which is slower than both at matched accuracy.",
    },
    {
        "nvar": "high", "stiff": True, "need_grad": False,
        "dim": 1000, "engine": _DIFFRAX, "gated_engine": _RODAS5P_REPLAY,
        "evidence": "The fused kernel is out of register budget at high NVAR, so a forward-only "
                    "stiff request goes to diffrax Kvaerno5 (honours rtol/atol) when installed "
                    "and otherwise to rodas5p_replay; the order-1 fixed IMEX scan is explicit "
                    "engine= only because on a stiff problem it is both slower and far less "
                    "accurate than the adaptive engines at matched tolerance.",
    },
    {
        "nvar": "high", "stiff": True, "need_grad": True,
        "dim": 1000, "engine": _RODAS5P_REPLAY, "gated_engine": None,
        "evidence": "The general-RHS Rodas5P record-and-replay (forward-mode Jacobian via "
                    "jacfwd) is an exact frozen-mesh adjoint that honours rtol/atol, uses remat "
                    "at dim >= STIFF_REMAT_DIM (choose_remat), and costs less per gradient than "
                    "an adaptive implicit reverse solve at matched accuracy.",
    },
]
