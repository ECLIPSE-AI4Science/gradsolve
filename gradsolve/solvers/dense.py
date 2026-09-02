"""Dense output (``saveat``) for the JAX scan/replay lanes — streaming bracket-and-re-step.

The shared core behind ``saveat=`` on every JAX lane: the fixed-step scans
(``fixed_step_tsit5``, ``fixed_step_imex``) and the record-and-replay adjoints
(``warp_replay``'s Tsit5 and Rosenbrock23 replays, and the general-RHS
``tsit5_replay``). Fused kernels and the hand-CUDA lane stay final-state-only by design —
see ``docs/api.md`` and the router in ``gradsolve/api.py``.

Why streaming. The obvious implementation stacks every step's state and interpolates
afterwards, which costs ``O(n*S*dim)`` — for a 10,000-step ensemble that dwarfs the solve
itself. Instead the scan carries only the active bracketing pair plus the ``k`` accumulating
outputs, and each step writes the saveat times that fall inside it:

    carry = (t, y, y_left[k, dim], t_left[k], written[k])  ->  O(n*k*dim)
    mesh  = dts[n, S]                                      ->  O(n*S)

No ``(S, dim)`` state history is ever materialized. Peak scaling is therefore today's
mesh/tape terms plus ``k*n*dim*8`` bytes of outputs. Per-step cost is ``O(k*dim)``: a scan
cannot early-exit, so the write mask is evaluated over all k saveat times each step.

How a state is produced — bracket, then re-step (not an interpolant).

The scan records, for each requested time, the state at the start of the step that brackets
it. After the scan, one genuine solver step of size ``ts - t_left`` is taken from each
bracket straight to the requested time, using the lane's own step function. So a saved state
is not interpolated at all: it is integrated, by the same method, to exactly the time asked
for.

A generic cubic Hermite interpolant is not used, because its interior error can exceed the
record tolerance on adaptive meshes: the controller takes wide steps where the problem is
easy, and a cubic across a wide bracket is far less accurate than the 5th-order states at
its ends. A caller who records at rtol=1e-6 and asks for y(0.5) should not silently get an
interior error orders of magnitude larger. Re-stepping avoids that at lower cost than an
interpolant: it needs no extra RHS evaluation per step (a Hermite interpolant needs one,
since ``tsit5_step``/``rosenbrock23_step`` return only
``y_next`` and Tsit5's FSAL is deliberately unexploited for bit-parity), and only ``k``
extra steps in total after the scan, vectorized over the requested times. It also needs no
new tableau: every lane reuses the step function it already integrates with, so the stiff
Rosenbrock and order-1 IMEX lanes get the same treatment for free rather than each needing
its own interpolant.

Two subtleties, both essential:

* **dt == 0 padding.** Replay meshes are zero-padded to a common length S. A padding step is
  an exact identity step, and the bracket mask ``(dt > 0) & (ts > t) & (ts <= t + dt)`` is
  empty for dt == 0, so padding is skipped by construction and claims no output time.
* **fp drift at t1.** A replayed mesh accumulates ``t0 + sum(dts)``, which can land an ulp
  below ``t1``; a saveat entry of exactly ``t1`` would then fall in no bracket. The
  ``written`` flags let unmatched entries resolve after the scan — to ``y0`` at or before
  ``t0``, else to the terminal state. Times at or past the final accumulated time resolve to
  ``y_final`` outright, which is what makes ``y_saved[:, -1] == y_final`` hold bitwise (a
  re-step of ``t1 - t_left`` would otherwise differ from the scan's own last step by an ulp)
  and what makes duplicate terminal times resolve to the terminal state.

An opt-in interpolating lane, for methods that publish a continuous extension.

Re-stepping is accurate but it is not free: it executes one extra solver step per requested
time, which for a dense saveat grid can be a large fraction of the mesh itself.
``scan_saveat_dense_one``/``vmap_saveat_dense`` delete those re-steps by evaluating the
step's own interpolating polynomial instead — but only with a method-specific one. A
generic cubic Hermite interpolant is not accurate enough (see above); it survives only as
the comparison arm in ``tests/test_rodas5p_dense_saveat.py``.

Rodas5P's published order-4 extension does not have that defect: on adaptive meshes its
interior error is inside the record tolerance, where cubic Hermite is outside it. The
trade-off is that the polynomial is less accurate than a genuine re-step on the same mesh —
both well inside the record tolerance — which is why the lane is opt-in and the re-step pair
below is left untouched.

Scoring is against the record tolerance, never against the endpoint error: an order-4
extension on an order-5 step has an interior/endpoint ratio that grows like 1/h, so a
constant-factor-of-endpoint gate would fail a correct method at tight tolerance.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def validate_saveat(saveat, t0: float, t1: float) -> np.ndarray:
    """Validate a saveat grid against the problem domain; return it as host float64.

    Sorted (non-decreasing — duplicates are allowed and resolve consistently) and inside
    ``[t0, t1]``. Anything else raises ``ValueError``: silently clipping or reordering a
    caller's requested output times would return a time series that is not the one asked
    for.
    """
    ts = np.asarray(saveat, dtype=np.float64)
    if ts.ndim != 1:
        raise ValueError(f"saveat must be 1-D (k,); got shape {ts.shape}")
    if ts.size == 0:
        raise ValueError("saveat must contain at least one time")
    if not np.all(np.isfinite(ts)):
        raise ValueError("saveat contains non-finite times")
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"saveat must be sorted non-decreasing; got {ts}")
    lo, hi = (t0, t1) if t1 >= t0 else (t1, t0)
    if ts[0] < lo or ts[-1] > hi:
        raise ValueError(
            f"saveat times must lie within the problem domain [{lo}, {hi}]; got "
            f"[{ts[0]}, {ts[-1]}]")
    return ts


def scan_saveat_one(f, p, t0, y0, dts, ts, step_fn):
    """Integrate one trajectory over ``dts``, emitting the states requested by ``ts``.

    Bracket-and-re-step (see the module docstring): the scan records each requested time's
    bracketing left state, then one real ``step_fn`` step of size ``ts - t_left`` carries it
    to the requested time. Saved states are integrated by the lane's own method, not
    interpolated.

    Parameters
    ----------
    f : callable
        RHS ``f(t, y, p)`` (``problem.f_jax``).
    p : array
        This trajectory's parameter vector, ``(P,)``.
    t0 : float
        Start time.
    y0 : array
        This trajectory's initial state, ``(dim,)``.
    dts : array
        Step sequence ``(S,)`` — a recorded accepted-dt mesh (zero-padded past the
        trajectory's accepted count) or a constant fixed-step grid.
    ts : array
        Sorted output times ``(k,)``, already validated against the domain.
    step_fn : callable
        ``step_fn(f, t, y, dt, p) -> y_next`` — ``tsit5_step``, ``rosenbrock23_step`` or
        ``imex_euler_step``. The same per-step arithmetic the lane uses without ``saveat``.

    Returns
    -------
    (y_final, ys_saved) : (array ``(dim,)``, array ``(k, dim)``)
    """
    def body(carry, dt):
        t, y, y_left, t_left, written = carry
        t_next = t + dt
        # Which requested times does THIS step bracket? Empty for a dt == 0 padding step.
        hit = (dt > 0) & (ts > t) & (ts <= t_next)
        y_left = jnp.where(hit[:, None], y, y_left)
        t_left = jnp.where(hit, t, t_left)
        written = written | hit
        return (t_next, step_fn(f, t, y, dt, p), y_left, t_left, written), None

    k = ts.shape[0]
    t0_j = jnp.asarray(t0, dtype=y0.dtype)
    init = (t0_j, y0,
            jnp.broadcast_to(y0, (k, y0.shape[0])),
            jnp.full((k,), t0_j),
            jnp.zeros((k,), dtype=bool))
    (t_fin, y_fin, y_left, t_left, written), _ = jax.lax.scan(body, init, dts)

    # One genuine step per requested time, from its bracket to exactly that time.
    ys = jax.vmap(lambda tl, yl, d: step_fn(f, tl, yl, d, p))(t_left, y_left, ts - t_left)

    # Times no step bracketed: at or before t0 -> y0 (never stepped).
    ys = jnp.where(written[:, None], ys, jnp.broadcast_to(y0, (k, y0.shape[0])))
    # At or past the final accumulated time -> the terminal state itself. Covers fp drift
    # below t1 and duplicate terminal times, and makes y_saved[-1] == y_final BITWISE.
    ys = jnp.where((ts >= t_fin)[:, None], y_fin, ys)
    return y_fin, ys


def scan_saveat_dense_one(f, p, t0, y0, dts, ts, stages_fn, weights_fn):
    """Streaming saveat by method-specific continuous extension.

    Same bracket bookkeeping as :func:`scan_saveat_one`, but the state at a requested time is
    produced by evaluating the step's own interpolating polynomial rather than by taking an
    extra solver step. Deletes the ``k`` post-scan re-steps; costs ~5*k*dim per step instead
    of ~2*k*dim in the mask.

    Not a generic interpolant: this is Rodas5P's published order-4 continuous extension (see
    the module docstring for why a generic cubic Hermite is not used); covered by
    tests/test_rodas5p_dense_saveat.py.

    ``theta`` uses the safe-denominator form: ``jnp.where(dt > 0, (ts - tl)/dt, 0.0)`` has the
    right value but a NaN derivative at the dt == 0 padding steps, so only the double-where
    survives reverse mode.

    Parameters
    ----------
    stages_fn : callable
        ``stages_fn(f, t, y, dt, p) -> (y_next, coeffs)`` — one step plus its dense-output
        coefficient bundle (``rodas5p_stages_and_dense``). ``y_next`` must be byte-identical
        to the lane's own step, so opting in cannot perturb the mesh.
    weights_fn : callable
        ``weights_fn(coeffs, theta) -> y(theta)`` — evaluates that bundle
        (``rodas5p_dense_eval``). ``theta`` arrives as a ``(k, 1)`` column, so all k requested
        times of a step are evaluated at once.

    Returns
    -------
    (y_final, ys_saved) : (array ``(dim,)``, array ``(k, dim)``)
    """
    def body(carry, dt):
        t, y, ys, written = carry
        t_next = t + dt
        y_next, coeffs = stages_fn(f, t, y, dt, p)
        # Which requested times does THIS step bracket? Empty for a dt == 0 padding step.
        hit = (dt > 0) & (ts > t) & (ts <= t_next)
        # Safe denominator: the naive single where has value 0.0 but derivative NaN at dt == 0,
        # and replay meshes are zero-padded, so it would NaN every gradient through a padded
        # trajectory while looking perfectly correct in forward mode.
        dt_safe = jnp.where(dt > 0, dt, 1.0)
        theta = jnp.where(dt > 0, (ts - t) / dt_safe, 0.0)
        # Non-bracketing entries are discarded by the mask below anyway; zeroing theta first
        # keeps the polynomial from evaluating at a wildly extrapolated argument.
        theta = jnp.where(hit, theta, 0.0)
        ys = jnp.where(hit[:, None], weights_fn(coeffs, theta[:, None]), ys)
        return (t_next, y_next, ys, written | hit), None

    k = ts.shape[0]
    t0_j = jnp.asarray(t0, dtype=y0.dtype)
    init = (t0_j, y0,
            jnp.broadcast_to(y0, (k, y0.shape[0])),
            jnp.zeros((k,), dtype=bool))
    (t_fin, y_fin, ys, written), _ = jax.lax.scan(body, init, dts)

    # The two post-rules are shared verbatim with the re-step path above: times no step
    # bracketed resolve to y0, and times at or past the final accumulated time resolve to the
    # terminal state itself. The latter is what keeps y_saved[-1] == y_final BITWISE — with a
    # naive theta, fp drift in the accumulated mesh lands the terminal save at 1 + O(ulp).
    ys = jnp.where(written[:, None], ys, jnp.broadcast_to(y0, (k, y0.shape[0])))
    ys = jnp.where((ts >= t_fin)[:, None], y_fin, ys)
    return y_fin, ys


def validate_dense_thetas(dts, ts, t0, *, bracket=None, ulp_slack=64):
    """Host-side guard: every interpolated time must land at ``theta`` in [0, 1] of the step
    that brackets it. Raises ``ValueError`` otherwise.

    Outside [0, 1] a continuous extension extrapolates, and a traced guard cannot raise from
    inside a scan — so this runs once on the concrete mesh, in the kernel builders.

    Two exemptions, both because the time is never interpolated:

    * **terminal saves** — a time at or past the final accumulated node takes the
      ``ts >= t_fin -> y_fin`` substitution before any theta is used, and fp drift in the
      accumulated mesh legitimately puts it at ``theta = 1 + O(ulp)``;
    * **times at or before t0** — they resolve to ``y0``.

    ``bracket`` selects which rule is being checked, and this is the point of the guard:

    * ``None`` (default) — derive the bracket with the streaming scan's own rule
      (``dt > 0 & ts > t & ts <= t_next``, last hit wins). Under that rule ``theta`` is in
      (0, 1] *by construction*, so the range check cannot fire and the reachable failure is
      a mesh that brackets one time twice (only possible if it is non-monotonic), where
      last-write-wins silently picks one of them. That case is rejected.
    * an int array ``(k,)`` or ``(n, k)`` of step indices — check a bracket computed
      elsewhere. This is the case the range check exists for: a host-precomputed map
      (the mesh-only variant flips to "the step whose right endpoint is >= ts") is not self-limiting, and an
      off-by-one there would silently extrapolate.

    Any consumer that computes its own bracket map must call this with that map via
    ``bracket=``: a host-precomputed entry table whose rule is not the scan's is exactly such a
    consumer, and the default derivation above would otherwise validate a bracket it does not
    actually use.

    ``ulp_slack`` is the allowance, in units of the local ulp, on the [0, 1] endpoints.
    No-op on traced input: a jit'd mesh has no host value to check.
    """
    if any(isinstance(a, jax.core.Tracer) for a in (dts, ts, t0, bracket)):
        return
    dts_h = np.asarray(dts, dtype=np.float64)
    ts_h = np.asarray(ts, dtype=np.float64)
    if dts_h.ndim == 1:
        dts_h = dts_h[None, :]
    if bracket is not None:
        bracket = np.asarray(bracket, dtype=np.int64)
        if bracket.ndim == 1:
            bracket = np.broadcast_to(bracket, (dts_h.shape[0], bracket.shape[0]))
        if bracket.shape != (dts_h.shape[0], ts_h.shape[0]):
            raise ValueError(
                f"bracket must be (k,) or (n, k) = {(dts_h.shape[0], ts_h.shape[0])}; "
                f"got {bracket.shape}")

    for row in range(dts_h.shape[0]):
        steps = dts_h[row]
        nodes = float(t0) + np.concatenate([[0.0], np.cumsum(steps)])
        t_left, t_right, t_fin = nodes[:-1], nodes[1:], nodes[-1]
        for j, tj in enumerate(ts_h):
            if tj <= nodes[0] or tj >= t_fin:
                continue                                    # y0 / terminal — never interpolated
            if bracket is None:
                hits = np.nonzero((steps > 0) & (tj > t_left) & (tj <= t_right))[0]
                if hits.size == 0:
                    raise ValueError(
                        f"saveat[{j}]={tj!r} is strictly inside trajectory {row}'s mesh "
                        f"[{nodes[0]!r}, {t_fin!r}] but no step brackets it; the dense lane "
                        "would silently resolve it to y0.")
                if hits.size > 1:
                    raise ValueError(
                        f"saveat[{j}]={tj!r} is bracketed by {hits.size} steps of trajectory "
                        f"{row} ({list(hits)}) — the mesh is not monotonic, and the scan's "
                        "last-write-wins mask would silently keep only one of them.")
                i = int(hits[0])
            else:
                i = int(bracket[row, j])
                if not 0 <= i < steps.shape[0]:
                    raise ValueError(
                        f"bracket[{row}, {j}]={i} is out of range for a {steps.shape[0]}-step "
                        "mesh.")
                if not steps[i] > 0:
                    raise ValueError(
                        f"bracket[{row}, {j}]={i} points at a dt={steps[i]!r} step; a padding "
                        "or non-advancing step cannot bracket a requested time.")
            theta = (tj - t_left[i]) / steps[i]
            slack = ulp_slack * np.spacing(max(1.0, abs(theta)))
            if not (-slack <= theta <= 1.0 + slack):
                raise ValueError(
                    f"saveat[{j}]={tj!r} brackets to theta={theta!r} on step {i} of trajectory "
                    f"{row} (dt={steps[i]!r}); outside [0, 1] the continuous extension "
                    "EXTRAPOLATES. Mesh and saveat grid disagree.")


def vmap_saveat(f, t0, y0, params, dts, ts, step_fn):
    """Ensemble form of :func:`scan_saveat_one`: ``-> (y_final[n,dim], ys[n,k,dim])``.

    ``dts`` is either per-trajectory ``(n, S)`` (a recorded, zero-padded replay mesh) or a
    single shared ``(S,)`` grid (the fixed-step lanes). A shared grid is broadcast by vmap's
    ``in_axes=None`` rather than tiled — tiling a 10,000-step grid across the ensemble would
    allocate ``n*S`` floats for no information.
    """
    ts_j = jnp.asarray(ts)
    dts_j = jnp.asarray(dts)
    if dts_j.ndim not in (1, 2):
        raise ValueError(f"dts must be (S,) shared or (n, S) per-trajectory; got {dts_j.shape}")

    def one(y0_j, p_j, dts_row):
        return scan_saveat_one(f, p_j, t0, y0_j, dts_row, ts_j, step_fn)

    in_axes = (0, 0, None) if dts_j.ndim == 1 else (0, 0, 0)
    return jax.vmap(one, in_axes=in_axes)(jnp.asarray(y0), jnp.asarray(params), dts_j)


def vmap_saveat_dense(f, t0, y0, params, dts, ts, stages_fn, weights_fn):
    """Ensemble form of :func:`scan_saveat_dense_one`: ``-> (y_final[n,dim], ys[n,k,dim])``.

    ``in_axes`` handling mirrors :func:`vmap_saveat` exactly — ``dts`` is either
    per-trajectory ``(n, S)`` or a single shared ``(S,)`` grid broadcast by ``in_axes=None``.

    Does not call :func:`validate_dense_thetas`: this is the per-call hot path, and the mesh
    it would check is frozen. The kernel builders run that guard once, at build time.
    """
    ts_j = jnp.asarray(ts)
    dts_j = jnp.asarray(dts)
    if dts_j.ndim not in (1, 2):
        raise ValueError(f"dts must be (S,) shared or (n, S) per-trajectory; got {dts_j.shape}")

    def one(y0_j, p_j, dts_row):
        return scan_saveat_dense_one(f, p_j, t0, y0_j, dts_row, ts_j, stages_fn, weights_fn)

    in_axes = (0, 0, None) if dts_j.ndim == 1 else (0, 0, 0)
    return jax.vmap(one, in_axes=in_axes)(jnp.asarray(y0), jnp.asarray(params), dts_j)
