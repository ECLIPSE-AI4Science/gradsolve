"""The Backend contract for ensemble solves of differential equations.

A Backend wraps one framework's idiomatic ensemble ODE solve. ``solve`` returns the
per-trajectory final state plus per-trajectory step statistics, with the same result
layout from every engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np


class Problem(Protocol):
    """Structural (duck-typed) contract a gradsolve problem must satisfy.

    gradsolve reads only the members below off a problem, so any object exposing
    them satisfies this Protocol with no inheritance — a plain dataclass, a module,
    or your own problem type all match automatically. Anything else a problem object
    may carry (batch factories, reference solutions, alternative RHS implementations,
    …) is outside this contract and gradsolve never touches it.

    Attributes
    ----------
    name : str
        Human-readable problem identifier; the fused engines match registered fields by it.
    dim : int
        State dimension (NVAR) — the axis ``dispatch.choose_engine`` routes on.
    t0 : float
        Integration start time.
    t1 : float
        Integration end time.
    """

    name: str
    dim: int
    t0: float
    t1: float

    @property
    def is_stiff(self) -> bool:
        """Whether the RHS is stiff (routes explicit vs implicit engines)."""
        ...

    def f_jax(self, t: float, y: Any, params: Any) -> Any:
        """JAX-traceable RHS for a single trajectory.

        gradsolve vmaps this over the ensemble, so it is written per-trajectory:
        ``y`` is one state vector ``(dim,)`` and ``params`` is that trajectory's
        parameter vector ``(P,)`` — never the batched ``(n, dim)`` / ``(n, P)``.

        Parameters
        ----------
        t : float
            Current integration time (scalar).
        y : Any
            State for one trajectory, shape ``(dim,)``.
        params : Any
            Parameters for one trajectory, shape ``(P,)``.

        Returns
        -------
        Any
            ``dy/dt`` for this trajectory, same shape as ``y`` — ``(dim,)``.
        """
        ...


@dataclass(frozen=True)
class Route:
    """Where a ``solve`` or ``grad_closure`` request actually went. Attached as ``.route``
    to every returned closure and to every ``SolveResult``.

    The requested engine and the engine that ran are different things — an engine can be
    rerouted because it does not ``supports()`` the problem, because the problem matches no
    registered Warp field, or because the request asked for something the engine cannot do
    (a y0 gradient from the params-only fused backward). Callers and tests read the actual
    route here instead of assuming the request was honoured.

    Attributes
    ----------
    requested : str
        The ``engine=`` argument as passed (``'auto'`` or an engine name).
    actual : str
        The engine that actually ran. Note this is a descriptive lane name, not necessarily
        an ``ENGINE_REGISTRY`` key: ``'warp_replay'`` is the one non-registry lane name.
    reason : str, optional
        Why ``actual`` differs from ``requested``; ``None`` when the request was honoured.
        Multiple reasons are joined with ``'; '``.
    """
    requested: str
    actual: str
    reason: Optional[str] = None


@dataclass
class SolveResult:
    """Result of one batched ensemble ODE solve.

    Every :class:`Backend` returns one of these from ``solve()``, so callers see the same
    result layout regardless of which engine produced it.

    Attributes
    ----------
    y_final : np.ndarray
        Final state per trajectory, shape ``(n, dim)``.
    accepted_steps : np.ndarray
        Per-trajectory accepted step count, shape ``(n,)``. Per-trajectory counts let
        callers measure how unevenly work is spread across the ensemble. Contract:
        adaptive backends
        fill the true per-trajectory count; fixed-step backends fill the
        constant step count repeated ``n`` times (not empty). The length-0
        default is reserved for the genuinely unreportable case — a backend
        whose API exposes no step counts at all.
    rejected_steps : np.ndarray
        Per-trajectory rejected step count, shape ``(n,)``; same fill contract
        as ``accepted_steps``.
    solver : str
        The name of the engine that produced this result.
    y_saved : np.ndarray, optional
        Dense output, shape ``(n, k, dim)`` — the states at ``ts_saved``. ``None`` unless
        ``solve(..., saveat=ts)`` requested it. Host NumPy, like ``y_final``.
    ts_saved : np.ndarray, optional
        The requested output times, shape ``(k,)``; ``None`` unless ``saveat`` was passed.
        Echoed back so a caller holding only the result knows what the rows mean.
    route : Route, optional
        The routing record (see :class:`Route`); ``None`` unless produced by
        ``gradsolve.solve``.
    """

    # Final state per trajectory, shape (n, dim).
    y_final: np.ndarray
    # Per-trajectory accepted / rejected step counts, shape (n,). Per-trajectory counts let
    # callers measure how unevenly work is spread across the ensemble. Contract:
    #   * adaptive backends fill the true per-trajectory accepted / rejected counts;
    #   * fixed-step backends have no per-trajectory variance, so they fill the constant
    #     step count repeated n times (accepted) and zeros (rejected) — NOT empty;
    #   * the empty (length-0) default is reserved for the genuinely unreportable case
    #     (a backend whose API exposes no step counts at all).
    accepted_steps: np.ndarray = field(default_factory=lambda: np.empty(0))
    rejected_steps: np.ndarray = field(default_factory=lambda: np.empty(0))
    # The name of the engine that produced this result.
    solver: str = ""
    # Dense output (n, k, dim) at ts_saved (k,) — both None unless solve(saveat=ts) asked
    # for it. Optional + defaulted so every existing Backend construction stays valid.
    y_saved: Optional[np.ndarray] = None
    ts_saved: Optional[np.ndarray] = None
    # Where the request went (requested engine, engine that ran, why they differ). Stamped by
    # gradsolve.solve(); None when a Backend's own solve() built the result directly.
    route: Optional[Route] = None


class Backend(Protocol):
    """Structural (duck-typed) contract one framework's ensemble ODE backend must satisfy.

    A Backend wraps one engine's batched solve (a JAX scan or replay solver, diffrax, a
    fused NVIDIA Warp kernel, a hand-written CUDA kernel, …) behind one call, so every
    engine is driven the same way. Duck-typed like :class:`Problem`: no inheritance
    required, only ``name``/``supports``/``solve``.

    Attributes
    ----------
    name : str
        Backend identifier, recorded in ``SolveResult.route``.
    """

    name: str

    def supports(self, problem: Problem) -> bool:
        """Whether this backend can solve the given problem (e.g. a non-stiff-only engine).

        Parameters
        ----------
        problem : Problem
            The problem to check.

        Returns
        -------
        bool
            True if ``solve`` can handle ``problem``.
        """
        ...

    def solve(
        self,
        problem: Problem,
        y0: np.ndarray,
        params: np.ndarray,
        *,
        rtol: float,
        atol: float,
        device: str,
    ) -> SolveResult:
        """Solve the ensemble. Must block until the device is done before returning.

        Parameters
        ----------
        problem : Problem
            The problem to solve.
        y0 : np.ndarray
            Initial state per trajectory, shape ``(n, dim)``.
        params : np.ndarray
            Per-trajectory parameters, shape ``(n, P)``.
        rtol, atol : float
            Solver tolerances.
        device : str
            ``'cpu'`` or ``'cuda'``.

        Returns
        -------
        SolveResult
            Final state plus per-trajectory step statistics.
        """
        ...
