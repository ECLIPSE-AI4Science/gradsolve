"""IMEX stiff spine member, exposed under the solvers API.

This is a thin facade. It exposes the structured fixed-step IMEX method — run as a single
``jax.lax.scan`` (Newton-free, natively reverse-mode differentiable, zero warp divergence by
construction) — under the solvers spine namespace. The generic stiff member already exists as
``gradsolve/solvers/fixed_step_imex.py`` (order-1 linearly-implicit / Rosenbrock-Euler): this
module simply re-exposes it and does not reimplement any numerics. It is the generic,
problem-agnostic stiff spine member.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gradsolve.base import SolveResult

if TYPE_CHECKING:  # gradsolve imports standalone; Problem is annotation-only
    from gradsolve.base import Problem

name = "imex_euler"


def supports(problem: Problem) -> bool:
    """Report whether this facade can solve `problem` (delegates to `fixed_step_imex`).

    Parameters
    ----------
    problem : Problem
        Candidate problem, forwarded unchanged to
        `gradsolve.solvers.fixed_step_imex.supports`.

    Returns
    -------
    bool
        Same as `fixed_step_imex.supports` -- always ``True`` (L-stable,
        small dense Jacobian; both stiff and non-stiff problems supported).
    """
    from gradsolve.solvers import fixed_step_imex
    return fixed_step_imex.supports(problem)


def solve_jax(problem, y0, params, *, n_steps: int = None):
    """Pure-JAX differentiable solve: (y0[n,dim], params[n,P]) -> y_final[n,dim]."""
    from gradsolve.solvers import fixed_step_imex
    if n_steps is None:
        n_steps = fixed_step_imex.DEFAULT_N_STEPS
    return fixed_step_imex.solve_jax(problem, y0, params, n_steps=n_steps)


def solve(
    problem: Problem,
    y0: np.ndarray,
    params: np.ndarray,
    *,
    rtol: float,
    atol: float,
    device: str = "cpu",
    n_steps: int = None,
) -> SolveResult:
    """Numpy Backend-protocol entry point (delegates to the stiff backend)."""
    from gradsolve.solvers import fixed_step_imex
    if n_steps is None:
        n_steps = fixed_step_imex.DEFAULT_N_STEPS
    return fixed_step_imex.solve(
        problem, y0, params, rtol=rtol, atol=atol, device=device, n_steps=n_steps
    )
