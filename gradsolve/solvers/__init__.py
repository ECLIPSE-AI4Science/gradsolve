"""Solver spine of the ``gradsolve`` library: fixed-step solvers for differential equations.

The spine consists of structured fixed-step solvers run as a single ``jax.lax.scan``:
reverse-mode differentiable by construction (bounded static tape, no ``while_loop``),
XLA-friendly, and an execution model that still scales at high state dimension
(around 80-256 state variables).

Members (each has forward-correctness, reverse-mode, and determinism tests):
  - fixed_step_explicit : fixed-step explicit Runge-Kutta in lax.scan (non-stiff).
  - fixed_step_tsit5    : Tsit5 in lax.scan (non-stiff, order 5).
  - imex                : linearly-implicit facade over ``gradsolve/solvers/fixed_step_imex.py``
                          (stiff; linearly-implicit Euler).

Three verification checks every solver must pass:
  Forward correctness   : max err < tol vs analytic / scipy Radau-LSODA (float64, CPU).
  Reverse mode          : jax.grad of a scalar functional == central finite-diff (~1e-5).
  Fixed-step determinism: identical step count across the ensemble (CV == 0), so no warp
                          divergence.
"""
from __future__ import annotations

# Opt-in high-order nonstiff record-and-replay engine. Re-exported for
# discoverability only — deliberately NOT added to the spine _MODULES/REGISTRY below (those
# hold fixed-step solve_jax spine modules; vern7_replay is a record-and-replay reverse engine).
# Opt-in high-order STIFF record-and-replay engine. Same contract as vern7_replay:
# re-exported for discoverability only, NOT a spine _MODULES/REGISTRY member.
from gradsolve.solvers import (
    fixed_step_explicit,
    fixed_step_tsit5,
    imex,
    rodas5p_replay,
    rodas5p_step,
    vern7_replay,
    vern7_step,
)

# The spine registry (duck-typed modules exposing name/supports/solve/solve_jax).
#   fixed_step_explicit : RK4   (non-stiff)
#   fixed_step_tsit5    : Tsit5 (non-stiff, order 5)
#   imex                : linearly-implicit Euler facade (stiff; structured fixed-step IMEX)
_MODULES = [fixed_step_explicit, fixed_step_tsit5, imex]
REGISTRY: dict[str, object] = {m.name: m for m in _MODULES}


def get_solver(name: str):
    """Look up a spine solver module by its registered name.

    Parameters
    ----------
    name : str
        Solver key as registered in `REGISTRY` -- each module's own ``name``
        attribute (e.g. ``"fixed_step_explicit"``, ``"fixed_step_tsit5"``,
        ``"imex_euler"``), not the module's Python identifier.

    Returns
    -------
    object
        The duck-typed solver module exposing ``name``/``supports``/
        ``solve``/``solve_jax``.

    Raises
    ------
    KeyError
        If `name` is not a key of `REGISTRY`.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown solver {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["REGISTRY", "get_solver", "fixed_step_explicit", "fixed_step_tsit5", "imex",
           "vern7_replay", "vern7_step", "rodas5p_replay", "rodas5p_step"]
