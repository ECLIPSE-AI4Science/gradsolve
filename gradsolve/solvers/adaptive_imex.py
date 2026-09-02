"""CFL-sized uniform grids and a fixed-length ``lax.scan`` replay for IMEX-type steps.

Standalone module (jax is imported lazily). It holds two reusable, testable pieces:

  * `cfl_n_steps` / `cfl_grid` — the k-only CFL step-count rule and the uniform grid it
    sizes. A transport (free-streaming) term with a constant factor k has a CFL stability
    limit k*dtau < cfl that is tau-independent, so a uniform grid with N(k) =
    clip(ceil(span*k/cfl), n_floor, n_cap) steps is already CFL-optimal. N is a static
    constant independent of the model parameters -> exact reverse-mode gradients with no
    stop_gradient.

  * `solve_replay` — a generic fixed-length `lax.scan` "replay" over a frozen explicit
    grid, given any single-step function step_fn(tau, dtau, y) -> y_next. Reverse-diff by
    construction (bounded static tape, no while_loop).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

name = "adaptive_imex"

# Two Courant numbers govern the uniform grid: k*dtau below ~0.30 keeps the high-k transport
# stable; k*dtau below ~0.03 is needed for 1e-5 accuracy. The grid is sized for accuracy, the
# binding constraint.
DEFAULT_CFL = 0.03        # accuracy Courant number (stability limit is ~0.30)
DEFAULT_N_FLOOR = 2000    # minimum step count, keeps low/mid-k accurate
DEFAULT_N_CAP = 26000     # maximum step count


def cfl_n_steps(k, tau_start, tau_end, *, cfl: float = DEFAULT_CFL,
                n_floor: int = DEFAULT_N_FLOOR, n_cap: int | None = None) -> int:
    """k-only CFL step count: N(k) = clip(ceil(span*k/cfl), n_floor, n_cap).

    A uniform grid of N steps over [tau_start, tau_end] then has dtau = span/N <= cfl/k,
    so the transport CFL condition k*dtau <= cfl holds everywhere. N depends on k only
    (cfl/n_floor/n_cap are fixed structural constants), never on the model parameters,
    so reverse-mode gradients through the replayed solve are exact.
    """
    span = float(tau_end) - float(tau_start)
    if span <= 0:
        raise ValueError(f"tau_end ({tau_end}) must exceed tau_start ({tau_start})")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n_cfl = math.ceil(span * float(k) / float(cfl))
    N = max(int(n_floor), int(n_cfl))
    if n_cap is not None:
        N = min(N, int(n_cap))
    return int(N)


# --- k-aware accuracy floor ---------------------------------------------------------
# A flat n_floor over-resolves smooth low-k modes: the step count needed for accuracy rises
# from ~250 at k=1e-5 to ~2000 at k=2.236e-3, rather than sitting at a flat 2000. The sub-CFL
# accuracy requirement is modelled as a weak power law in k, anchored at the mid-k value
# (N=2000 at k=2.236e-3) with log-log slope 0.38, which upper-bounds the low/mid-k
# requirement. Above ~k=8e-3 the acoustic CFL term (span*k/c_acc) takes over. N(k) stays
# k-only (independent of the model parameters) -> exact reverse-mode gradients, no
# stop_gradient.
_ACC_KNEE_N = 2000.0      # mid-k accuracy step count, the anchor of the power law
_ACC_KNEE_K = 2.236e-3    # k at which the anchor applies
_ACC_FLOOR_SLOPE = 0.38   # log-log slope of the sub-CFL accuracy floor
DEFAULT_N_MIN = 250       # accuracy floor at the lowest k


def accuracy_n_steps(k, tau_start, tau_end, *, cfl: float = DEFAULT_CFL,
                     n_min: int = DEFAULT_N_MIN, n_cap: int = DEFAULT_N_CAP) -> int:
    """k-aware step count: max(acoustic CFL term, the power-law accuracy floor), clamped.

    N(k) = clip( max( ceil(span*k/cfl),  ceil(N_knee*(k/k_knee)**slope) ), n_min, n_cap ).
    The first term is the high-k acoustic-resolution CFL (c_acc~0.03); the second is the
    low/mid-k accuracy floor anchored at the mid-k value. Reclaims the low-k modes a flat
    floor over-resolved (e.g. ~256 vs 2000 at k=1e-5). k-only -> exact reverse-mode
    gradients.
    """
    span = float(tau_end) - float(tau_start)
    if span <= 0:
        raise ValueError(f"tau_end ({tau_end}) must exceed tau_start ({tau_start})")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    acoustic = math.ceil(span * float(k) / float(cfl))
    floor = math.ceil(_ACC_KNEE_N * (float(k) / _ACC_KNEE_K) ** _ACC_FLOOR_SLOPE)
    N = max(int(acoustic), int(floor))
    return int(min(max(N, int(n_min)), int(n_cap)))


def cfl_grid(k, tau_start, tau_end, *, cfl: float = DEFAULT_CFL,
             n_floor: int = DEFAULT_N_FLOOR, n_cap: int | None = None) -> np.ndarray:
    """The (uniform) frozen CFL grid: N(k)+1 tau points, endpoints pinned, CFL-stable.

    Uniform because the transport CFL limit is tau-independent (transport ~ const*k), so
    no interval may be coarser than cfl/k -> uniform dtau=cfl/k is already optimal.
    """
    N = cfl_n_steps(k, tau_start, tau_end, cfl=cfl, n_floor=n_floor, n_cap=n_cap)
    return np.linspace(float(tau_start), float(tau_end), N + 1)


def solve_replay(step_fn: Callable, tau_grid, y0):
    """Replay a frozen explicit tau grid through a fixed-length reverse-diff lax.scan.

    step_fn(tau, dtau, y) -> y_next is any single-step kernel; tau_grid is an explicit
    (possibly non-uniform) 1-D array of tau nodes. Returns the full history
    (len(tau_grid), *y0.shape). Pure lax.scan of pure functions -> jax.grad flows.
    """
    import jax
    import jax.numpy as jnp

    tau_grid = jnp.asarray(tau_grid)
    dtaus = jnp.diff(tau_grid)

    def step(y_prev, inp):
        tau, dtau = inp
        y_next = step_fn(tau, dtau, y_prev)
        return y_next, y_next

    _, y_seq = jax.lax.scan(step, y0, (tau_grid[:-1], dtaus))
    return jnp.concatenate([y0[None, ...], y_seq], axis=0)
