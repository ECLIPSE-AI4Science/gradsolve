"""Tests for gradsolve/dispatch.py — the (dim, stiff, need_grad) engine dispatcher.

Covers:
  * ``choose_engine()`` over the full (dim, stiff, need_grad) grid, including the
    ``CUDA_NVAR_CEILING`` / ``NVAR_CEILING`` sub-band boundaries and the engine-enable
    gate overrides (``stiff_fused_enabled`` / ``cuda_tsit5_enabled``).
  * ``choose_remat()`` policy (nonstiff always remat; stiff gated at ``STIFF_REMAT_DIM``).
  * ``DECISION_MAP`` internal consistency: schema, every row agrees with
    ``choose_engine()`` at default gate settings, and with the *opposite*
    gate setting wherever a ``gated_engine`` is recorded — and, on the rows that route to
    diffrax, the install gate applied by ``api._fallback`` — so the routing table
    can never silently drift from the code it documents.

Imports only gradsolve (+ numpy/jax/pytest).
``choose_engine``/``choose_remat`` are pure functions of plain (dim, stiff, need_grad)
values, so no Problem instance is required for most of this file; one tiny duck-typed
Problem (matching gradsolve/base.py's Protocol) is defined at the bottom to confirm the
dispatcher round-trips correctly off real ``.dim``/``.is_stiff`` attributes, the way
``gradsolve/api.py``'s ``_resolve_engine`` actually drives it.
"""
from __future__ import annotations

import numpy as np
import pytest

from gradsolve import dispatch as gd
from gradsolve.dispatch import DECISION_MAP, choose_engine, choose_remat

_KNOWN_ENGINES = {
    "warp_ode",
    "cuda_tsit5",
    "cuda_rosenbrock23",
    "warp_replay",
    "fixed_step_tsit5",
    "fixed_step_imex",
    "warp_rosenbrock",
    "diffrax",
    "tsit5_replay",
    "rodas5p_replay",
}


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

def test_contract_constants_pinned():
    """The measured contract thresholds/gates, pinned so a silent drift is caught."""
    assert gd.NVAR_CEILING == 64
    assert gd.CUDA_NVAR_CEILING == 16
    assert gd.CUDA_ROSEN_NVAR_CEILING == 16
    assert gd.STIFF_REMAT_DIM == 16
    assert gd.STIFF_FUSED_ENABLED is True
    assert gd.CUDA_TSIT5_ENABLED is True
    # cuda_rosenbrock23 (the hand-CUDA stiff forward lane) is disabled by default — built and
    # CPU-validated but not yet measured on a GPU, so the stiff-forward cell routes unchanged.
    assert gd.CUDA_ROSENBROCK23_ENABLED is False
    # Ordering invariant the routing policy depends on: the hand-CUDA lanes each own a
    # STRICT sub-band of their low-NVAR forward cell, not the whole thing.
    assert 0 < gd.CUDA_NVAR_CEILING < gd.NVAR_CEILING
    assert 0 < gd.CUDA_ROSEN_NVAR_CEILING < gd.NVAR_CEILING


def test_nvar_ceiling_matches_warp_gpu_register_ceiling():
    """NVAR_CEILING must equal the fused kernel's own register-vector limit — the
    map's low/high-NVAR split is that crossover, not an independently-chosen number
    (module docstring: "MUST equal warp_ode._GPU_VEC_DIM_CEILING")."""
    from gradsolve.warp import warp_ode
    assert gd.NVAR_CEILING == warp_ode._GPU_VEC_DIM_CEILING


# ---------------------------------------------------------------------------
# choose_engine — explicit per-cell coverage (the documented decision map)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dim,stiff,need_grad,expected",
    [
        # non-stiff forward: cuda_tsit5 (dim<=16) -> warp_ode (16<dim<=64)
        #                    -> diffrax (dim>64, forward: general engine)
        (1, False, False, "cuda_tsit5"),
        (3, False, False, "cuda_tsit5"),
        (16, False, False, "cuda_tsit5"),
        (17, False, False, "warp_ode"),
        (32, False, False, "warp_ode"),
        (64, False, False, "warp_ode"),
        (65, False, False, "diffrax"),
        (1000, False, False, "diffrax"),
        # non-stiff grad: warp_replay (dim<=64) -> tsit5_replay (dim>64). The
        # cuda_tsit5/warp_ode split is forward-only and does not apply to this cell.
        (1, False, True, "warp_replay"),
        (3, False, True, "warp_replay"),
        (16, False, True, "warp_replay"),
        (17, False, True, "warp_replay"),
        (64, False, True, "warp_replay"),
        (65, False, True, "tsit5_replay"),
        (1000, False, True, "tsit5_replay"),
        # stiff forward (default gate True): warp_rosenbrock (dim<=64) -> diffrax (dim>64)
        (1, True, False, "warp_rosenbrock"),
        (3, True, False, "warp_rosenbrock"),
        (64, True, False, "warp_rosenbrock"),
        (65, True, False, "diffrax"),
        (1000, True, False, "diffrax"),
        # stiff grad (default gate True): warp_rosenbrock (dim<=64) -> rodas5p_replay (dim>64)
        (1, True, True, "warp_rosenbrock"),
        (3, True, True, "warp_rosenbrock"),
        (64, True, True, "warp_rosenbrock"),
        (65, True, True, "rodas5p_replay"),
        (1000, True, True, "rodas5p_replay"),
    ],
)
def test_choose_engine_grid(dim, stiff, need_grad, expected):
    assert choose_engine(dim=dim, stiff=stiff, need_grad=need_grad) == expected


def test_choose_engine_cuda_ceiling_boundary_is_le_not_lt():
    """The CUDA_NVAR_CEILING split is a <= boundary."""
    c = gd.CUDA_NVAR_CEILING
    assert choose_engine(dim=c, stiff=False, need_grad=False) == "cuda_tsit5"
    assert choose_engine(dim=c + 1, stiff=False, need_grad=False) == "warp_ode"


def test_choose_engine_nvar_ceiling_boundary_is_le_not_lt():
    """The NVAR_CEILING split is a <= boundary, across every (stiff, need_grad) class."""
    n = gd.NVAR_CEILING
    assert choose_engine(dim=n, stiff=False, need_grad=False) == "warp_ode"
    assert choose_engine(dim=n + 1, stiff=False, need_grad=False) == "diffrax"
    assert choose_engine(dim=n, stiff=False, need_grad=True) == "warp_replay"
    assert choose_engine(dim=n + 1, stiff=False, need_grad=True) == "tsit5_replay"
    assert choose_engine(dim=n, stiff=True, need_grad=False) == "warp_rosenbrock"
    assert choose_engine(dim=n + 1, stiff=True, need_grad=False) == "diffrax"
    assert choose_engine(dim=n, stiff=True, need_grad=True) == "warp_rosenbrock"
    assert choose_engine(dim=n + 1, stiff=True, need_grad=True) == "rodas5p_replay"


# ---------------------------------------------------------------------------
# choose_engine — gate overrides
# ---------------------------------------------------------------------------

def test_cuda_tsit5_override_false_restores_warp_ode_across_whole_low_band():
    for dim in (1, 3, 16, 32, 64):
        assert choose_engine(dim=dim, stiff=False, need_grad=False,
                              cuda_tsit5_enabled=False) == "warp_ode"
    # High NVAR is unaffected by the override either way.
    assert choose_engine(dim=1000, stiff=False, need_grad=False,
                          cuda_tsit5_enabled=False) == "diffrax"
    # The reverse (need_grad) cell is untouched by this override, on or off.
    for flag in (True, False):
        assert choose_engine(dim=3, stiff=False, need_grad=True,
                              cuda_tsit5_enabled=flag) == "warp_replay"


def test_cuda_rosenbrock23_override_true_owns_the_low_nvar_stiff_forward_sub_band():
    """With cuda_rosenbrock23_enabled=True the stiff + forward + dim<=CUDA_ROSEN_NVAR_CEILING
    sub-band routes to cuda_rosenbrock23; warp_rosenbrock keeps the rest of the low-NVAR
    stiff band. Mirrors the cuda_tsit5 forward sub-band split."""
    c = gd.CUDA_ROSEN_NVAR_CEILING
    for dim in (1, 3, c):
        assert choose_engine(dim=dim, stiff=True, need_grad=False,
                             cuda_rosenbrock23_enabled=True) == "cuda_rosenbrock23"
    for dim in (c + 1, 32, gd.NVAR_CEILING):
        assert choose_engine(dim=dim, stiff=True, need_grad=False,
                             cuda_rosenbrock23_enabled=True) == "warp_rosenbrock"
    # High NVAR is unaffected (the register budget rules the fused/CUDA kernels out).
    assert choose_engine(dim=1000, stiff=True, need_grad=False,
                         cuda_rosenbrock23_enabled=True) == "diffrax"
    # The reverse (need_grad) stiff cell is untouched by this override, on or off.
    for flag in (True, False):
        assert choose_engine(dim=3, stiff=True, need_grad=True,
                             cuda_rosenbrock23_enabled=flag) == "warp_rosenbrock"


def test_cuda_rosenbrock23_default_off_leaves_stiff_forward_on_warp_rosenbrock():
    """Default (flag False): the whole low-NVAR stiff-forward band stays on
    warp_rosenbrock — no routing change from adding the disabled CUDA stiff lane."""
    for dim in (1, 3, 16, 17, 32, 64):
        assert choose_engine(dim=dim, stiff=True, need_grad=False) == "warp_rosenbrock"
        assert choose_engine(dim=dim, stiff=True, need_grad=False,
                             cuda_rosenbrock23_enabled=False) == "warp_rosenbrock"


def test_stiff_fused_override_false_routes_to_the_general_stiff_engines():
    """Gate off (or no registered field, via api._fallback): forward -> diffrax, gradient ->
    rodas5p_replay — the same engines as the high-NVAR stiff cell."""
    for dim in (1, 3, 32, 64):
        assert choose_engine(dim=dim, stiff=True, need_grad=False,
                              stiff_fused_enabled=False) == "diffrax"
        assert choose_engine(dim=dim, stiff=True, need_grad=True,
                              stiff_fused_enabled=False) == "rodas5p_replay"


def test_stiff_fused_confined_to_low_nvar_even_when_forced_on():
    """Forcing stiff_fused_enabled=True at high NVAR must not route to warp_rosenbrock —
    the register budget rules the fused kernel out regardless of the gate flag."""
    for need_grad, engine in ((False, "diffrax"), (True, "rodas5p_replay")):
        assert choose_engine(dim=1000, stiff=True, need_grad=need_grad,
                              stiff_fused_enabled=True) == engine


# ---------------------------------------------------------------------------
# choose_engine — extra runtime keys are accepted but do not (yet) change routing
# ---------------------------------------------------------------------------

def test_choose_engine_batch_n_accuracy_target_accepted_no_routing_change():
    base = choose_engine(dim=3, stiff=True, need_grad=True)
    for batch_n in (1, 100, 10_000, 1_000_000):
        for accuracy_target in (1e-3, 1e-6, 1e-9):
            assert choose_engine(dim=3, stiff=True, need_grad=True,
                                  batch_n=batch_n,
                                  accuracy_target=accuracy_target) == base


# ---------------------------------------------------------------------------
# choose_engine — purity / totality over the full grid (never raises, deterministic)
# ---------------------------------------------------------------------------

def test_choose_engine_total_and_pure_over_full_grid():
    dims = (1, 2, 3, 15, 16, 17, 32, 63, 64, 65, 128, 1000, 26000)
    for dim in dims:
        for stiff in (False, True):
            for need_grad in (False, True):
                out1 = choose_engine(dim=dim, stiff=stiff, need_grad=need_grad)
                out2 = choose_engine(dim=dim, stiff=stiff, need_grad=need_grad)
                assert out1 == out2, "choose_engine must be a pure function of its inputs"
                assert out1 in _KNOWN_ENGINES
                assert isinstance(out1, str)


def test_need_grad_cells_never_route_to_diffrax_or_a_forward_only_lane():
    """Gradient requests stay on the record-and-replay lanes: diffrax is the forward-only general
    engine (its adjoint is a comparison baseline, not a routing target), and
    cuda_tsit5 / warp_ode have no reverse path."""
    for dim in (1, 3, 16, 17, 64, 65, 1000):
        for stiff in (False, True):
            assert choose_engine(dim=dim, stiff=stiff, need_grad=True) not in (
                "diffrax", "cuda_tsit5", "warp_ode")


def test_choose_engine_never_raises_on_degenerate_dims():
    for dim in (0, 1, -1):
        for stiff in (False, True):
            for need_grad in (False, True):
                out = choose_engine(dim=dim, stiff=stiff, need_grad=need_grad)
                assert out in _KNOWN_ENGINES


# ---------------------------------------------------------------------------
# choose_remat — remat policy (nonstiff always remat; stiff gated
# on STIFF_REMAT_DIM, where the per-step replay tape would otherwise blow up).
# ---------------------------------------------------------------------------

def test_choose_remat_nonstiff_always_true():
    for dim in (1, 3, 16, 32, 64, 128, 1000):
        assert choose_remat(dim, stiff=False) is True


def test_choose_remat_stiff_gated_on_stiff_remat_dim_boundary():
    d = gd.STIFF_REMAT_DIM
    assert choose_remat(d - 1, stiff=True) is False   # fits anyway -> no remat
    assert choose_remat(d, stiff=True) is True        # threshold -> remat
    assert choose_remat(d + 1, stiff=True) is True
    assert choose_remat(1000, stiff=True) is True      # would OOM without it -> remat


def test_choose_remat_accepts_batch_n_without_changing_policy():
    for batch_n in (None, 1, 1000, 1_000_000):
        assert choose_remat(3, stiff=True, batch_n=batch_n) is False
        assert choose_remat(3, stiff=False, batch_n=batch_n) is True


def test_choose_remat_returns_a_real_bool():
    # call-sites branch on this with `if remat:` / feed it straight to jax.checkpoint —
    # guard against a numpy bool_ (or similar near-bool) leaking through by luck.
    assert type(choose_remat(3, stiff=False)) is bool
    assert type(choose_remat(3, stiff=True)) is bool
    assert type(choose_remat(1000, stiff=True)) is bool


# ---------------------------------------------------------------------------
# DECISION_MAP — schema + internal consistency (the routing table
# choose_engine must never silently diverge from).
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"nvar", "stiff", "need_grad", "dim", "engine", "gated_engine", "evidence"}


def test_decision_map_row_schema():
    assert len(DECISION_MAP) == 9  # pinned to the current documented table
    for row in DECISION_MAP:
        assert _REQUIRED_KEYS <= set(row.keys())
        assert row["nvar"] in ("low", "high")
        assert isinstance(row["stiff"], bool)
        assert isinstance(row["need_grad"], bool)
        assert isinstance(row["dim"], int) and row["dim"] > 0
        assert row["engine"] in _KNOWN_ENGINES
        assert row["gated_engine"] is None or row["gated_engine"] in _KNOWN_ENGINES
        assert row["gated_engine"] != row["engine"]  # a "gate" that changes nothing is not a gate
        assert isinstance(row["evidence"], str) and len(row["evidence"]) > 20


def test_decision_map_nvar_label_matches_dim_vs_ceiling():
    for row in DECISION_MAP:
        is_low = row["dim"] <= gd.NVAR_CEILING
        assert (row["nvar"] == "low") == is_low, (
            f"row {row!r}: nvar label disagrees with dim vs NVAR_CEILING")


def test_decision_map_rows_agree_with_choose_engine_default_gates():
    """Every row's `engine` must equal what choose_engine() actually returns at
    default gate settings — the table cannot silently drift from the code."""
    for row in DECISION_MAP:
        got = choose_engine(dim=row["dim"], stiff=row["stiff"], need_grad=row["need_grad"])
        assert got == row["engine"], (
            f"DECISION_MAP row {row!r} disagrees with choose_engine -> {got!r}")


def test_decision_map_gated_engine_matches_the_gate():
    """Where a row records a `gated_engine`, the other side of its gate must land exactly there.
    Two kinds of gate: a choose_engine override (the fused-kernel enable flags), and — on the
    rows that route to diffrax — the diffrax optional extra, applied by api._fallback with the rule
    "a forward request diffrax cannot take goes where the gradient request goes", i.e. the same
    cell with need_grad=True."""
    seen_gated = 0
    for row in DECISION_MAP:
        if row["gated_engine"] is None:
            continue
        seen_gated += 1
        if row["engine"] == "diffrax":
            got = choose_engine(dim=row["dim"], stiff=row["stiff"], need_grad=True)
        elif row["stiff"]:
            got = choose_engine(dim=row["dim"], stiff=True, need_grad=row["need_grad"],
                                 stiff_fused_enabled=False)
        else:
            got = choose_engine(dim=row["dim"], stiff=False, need_grad=row["need_grad"],
                                 cuda_tsit5_enabled=False)
        assert got == row["gated_engine"], (
            f"DECISION_MAP row {row!r}: gated alternative -> {got!r}, "
            f"expected {row['gated_engine']!r}")
    assert seen_gated == 5  # cuda_tsit5 row, the two warp_rosenbrock rows, the two diffrax rows


def test_decision_map_rows_with_no_gate_are_override_invariant():
    """Rows recording gated_engine=None must be genuinely gate-independent: neither
    override changes their routing (there is nothing left to gate at that cell)."""
    for row in DECISION_MAP:
        if row["gated_engine"] is not None:
            continue
        for stiff_flag in (True, False):
            for cuda_flag in (True, False):
                got = choose_engine(dim=row["dim"], stiff=row["stiff"],
                                     need_grad=row["need_grad"],
                                     stiff_fused_enabled=stiff_flag,
                                     cuda_tsit5_enabled=cuda_flag)
                assert got == row["engine"], (
                    f"row {row!r} unexpectedly changed under override "
                    f"stiff_fused_enabled={stiff_flag}, cuda_tsit5_enabled={cuda_flag} "
                    f"-> {got!r}")


def test_decision_map_covers_every_nvar_stiff_need_grad_cell_at_least_once():
    """The 8 (nvar, stiff, need_grad) cells of the cube are all represented (some, like
    the low+nonstiff+forward cell, are split into sub-bands -> more than one row)."""
    cells = {(row["nvar"], row["stiff"], row["need_grad"]) for row in DECISION_MAP}
    expected = {
        (nvar, stiff, need_grad)
        for nvar in ("low", "high")
        for stiff in (False, True)
        for need_grad in (False, True)
    }
    assert cells == expected


def test_decision_map_no_dim_stiff_grad_key_has_conflicting_engines():
    """Sanity: choose_engine is a function of only (dim, stiff, need_grad) [+ gate
    overrides held at default], so two rows sharing that key can never legitimately
    disagree on `engine`."""
    seen: dict[tuple, str] = {}
    for row in DECISION_MAP:
        key = (row["dim"], row["stiff"], row["need_grad"])
        if key in seen:
            assert seen[key] == row["engine"], f"conflicting engine recorded for {key}"
        seen[key] = row["engine"]


def test_decision_map_low_nonstiff_forward_sub_band_split_holds_across_the_whole_band():
    """The low+nonstiff+forward cell is split by CUDA_NVAR_CEILING into two rows; verify
    choose_engine agrees with each row across its *whole* sub-band, not just at the row's
    single representative dim."""
    low_rows = [r for r in DECISION_MAP
                if r["nvar"] == "low" and not r["stiff"] and not r["need_grad"]]
    assert len(low_rows) == 2
    cuda_row = next(r for r in low_rows if r["dim"] <= gd.CUDA_NVAR_CEILING)
    warp_row = next(r for r in low_rows if r["dim"] > gd.CUDA_NVAR_CEILING)
    assert cuda_row["engine"] == "cuda_tsit5"
    assert warp_row["engine"] == "warp_ode"
    for dim in range(1, gd.CUDA_NVAR_CEILING + 1):
        assert choose_engine(dim=dim, stiff=False, need_grad=False) == cuda_row["engine"]
    for dim in range(gd.CUDA_NVAR_CEILING + 1, gd.NVAR_CEILING + 1):
        assert choose_engine(dim=dim, stiff=False, need_grad=False) == warp_row["engine"]


# ---------------------------------------------------------------------------
# Duck-typed Problem contract (gradsolve/base.py's Problem Protocol) feeding
# choose_engine, mirroring how gradsolve/api.py's _resolve_engine drives the
# dispatcher off real problem.dim / problem.is_stiff attributes in practice.
# ---------------------------------------------------------------------------

class _ToyProblem:
    """Minimal duck-typed Problem: only the members gradsolve/base.py's Protocol
    specifies (name/dim/t0/t1/is_stiff/f_jax) — no external Problem type."""

    def __init__(self, name: str, dim: int, stiff: bool):
        self.name = name
        self.dim = dim
        self.t0 = 0.0
        self.t1 = 1.0
        self._stiff = stiff

    @property
    def is_stiff(self) -> bool:
        return self._stiff

    def f_jax(self, t, y, params):
        # Trivial linear-decay field: y is (dim,), params is (1,). Never solved here —
        # this test file only exercises routing — but exercising f_jax once below
        # confirms the shape contract actually holds for this duck-type.
        import jax.numpy as jnp
        return -jnp.abs(params[0]) * y


def test_toy_problem_duck_type_satisfies_base_problem_shape_contract():
    """f_jax(t, y, params) must accept y:(dim,), params:(P,) and return dy/dt:(dim,) —
    the shape contract gradsolve/base.py's Problem Protocol specifies."""
    import jax.numpy as jnp
    prob = _ToyProblem("toy", dim=4, stiff=False)
    y = jnp.ones((4,), dtype=jnp.float64)
    params = jnp.array([2.0], dtype=jnp.float64)
    dydt = prob.f_jax(0.0, y, params)
    assert dydt.shape == (4,)
    assert dydt.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(dydt), -2.0 * np.ones(4), rtol=0, atol=1e-14)


def test_choose_engine_from_duck_typed_problem_attrs():
    """choose_engine is driven off `problem.dim` / `problem.is_stiff`, exactly as
    gradsolve.api._resolve_engine does; confirm it round-trips through a Problem
    duck-type that only defines the base.Problem Protocol members."""
    nonstiff_lowdim = _ToyProblem("toy_nonstiff_low", dim=3, stiff=False)
    stiff_lowdim = _ToyProblem("toy_stiff_low", dim=3, stiff=True)
    nonstiff_highdim = _ToyProblem("toy_nonstiff_high", dim=1000, stiff=False)

    assert choose_engine(dim=nonstiff_lowdim.dim, stiff=nonstiff_lowdim.is_stiff,
                          need_grad=False) == "cuda_tsit5"
    assert choose_engine(dim=stiff_lowdim.dim, stiff=stiff_lowdim.is_stiff,
                          need_grad=True) == "warp_rosenbrock"
    assert choose_engine(dim=nonstiff_highdim.dim, stiff=nonstiff_highdim.is_stiff,
                          need_grad=False) == "diffrax"
