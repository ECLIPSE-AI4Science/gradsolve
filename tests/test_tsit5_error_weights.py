"""Tsit5 embedded-error weights must match diffrax's tableau exactly."""
import numpy as np
import pytest


def _diffrax_tableau():
    diffrax = pytest.importorskip("diffrax")
    return diffrax.Tsit5.tableau  # ButcherTableau: .c, .b_sol, .b_error, .a_lower


def test_error_weights_match_diffrax():
    from gradsolve.solvers import fixed_step_tsit5 as m
    tab = _diffrax_tableau()
    ours_b = np.array([m._B1, m._B2, m._B3, m._B4, m._B5, m._B6, m._B7])
    ours_e = np.array([m._E1, m._E2, m._E3, m._E4, m._E5, m._E6, m._E7])
    np.testing.assert_allclose(ours_b, np.asarray(tab.b_sol, dtype=np.float64), rtol=0, atol=1e-15)
    np.testing.assert_allclose(ours_e, np.asarray(tab.b_error, dtype=np.float64), rtol=0, atol=1e-15)
