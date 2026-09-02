"""Regression: gradsolve must be importable and runnable *standalone*, with no ``examples``
package on the import path.

A fresh subprocess sets ``sys.modules["examples"] = None`` *before* importing gradsolve
(the standard trick that makes any subsequent ``import examples`` raise ImportError
immediately, instead of silently resolving off ``sys.path``), then exercises
``gradsolve.solve`` + ``gradsolve.grad_closure`` on an inline duck-typed ``Problem`` (linear
decay, dy/dt = -k*y — analytic solution, so both the forward state and the reverse
gradient have closed-form ground truth to check against, no scipy/FD-only smoke test).

Imports only ``gradsolve`` (+ stdlib/numpy/jax/pytest). The
child script is deliberately a subprocess (not an in-process ``sys.modules`` patch):
patching ``sys.modules`` in the *test* process would leak into every other test in the
session, so isolation requires a fresh interpreter.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The forward-only general nonstiff engine on this interpreter. The child runs the same
#: interpreter, so it sees the same install: diffrax is an optional extra, and without it
#: api._fallback serves the forward request from the Tsit5 record-and-replay lane.
_FWD_NONSTIFF = "diffrax" if importlib.util.find_spec("diffrax") is not None else "tsit5_replay"

# NOTE: kept as a single triple-quoted script (not a helper module on disk) so there is
# no on-disk import path for the child to accidentally resolve back to this test file's
# package (tests/ has no __init__.py) or to any ``examples`` package.
_CHILD_SCRIPT = r'''
import sys
# Block any ``examples`` package before importing gradsolve. If gradsolve's import chain (or
# anything it lazily reaches for) ever depends on one, the very next "import examples"
# anywhere below raises ImportError instead of quietly resolving off sys.path.
sys.modules["examples"] = None

import json

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve  # noqa: E402  -- must come after the sys.modules block above


class LinearDecay:
    """dy/dt = -k*y -> analytic y(t) = y0 * exp(-k*(t - t0)).

    Minimal duck-typed Problem (gradsolve.base.Problem contract): name/dim/t0/t1/is_stiff
    + f_jax(t, y, params) called per trajectory (y is (dim,), params is (P,) — gradsolve
    vmaps over the ensemble). Deliberately not one of gradsolve's registered field names
    (lorenz/vdp/robertson/linear_ladder_*), so it exercises the general-RHS engines that
    differentiate through f_jax directly (not a fused/registered kernel) -- the
    standalone-user path.
    """

    name = "toy_linear_decay"
    dim = 1
    t0 = 0.0
    t1 = 2.0
    is_stiff = False

    def f_jax(self, t, y, p):
        k = p[..., 0]
        return (-k * y[..., 0])[..., None]


def main():
    prob = LinearDecay()
    T = prob.t1 - prob.t0

    n = 6
    y0 = np.array([[1.0], [2.0], [0.5], [3.0], [1.0], [0.2]], dtype=np.float64)
    k = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8], dtype=np.float64)
    params = k[:, None]
    assert y0.shape == (n, 1) and params.shape == (n, 1)

    # --- 1) forward ensemble solve, auto-routed, CPU only ---------------------------
    res = gradsolve.solve(prob, y0, params, engine="auto", device="cpu",
                        rtol=1e-10, atol=1e-13)
    assert res.y_final.shape == (n, 1), res.y_final.shape
    assert bool(np.all(np.isfinite(res.y_final)))

    analytic = y0 * np.exp(-k[:, None] * T)
    fwd_rel_err = float(np.max(np.abs(res.y_final - analytic) / np.abs(analytic)))
    assert fwd_rel_err < 1e-9, f"forward vs analytic rel err too large: {fwd_rel_err:.3e}"

    # --- 2) reverse-mode gradient THROUGH the solve ----------------------------------
    f = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cpu")
    loss = lambda p: jnp.sum(f(p) ** 2)
    g = np.asarray(jax.grad(loss)(jnp.asarray(params)))
    assert g.shape == params.shape
    assert bool(np.all(np.isfinite(g)))

    # d(loss)/dk_i = 2*y_final_i * d(y_final_i)/dk_i, d(y_final_i)/dk_i = -T*y_final_i
    # (trajectories are independent -> no cross terms) => -2*T*y_final_i**2.
    analytic_grad = -2.0 * T * analytic ** 2
    grad_rel_err = float(np.max(np.abs(g - analytic_grad) / np.abs(analytic_grad)))
    assert grad_rel_err < 1e-4, f"grad vs analytic rel err too large: {grad_rel_err:.3e}"

    # Central finite-difference cross-check on one trajectory -- independent of the
    # hand-derived analytic-gradient algebra above (catches an algebra mistake there).
    i, eps = 2, 1e-5
    pp = params.copy()
    pp[i, 0] += eps
    lp = float(loss(jnp.asarray(pp)))
    pp[i, 0] -= 2 * eps
    lm = float(loss(jnp.asarray(pp)))
    fd = (lp - lm) / (2 * eps)
    fd_rel_err = abs(float(g[i, 0]) - fd) / abs(fd)
    assert fd_rel_err < 1e-6, f"grad vs central-FD rel err too large: {fd_rel_err:.3e}"

    # --- 3) the block is actually in effect (not a vacuous pass) ---------------------
    try:
        import examples  # noqa: F401
        raise SystemExit("FAIL: examples import should have been blocked but succeeded")
    except ImportError:
        pass

    print("RESULT_JSON:" + json.dumps({
        "solver": res.solver,
        "fwd_rel_err": fwd_rel_err,
        "grad_rel_err": grad_rel_err,
        "fd_rel_err": fd_rel_err,
    }))
    print("STANDALONE_IMPORT_OK")


main()
'''


def test_gradsolve_standalone_blocks_examples():
    """Fresh subprocess, ``examples`` blocked in sys.modules: gradsolve still solves
    + differentiates an inline (non-registered) Problem, matching a closed-form solution."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"child subprocess failed (returncode={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "STANDALONE_IMPORT_OK" in proc.stdout, proc.stdout

    result_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON:")
    )
    payload = json.loads(result_line[len("RESULT_JSON:"):])

    # Re-check the numeric claims in this process too (don't just trust the child's
    # own asserts) -- tight tolerances, with a safety margin above the typical values
    # (forward ~2.7e-11 at rtol=1e-10 on diffrax and tsit5_replay alike, grad-vs-analytic
    # ~6.1e-7, grad-vs-FD ~2.3e-10).
    assert payload["fwd_rel_err"] < 1e-9
    assert payload["grad_rel_err"] < 1e-4
    assert payload["fd_rel_err"] < 1e-6

    # Routing sanity: dim=1 nonstiff forward-only auto-routes to cuda_tsit5/warp_ode, but neither
    # has a registered field for this problem name, so solve() falls back to the general forward
    # engine (diffrax when installed, else the tsit5 record-and-replay) -- the standalone-user path.
    assert payload["solver"] == _FWD_NONSTIFF, payload
