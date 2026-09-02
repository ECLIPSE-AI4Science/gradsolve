"""Regression: routing a stiff problem through gradsolve imports no ``examples`` module.

The package must be standalone: a stiff solve routes entirely within ``gradsolve`` (diffrax, or
the general-RHS stiff record-and-replay), never reaching for an ``examples.*`` module. This
test checks it: a fresh subprocess blocks ``examples`` from the import system before importing
gradsolve, routes
an inline stiff Problem through ``gradsolve.solve``, asserts the solve succeeds, and asserts
that no ``examples.*`` module ended up in ``sys.modules`` as a side effect.

A subprocess (not an in-process ``sys.modules`` patch) is used so the block cannot leak into
the rest of the pytest session. The stiff Problem uses a non-registered ``name`` so it routes
to gradsolve's general stiff engine (diffrax when installed, else ``rodas5p_replay``) regardless
of whether Warp is installed.

Imports only ``gradsolve`` (+ stdlib/numpy/jax/pytest).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The forward-only general stiff engine on this interpreter. The child runs the same
#: interpreter, so it sees the same install: diffrax is an optional extra, and without it
#: api._fallback serves the forward request from the stiff record-and-replay lane.
_FWD_STIFF = "diffrax" if importlib.util.find_spec("diffrax") is not None else "rodas5p_replay"

_CHILD_SCRIPT = r'''
import sys
# Block any ``examples`` package before importing gradsolve: any subsequent "import examples"
# (or "import examples.<sub>") raises ImportError instead of resolving off sys.path, so if
# gradsolve's stiff-solve path depended on such a module this child would fail loudly.
sys.modules["examples"] = None

import json

import numpy as np

import gradsolve  # noqa: E402  -- must come after the sys.modules block above


class StiffDecay:
    """dim=2 diagonal decay dy/dt = -k*y, is_stiff=True. Analytic y(t)=y0*exp(-k*(t-t0)).

    ``name`` matches no registered Warp/CUDA field, so the stiff route (choose_engine ->
    warp_rosenbrock; unsupported -> fallback) lands on gradsolve's general stiff engine --
    diffrax when installed, else ``rodas5p_replay`` -- no GPU.
    """

    name = "toy_stiff_decay"
    dim = 2
    t0 = 0.0
    t1 = 0.5
    is_stiff = True

    def f_jax(self, t, y, p):
        return -p * y


def main():
    prob = StiffDecay()
    T = prob.t1 - prob.t0

    n = 4
    y0 = np.array([[2.0, 1.0], [1.0, 0.5], [0.5, 2.0], [3.0, 1.5]], dtype=np.float64)
    k = np.array([[40.0, 30.0], [25.0, 50.0], [60.0, 20.0], [35.0, 45.0]], dtype=np.float64)

    res = gradsolve.solve(prob, y0, k, engine="auto", device="cpu")
    assert res.y_final.shape == (n, prob.dim), res.y_final.shape
    assert bool(np.all(np.isfinite(res.y_final)))

    analytic = y0 * np.exp(-k * T)
    rel_err = float(np.max(np.abs(res.y_final - analytic) / np.abs(analytic)))
    assert rel_err < 5e-2, f"stiff forward vs analytic rel err too large: {rel_err:.3e}"

    # The block is not vacuous: a real "import examples" must still be blocked.
    try:
        import examples  # noqa: F401
        raise SystemExit("FAIL: examples import should have been blocked but succeeded")
    except ImportError:
        pass

    # No examples.* module leaked into sys.modules via the solve. (The "examples" key is
    # our own None sentinel, which is the block itself -- not a real import.)
    leaked = sorted(
        m for m in sys.modules
        if m == "examples" and sys.modules[m] is not None
        or m.startswith("examples.")
    )
    assert not leaked, f"examples module leaked into sys.modules: {leaked}"

    print("RESULT_JSON:" + json.dumps({
        "solver": res.solver,
        "rel_err": rel_err,
        "leaked": leaked,
    }))
    print("NO_EXAMPLES_IMPORT_OK")


main()
'''


def test_stiff_solve_pulls_in_no_examples_module():
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"child subprocess failed (returncode={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "NO_EXAMPLES_IMPORT_OK" in proc.stdout, proc.stdout

    result_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON:")
    )
    payload = json.loads(result_line[len("RESULT_JSON:"):])

    assert payload["leaked"] == [], payload
    assert payload["rel_err"] < 5e-2, payload
    # Stiff, non-registered problem -> the general stiff engine (diffrax or rodas5p_replay).
    assert payload["solver"] == _FWD_STIFF, payload
