"""Regression: the Warp-backed modules import safely when NVIDIA Warp is absent.

``gradsolve`` must ``pip install`` and run without the optional ``[warp]`` extra. That means
every Warp-backed module has to survive a bare ``import`` in a warp-less env: the ``import
warp`` at module load is guarded behind a shared availability flag, the flag reads False, a
stub/blocked ``warp`` must not make any fused lane report itself available, and ``gradsolve``'s
engine dispatch must fall back to a general engine (diffrax, or the stiff record-and-replay lane)
instead of crashing.

A fresh subprocess blocks ``warp`` from the import system (``sys.modules["warp"] = None``, so
``import warp`` raises ImportError) before importing gradsolve, then checks:
  1. ``import gradsolve.warp.fused_rosenbrock`` succeeds (no unguarded top-level ``import warp``).
  2. its availability flag(s) read False (the fused lane reports itself unavailable).
  3. ``gradsolve.warp.fused_backward`` reports its fused lane unavailable (a blocked warp must
     not let the stub masquerade as the real thing).
  4. routing a stiff problem through ``gradsolve.solve`` falls back cleanly (no crash).

Imports only ``gradsolve`` (+ stdlib/numpy/pytest).
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
# Block NVIDIA Warp BEFORE importing gradsolve: any "import warp" now raises ImportError, so
# an unguarded top-level warp import in a gradsolve.warp.* module would surface immediately.
sys.modules["warp"] = None

import json

import numpy as np

# (1) The Warp-backed fused-stiff module must import even with warp blocked.
import gradsolve  # noqa: E402
import gradsolve.warp.fused_rosenbrock as fr  # noqa: E402
import gradsolve.warp.fused_backward as fb    # noqa: E402

# (2) Its availability flag(s) must read False. Discover any module-level *_AVAILABLE flag
# (the sibling convention is `_WARP_AVAILABLE`, mirrored across the warp lanes) rather than
# hard-coding one name, and require at least one to exist and all of them to be False.
avail_flags = {k: v for k, v in vars(fr).items() if k.endswith("_AVAILABLE")}
assert avail_flags, "fused_rosenbrock exposes no *_AVAILABLE flag to gate the warp import"
assert all(v is False for v in avail_flags.values()), (
    f"fused_rosenbrock availability flag(s) not False with warp blocked: {avail_flags}")

# (3) A blocked/stub warp must NOT let fused_backward report the fused lane available.
assert fb._FUSED_AVAILABLE is False, (
    f"fused_backward._FUSED_AVAILABLE should be False with warp blocked, "
    f"got {fb._FUSED_AVAILABLE!r}")


class StiffDecay:
    """dim=2 diagonal stiff decay; name matches no registered field, so warp engines are
    unsupported and gradsolve must fall back to the general stiff engine."""

    name = "toy_stiff_decay"
    dim = 2
    t0 = 0.0
    t1 = 0.5
    is_stiff = True

    def f_jax(self, t, y, p):
        return -p * y


# (4) Routing a stiff problem must fall back cleanly (warp unavailable -> diffrax / rodas5p_replay).
prob = StiffDecay()
n = 4
y0 = np.array([[2.0, 1.0], [1.0, 0.5], [0.5, 2.0], [3.0, 1.5]], dtype=np.float64)
k = np.array([[40.0, 30.0], [25.0, 50.0], [60.0, 20.0], [35.0, 45.0]], dtype=np.float64)
res = gradsolve.solve(prob, y0, k, engine="auto", device="cpu")
assert res.y_final.shape == (n, prob.dim), res.y_final.shape
assert bool(np.all(np.isfinite(res.y_final)))

print("RESULT_JSON:" + json.dumps({
    "avail_flags": avail_flags,
    "fused_available": fb._FUSED_AVAILABLE,
    "solver": res.solver,
}))
print("WARP_IMPORT_GUARD_OK")
'''


def test_warp_backed_modules_import_and_route_without_warp():
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
    assert "WARP_IMPORT_GUARD_OK" in proc.stdout, proc.stdout

    result_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("RESULT_JSON:")
    )
    payload = json.loads(result_line[len("RESULT_JSON:"):])

    assert payload["avail_flags"], payload
    assert all(v is False for v in payload["avail_flags"].values()), payload
    assert payload["fused_available"] is False, payload
    # Stiff, non-registered problem with warp blocked -> the general forward engine: diffrax when
    # installed, else the stiff record-and-replay lane (never the order-1 fixed scan).
    assert payload["solver"] == _FWD_STIFF, payload
