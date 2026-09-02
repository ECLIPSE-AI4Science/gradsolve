"""Example 06: stiff differential equations -- what ``is_stiff`` changes, and gradsolve's two stiff engines.

Stiffness is the one flag on the Problem contract (see 05_bring_your_own_ode.py) that
actually changes which family of engine gradsolve reaches for: nonstiff problems route to
explicit Runge-Kutta engines (Tsit5), stiff problems route to an L-stable linearly-implicit
engine instead (explicit methods either need a microscopic step or blow up on a stiff RHS).

This example solves the textbook Robertson chemical-kinetics system (three species, rate
constants spanning 1 : 1e4 : 1e7 -- a standard stiff test problem) twice, with the
identical physics, to show both stiff paths:

  Part A -- ``name="my_stiff_kinetics"`` (unregistered): gradsolve falls back to its general
    stiff engines, which work for any stiff ``f_jax`` with no per-problem registration --
    diffrax (adaptive Kvaerno5) for a forward-only solve when that optional extra is
    installed, and ``rodas5p_replay`` (an adaptive Rodas5P recorded once, then replayed on
    a frozen mesh, dense per-step Jacobian via ``jax.jacfwd``) for the gradient and whenever
    diffrax is absent. These are the engines every bring-your-own stiff model gets by
    default.

  Part B -- ``name="robertson"`` (matches gradsolve's own registered fused field, in
    ``gradsolve/warp/_warp_rosenbrock.py``): the same physics now auto-routes to
    ``warp_rosenbrock``, a fused adaptive Rosenbrock23 kernel (Shampine & Reichelt 1997)
    whose reverse pass is again a record-and-replay adjoint.

Both paths give the same physical answer (species mass is conserved: y0+y1+y2 == 1) and
both are reverse-mode differentiable; only the engine differs. Choosing between them is
what gradsolve/dispatch.py automates.

Run with:
    python examples/06_stiff.py

See 09_gpu.py for the forward solve and the gradient on a GPU.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import gradsolve

# Classic Robertson rate constants (k1, k2, k3) and the standard [1, 0, 0] initial
# condition, integrated to the textbook horizon t1=1e4.
_K = np.array([0.04, 3.0e7, 1.0e4])
_T0, _T1 = 0.0, 1.0e4


def _robertson_rhs(y, p):
    y0, y1, y2 = y[..., 0], y[..., 1], y[..., 2]
    k1, k2, k3 = p[..., 0], p[..., 1], p[..., 2]
    d0 = -k1 * y0 + k3 * y1 * y2
    d1 = k1 * y0 - k3 * y1 * y2 - k2 * y1 * y1
    d2 = k2 * y1 * y1
    return jnp.stack([d0, d1, d2], axis=-1)


class MyStiffKinetics:
    """Robertson kinetics under an unregistered name -- the general bring-your-own path."""
    name = "my_stiff_kinetics"
    dim = 3
    t0 = _T0
    t1 = _T1
    is_stiff = True  # <- the flag that steers routing towards the L-stable engines

    def f_jax(self, t, y, p):
        return _robertson_rhs(y, p)


class Robertson:
    """The identical physics, named to match gradsolve's registered fused stiff field."""
    name = "robertson"
    dim = 3
    t0 = _T0
    t1 = _T1
    is_stiff = True

    def f_jax(self, t, y, p):
        return _robertson_rhs(y, p)


def _run(prob, label):
    n = 4
    y0 = np.zeros((n, 3))
    y0[:, 0] = 1.0
    params = np.tile(_K, (n, 1))

    res = gradsolve.solve(prob, y0, params, engine="auto", device="cpu")
    mass = res.y_final.sum(axis=-1)  # conserved quantity: y0+y1+y2 == 1 for all time
    print(f"[{label}] forward engine={res.solver}  "
          f"mass_err={np.max(np.abs(mass - 1.0)):.2e}  "
          f"accepted_steps[0]={int(res.accepted_steps[0]) if res.accepted_steps.size else 'n/a'}")
    assert bool(np.all(np.isfinite(res.y_final))), f"[{label}] y_final has non-finite values"
    assert np.max(np.abs(mass - 1.0)) < 1e-4, f"[{label}] mass conservation violated"

    closure = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cpu")
    params_j = jnp.asarray(params)
    loss = lambda q: jnp.sum(closure(q) ** 2)
    g = jax.grad(loss)(params_j)
    assert g.shape == params_j.shape
    assert bool(jnp.all(jnp.isfinite(g))), f"[{label}] gradient has non-finite values"

    # Finite-difference spot-check on trajectory 0's k1.
    i, j, eps = 0, 0, 1e-6
    pp = np.asarray(params, dtype=np.float64).copy()
    pp[i, j] += eps
    lp = float(loss(jnp.asarray(pp)))
    pp[i, j] -= 2 * eps
    lm = float(loss(jnp.asarray(pp)))
    fd = (lp - lm) / (2 * eps)
    rel = abs(float(g[i, j]) - fd) / (abs(fd) + 1e-12)
    print(f"[{label}] reverse: grad[0,0]={float(g[i, j]):.6g}  finite-diff={fd:.6g}  "
          f"rel_err={rel:.2e}")
    assert rel < 5e-2, f"[{label}] gradient disagrees with FD (rel {rel:.2e})"
    return res.solver


def main():
    engine_a = _run(MyStiffKinetics(), "unregistered")
    engine_b = _run(Robertson(), "registered")

    import importlib.util
    general = "diffrax" if importlib.util.find_spec("diffrax") is not None else "rodas5p_replay"
    assert engine_a == general, f"expected the general forward engine {general!r}, got {engine_a!r}"
    assert engine_b == "warp_rosenbrock", f"expected the fused stiff kernel, got {engine_b!r}"
    print(f"\nsame physics, two engines: {engine_a!r} (general engine) "
          f"vs {engine_b!r} (fused, registered field)")
    print("OK — stiff forward + reverse-mode gradient correct on both paths")


if __name__ == "__main__":
    main()
