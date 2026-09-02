"""Regression: the ``Problem.f_jax`` contract is per-trajectory.

gradsolve vmaps its solvers over the ensemble, so a Problem's ``f_jax(t, y, params)`` is
called with ``y`` of shape ``(dim,)`` and ``params`` of shape ``(P,)`` — not the batched
``(n, dim)`` / ``(n, P)`` the old docstrings implied. This test pins that contract by
recording the shapes ``f_jax`` actually observes when driven through ``gradsolve.solve`` on
a small ensemble, and asserting they are the per-trajectory shapes.

The recording is a Python side effect inside ``f_jax``; under gradsolve's ``jax`` trace it
fires at trace time with the per-trajectory abstract values, whose ``.shape`` is exactly
the shape the RHS is contractually handed. A non-registered problem ``name`` is used on
purpose so routing lands on a general-RHS engine (diffrax, or the tsit5 record-and-replay)
that calls ``problem.f_jax`` per trajectory (a fused/registered-field kernel would bypass
``f_jax`` entirely).

Imports only ``gradsolve`` (+ numpy/jax/pytest).
"""
from __future__ import annotations

import numpy as np
import pytest

import gradsolve

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


class _ShapeRecordingProblem:
    """dim=3, P=2 linear decay. ``f_jax`` records the shapes it is called with.

    ``name`` deliberately matches no registered Warp/CUDA field, so ``solve()`` routes to a
    general-RHS engine (diffrax, or the tsit5 record-and-replay) which calls this ``f_jax``
    per trajectory — the path whose shape contract we are pinning.
    """

    name = "shape_probe_decay"
    dim = 3
    t0 = 0.0
    t1 = 0.5
    is_stiff = False

    def __init__(self):
        # (y.shape, params.shape) tuples observed at each f_jax invocation (trace time).
        self.seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def f_jax(self, t, y, params):
        self.seen.append((tuple(y.shape), tuple(params.shape)))
        k = params[0]  # per-trajectory scalar drawn from the (P,) param vector
        return -k * y  # shape (dim,)


_N = 4


def _ensemble():
    y0 = np.array(
        [[2.0, 1.0, 0.5], [1.0, 0.5, 2.0], [0.5, 2.0, 1.0], [3.0, 1.5, 0.25]],
        dtype=np.float64,
    )
    params = np.array(
        [[0.7, 0.1], [1.3, 0.2], [2.0, 0.3], [0.4, 0.4]], dtype=np.float64
    )
    assert y0.shape == (_N, 3) and params.shape == (_N, 2)
    return y0, params


def test_fjax_called_per_trajectory_on_solve():
    prob = _ShapeRecordingProblem()
    y0, params = _ensemble()

    res = gradsolve.solve(prob, y0, params, engine="auto", device="cpu")

    # Sanity: the solve actually ran and produced a finite ensemble of the right shape.
    assert res.y_final.shape == (_N, prob.dim)
    assert bool(np.all(np.isfinite(res.y_final)))

    # The RHS was invoked (trace time) and EVERY invocation saw the per-trajectory shapes:
    # y is (dim,), params is (P,) — never the batched (n, dim) / (n, P).
    assert prob.seen, "f_jax was never called — cannot verify the shape contract"
    for y_shape, p_shape in prob.seen:
        assert y_shape == (prob.dim,), (
            f"f_jax got y.shape={y_shape}, expected per-trajectory ({prob.dim},)"
        )
        assert p_shape == (params.shape[1],), (
            f"f_jax got params.shape={p_shape}, expected per-trajectory "
            f"({params.shape[1]},)"
        )


def test_fjax_called_per_trajectory_on_grad_closure():
    """Same contract on the reverse path: the gradient closure differentiates through an
    f_jax that is still handed per-trajectory ``(dim,)`` / ``(P,)`` shapes."""
    prob = _ShapeRecordingProblem()
    y0, params = _ensemble()

    closure = gradsolve.grad_closure(prob, y0, params, engine="auto", device="cpu")
    loss = lambda p: jnp.sum(closure(p) ** 2)  # noqa: E731
    g = np.asarray(jax.grad(loss)(jnp.asarray(params)))

    assert g.shape == params.shape
    assert bool(np.all(np.isfinite(g)))

    assert prob.seen, "f_jax was never called through grad_closure"
    for y_shape, p_shape in prob.seen:
        assert y_shape == (prob.dim,), y_shape
        assert p_shape == (params.shape[1],), p_shape
