"""Adaptive Rosenbrock23 (ode23s) reference integrator: the semantic specification for the
stiff Warp kernel.

Linearly-implicit 2(3) Rosenbrock pair (Shampine & Reichelt 1997, "The MATLAB ODE
Suite", ode23s), specialized to autonomous systems (the registered stiff problems are autonomous,
so the d(f)/dt term that ode23s carries for non-autonomous f vanishes — it is dropped
here and in the kernel, identically).

Method core (per step h at state (t, y), with J = df/dy(y) — the analytic Jacobian):

    d   = 1 / (2 + sqrt(2))
    e32 = 6 + sqrt(2)
    W   = I - h*d*J                                  # DxD, the (frozen) Newton matrix
    F0  = f(t, y);              k1 = solve(W, F0)
    F1  = f(t, y + 0.5*h*k1);   k2 = solve(W, F1 - k1) + k1
    ynew = y + h*k2                                  # 2nd-order propagated solution
    F2  = f(t, ynew);           k3 = solve(W, F2 - e32*(k2 - F1) - 2*(k1 - F0))
    err_vec = (h/6) * (k1 - 2*k2 + k3)               # 3rd-order embedded estimate

Adaptive I-controller (order-2 propagated solution => error-exponent -1/3, not Tsit5's
-1/5; the propagated order p=2 gives the standard exponent -1/(p+1) = -1/3):

    sc  = atol + rtol*max(|y_i|, |ynew_i|)           # WRMS scale, per component
    err = sqrt(mean_i (err_vec_i / sc_i)^2)
    accept iff err <= 1.0
    on accept:  t += h, y = ynew, record h
    dt <- h * clamp(0.9 * max(err, 1e-10)**(-1.0/3.0), 0.2, 5.0)   # after both accept+reject
    last step clamped: h = min(dt, t1 - t); finish when t >= t1

This oracle defines the constants and the arithmetic; the Warp kernel matches it
step-for-step (same accept/reject sequence, same accepted-dt sequence). The matrix
W-solves use ``numpy.linalg.solve`` (an LU factorization), so kernel-vs-oracle value
parity is floored at the LU/FMA level (~1e-8 documented in the test), while the
accept/reject control sequence is the strict claim.

Implemented from the published method (Shampine & Reichelt 1997) with the I-controller of
Hairer & Wanner; independent of any other solver library.
"""
import numpy as np

# Controller constants (shared spec for the kernel; -1/3 is the order-2 error exponent).
SAFETY = 0.9
ERR_FLOOR = 1e-10
FAC_MIN = 0.2
FAC_MAX = 5.0
ORDER_EXP = -1.0 / 3.0  # propagated order p=2 -> -1/(p+1) = -1/3 (vs Tsit5's -0.2)

# Rosenbrock23 coefficients.
_D = 1.0 / (2.0 + np.sqrt(2.0))
_E32 = 6.0 + np.sqrt(2.0)


def robertson_jac(y, p):
    """Analytic Jacobian df/dy for the Robertson field (3x3), params p=[k1,k2,k3].

    f0 = -k1*y0 + k3*y1*y2
    f1 =  k1*y0 - k3*y1*y2 - k2*y1*y1
    f2 =  k2*y1*y1
    """
    k1, k2, k3 = float(p[0]), float(p[1]), float(p[2])
    # The Jacobian does not depend on y0 (only y1, y2 appear in df/dy).
    y1, y2 = float(y[1]), float(y[2])
    return np.array([
        [-k1,            k3 * y2,                  k3 * y1],
        [k1,  -k3 * y2 - 2.0 * k2 * y1,           -k3 * y1],
        [0.0,           2.0 * k2 * y1,                 0.0],
    ], dtype=np.float64)


def hires_jac(y, p):
    """Analytic Jacobian df/dy for the HIRES-8 field (8x8); ``p`` is ignored (HIRES is
    autonomous, no free params). Equations from Hairer & Wanner, Solving ODEs II, Section IV.10
    (0-indexed):

        d0 = -1.71*y0 + 0.43*y1 + 8.32*y2 + 0.0007
        d1 =  1.71*y0 - 8.75*y1
        d2 = -10.03*y2 + 0.43*y3 + 0.035*y4
        d3 =  8.32*y1 + 1.71*y2 - 1.12*y3
        d4 = -1.745*y4 + 0.43*y5 + 0.43*y6
        d5 = -280*y5*y7 + 0.69*y3 + 1.71*y4 - 0.43*y5 + 0.69*y6
        d6 =  280*y5*y7 - 1.81*y6
        d7 = -280*y5*y7 + 1.81*y6

    Only y5 and y7 enter df/dy (the bimolecular ``280*y5*y7`` coupling); all other
    partials are constants. Agrees with ``jax.jacfwd`` of the HIRES ``f_jax`` (argnums=1) to
    0.0.
    """
    del p  # HIRES is autonomous; the Jacobian carries no params
    y5 = float(y[5])
    y7 = float(y[7])
    J = np.zeros((8, 8), dtype=np.float64)
    # Row 0
    J[0, 0] = -1.71
    J[0, 1] = 0.43
    J[0, 2] = 8.32
    # Row 1
    J[1, 0] = 1.71
    J[1, 1] = -8.75
    # Row 2
    J[2, 2] = -10.03
    J[2, 3] = 0.43
    J[2, 4] = 0.035
    # Row 3
    J[3, 1] = 8.32
    J[3, 2] = 1.71
    J[3, 3] = -1.12
    # Row 4
    J[4, 4] = -1.745
    J[4, 5] = 0.43
    J[4, 6] = 0.43
    # Row 5
    J[5, 3] = 0.69
    J[5, 4] = 1.71
    J[5, 5] = -280.0 * y7 - 0.43
    J[5, 6] = 0.69
    J[5, 7] = -280.0 * y5
    # Row 6
    J[6, 5] = 280.0 * y7
    J[6, 6] = -1.81
    J[6, 7] = 280.0 * y5
    # Row 7
    J[7, 5] = -280.0 * y7
    J[7, 6] = 1.81
    J[7, 7] = -280.0 * y5
    return J


def constant_jac(A):
    """Build a ``jac(y, p) -> A`` for a linear field f(y) = A y (constant Jacobian)."""
    A = np.asarray(A, dtype=np.float64)

    def jac(y, p):
        return A

    return jac


def rosenbrock23_adaptive(f, jac, y0, t0, t1, rtol, atol, dt0, max_steps):
    """Adaptive Rosenbrock23 integration of dy/dt = f(t, y) with Jacobian ``jac(y, p)``.

    Mirrors ``gradsolve.solvers.tsit5_replay.tsit5_adaptive``'s signature
    and return shape exactly:

        f:     callable f(t, y) -> dy/dt (numpy), the autonomous RHS (closes over p).
        jac:   callable jac(y, p) -> DxD analytic Jacobian. (p threaded for the
               registered helpers; ``f`` already closes over the same p.)
        returns ``(y_final, dts, n_rej, status)`` with ``dts`` the accepted-step
        sequence (numpy), ``status == 0`` iff t reached t1 within max_steps else 1.

    ``jac`` is called as ``jac(y, None)`` here — the registered helpers
    (``robertson_jac`` via a partial, ``constant_jac``) bind p before this call, so the
    oracle stays p-agnostic and matches the kernel (which closes over the analytic J).
    """
    y = np.array(y0, dtype=np.float64)
    t = float(t0)
    dt = float(dt0)
    D = y.shape[0]
    eye = np.eye(D, dtype=np.float64)
    dts, n_rej = [], 0
    for _ in range(max_steps):
        if t >= t1:
            break
        h = min(dt, t1 - t)
        J = jac(y, None)
        W = eye - h * _D * J
        F0 = f(t, y)
        k1 = np.linalg.solve(W, F0)
        F1 = f(t, y + 0.5 * h * k1)
        k2 = np.linalg.solve(W, F1 - k1) + k1
        ynew = y + h * k2
        F2 = f(t, ynew)
        k3 = np.linalg.solve(W, F2 - _E32 * (k2 - F1) - 2.0 * (k1 - F0))
        err_vec = (h / 6.0) * (k1 - 2.0 * k2 + k3)
        sc = atol + rtol * np.maximum(np.abs(y), np.abs(ynew))
        err = float(np.sqrt(np.mean((err_vec / sc) ** 2)))
        if err <= 1.0:
            t, y = t + h, ynew
            dts.append(h)
        else:
            n_rej += 1
        fac = SAFETY * max(err, ERR_FLOOR) ** ORDER_EXP
        dt = h * min(max(fac, FAC_MIN), FAC_MAX)
    status = 0 if t >= t1 else 1
    return y, np.array(dts), n_rej, status
