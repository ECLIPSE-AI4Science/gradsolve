"""Single Rodas5P step + tableau — the method core of the stiff high-order replay lane.

Implements the Steinebach (2023) Rodas5P Rosenbrock-Wanner order-5(4) pair. The tableau is
stored in a/C form and converted at import to the alpha/Gamma form the driver uses.
Mirrors gradsolve/solvers/rosenbrock23_step.py (in-step jax.jacfwd Jacobian, jnp.linalg.solve,
dt==0 identity padding) and the two-tier shape. Recorder is rodas5p_replay.record_rodas5p
(jitted host loop), not record_adaptive (RODAS5P_METHOD.needs_jacobian=True enforces this).

Reference: G. Steinebach, BIT Numer. Math. 63:27 (2023), doi:10.1007/s10543-023-00967-x.
Stored-form convention: a = alpha*Gamma^-1, C = diag(Gamma^-1) - Gamma^-1, b = m^T Gamma^-1,
btilde = (m-mhat)^T Gamma^-1.
Verified by the shape/stage-count, node (alpha-rowsum==c) + Gamma-rowsum==d checksums, the ROW
stability-function order series (order 5 / 4), and observed order 5 on nonlinear + non-autonomous
problems (test_rodas5p_step.py) + Radau parity (test_rodas5p_replay.py).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gradsolve.solvers.method import Method, StepResult

# ============================ TABLEAU (stored a/C form) ==============================
# Rodas5P, 8 stages (Steinebach 2023). The conversion below reproduces the published c/d row
# sums to ~1e-14, the ROW step matches a Radau reference on a nonlinear stiff problem to
# ~8e-15, and observed order is 5.
# The C matrix is stored 8x7 in the source (strictly lower, trailing all-zero column dropped);
# it is padded to 8x8. b = (last row of A)[:7] + [1]; btilde = e8 (stiffly-accurate structure).
_S = 8
RODAS5P_GAMMA_DIAG = 0.21193756319429014
_A_STORED = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [3.0, 0, 0, 0, 0, 0, 0, 0],
    [2.849394379747939, 0.45842242204463923, 0, 0, 0, 0, 0, 0],
    [-6.954028509809101, 2.489845061869568, -10.358996098473584, 0, 0, 0, 0, 0],
    [2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641, 0, 0, 0, 0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 0, 0, 0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1, 0, 0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1, 1, 0],
], dtype=np.float64)
_C7 = np.array([   # stored 8x7 strictly-lower C (trailing zero column dropped)
    [0, 0, 0, 0, 0, 0, 0],
    [-14.155112264123755, 0, 0, 0, 0, 0, 0],
    [-17.97296035885952, -2.859693295451294, 0, 0, 0, 0, 0],
    [147.12150275711716, -1.41221402718213, 71.68940251302358, 0, 0, 0, 0],
    [165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306, 0, 0, 0],
    [24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125, -0.6610691825249471, 0, 0],
    [30.91273214028599, -3.1208243349937974, 77.79954646070892, 34.28646028294783, -19.097331116725623, -28.087943162872662, 0],
    [37.80277123390563, -3.2571969029072276, 112.26918849496327, 66.9347231244047, -40.06618937091002, -54.66780262877968, -9.48861652309627],
], dtype=np.float64)
_C_STORED = np.zeros((_S, _S))
_C_STORED[:, :7] = _C7
_B_STORED = np.concatenate([_A_STORED[7, :7], [1.0]])          # b = last A row + 1
_BTILDE_STORED = np.eye(_S)[7]                                  # btilde = e8
RODAS5P_C = np.array([0.0, 0.6358126895828704, 0.4095798393397535, 0.9769306725060716,
                      0.4288403609558664, 1.0, 1.0, 1.0], dtype=np.float64)      # abscissae
RODAS5P_D = np.array([0.21193756319429014, -0.42387512638858027, -0.3384627126235924,
                      1.8046452872882734, 2.325825639765069, 0.0, 0.0, 0.0], dtype=np.float64)  # df/dt
# ==================================================================================


def _convert_stored_to_alpha_gamma():
    """a/C (stored) -> alpha/Gamma/m/mhat (driver). Gamma^-1 = (1/gamma)I - C ; Gamma = inv ;
    alpha = a@Gamma ; m = b@Gamma ; mhat = m - btilde@Gamma. Run once at import."""
    g = RODAS5P_GAMMA_DIAG
    ginv = np.eye(_S) / g - _C_STORED
    Gam = np.linalg.inv(ginv)                 # lower triangular, diagonal gamma
    Alpha = _A_STORED @ Gam                   # strictly lower
    M = _B_STORED @ Gam
    Mhat = M - _BTILDE_STORED @ Gam
    return Alpha, Gam, M, Mhat


RODAS5P_ALPHA, RODAS5P_GAMMA, RODAS5P_M, RODAS5P_MHAT = _convert_stored_to_alpha_gamma()
_N_STAGES = _S
_GAMMA_ROWSUM = RODAS5P_GAMMA.sum(axis=1)     # gamma_i (incl. the diagonal gamma) == RODAS5P_D

# ==================== DENSE-OUTPUT TABLEAU (stored H form) ============================
# Rodas5P's continuous-extension weights H form a 3x8 array in the same stored convention
# as A/C. Three dense vectors D_r = sum_i H[r,i] * u_i are formed from the a/C-form stage
# vectors u_i and interpolated as
#     y(Theta) = (1-Theta)*y0 + Theta*(y1 + (1-Theta)*(D1 + Theta*(D2 + Theta*D3)))
# This driver produces alpha/Gamma increments k_i with u = Gamma @ k (verified against a
# numpy implementation of the stored-form driver to 1.5e-16 absolute / 4.1e-15 relative),
# so the conversion for these increments is hd = H @ Gamma.
_H_STORED = np.array([
    [25.948786856663858, -2.5579724845846235, 10.433815404888879, -2.3679251022685204,
     0.524948541321073, 1.1241088310450404, 0.4272876194431874, -0.17202221070155493],
    [-9.91568850695171, -0.9689944594115154, 3.0438037242978453, -24.495224566215796,
     20.176138334709044, 15.98066361424651, -6.789040303419874, -6.710236069923372],
    [11.419903575922262, 2.8879645146136994, 72.92137995996029, 80.12511834622643,
     -52.072871366152654, -59.78993625266729, -0.15582684282751913, 4.883087185713722],
], dtype=np.float64)
# ======================================================================================

#: Dense weights in this driver's k_i convention, (3, 8). Row sums are zero by the extension's own
#: consistency condition (see test_dense_consistency_invariants).
RODAS5P_HD = _H_STORED @ RODAS5P_GAMMA


def rodas5p_dense_weights(theta):
    """b_i(theta) for y(t0 + theta*dt) = y0 + sum_i b_i(theta) * k_i.

    Quartic in theta. The theta*(1-theta) factorization is required, not cosmetic: it makes
    ``b_i(1) == m_i`` hold bitwise. Order 4 continuous extension on the order-5 step
    (measured).

    That bitwise weight identity does not make a save landing on a mesh node bitwise equal to
    the step output: the streaming lane reaches it through ``theta = (ts - t)/dt``, which
    rounds to ``1 +/- 1-2 ulp``, so a node save agrees to a few ulp (measured <= 4 eps
    relative), not exactly. Only an exactly-supplied ``th == 1.0`` returns ``y_r`` bitwise.
    """
    th = jnp.asarray(theta)
    inner = RODAS5P_HD[0] + th * (RODAS5P_HD[1] + th * RODAS5P_HD[2])
    return th * RODAS5P_M + th * (1.0 - th) * inner


def _rodas5p_stages(f, t, y, dt, p):
    """Evaluate the s ROW stages; AD Jacobian df/dy + df/dt. Returns list k[0..s-1]."""
    D = y.shape[-1]
    eye = jnp.eye(D, dtype=y.dtype)
    g = jnp.asarray(RODAS5P_GAMMA_DIAG, dtype=y.dtype)
    J = jax.jacfwd(f, argnums=1)(t, y, p)
    f_t = jax.jacfwd(f, argnums=0)(t, y, p)
    # dt==0 padding must be an identity in value AND gradient. Every rhs term carries a dt
    # factor, but that is only an ALGEBRAIC identity: in IEEE arithmetic 0 * inf = NaN, so a
    # non-finite J or f_t at the padded node poisons the row. Unlike the explicit lanes, a
    # successful record does not rule that out here — the last real step evaluates f_t at its
    # LEFT endpoint, never at t1, where a padded row sits. Masking is bitwise-neutral on any
    # real step (dt != 0).
    nz = dt != 0
    J = jnp.where(nz, J, jnp.zeros_like(J))
    f_t = jnp.where(nz, f_t, jnp.zeros_like(f_t))
    W = eye - dt * g * J
    k = [None] * _N_STAGES
    for i in range(_N_STAGES):
        y_arg = y
        accum = jnp.zeros_like(y)
        for j in range(i):
            y_arg = y_arg + float(RODAS5P_ALPHA[i, j]) * k[j]
            accum = accum + float(RODAS5P_GAMMA[i, j]) * k[j]
        t_arg = t + float(RODAS5P_C[i]) * dt
        rhs = dt * f(t_arg, y_arg, p) + dt * (J @ accum) + (dt * dt * float(_GAMMA_ROWSUM[i])) * f_t
        k[i] = jnp.linalg.solve(W, rhs)
    return k


def rodas5p_advance(f, t, y, dt, p):
    """One Rodas5P step (value-only). dt==0 is an exact identity in value and gradient."""
    k = _rodas5p_stages(f, t, y, dt, p)
    out = y
    for i in range(_N_STAGES):
        out = out + float(RODAS5P_M[i]) * k[i]
    return out


def rodas5p_trial_step(f, t, y, dt, p, k1=None):
    """One Rodas5P trial step -> StepResult(y_next, sum((m_i-mhat_i)k_i), fsal=None). y_next
    byte-identical to rodas5p_advance. k1 ignored (Rosenbrock is not FSAL)."""
    del k1
    k = _rodas5p_stages(f, t, y, dt, p)
    y_next = y
    y_err = jnp.zeros_like(y)
    for i in range(_N_STAGES):
        y_next = y_next + float(RODAS5P_M[i]) * k[i]
        y_err = y_err + float(RODAS5P_M[i] - RODAS5P_MHAT[i]) * k[i]
    return StepResult(y_next=y_next, y_err=y_err, fsal=None)


def rodas5p_dense_eval(coeffs, theta):
    """Evaluate the continuous extension from the 5-vector bundle produced by
    :func:`rodas5p_stages_and_dense`, in Horner form:

        y(t0 + theta*dt) = (1-th)*y_l + th*(y_r + (1-th)*(D1 + th*(D2 + th*D3)))

    Algebraically identical to ``y_l + sum_i b_i(theta) k_i`` with the
    :func:`rodas5p_dense_weights` weights -- the two are welded by a test -- but it costs ~5
    vector operations per requested time instead of 8, and the three D_r contractions over
    the 8 stages are paid once per step rather than once per requested time. That is what
    makes the streaming saveat lane ~5*k*dim per step.

    ``theta`` broadcasts against the (dim,) bundle vectors, so a (k, 1) column evaluates all
    k requested times of a step at once.

    Endpoints are exact, not merely accurate: th == 1.0 makes (1-th) exactly 0.0 and returns
    ``y_r`` -- the step output itself; th == 0.0 likewise returns ``y_l``. Note this holds for
    an exactly-supplied endpoint: the streaming lane computes ``theta = (ts - t)/dt``, which
    at a mesh node rounds to ``1 +/- 1-2 ulp``, so a node save is continuous to a few ulp
    (measured <= 4 eps relative) rather than bitwise.
    """
    y_l, y_r, d1, d2, d3 = coeffs
    th = jnp.asarray(theta)
    om = 1.0 - th
    return om * y_l + th * (y_r + om * (d1 + th * (d2 + th * d3)))


def rodas5p_stages_and_dense(f, t, y, dt, p):
    """One Rodas5P step plus its continuous-extension coefficients.

    Returns ``(y_next, coeffs)`` with ``coeffs = (y_left, y_right, D1, D2, D3)`` -- the
    5-vector Horner bundle, where ``D_r = sum_i hd[r, i] * k_i`` are the dense vectors in
    this driver's alpha/Gamma increment convention. Feed the bundle to :func:`rodas5p_dense_eval`.

    ``y_next`` is byte-identical to :func:`rodas5p_advance` (same weights, same accumulation
    order), so opting into the dense lane cannot perturb the mesh itself.
    """
    k = _rodas5p_stages(f, t, y, dt, p)
    y_next = y
    for i in range(_N_STAGES):
        y_next = y_next + float(RODAS5P_M[i]) * k[i]
    dense = []
    for r in range(3):
        acc = jnp.zeros_like(y)
        for i in range(_N_STAGES):
            acc = acc + float(RODAS5P_HD[r, i]) * k[i]
        dense.append(acc)
    return y_next, (y, y_next, dense[0], dense[1], dense[2])


#: Rodas5P on the two-tier contract. needs_jacobian=True -> records via record_rodas5p.
RODAS5P_METHOD = Method(
    name="rodas5p", order=5, error_order=4, fsal=False,
    advance=rodas5p_advance, trial_step=rodas5p_trial_step,
    beta1=1.0 / 5.0, beta2=0.0, needs_jacobian=True,
)
