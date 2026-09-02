"""Single Vern7 step + tableau — the method core of the nonstiff high-order replay lane.

Implements J. H. Verner's "most efficient" order 7(6) explicit Runge-Kutta pair (10-stage
main method). The lazy seventh-order interpolant is deliberately not
implemented — gradsolve keeps its re-step ``saveat`` (see gradsolve/solvers/dense.py), so no
interpolation coefficients are transcribed. Mirrors gradsolve/solvers/tsit5_step.py's layout:
array-library agnostic pure arithmetic (jit/grad/vmap-safe and valid on numpy), the same
``dt == 0`` identity-step padding contract for the replay adjoint, and the same two-tier
(advance/trial_step) shape as the method descriptor.

Reference:
    J. H. Verner, "Numerically optimal Runge-Kutta pairs with interpolants",
    Numer. Algorithms 53, 383-396 (2010); coefficient tables at https://www.sfu.ca/~jverner/
    Specifically the "more efficient" RK 7(6) pair (April 2007, corrected July 2024),
    floating-point coefficient file RKV76.IIa.Efficient.00001675585.240712.CoeffsOnlyFLOAT.

Verification (tests/test_vern7_step.py): node-consistency sum_j a_ij == c_i; local order-8
convergence (single wrong constant collapses the slope); embedded error order ~7; dt == 0
value/JVP/VJP identity. diffrax has no Vern7 to compare against, so these mechanical checks
are the transcription check.
"""
from __future__ import annotations

from gradsolve.solvers.method import Method, StepResult

# ============================ TRANSCRIBED TABLEAU ====================================
# Verner "most efficient" RK7(6), 10 stages, main method only (interpolant stages c[11]..c[16]
# / bi6 / bi7 from the source file are the lazy interpolant and are intentionally dropped).
# Transcribed verbatim from the cited CoeffsOnlyFLOAT file, remapping Verner's 1-based
# a[i,j]/b/bh notation to 0-based tuples. The node-consistency test (sum of each A-row == the
# c-node) and the order-8 convergence test are the acceptance gates: they FAIL loudly on any
# transcription error. Stage indices are 0-based (stage 0 is at the step start, c_0 == 0.0,
# A row 0 is empty).
#
# VERN7_C  : tuple of 10 c-nodes,      VERN7_C[0] == 0.0
# VERN7_A  : 10 lower-triangular rows; VERN7_A[i] has length i (feeds stage i)
# VERN7_B  : 10 propagated (7th-order) weights
# VERN7_BHAT : 10 embedded (6th-order) weights
# VERN7_E  : 10 error weights, VERN7_E[j] == VERN7_B[j] - VERN7_BHAT[j]  (precompute here)
VERN7_C = (
    0.0,
    0.5000000000000000000000000000000000000000e-2,
    0.1088888888888888888888888888888888888889,
    0.1633333333333333333333333333333333333333,
    0.4555000000000000000000000000000000000000,
    0.6095094489978381317087004421486024949638,
    0.8840000000000000000000000000000000000000,
    0.9250000000000000000000000000000000000000,
    1.0,
    1.0,
)
VERN7_A = (
    (),
    (0.5000000000000000000000000000000000000000e-2,),
    (-1.076790123456790123456790123456790123457,
     1.185679012345679012345679012345679012346),
    (0.4083333333333333333333333333333333333333e-1,
     0.0,
     0.1225000000000000000000000000000000000000),
    (0.6389139236255726780508121615993336109954,
     0.0,
     -2.455672638223656809662640566430653894211,
     2.272258714598084131611828404831320283215),
    (-2.661577375018757131119259297861818119279,
     0.0,
     10.80451388645613769565396655365532838482,
     -8.353914657396199411968048547819291691541,
     0.8204875949566569791420417341743839209619),
    (6.067741434696770992718360183877276714679,
     0.0,
     -24.71127363591108579734203485290746001803,
     20.42751793078889394045773111748346612697,
     -1.906157978816647150624096784352757010879,
     1.006172249242068014790040335899474187268),
    (12.05467007625320299509109452892778311648,
     0.0,
     -49.75478495046898932807257615331444758322,
     41.14288863860467663259698416710157354209,
     -4.461760149974004185641911603484815375051,
     2.042334822239174959821717077708608543738,
     -0.9834843665406107379530801693870224403537e-1),
    (10.13814652288180787641845141981689030769,
     0.0,
     -42.64113603171750214622846006736635730625,
     35.76384003992257007135021178023160054034,
     -4.348022840392907653340370296908245943710,
     2.009862268377035895441943593011827554771,
     0.3487490460338272405953822853053145879140,
     -0.2714390051048312842371587140910297407572),
    (-45.03007203429867712435322405073769635151,
     0.0,
     187.3272437654588840752418206154201997384,
     -154.0288236935018690596728621034510402582,
     18.56465306347536233859492332958439136765,
     -7.141809679295078854925420496823551192821,
     1.308808578161378625114762706007696696508,
     0.0,
     0.0),
)
VERN7_B = (
    0.4715561848627222170431765108838175679569e-1,
    0.0,
    0.0,
    0.2575056429843415189596436101037687580986,
    0.2621665397741262047713863095764527711129,
    0.1521609265673855740323133199165117535523,
    0.4939969170032484246907175893227876844296,
    -0.2943031171403250441557244744092703429139,
    0.8131747232495109999734599440136761892478e-1,
    0.0,
)
VERN7_BHAT = (
    0.4460860660634117628731817597479197781432e-1,
    0.0,
    0.0,
    0.2671640378571372680509102260943837899738,
    0.2201018300177293019979715776650753096323,
    0.2188431703143156830983120833512893824578,
    0.2289871705411202883378173889763552365362,
    0.0,
    0.0,
    0.2029518466335628222767054793810430358554e-1,
)
VERN7_E = tuple(b - bh for b, bh in zip(VERN7_B, VERN7_BHAT))
# ==================================================================================


_N_STAGES = len(VERN7_C)


def _stages(f, t, y, dt, p, k1=None):
    """Evaluate the 10 explicit stages; returns the list of stage derivatives k[0..9].

    ``k1`` (stage 0) may be supplied for FSAL reuse; Verner Vern7's first stage is
    ``f(t, y)`` so a caller holding it from the previous accepted step can pass it in.
    """
    k = [None] * _N_STAGES
    k[0] = f(t, y, p) if k1 is None else k1
    for i in range(1, _N_STAGES):
        acc = k[0] * VERN7_A[i][0]
        for j in range(1, i):
            acc = acc + k[j] * VERN7_A[i][j]
        k[i] = f(t + VERN7_C[i] * dt, y + dt * acc, p)
    return k


def vern7_advance(f, t, y, dt, p):
    """One Vern7 step (value-only): ``y(t) -> y(t + dt)`` for ``dy/dt = f(t, y, p)``.

    Padding contract for the replay adjoint: ``dt == 0`` is exactly an identity step in value
    and gradient — every stage collapses to ``f(t, y, p)``-argument ``y`` and the update
    ``y + dt*sum(b_i k_i) == y`` with ``d y_next/dy == I`` and ``d y_next/dp == 0`` (verified
    in value, JVP and VJP by tests/test_vern7_step.py). Traceable under jit/grad/vmap.
    """
    k = _stages(f, t, y, dt, p)
    acc = k[0] * VERN7_B[0]
    for i in range(1, _N_STAGES):
        acc = acc + k[i] * VERN7_B[i]
    return y + dt * acc


def vern7_trial_step(f, t, y, dt, p, k1=None):
    """One Vern7 trial step -> StepResult(y_next, dt*sum(E_i k_i), fsal=k[-1]).

    ``y_next`` is byte-identical to :func:`vern7_advance` (same stages, same b-weights). The
    error vector uses the precomputed ``VERN7_E = b - bhat`` weights. The FSAL slot carries the
    last stage; whether the recorder reuses it is set by ``VERN7_METHOD.fsal``.
    """
    k = _stages(f, t, y, dt, p, k1=k1)
    b_acc = k[0] * VERN7_B[0]
    e_acc = k[0] * VERN7_E[0]
    for i in range(1, _N_STAGES):
        b_acc = b_acc + k[i] * VERN7_B[i]
        e_acc = e_acc + k[i] * VERN7_E[i]
    y_next = y + dt * b_acc
    y_err = dt * e_acc
    return StepResult(y_next=y_next, y_err=y_err, fsal=k[-1])


#: Vern7 on the two-tier contract. fsal=False: Verner Vern7's stage-0 is f(t,y) but the
#: last stage is not guaranteed == the propagated point, so the recorder recomputes k1 each
#: step (correct; one extra RHS evaluation per step compared with a true FSAL method).
VERN7_METHOD = Method(
    name="vern7", order=7, error_order=6, fsal=False,
    advance=vern7_advance, trial_step=vern7_trial_step,
    beta1=1.0 / 7.0, beta2=0.0,   # I-controller exponent 1/(error_order+1); PI gain off
)
