"""Adaptive Tsit5 + I-controller reference integrator: the semantic specification for the
Warp kernel.

Controller (Hairer, Norsett & Wanner, Solving ODEs I, Section II.4, I-controller):
  err  = sqrt(mean_i((e_i / (atol + rtol*max(|y_i|,|y5_i|)))^2))   # WRMS
  accept iff err <= 1.0
  dt   <- dt * clamp(0.9 * max(err, 1e-10)**(-0.2), 0.2, 5.0)      # after both accept and reject
  last step clamped: dt_try = min(dt, t1 - t); finish when t >= t1
FSAL: k1 of the next step = k7 = f(t+dt, y5) of the accepted step; k1 unchanged on reject.
"""
# The implementation lives in the gradsolve library (it is the reference record-and-replay
# recorder, general-RHS). It is re-exported here so this module keeps its role as the
# Warp-kernel semantic specification and existing import paths keep working.
from gradsolve.solvers.tsit5_replay import tsit5_adaptive  # noqa: F401
