"""Fixed-step Rosenbrock23 Warp kernel differentiated by ``wp.Tape``.

Three design points:

- Carry step state through an array (y_traj[j,s] -> [j,s+1]); register carry breaks the
  wp.Tape parameter adjoint.

- The solve adjoint. wp.Tape cannot differentiate the in-kernel partial-pivot Gaussian
  elimination: the in-place local matrix/vector mutation (M[r,c]=..., x[r]=..., row swaps)
  makes wp.Tape mis-accumulate the adjoint. The fix is a custom @wp.func_grad on the
  solve: the forward is the elimination, but the backward is the analytic linear-solve
  adjoint (dL/db = W^-T dL/dx; dL/dW = -(dL/db) outer x), computed with non-differentiated
  solves. wp.Tape never differentiates the elimination -> exact gradients for general D
  (checked to D=8; wp.inverse(W)*b also works but only for D<=4).

- Forward conditioning. A fixed-step replay of an aggressive adaptive stiff mesh amplifies
  per-step arithmetic differences (about 1.12x per step in Robertson's large-h tail). With
  Warp's CPU-LLVM FMA, the ~1e-10/step deviation from the float64 reference (numpy/JAX)
  amplifies to a forward blow-up on the default rtol=1e-6 mesh, while numpy/JAX (~1e-16)
  stay bounded. fuse_fp=False is a no-op on the LLVM-CPU backend (CUDA/NVCC only), so on
  CPU the cure is a fine-enough mesh: robertson rtol~1e-12, hires rtol~1e-10 -> the forward
  is bounded and matches the reference to ~3e-4. This is a conditioning property of
  fixed-step stiff replay, not a gradient bug.

Reuses the field + analytic-Jacobian @wp.func builders from _warp_rosenbrock.
"""
import functools

import numpy as np

# Guarded Warp import boundary (mirrors warp_ode's _WARP_AVAILABLE pattern) with ONE shared
# availability flag, ``_FUSED_AVAILABLE``, re-used by fused_backward. A warp-less environment
# imports this module as a stub (flag False) instead of raising, so ``import gradsolve`` succeeds
# everywhere and the fused reverse lane correctly reports itself unavailable rather than
# importable-but-broken.
try:
    import warp as wp

    from gradsolve.warp._warp_rosenbrock import _field_entry, _precision_types

    # Disable FMA contraction (-ffp-contract=fast, Warp's default fuse_fp=True). The fixed-step
    # replay of an AGGRESSIVE adaptive stiff mesh amplifies per-step perturbations (~1.12x/step
    # in Robertson's late large-h region); with FMA on, Warp's ~1e-10 arithmetic deviation from
    # the clean-float64 reference (numpy/JAX) is amplified to a forward BLOW-UP (2e5 by step
    # ~200), while numpy/JAX's ~1e-16 rounding stays bounded. fuse_fp=False makes the kernel
    # match the reference to ~1e-14/step -> the replay stays stable.
    wp.set_module_options({"fuse_fp": False})

    _FUSED_AVAILABLE = True
except ImportError:  # warp-less env: import as a stub; the fused lane reports unavailable
    wp = None
    _field_entry = None
    _precision_types = None
    _FUSED_AVAILABLE = False

_D_COEF = 1.0 / (2.0 + np.sqrt(2.0))


@functools.lru_cache(maxsize=None)
def _build(field_key, dim, precision="float64"):
    if not _FUSED_AVAILABLE:
        raise RuntimeError(
            "NVIDIA Warp is not installed: the fused Rosenbrock23 kernel needs it. "
            "Install the GPU extra with `pip install 'gradsolve[cuda12]'`.")
    np_dtype, wp_scalar = _precision_types(precision)
    builder, reg_dim = _field_entry(field_key)
    D = int(reg_dim)
    if int(dim) != D:
        raise ValueError(f"stiff field {field_key!r} has dim {D}, asked for {dim}")
    field_func, jac_func, vecD, matDD = builder(wp_scalar)

    d_c = wp_scalar(_D_COEF)
    half = wp_scalar(0.5)
    one = wp_scalar(1.0)
    two = wp_scalar(2.0)

    @wp.func
    def _gelim(A: matDD, b: vecD):
        # Partial-pivot Gaussian elimination solving A x = b (same as ``_gauss_solve`` in
        # gradsolve/warp/_warp_rosenbrock.py, the adaptive kernel's solve).
        # Used in the FORWARD and inside the CUSTOM backward below -- never differentiated
        # by wp.Tape (the custom @wp.func_grad supplies the adjoint analytically).
        M = matDD(A)
        x = vecD(b)
        for col in range(D):
            piv = col
            big = wp.abs(M[col, col])
            for r in range(col + 1, D):
                v = wp.abs(M[r, col])
                if v > big:
                    big = v
                    piv = r
            if piv != col:
                for c in range(D):
                    tmp = M[col, c]
                    M[col, c] = M[piv, c]
                    M[piv, c] = tmp
                tb = x[col]
                x[col] = x[piv]
                x[piv] = tb
            for r in range(col + 1, D):
                factor = M[r, col] / M[col, col]
                for c in range(col, D):
                    M[r, c] = M[r, c] - factor * M[col, c]
                x[r] = x[r] - factor * x[col]
        out = vecD()
        for ii in range(D):
            i = D - 1 - ii
            s = x[i]
            for c in range(i + 1, D):
                s = s - M[i, c] * out[c]
            out[i] = s / M[i, i]
        return out

    @wp.func
    def _solve(A: matDD, b: vecD):
        # x = A^{-1} b. The forward is plain Gaussian elimination; wp.Tape uses the CUSTOM
        # adjoint below instead of differentiating the elimination.
        return _gelim(A, b)

    @wp.func_grad(_solve)
    def _solve_grad(A: matDD, b: vecD, adj_ret: vecD):
        # THE FIX (probed: wp.Tape mis-accumulates adjoints through the in-place local
        # matrix/vector mutation of Gaussian elimination -- isolated 3x3 grad was rel 0.5-11
        # vs FD, same family as the register-carry bug). Analytic linear-solve adjoint:
        # for x = A^{-1} b and incoming cotangent adj_ret = dL/dx,
        #   dL/db = A^{-T} adj_ret ;   dL/dA = -(dL/db) outer x .
        # The two solves here are NOT differentiated (terminal adjoint computation), so their
        # in-place mutation is harmless. General D (probed exact to D=8, ~2e-9 vs FD).
        x = _gelim(A, b)
        ab = _gelim(wp.transpose(A), adj_ret)
        wp.adjoint[b] += ab
        for i in range(D):
            for jc in range(D):
                wp.adjoint[A][i, jc] += -ab[i] * x[jc]

    @wp.kernel
    def fused_rosenbrock_kernel(
        y0: wp.array(dtype=vecD),
        params: wp.array(dtype=vecD),
        dts: wp.array(dtype=wp_scalar, ndim=2),    # [n, S] frozen recorded mesh (data)
        t0: wp_scalar,
        n_steps: wp.int32,
        y_traj: wp.array(dtype=vecD, ndim=2),      # [n, S+1] state carry + output
    ):
        """Fixed-step Rosenbrock23 replay of a FROZEN recorded dt mesh, one thread per trajectory.

        The stiff prototype kernel (see module docstring): replays the
        accepted-dt sequence ``dts`` — as recorded by the adaptive kernel in
        ``_warp_rosenbrock.py`` — as a fixed-step loop entirely inside Warp, carrying
        state THROUGH the ``y_traj`` array rather than a register (the carry-through-array
        fix this module's docstring documents) so ``wp.Tape`` sees every intermediate
        state to differentiate against. Each step builds ``W = I - d_c*h*J`` from the
        closed-over analytic Jacobian ``jac_func`` and solves the two Rosenbrock23 stage
        systems via the CUSTOM-adjoint ``_solve`` (never the plain Gaussian elimination,
        which ``wp.Tape`` would mis-differentiate — see ``_solve``/``_solve_grad`` above).
        A zero row of ``dts`` (``h == 0``) collapses ``W`` to the identity and both stage
        solves to the identity step, so the zero-padded tail of a shorter-than-``S``
        trajectory replays as a no-op — the same zero-padding contract as the adaptive
        kernel's ``dt_rec``.

        Parameters
        ----------
        y0 : wp.array(dtype=vecD)
            Initial states, one ``vecD`` per trajectory (length n).
        params : wp.array(dtype=vecD)
            Per-trajectory param vector, padded/truncated to width D by ``_inputs``.
        dts : wp.array(dtype=wp_scalar, ndim=2)
            ``[n, S]`` frozen recorded accepted-dt mesh — DATA, not a differentiable
            function of ``params`` (the frozen-controller note in
            ``fused_backward.py``); zero-padded past each trajectory's accepted-step
            count.
        t0 : wp_scalar
            Shared initial time for every trajectory.
        n_steps : wp.int32
            ``S``, the (static, per-launch) number of replay steps.
        y_traj : wp.array(dtype=vecD, ndim=2)
            ``[n, S+1]`` output AND ``wp.Tape`` carry array: column 0 is ``y0``,
            column ``s + 1`` is the state after replaying ``dts[:, s]``. Column
            ``S`` is the final state ``fused_forward``/``fused_sq_grad``/``fused_vjp``
            read back; must be a full array (not a register) for the reverse pass to
            see every intermediate step.
        """
        j = wp.tid()
        p = params[j]
        t = t0
        y_traj[j, 0] = y0[j]
        for s in range(n_steps):
            y = y_traj[j, s]              # read carry FROM array (not a register)
            h = dts[j, s]                 # dt==0 -> W=I -> identity step
            J = jac_func(t, y, p)
            W = matDD()
            for a in range(D):
                for b in range(D):
                    W[a, b] = -d_c * h * J[a, b]
                W[a, a] = W[a, a] + one
            F0 = field_func(t, y, p)
            k1 = _solve(W, F0)
            F1 = field_func(t, y + half * h * k1, p)
            k2 = _solve(W, F1 - k1) + k1
            y_traj[j, s + 1] = y + h * k2
            t = t + h

    @wp.kernel
    def seed_sq_grad(y_traj: wp.array(dtype=vecD, ndim=2), s_final: wp.int32,
                     grad_out: wp.array(dtype=vecD, ndim=2)):
        j = wp.tid()
        grad_out[j, s_final] = two * y_traj[j, s_final]

    @wp.kernel
    def seed_vjp(gy: wp.array(dtype=vecD), s_final: wp.int32,
                 grad_out: wp.array(dtype=vecD, ndim=2)):
        # Seed the INCOMING output cotangent gy[j] (shape [n, D]) onto the final state
        # adjoint y_traj.grad[:, s_final, :], generalizing seed_sq_grad's hard-coded
        # 2*y_final (the sum(y**2) loss cotangent) to an arbitrary downstream cotangent.
        j = wp.tid()
        grad_out[j, s_final] = gy[j]

    return fused_rosenbrock_kernel, seed_sq_grad, seed_vjp, vecD, np_dtype, wp_scalar, D


def _inputs(field_key, dim, y0, pvec, dts, precision, device, requires_grad):
    kern, seed_kern, seed_vjp_kern, vecD, np_dtype, wp_scalar, D = _build(field_key, dim, precision)
    y0 = np.ascontiguousarray(y0, dtype=np_dtype)
    # The kernel's `params` arg is a per-trajectory vecD (inner dim == D). `_params_of`
    # returns the problem's RAW param vector (P columns: robertson P=3==D; hires/linstiff
    # P=1 placeholder), so right-pad with zeros / truncate to width D -- exactly as the
    # real backend's `_launch` does. The padded columns are ignored by autonomous fields.
    pvec = np.ascontiguousarray(pvec, dtype=np_dtype)
    if pvec.ndim == 1:
        pvec = pvec.reshape(-1, 1)
    P = pvec.shape[1]
    if P < D:
        pvec = np.pad(pvec, ((0, 0), (0, D - P)))
    elif P > D:
        pvec = pvec[:, :D]
    pvec = np.ascontiguousarray(pvec, dtype=np_dtype)
    dts = np.ascontiguousarray(dts, dtype=np_dtype)
    n, S = dts.shape
    y0_w = wp.array(y0, dtype=vecD, device=device, requires_grad=requires_grad)
    p_w = wp.array(pvec, dtype=vecD, device=device, requires_grad=requires_grad)
    dts_w = wp.array(dts, dtype=wp_scalar, ndim=2, device=device)
    y_traj = wp.zeros((n, S + 1), dtype=vecD, device=device, requires_grad=requires_grad)
    return kern, seed_kern, seed_vjp_kern, vecD, D, n, S, y0_w, p_w, dts_w, y_traj


def fused_forward(field_key, dim, y0, pvec, dts, t0, *, device="cpu", precision="float64"):
    kern, _seed, _seed_vjp, vecD, D, n, S, y0_w, p_w, dts_w, y_traj = _inputs(
        field_key, dim, y0, pvec, dts, precision, device, requires_grad=False)
    wp.launch(kern, dim=n, inputs=[y0_w, p_w, dts_w, float(t0), int(S)], outputs=[y_traj], device=device)
    return y_traj.numpy()[:, S, :].astype(np.float64)


def fused_sq_grad(field_key, dim, y0, pvec, dts, t0, *, device="cpu", precision="float64"):
    """wp.Tape reverse of sum(y_final**2). Returns (grad_y0[n,D], grad_params[n,D])."""
    kern, seed_kern, _seed_vjp, vecD, D, n, S, y0_w, p_w, dts_w, y_traj = _inputs(
        field_key, dim, y0, pvec, dts, precision, device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(kern, dim=n, inputs=[y0_w, p_w, dts_w, float(t0), int(S)], outputs=[y_traj], device=device)
    y_traj.grad.zero_()
    wp.launch(seed_kern, dim=n, inputs=[y_traj, int(S)], outputs=[y_traj.grad], device=device)
    tape.backward()
    return y0_w.grad.numpy().astype(np.float64), p_w.grad.numpy().astype(np.float64)


def fused_vjp(field_key, dim, y0, pvec, dts, t0, gy_final, *, device="cpu", precision="float64"):
    """wp.Tape reverse with an ARBITRARY incoming output cotangent (the vjp generalizing
    fused_sq_grad). ``gy_final[n, D]`` is the downstream cotangent dL/dy_final; this seeds
    y_traj.grad[:, s_final, :] = gy_final (instead of fused_sq_grad's hard-coded
    2*y_final = the sum(y**2) cotangent), runs tape.backward(), and returns
    (grad_y0[n, D], grad_pvec[n, D]) = (dL/dy0, dL/dpvec). The param grad is in the
    PADDED pvec layout (width D); the caller maps it back to the native param shape.
    """
    kern, _seed, seed_vjp_kern, vecD, D, n, S, y0_w, p_w, dts_w, y_traj = _inputs(
        field_key, dim, y0, pvec, dts, precision, device, requires_grad=True)
    np_dtype = np.float32 if precision == "float32" else np.float64
    gy = np.ascontiguousarray(gy_final, dtype=np_dtype)
    gy_w = wp.array(gy, dtype=vecD, device=device)
    tape = wp.Tape()
    with tape:
        wp.launch(kern, dim=n, inputs=[y0_w, p_w, dts_w, float(t0), int(S)], outputs=[y_traj], device=device)
    y_traj.grad.zero_()
    wp.launch(seed_vjp_kern, dim=n, inputs=[gy_w, int(S)], outputs=[y_traj.grad], device=device)
    tape.backward()
    return y0_w.grad.numpy().astype(np.float64), p_w.grad.numpy().astype(np.float64)
