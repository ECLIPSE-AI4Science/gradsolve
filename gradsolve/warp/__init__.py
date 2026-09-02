"""gradsolve.warp — fused NVIDIA Warp engines for ensembles of differential equations.

``warp_ode`` (fused adaptive Tsit5), ``warp_rosenbrock`` (fused stiff Rosenbrock23), and
``warp_replay`` (record-and-replay reverse-mode adjoint), plus the internal kernel modules
``_warp_kernel`` / ``_warp_rosenbrock``. Each runs its fused CUDA kernel only on a GPU;
on CPU the modules import but their solves report unsupported.
"""
