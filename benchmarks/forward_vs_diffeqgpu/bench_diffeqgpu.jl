# Forward-only Lorenz ensemble solve with DiffEqGPU.jl's fused GPUTsit5 kernel, adaptive, float64.
#
# Same configuration as the Python side (benchmarks/forward_vs_diffeqgpu.py): sigma = 10, beta = 8/3,
# rho swept over [0, 21], u0 = (1, 0, 0), t in [0, 1]. The fused kernel is called through
# DiffEqGPU.vectorized_asolve, the entry the library's own GPU benchmarks use.
#
#   julia --project=benchmarks/forward_vs_diffeqgpu benchmarks/forward_vs_diffeqgpu/bench_diffeqgpu.jl \
#         <out_dir> <rtol> <n1> [n2 ...]
#
# Writes <out_dir>/diffeqgpu_rtol<rtol>.csv (n, min_ms, us_per_traj) and, for each n, the final states
# of an evenly spaced subsample of trajectories (<out_dir>/diffeqgpu_final_n<n>_rtol<rtol>.csv:
# index, x, y, z) for the accuracy check on the Python side. Timing: one warm-up, then BenchmarkTools'
# minimum over its samples of a CUDA-synchronised solve.
using DiffEqGPU, OrdinaryDiffEq, CUDA, StaticArrays, BenchmarkTools, DelimitedFiles, Printf

const T = Float64
const SUBSAMPLE = 4096

function lorenz(u, p, t)
    du1 = T(10.0) * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - T(8 / 3) * u[3]
    return @SVector T[du1, du2, du3]
end

function main()
    out = ARGS[1]; rtol = parse(T, ARGS[2]); ns = parse.(Int, ARGS[3:end])
    atol = rtol * T(1e-3)
    mkpath(out)
    println("Device: ", CUDA.name(CUDA.device()))
    prob = ODEProblem(lorenz, (@SVector T[1.0, 0.0, 0.0]), (T(0.0), T(1.0)), (@SArray T[21.0]))
    backend = CUDA.CUDABackend()
    CUDA.allowscalar(false)
    rows = []
    for n in ns
        rhos = range(T(0.0), stop = T(21.0), length = n)
        probs = [remake(prob, p = @SArray T[rhos[i]]) for i in 1:n]
        gprobs = DiffEqGPU.adapt(backend, DiffEqGPU.adapt.((backend,), probs))
        solve_once() = CUDA.@sync DiffEqGPU.vectorized_asolve(gprobs, prob, GPUTsit5();
                                                               dt = T(1e-3), reltol = rtol, abstol = atol,
                                                               save_everystep = false)
        ts, us = solve_once()                                   # warm-up (compile), untimed
        b = @benchmark $solve_once()
        min_ms = minimum(b.times) / 1e6
        @printf("n=%8d  min %.3f ms  %.5f us/traj\n", n, min_ms, min_ms * 1e3 / n)
        push!(rows, (n, min_ms, min_ms * 1e3 / n))
        # final states of an evenly spaced subsample, for the accuracy check
        uh = Array(us)
        finals = uh[end, :]
        idx = n <= SUBSAMPLE ? (1:n) : round.(Int, range(1, n, length = SUBSAMPLE))
        open(joinpath(out, @sprintf("diffeqgpu_final_n%d_rtol%g.csv", n, rtol)), "w") do io
            println(io, "index,x,y,z")
            for i in idx
                u = finals[i]
                @printf(io, "%d,%.17g,%.17g,%.17g\n", i - 1, u[1], u[2], u[3])
            end
        end
        GC.gc(); CUDA.reclaim()
    end
    open(joinpath(out, @sprintf("diffeqgpu_rtol%g.csv", rtol)), "w") do io
        println(io, "n,min_ms,us_per_traj")
        for (n, ms, us) in rows
            @printf(io, "%d,%.6f,%.6f\n", n, ms, us)
        end
    end
end

main()
