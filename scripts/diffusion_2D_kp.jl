# https://github.com/JuliaGPU/Metal.jl
# https://juliagpu.org/post/2022-06-24-metal/
# Perf on MacBookAir M2: memcopy 86 GB/s, 2D diffusion 83 GB/s (96%)
# NOTE: soon, grid will become groups in kernel launch params
using Metal
using Plots
using BenchmarkTools

function memcopy!(T2, T, D, dt)
    ix, iy = thread_position_in_grid_2d()
    @inbounds T2[ix, iy] = T[ix, iy] + dt * D[ix, iy] # memcopy
    return
end

function laplace!(T2, T, D, dt, _dx, _dy)
    xr, yr = Int32.(size(T))
    ix, iy = thread_position_in_grid_2d()
    if (ix>1 && ix<xr && iy>1 && iy<yr)
        @inbounds T2[ix, iy] = T[ix, iy] + dt * (D[ix, iy] * (
                              - ((-(T[ix+1, iy] - T[ix, iy])*_dx) - (-(T[ix, iy] - T[ix-1, iy])*_dx))
                              - ((-(T[ix, iy+1] - T[ix, iy])*_dy) - (-(T[ix, iy] - T[ix, iy-1])*_dy))))
    end
    return
end

function compute!(T2, T, D, dt, _dx, _dy, nthreads, nblocks)
        Metal.@sync @metal threads=nthreads grid=nblocks laplace!(T2, T, D, dt, _dx, _dy)
    return
end

function compute!(T2, T, D, dt, nthreads, nblocks)
        Metal.@sync @metal threads=nthreads grid=nblocks memcopy!(T2, T, D, dt)
    return
end

function main(; do_visu=false, do_bench=false)
    # Metal.versioninfo()
    # current_device()
    # physics
    lx, ly = 10.0f0, 10.0f0
    D0 = 1.0f0
    T0 = 2.0f0
    # numerics
    nx = ny = 8192
    nt = 3nx
    nout = ceil(Int32, 0.3nx)
    nthreads = (32, 8)
    nblocks = cld.((nx, ny), nthreads)
    # preprocess
    dx, dy = lx/nx, ly/ny
    dt = min(dx,dy)^2/D0/4.1f0
    _dx, _dy = 1.0f0/dx, 1.0f0/dy
    # init
    xc, yc = LinRange(-lx/2+dx/2, lx/2-dx/2, nx), LinRange(-ly/2+dy/2, ly/2-dy/2, ny)
    T = MtlArray{Float32}(undef, nx, ny); copy!(T, T0 .* exp.( .- xc.^2 .- yc'.^2))
    D = MtlArray{Float32}(undef, nx, ny); copy!(D, D0 .* ones(nx,ny))
    T2 = copy(T)
    if !do_bench
        for it = 1:nt
            Metal.@sync @metal threads=nthreads grid=nblocks laplace!(T2, T, D, dt, _dx, _dy)
            T, T2 = T2, T
            # visu
            if (it % nout == 0) && do_visu
                display(heatmap(Array(T)', c=:turbo, clims=(0, T0), aspect_ratio=1, xlims=(1, nx), ylims=(1, ny), title="Metal.jl diffusion step $it"))
            end
        end
    else    
        t_it = @belapsed compute!($T2, $T, $D, $dt, $nthreads, $nblocks)
        t_eff1 = sizeof(eltype(T)) * 3 * length(T) * 1e-9 / t_it
        println(" Perf memcopy: time (s) = $(round(t_it, digits=5)), T_eff (GB/s) = $(round(t_eff1, digits=2))")
        
        t_it = @belapsed compute!($T2, $T, $D, $dt, $_dx, $_dy, $nthreads, $nblocks)
        t_eff2 = sizeof(eltype(T)) * 3 * length(T) * 1e-9 / t_it
        println(" Perf Laplace: time (s) = $(round(t_it, digits=5)), T_eff (GB/s) = $(round(t_eff2, digits=2)) ($(round(t_eff2/t_eff1, digits=2))% of memcopy)")
    end

    finalize(T)
    finalize(D)
    finalize(T2)
    return
end

main(; do_visu=false, do_bench=true)
