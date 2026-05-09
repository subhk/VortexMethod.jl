module Workspace

using StaticArrays
using ..DomainImpl
using ..Circulation: TriangleGeometry

export VortexWorkspace

struct VortexWorkspace{T<:AbstractFloat}
    # Timestep buffers (rk2_step!)
    xh::Vector{T}
    yh::Vector{T}
    zh::Vector{T}
    triXC::Matrix{T}
    triYC::Matrix{T}
    triZC::Matrix{T}
    triXC_new::Matrix{T}
    triYC_new::Matrix{T}
    triZC_new::Matrix{T}
    eleGma_mid::Matrix{T}
    eleGma_new::Matrix{T}
    nodeΓ::Matrix{T}
    u1::Vector{T}
    v1::Vector{T}
    w1::Vector{T}
    u2::Vector{T}
    v2::Vector{T}
    w2::Vector{T}

    # Geometry cache — invalidated when triXC is overwritten
    triC::Matrix{T}           # (ntri, 3) element centroids
    subC::Array{T,3}          # (ntri, nsub_max, 3) sub-triangle centroids
    areas::Vector{T}          # (ntri,)
    geom::TriangleGeometry{T} # edge vectors + areas + centroids, shared by circ and spread
    geom_dirty::Ref{Bool}     # true → recompute on next spread/circ call
    nsub::Ref{Int}            # current sub-segment count (≤ nsub_max)
    nsub_max::Int             # max sub-segments^2 (allocated size of subC dim 2)

    # Spread / interpolation buffers
    local_buf::Matrix{T}      # (nx*ny*nz, 3) — MPI local accumulation
    global_buf::Matrix{T}     # (nx*ny*nz, 3) — post-Allreduce result
    ζx::Array{T,3}          # (nz, ny, nx)
    ζy::Array{T,3}
    ζz::Array{T,3}
    rhs_x::Array{T,3}
    rhs_y::Array{T,3}
    rhs_z::Array{T,3}
    gridUx::Array{T,3}
    gridUy::Array{T,3}
    gridUz::Array{T,3}
    fft_x::Array{Complex{T},3}
    fft_y::Array{Complex{T},3}
    fft_z::Array{Complex{T},3}
    pencil_poisson::Base.RefValue{Any}

    # Nearby element index scratch (used by find_elements_nearby!)
    nearby_buf::Vector{Int}

    # Periodic tile shifts — NTuple → compiler unrolls 27-tile loop
    shifts::NTuple{27, SVector{3,T}}
end

function VortexWorkspace(::Type{T}, nnodes::Int, ntri::Int,
                         nx::Int, ny::Int, nz::Int,
                         domain::DomainSpec;
                         max_subsegments::Int=8) where T<:AbstractFloat
    nsub_max = max_subsegments * max_subsegments
    grid_size = nx * ny * nz

    # Build shifts NTuple from domain — computed once, never reallocated
    raw_shifts = periodic_shifts(domain)  # Vector{NTuple{3,Float64}} from domain.jl
    shifts = ntuple(i -> SVector{3,T}(T(raw_shifts[i][1]),
                                      T(raw_shifts[i][2]),
                                      T(raw_shifts[i][3])), 27)

    return VortexWorkspace{T}(
        # Timestep buffers
        Vector{T}(undef, nnodes), Vector{T}(undef, nnodes), Vector{T}(undef, nnodes),
        Matrix{T}(undef, ntri, 3), Matrix{T}(undef, ntri, 3), Matrix{T}(undef, ntri, 3),
        Matrix{T}(undef, ntri, 3), Matrix{T}(undef, ntri, 3), Matrix{T}(undef, ntri, 3),
        Matrix{T}(undef, ntri, 3), Matrix{T}(undef, ntri, 3),
        Matrix{T}(undef, ntri, 3),   # nodeΓ
        Vector{T}(undef, nnodes), Vector{T}(undef, nnodes), Vector{T}(undef, nnodes),
        Vector{T}(undef, nnodes), Vector{T}(undef, nnodes), Vector{T}(undef, nnodes),
        # Geometry cache
        Matrix{T}(undef, ntri, 3),
        Array{T}(undef, ntri, nsub_max, 3),
        Vector{T}(undef, ntri),
        TriangleGeometry(T, ntri),
        Ref(true),          # geom_dirty = true initially
        Ref(0),
        nsub_max,
        # Spread/interp buffers
        Matrix{T}(undef, max(grid_size, nnodes), 3),
        Matrix{T}(undef, max(grid_size, nnodes), 3),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{T}(undef, nz, ny, nx),
        Array{Complex{T}}(undef, nz, ny, nx),
        Array{Complex{T}}(undef, nz, ny, nx),
        Array{Complex{T}}(undef, nz, ny, nx),
        Ref{Any}(nothing),
        # Nearby element index scratch
        Vector{Int}(undef, ntri),
        # Shifts NTuple
        shifts
    )
end

end # module

using .Workspace: VortexWorkspace
