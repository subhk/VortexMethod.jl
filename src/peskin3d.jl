# 3D Peskin spreading and interpolation with MPI

module Peskin3D

using ..DomainImpl
using ..Kernels
using ..Circulation: compute_triangle_geometry!
using MPI
using StaticArrays
using ..Workspace: VortexWorkspace

export init_mpi!, finalize_mpi!,
       triangle_centroids, triangle_areas,
       subtriangle_centroids, subtriangle_centroids4,
       spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_mpi!,
       spread_vorticity_to_grid_kernel_mpi,
       interpolate_node_velocity_mpi, interpolate_node_velocity_mpi!,
       interpolate_node_velocity_kernel_mpi,
       find_elements_nearby!

init_mpi!() = (MPI.Initialized() || MPI.Init(); nothing)
finalize_mpi!() = (MPI.Finalized() || MPI.Finalize(); nothing)

# Basic geometry helpers
triangle_centroids(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                   domain::Union{Nothing,DomainSpec}=nothing) = begin
    nt = size(triXC, 1)
    C = zeros(Float64, nt, 3)

    @inbounds for t in 1:nt
        p1 = (triXC[t, 1], triYC[t, 1], triZC[t, 1])
        p2 = (triXC[t, 2], triYC[t, 2], triZC[t, 2])
        p3 = (triXC[t, 3], triYC[t, 3], triZC[t, 3])
        if domain === nothing
            C[t, 1] = (p1[1] + p2[1] + p3[1]) / 3
            C[t, 2] = (p1[2] + p2[2] + p3[2]) / 3
            C[t, 3] = (p1[3] + p2[3] + p3[3]) / 3
        else
            C[t, 1], C[t, 2], C[t, 3] = periodic_centroid(p1, p2, p3, domain)
        end
    end
    C
end

function triangle_areas(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                        domain::Union{Nothing,DomainSpec}=nothing)
    nt = size(triXC, 1)
    A = zeros(Float64, nt)

    @inbounds for t in 1:nt
        p1 = (triXC[t, 1], triYC[t, 1], triZC[t, 1])
        p2 = (triXC[t, 2], triYC[t, 2], triZC[t, 2])
        p3 = (triXC[t, 3], triYC[t, 3], triZC[t, 3])

        if domain !== nothing
            A[t] = periodic_triangle_area(p1, p2, p3, domain)
            continue
        end

        a = hypot(hypot(p1[1]-p2[1], p1[2]-p2[2]), p1[3]-p2[3])
        b = hypot(hypot(p2[1]-p3[1], p2[2]-p3[2]), p2[3]-p3[3])
        c = hypot(hypot(p3[1]-p1[1], p3[2]-p1[2]), p3[3]-p1[3])

        # Heron's formula
        s = (a+b+c)/2
        A[t] = sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
    end
    A
end

@inline function bary_point(a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T},
                             i::Int, j::Int, M::Int) where T
    rb = T(i) / T(M)
    rc = T(j) / T(M)
    return a + rb * (b - a) + rc * (c - a)
end

@inline function centroid3(a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T}) where T
    return (a + b + c) / T(3)
end

@inline function maybe_wrap(c::SVector{3,T}, domain::Union{Nothing,DomainSpec}) where T
    domain === nothing && return c
    xw, yw, zw = wrap_point(c[1], c[2], c[3], domain)
    return SVector{3,T}(T(xw), T(yw), T(zw))
end

@inline function periodic_shifts_svector(::Type{T}, domain::DomainSpec) where T
    return ntuple(Val(27)) do i
        q = i - 1
        sx = fld(q, 9) - 1
        sy = fld(mod(q, 9), 3) - 1
        sz = mod(q, 3) - 1
        SVector{3,T}(T(sx) * T(domain.Lx),
                     T(sy) * T(domain.Ly),
                     T(sz) * T(2 * domain.Lz))
    end
end

function subtriangle_centroids(p1::NTuple{3,T}, p2::NTuple{3,T},
                               p3::NTuple{3,T}, M::Integer;
                               domain::Union{Nothing,DomainSpec}=nothing) where {T<:AbstractFloat}
    M >= 1 || throw(ArgumentError("subtriangle segment count M must be >= 1"))
    # pack into SVector for use with updated bary_point/centroid3
    raw_a, raw_b, raw_c = domain === nothing ? (p1, p2, p3) : unwrap_triangle(p1, p2, p3, domain)
    a = SVector{3,T}(T(raw_a[1]), T(raw_a[2]), T(raw_a[3]))
    b = SVector{3,T}(T(raw_b[1]), T(raw_b[2]), T(raw_b[3]))
    c = SVector{3,T}(T(raw_c[1]), T(raw_c[2]), T(raw_c[3]))
    C = Array{T}(undef, M * M, 3)
    idx = 1

    if M == 2
        order = (
            ((0, 0), (1, 0), (0, 1)),
            ((1, 0), (1, 1), (0, 1)),
            ((1, 0), (2, 0), (1, 1)),
            ((1, 1), (0, 2), (0, 1)),
        )
        @inbounds for tri_idx in order
            v1 = bary_point(a, b, c, tri_idx[1][1], tri_idx[1][2], M)
            v2 = bary_point(a, b, c, tri_idx[2][1], tri_idx[2][2], M)
            v3 = bary_point(a, b, c, tri_idx[3][1], tri_idx[3][2], M)
            ct = maybe_wrap(centroid3(v1, v2, v3), domain)
            C[idx, 1] = ct[1]; C[idx, 2] = ct[2]; C[idx, 3] = ct[3]
            idx += 1
        end
        return C
    end

    @inbounds for i in 0:M-1, j in 0:M-1-i
        v1 = bary_point(a, b, c, i, j, M)
        v2 = bary_point(a, b, c, i + 1, j, M)
        v3 = bary_point(a, b, c, i, j + 1, M)
        ct = maybe_wrap(centroid3(v1, v2, v3), domain)
        C[idx, 1] = ct[1]; C[idx, 2] = ct[2]; C[idx, 3] = ct[3]
        idx += 1
        if i + j <= M - 2
            v1 = bary_point(a, b, c, i + 1, j, M)
            v2 = bary_point(a, b, c, i + 1, j + 1, M)
            v3 = bary_point(a, b, c, i, j + 1, M)
            ct = maybe_wrap(centroid3(v1, v2, v3), domain)
            C[idx, 1] = ct[1]; C[idx, 2] = ct[2]; C[idx, 3] = ct[3]
            idx += 1
        end
    end
    return C
end

function write_subtriangle_centroids!(subC::Array{T,3}, t::Int,
                                      p1::NTuple{3,T}, p2::NTuple{3,T},
                                      p3::NTuple{3,T}, M::Integer,
                                      domain::Union{Nothing,DomainSpec}) where {T<:AbstractFloat}
    raw_a, raw_b, raw_c = domain === nothing ? (p1, p2, p3) : unwrap_triangle(p1, p2, p3, domain)
    a = SVector{3,T}(T(raw_a[1]), T(raw_a[2]), T(raw_a[3]))
    b = SVector{3,T}(T(raw_b[1]), T(raw_b[2]), T(raw_b[3]))
    c = SVector{3,T}(T(raw_c[1]), T(raw_c[2]), T(raw_c[3]))
    idx = 1

    if M == 2
        order = (
            ((0, 0), (1, 0), (0, 1)),
            ((1, 0), (1, 1), (0, 1)),
            ((1, 0), (2, 0), (1, 1)),
            ((1, 1), (0, 2), (0, 1)),
        )
        @inbounds for tri_idx in order
            v1 = bary_point(a, b, c, tri_idx[1][1], tri_idx[1][2], M)
            v2 = bary_point(a, b, c, tri_idx[2][1], tri_idx[2][2], M)
            v3 = bary_point(a, b, c, tri_idx[3][1], tri_idx[3][2], M)
            ct = maybe_wrap(centroid3(v1, v2, v3), domain)
            subC[t, idx, 1] = ct[1]
            subC[t, idx, 2] = ct[2]
            subC[t, idx, 3] = ct[3]
            idx += 1
        end
        return nothing
    end

    @inbounds for i in 0:M-1, j in 0:M-1-i
        v1 = bary_point(a, b, c, i, j, M)
        v2 = bary_point(a, b, c, i + 1, j, M)
        v3 = bary_point(a, b, c, i, j + 1, M)
        ct = maybe_wrap(centroid3(v1, v2, v3), domain)
        subC[t, idx, 1] = ct[1]
        subC[t, idx, 2] = ct[2]
        subC[t, idx, 3] = ct[3]
        idx += 1
        if i + j <= M - 2
            v1 = bary_point(a, b, c, i + 1, j, M)
            v2 = bary_point(a, b, c, i + 1, j + 1, M)
            v3 = bary_point(a, b, c, i, j + 1, M)
            ct = maybe_wrap(centroid3(v1, v2, v3), domain)
            subC[t, idx, 1] = ct[1]
            subC[t, idx, 2] = ct[2]
            subC[t, idx, 3] = ct[3]
            idx += 1
        end
    end
    return nothing
end

subtriangle_centroids4(p1::NTuple{3,T}, p2::NTuple{3,T}, p3::NTuple{3,T}) where T<:AbstractFloat =
    subtriangle_centroids(p1, p2, p3, 2)

# Build 4-subtriangle centroids for all triangles
function build_all_subcentroids(triXC, triYC, triZC;
                                subsegments::Int=2,
                                domain::Union{Nothing,DomainSpec}=nothing)
    nt = size(triXC,1)
    C = Array{Float64}(undef, nt, subsegments * subsegments, 3)

    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        T = subtriangle_centroids(p1, p2, p3, subsegments; domain=domain)

        C[t, :, 1] = view(T, :, 1)
        C[t, :, 2] = view(T, :, 2)
        C[t, :, 3] = view(T, :, 3)
    end
    C
end

function subtriangle_segment_count(triXC, triYC, triZC, ds;
                                   domain::Union{Nothing,DomainSpec}=nothing,
                                   min_segments::Int=2,
                                   max_segments::Int=8)
    h = max(minimum(ds), eps(Float64))
    max_edge = 0.0
    @inbounds for t in 1:size(triXC, 1)
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        q1, q2, q3 = domain === nothing ? (p1, p2, p3) : unwrap_triangle(p1, p2, p3, domain)
        l12 = hypot(hypot(q2[1]-q1[1], q2[2]-q1[2]), q2[3]-q1[3])
        l23 = hypot(hypot(q3[1]-q2[1], q3[2]-q2[2]), q3[3]-q2[3])
        l31 = hypot(hypot(q1[1]-q3[1], q1[2]-q3[2]), q1[3]-q3[3])
        max_edge = max(max_edge, l12, l23, l31)
    end
    return clamp(ceil(Int, max_edge / h), min_segments, max_segments)
end

# Find elements near a point within epsx,epsy,epsz (Python _find_elements_nearby_)
function find_elements_nearby(x,y,z, epsx,epsy,epsz, triC::AbstractMatrix)
    nt = size(triC,1)
    nearby = Int[]
    @inbounds for t in 1:nt
        if abs(triC[t,1] - x) <= epsx &&
           abs(triC[t,2] - y) <= epsy &&
           abs(triC[t,3] - z) <= epsz
            push!(nearby, t)
        end
    end
    return nearby
end

# Non-allocating version with preallocated buffer
function find_elements_nearby!(buf::Vector{Int}, x::T, y::T, z::T,
                                epsx::T, epsy::T, epsz::T,
                                triC::AbstractMatrix{T}) where T
    empty!(buf)
    @inbounds for t in 1:size(triC,1)
        if abs(triC[t,1] - x) <= epsx &&
           abs(triC[t,2] - y) <= epsy &&
           abs(triC[t,3] - z) <= epsz
            push!(buf, t)
        end
    end
    return buf
end

# Core Peskin sum for a list of triangles (grid vorticity accumulation)
function peskin_add_ele!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix, 
                         subC, triAreas, tri_list, coord, delr, eps)
    sx, sy, sz = sum
    (epsx,epsy,epsz) = eps
    x = coord
    n_sub = size(subC,2)
    inv_n_sub = 1.0 / n_sub
    scale = 1.0 / (8.0 * delr^3)
    @inbounds for idx in tri_list
        S = 0.0
        @simd for s in 1:n_sub
            dx = x[1] - subC[idx,s,1]
            dy = x[2] - subC[idx,s,2]
            dz = x[3] - subC[idx,s,3]
            tmp = (1 + cos(pi*dx/epsx))*(1 + cos(pi*dy/epsy))*(1 + cos(pi*dz/epsz)) * scale
            S += tmp
        end
        w = triAreas[idx] * S * inv_n_sub
        sx += w * eleGma[idx,1]
        sy += w * eleGma[idx,2]
        sz += w * eleGma[idx,3]
    end
    return (sx,sy,sz)
end

function peskin_add_nearby!(acc::SVector{3,T}, eleGma::AbstractMatrix{T},
                            triC::AbstractMatrix{T}, subC::Array{T,3},
                            triAreas::AbstractVector{T},
                            coord::SVector{3,T}, delr::T,
                            eps::SVector{3,T},
                            shift::SVector{3,T}) where T<:AbstractFloat
    return peskin_add_nearby_nsub!(acc, eleGma, triC, subC, triAreas,
                                   size(subC, 2), coord, delr, eps, shift)
end

function peskin_add_nearby_nsub!(acc::SVector{3,T}, eleGma::AbstractMatrix{T},
                                 triC::AbstractMatrix{T}, subC::Array{T,3},
                                 triAreas::AbstractVector{T}, n_sub::Int,
                                 coord::SVector{3,T}, delr::T,
                                 eps::SVector{3,T},
                                 shift::SVector{3,T}) where T<:AbstractFloat
    sx = acc[1]; sy = acc[2]; sz = acc[3]
    x = coord[1]; y = coord[2]; z = coord[3]
    epsx = eps[1]; epsy = eps[2]; epsz = eps[3]
    inv_n_sub = one(T) / T(n_sub)
    scale = one(T) / (T(8) * delr^3)

    @inbounds for idx in 1:size(triC, 1)
        cx = triC[idx,1] + shift[1]
        cy = triC[idx,2] + shift[2]
        cz = triC[idx,3] + shift[3]
        if abs(cx - x) <= epsx && abs(cy - y) <= epsy && abs(cz - z) <= epsz
            S = zero(T)
            @simd for s in 1:n_sub
                dx = x - (subC[idx,s,1] + shift[1])
                dy = y - (subC[idx,s,2] + shift[2])
                dz = z - (subC[idx,s,3] + shift[3])
                S += (1 + cos(T(π)*dx/epsx))*(1 + cos(T(π)*dy/epsy))*(1 + cos(T(π)*dz/epsz)) * scale
            end
            w = triAreas[idx] * S * inv_n_sub
            sx += w * eleGma[idx,1]
            sy += w * eleGma[idx,2]
            sz += w * eleGma[idx,3]
        end
    end
    return SVector{3,T}(sx, sy, sz)
end

function peskin_add_nearby_kernel!(acc::SVector{3,T}, eleGma::AbstractMatrix{T},
                                   triC::AbstractMatrix{T}, subC::Array{T,3},
                                   triAreas::AbstractVector{T},
                                   coord::SVector{3,T}, kernel::KernelType,
                                   eps::SVector{3,T},
                                   shift::SVector{3,T}) where T<:AbstractFloat
    sx = acc[1]; sy = acc[2]; sz = acc[3]
    x = coord[1]; y = coord[2]; z = coord[3]
    epsx = eps[1]; epsy = eps[2]; epsz = eps[3]
    delr = T(kernel_support_radius(kernel))
    hx = epsx/delr; hy = epsy/delr; hz = epsz/delr
    n_sub = size(subC, 2)
    inv_n_sub = one(T) / T(n_sub)

    @inbounds for idx in 1:size(triC, 1)
        cx = triC[idx,1] + shift[1]
        cy = triC[idx,2] + shift[2]
        cz = triC[idx,3] + shift[3]
        if abs(cx - x) <= epsx && abs(cy - y) <= epsy && abs(cz - z) <= epsz
            S = zero(T)
            @simd for s in 1:n_sub
                dx = x - (subC[idx,s,1] + shift[1])
                dy = y - (subC[idx,s,2] + shift[2])
                dz = z - (subC[idx,s,3] + shift[3])
                S += kernel_function(kernel, dx, dy, dz, hx, hy, hz)
            end
            w = triAreas[idx] * S * inv_n_sub
            sx += w * eleGma[idx,1]
            sy += w * eleGma[idx,2]
            sz += w * eleGma[idx,3]
        end
    end
    return SVector{3,T}(sx, sy, sz)
end

# Accumulate vorticity at a coordinate from periodic tiles (9 tiles in xy like python)
function peskin_grid_sum(eleGma::AbstractMatrix{Float64}, triC::AbstractMatrix{Float64},
                         subC::Array{Float64,3}, coord::NTuple{3,Float64},
                         ds::NTuple{3,<:Real}, triAreas::AbstractVector{Float64};
                         delr::Real=4.0, domain::DomainSpec=default_domain())
    eps = SVector{3,Float64}(delr*ds[1], delr*ds[2], delr*ds[3])
    sv_coord = SVector{3,Float64}(coord[1], coord[2], coord[3])
    acc = SVector{3,Float64}(0.0, 0.0, 0.0)
    for shift in periodic_shifts_svector(Float64, domain)
        acc = peskin_add_nearby!(acc, eleGma, triC, subC, triAreas,
                                 sv_coord, Float64(delr), eps, shift)
    end
    return acc[1], acc[2], acc[3]
end

# Enhanced grid sum with kernel selection
function peskin_grid_sum_kernel(eleGma, triC, subC, coord::NTuple{3,Float64}, ds, triAreas,
                        kernel::KernelType; domain::DomainSpec=default_domain())
    delr = kernel_support_radius(kernel)
    eps = SVector{3,Float64}(delr*ds[1], delr*ds[2], delr*ds[3])
    sv_coord = SVector{3,Float64}(coord[1], coord[2], coord[3])
    acc = SVector{3,Float64}(0.0, 0.0, 0.0)
    for shift in periodic_shifts_svector(Float64, domain)
        acc = peskin_add_nearby_kernel!(acc, eleGma, triC, subC, triAreas,
                                       sv_coord, kernel, eps, shift)
    end
    return acc[1], acc[2], acc[3]
end

function _recompute_geometry!(ws::VortexWorkspace{T},
                              triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                              domain::DomainSpec, gr::GridSpec) where T<:AbstractFloat
    dx, dy, dz = grid_spacing(domain, gr)
    max_segments = isqrt(ws.nsub_max)
    segments = subtriangle_segment_count(triXC, triYC, triZC, (dx, dy, dz);
                                         domain=domain, max_segments=max_segments)
    nsub = segments * segments
    ws.nsub[] = nsub

    @inbounds for t in 1:size(triXC, 1)
        p1 = (T(triXC[t,1]), T(triYC[t,1]), T(triZC[t,1]))
        p2 = (T(triXC[t,2]), T(triYC[t,2]), T(triZC[t,2]))
        p3 = (T(triXC[t,3]), T(triYC[t,3]), T(triZC[t,3]))

        cx, cy, cz = periodic_centroid(p1, p2, p3, domain)
        ws.triC[t,1] = T(cx)
        ws.triC[t,2] = T(cy)
        ws.triC[t,3] = T(cz)
        ws.areas[t] = T(periodic_triangle_area(p1, p2, p3, domain))

        write_subtriangle_centroids!(ws.subC, t, p1, p2, p3, segments, domain)
    end

    compute_triangle_geometry!(ws.geom, triXC, triYC, triZC; domain=domain)
    return nothing
end

function copy_periodic_boundaries!(A::AbstractArray{T,3}) where T
    @views begin
        A[end, :, :] .= A[1, :, :]
        A[:, end, :] .= A[:, 1, :]
        A[:, :, end] .= A[:, :, 1]
    end
    return A
end

function spread_vorticity_to_grid_mpi!(ws::VortexWorkspace{T},
                                       eleGma::AbstractMatrix{T},
                                       triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                       domain::DomainSpec, gr::GridSpec) where T<:AbstractFloat
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if ws.geom_dirty[]
        _recompute_geometry!(ws, triXC, triYC, triZC, domain, gr)
        ws.geom_dirty[] = false
    end

    dx, dy, dz = grid_spacing(domain, gr)
    nx, ny, nz = gr.nx, gr.ny, gr.nz
    nsub = ws.nsub[]
    inv_cell_volume = one(T) / T(dx * dy * dz)
    delr = T(4)
    eps = SVector{3,T}(delr * T(dx), delr * T(dy), delr * T(dz))

    fill!(ws.local_buf, zero(T))

    idx = 1
    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if mod(idx - 1, nprocs) == rank
            coord = SVector{3,T}(T(i - 1) * T(dx),
                                 T(j - 1) * T(dy),
                                 -T(domain.Lz) + T(k - 1) * T(dz))
            acc = SVector{3,T}(zero(T), zero(T), zero(T))
            for shift in ws.shifts
                acc = peskin_add_nearby_nsub!(acc, eleGma, ws.triC, ws.subC,
                                              ws.areas, nsub, coord, delr, eps, shift)
            end
            ws.local_buf[idx,1] = acc[1] * inv_cell_volume
            ws.local_buf[idx,2] = acc[2] * inv_cell_volume
            ws.local_buf[idx,3] = acc[3] * inv_cell_volume
        end
        idx += 1
    end

    MPI.Allreduce!(ws.local_buf, ws.global_buf, MPI.SUM, comm)

    @inbounds for idx in 1:(nx * ny * nz)
        ws.ζx[idx] = ws.global_buf[idx,1]
        ws.ζy[idx] = ws.global_buf[idx,2]
        ws.ζz[idx] = ws.global_buf[idx,3]
    end

    copy_periodic_boundaries!(ws.ζx)
    copy_periodic_boundaries!(ws.ζy)
    copy_periodic_boundaries!(ws.ζz)
    return nothing
end

function interpolate_node_velocity_mpi!(ws::VortexWorkspace{T},
                                        gridUx::Array{T,3}, gridUy::Array{T,3}, gridUz::Array{T,3},
                                        nodeX::AbstractVector{T}, nodeY::AbstractVector{T},
                                        nodeZ::AbstractVector{T},
                                        domain::DomainSpec, gr::GridSpec) where T<:AbstractFloat
    return interpolate_node_velocity_mpi!(ws.u1, ws.v1, ws.w1, ws, gridUx, gridUy, gridUz,
                                          nodeX, nodeY, nodeZ, domain, gr)
end

function interpolate_node_velocity_mpi!(out_u::AbstractVector{T},
                                        out_v::AbstractVector{T},
                                        out_w::AbstractVector{T},
                                        ws::VortexWorkspace{T},
                                        gridUx::Array{T,3}, gridUy::Array{T,3}, gridUz::Array{T,3},
                                        nodeX::AbstractVector{T}, nodeY::AbstractVector{T},
                                        nodeZ::AbstractVector{T},
                                        domain::DomainSpec, gr::GridSpec) where T<:AbstractFloat
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nx, ny, nz = gr.nx, gr.ny, gr.nz
    dx, dy, dz = grid_spacing(domain, gr)
    delr = T(4)
    epsx = delr * T(dx)
    epsy = delr * T(dy)
    epsz = delr * T(dz)
    scale = one(T) / (T(8) * delr^3)

    N = length(nodeX)
    fill!(ws.local_buf, zero(T))

    @inbounds for i in (rank + 1):nprocs:N
        xc = nodeX[i]
        yc = nodeY[i]
        zc = nodeZ[i]
        sx = zero(T)
        sy = zero(T)
        sz = zero(T)
        for shift in ws.shifts
            xq = xc - shift[1]
            yq = yc - shift[2]
            zq = zc - shift[3]
            for k in 1:nz, j in 1:ny, ii in 1:nx
                dxv = xq - T(ii - 1) * T(dx)
                dyv = yq - T(j - 1) * T(dy)
                dzv = zq - (-T(domain.Lz) + T(k - 1) * T(dz))
                if abs(dxv) <= epsx && abs(dyv) <= epsy && abs(dzv) <= epsz
                    w = (one(T) + cos(T(π) * dxv / epsx)) *
                        (one(T) + cos(T(π) * dyv / epsy)) *
                        (one(T) + cos(T(π) * dzv / epsz)) * scale
                    sx += gridUx[k,j,ii] * w
                    sy += gridUy[k,j,ii] * w
                    sz += gridUz[k,j,ii] * w
                end
            end
        end
        ws.local_buf[i,1] = sx
        ws.local_buf[i,2] = sy
        ws.local_buf[i,3] = sz
    end

    MPI.Allreduce!(ws.local_buf, ws.global_buf, MPI.SUM, comm)

    @inbounds for i in 1:N
        out_u[i] = ws.global_buf[i,1]
        out_v[i] = ws.global_buf[i,2]
        out_w[i] = ws.global_buf[i,3]
    end
    return nothing
end

# MPI-parallel: spread element vorticity to grid with kernel selection
function spread_vorticity_to_grid_kernel_mpi(eleGma::AbstractMatrix,
                                        triXC::AbstractMatrix,
                                        triYC::AbstractMatrix,
                                        triZC::AbstractMatrix,
                                        domain::DomainSpec,
                                        gr::GridSpec,
                                        kernel::KernelType=PeskinStandard();
                                        subsegments::Union{Nothing,Int}=nothing,
                                        max_subsegments::Int=8)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    x, y, z = grid_vectors(domain, gr)
    (dx, dy, dz) = grid_spacing(domain, gr)
    segments = subsegments === nothing ?
               subtriangle_segment_count(triXC, triYC, triZC, (dx, dy, dz);
                                         domain=domain, max_segments=max_subsegments) :
               subsegments
    triC = triangle_centroids(triXC, triYC, triZC; domain=domain)
    subC = build_all_subcentroids(triXC, triYC, triZC; domain=domain, subsegments=segments)
    areas = triangle_areas(triXC, triYC, triZC; domain=domain)
    nx, ny, nz = gr.nx, gr.ny, gr.nz

    # Local buffers
    local_buf = zeros(Float64, nx*ny*nz, 3)
    inv_cell_volume = 1.0 / (dx*dy*dz)
    idx = 1
    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if mod(idx - 1, nprocs) == rank
            sx,sy,sz = peskin_grid_sum_kernel(eleGma, triC, subC, (x[i], y[j], z[k]), (dx,dy,dz), areas, kernel; domain=domain)
            local_buf[idx,1] = sx * inv_cell_volume
            local_buf[idx,2] = sy * inv_cell_volume
            local_buf[idx,3] = sz * inv_cell_volume
        end
        idx += 1
    end

    # Reduce across ranks
    global_buf = similar(local_buf)
    MPI.Allreduce!(local_buf, global_buf, MPI.SUM, comm)

    # Reshape to (nz,ny,nx) - column-major means nz varies fastest, matching loop order
    ζx = reshape(view(global_buf,:,1), nz, ny, nx)
    ζy = reshape(view(global_buf,:,2), nz, ny, nx)
    ζz = reshape(view(global_buf,:,3), nz, ny, nx)

    copy_periodic_boundaries!(ζx)
    copy_periodic_boundaries!(ζy)
    copy_periodic_boundaries!(ζz)

    return ζx, ζy, ζz
end

# MPI-parallel: spread element vorticity to grid (original function)
function spread_vorticity_to_grid_mpi(eleGma::AbstractMatrix,
                                      triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                      domain::DomainSpec, gr::GridSpec;
                                      subsegments::Union{Nothing,Int}=nothing,
                                      max_subsegments::Int=8)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    x, y, z   = grid_vectors(domain, gr)
    (dx, dy, dz) = grid_spacing(domain, gr)
    segments = subsegments === nothing ?
               subtriangle_segment_count(triXC, triYC, triZC, (dx, dy, dz);
                                         domain=domain, max_segments=max_subsegments) :
               subsegments
    triC = triangle_centroids(triXC, triYC, triZC; domain=domain)
    subC = build_all_subcentroids(triXC, triYC, triZC; domain=domain, subsegments=segments)
    areas = triangle_areas(triXC, triYC, triZC; domain=domain)

    # Flatten grid in order (nz,ny,nx) as in python final reshape
    nx, ny, nz = gr.nx, gr.ny, gr.nz

    # Local buffers
    local_buf = zeros(Float64, nx*ny*nz, 3)
    inv_cell_volume = 1.0 / (dx*dy*dz)
    idx = 1
    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if mod(idx - 1, nprocs) == rank
            sx,sy,sz = peskin_grid_sum(eleGma, triC, subC, (x[i], y[j], z[k]), (dx,dy,dz), areas; domain=domain)
            local_buf[idx,1] = sx * inv_cell_volume
            local_buf[idx,2] = sy * inv_cell_volume
            local_buf[idx,3] = sz * inv_cell_volume
        end
        idx += 1
    end

    # Reduce across ranks (sum), even though strided fill is disjoint this is safe
    global_buf = similar(local_buf)
    MPI.Allreduce!(local_buf, global_buf, MPI.SUM, comm)

    # Reshape to (nz,ny,nx)
    ζx = reshape(view(global_buf,:,1), nz, ny, nx)
    ζy = reshape(view(global_buf,:,2), nz, ny, nx)
    ζz = reshape(view(global_buf,:,3), nz, ny, nx)

    copy_periodic_boundaries!(ζx)
    copy_periodic_boundaries!(ζy)
    copy_periodic_boundaries!(ζz)

    return ζx, ζy, ζz
end

# Enhanced interpolation with kernel selection
function interpolate_node_velocity_kernel_mpi(gridUx::Array{Float64,3}, 
                                            gridUy::Array{Float64,3}, 
                                            gridUz::Array{Float64,3},
                                            nodeX::AbstractVector, 
                                            nodeY::AbstractVector, 
                                            nodeZ::AbstractVector,
                                            domain::DomainSpec, 
                                            gr::GridSpec, 
                                            kernel::KernelType=PeskinStandard())
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nx,ny,nz = gr.nx, gr.ny, gr.nz
    (dx,dy,dz) = grid_spacing(domain, gr)
    x,y,z = grid_vectors(domain, gr)

    delr = kernel_support_radius(kernel)
    epsx, epsy, epsz = delr*dx, delr*dy, delr*dz

    # local node buffers
    N = length(nodeX)
    local_buf = zeros(Float64, N, 3)

    tiles = periodic_shifts(domain)

    @inbounds for i in (rank+1):nprocs:N
        xc,yc,zc = nodeX[i], nodeY[i], nodeZ[i]
        sx=0.0; sy=0.0; sz=0.0
        for (dxL,dyL,dzL) in tiles
            xq = xc - dxL; yq = yc - dyL; zq = zc - dzL
            for k in 1:nz, j in 1:ny, ii in 1:nx
                dxv = xq - x[ii]
                dyv = yq - y[j]
                dzv = zq - z[k]
                if abs(dxv) <= epsx && abs(dyv) <= epsy && abs(dzv) <= epsz
                    w = interpolate_kernel_weight(kernel, dxv, dyv, dzv, dx, dy, dz)
                    sx += gridUx[k,j,ii]*w
                    sy += gridUy[k,j,ii]*w
                    sz += gridUz[k,j,ii]*w
                end
            end
        end
        local_buf[i,1]=sx; local_buf[i,2]=sy; local_buf[i,3]=sz
    end

    global_buf = similar(local_buf)
    MPI.Allreduce!(local_buf, global_buf, MPI.SUM, comm)
    return view(global_buf,:,1), view(global_buf,:,2), view(global_buf,:,3)
end

# Interpolate node velocities from grid (MPI parallel over nodes) - original function
function interpolate_node_velocity_mpi(gridUx::Array{Float64,3}, 
                                    gridUy::Array{Float64,3}, 
                                    gridUz::Array{Float64,3},
                                    nodeX::AbstractVector, 
                                    nodeY::AbstractVector, 
                                    nodeZ::AbstractVector,
                                    domain::DomainSpec, gr::GridSpec)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nx,ny,nz = gr.nx, gr.ny, gr.nz
    (dx,dy,dz) = grid_spacing(domain, gr)
    x,y,z = grid_vectors(domain, gr)

    delr = 4.0
    epsx,epsy,epsz = delr*dx, delr*dy, delr*dz

    # local node buffers
    N = length(nodeX)
    local_buf = zeros(Float64, N, 3)

    tiles = periodic_shifts(domain)

    @inbounds for i in (rank+1):nprocs:N
        xc,yc,zc = nodeX[i], nodeY[i], nodeZ[i]
        sx=0.0; sy=0.0; sz=0.0
        for (dxL,dyL,dzL) in tiles
            # find neighbors relative to shifted tile by querying around (xc-dxL, yc-dyL, zc)
            xq = xc - dxL; yq = yc - dyL; zq = zc - dzL
            for k in 1:nz, j in 1:ny, ii in 1:nx
                dxv = xq - x[ii]
                dyv = yq - y[j]
                dzv = zq - z[k]
                if abs(dxv) <= epsx && abs(dyv) <= epsy && abs(dzv) <= epsz
                    w = (1 + cos(pi*dxv/epsx))*(1 + cos(pi*dyv/epsy))*(1 + cos(pi*dzv/epsz)) / (8*delr^3)
                    sx += gridUx[k,j,ii]*w
                    sy += gridUy[k,j,ii]*w
                    sz += gridUz[k,j,ii]*w
                end
            end
        end
        local_buf[i,1]=sx; local_buf[i,2]=sy; local_buf[i,3]=sz
    end

    global_buf = similar(local_buf)
    MPI.Allreduce!(local_buf, global_buf, MPI.SUM, comm)
    return view(global_buf,:,1), view(global_buf,:,2), view(global_buf,:,3)
end

end # module

using .Peskin3D: init_mpi!, finalize_mpi!, triangle_centroids, triangle_areas, subtriangle_centroids, subtriangle_centroids4,
                          spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_mpi!,
                          interpolate_node_velocity_mpi, interpolate_node_velocity_mpi!,
                          spread_vorticity_to_grid_kernel_mpi, interpolate_node_velocity_kernel_mpi,
                          find_elements_nearby!
