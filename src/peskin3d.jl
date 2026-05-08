# 3D Peskin spreading and interpolation with MPI

module Peskin3D

using ..DomainImpl
using ..Kernels
using MPI

export init_mpi!, finalize_mpi!,
       triangle_centroids, triangle_areas,
       subtriangle_centroids, subtriangle_centroids4,
       spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_kernel_mpi,
       interpolate_node_velocity_mpi, interpolate_node_velocity_kernel_mpi

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

@inline function bary_point(a, b, c, i::Int, j::Int, M::Int)
    rb = i / M
    rc = j / M
    return (a[1] + rb * (b[1] - a[1]) + rc * (c[1] - a[1]),
            a[2] + rb * (b[2] - a[2]) + rc * (c[2] - a[2]),
            a[3] + rb * (b[3] - a[3]) + rc * (c[3] - a[3]))
end

@inline function centroid3(a, b, c)
    return ((a[1] + b[1] + c[1]) / 3,
            (a[2] + b[2] + c[2]) / 3,
            (a[3] + b[3] + c[3]) / 3)
end

@inline function maybe_wrap(c, domain::Union{Nothing,DomainSpec})
    domain === nothing && return c
    return wrap_point(c[1], c[2], c[3], domain)
end

function subtriangle_centroids(p1::NTuple{3,Float64}, p2::NTuple{3,Float64},
                               p3::NTuple{3,Float64}, M::Integer;
                               domain::Union{Nothing,DomainSpec}=nothing)
    M >= 1 || throw(ArgumentError("subtriangle segment count M must be >= 1"))
    a, b, c = domain === nothing ? (p1, p2, p3) : unwrap_triangle(p1, p2, p3, domain)
    T = Array{Float64}(undef, M * M, 3)
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
            T[idx, :] .= ct
            idx += 1
        end
        return T
    end

    @inbounds for i in 0:M-1, j in 0:M-1-i
        v1 = bary_point(a, b, c, i, j, M)
        v2 = bary_point(a, b, c, i + 1, j, M)
        v3 = bary_point(a, b, c, i, j + 1, M)
        ct = maybe_wrap(centroid3(v1, v2, v3), domain)
        T[idx, :] .= ct
        idx += 1
        if i + j <= M - 2
            v1 = bary_point(a, b, c, i + 1, j, M)
            v2 = bary_point(a, b, c, i + 1, j + 1, M)
            v3 = bary_point(a, b, c, i, j + 1, M)
            ct = maybe_wrap(centroid3(v1, v2, v3), domain)
            T[idx, :] .= ct
            idx += 1
        end
    end
    return T
end

subtriangle_centroids4(p1::NTuple{3,Float64}, p2::NTuple{3,Float64}, p3::NTuple{3,Float64}) =
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

function peskin_add_nearby!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix,
                            triC::AbstractMatrix, subC, triAreas,
                            coord::NTuple{3,Float64}, delr::Float64,
                            eps::NTuple{3,Float64},
                            shiftx::Float64, shifty::Float64, shiftz::Float64)
    sx, sy, sz = sum
    x, y, z = coord
    epsx, epsy, epsz = eps
    n_sub = size(subC, 2)
    inv_n_sub = 1.0 / n_sub
    scale = 1.0 / (8.0 * delr^3)

    @inbounds for idx in 1:size(triC, 1)
        cx = triC[idx,1] + shiftx
        cy = triC[idx,2] + shifty
        cz = triC[idx,3] + shiftz
        if abs(cx - x) <= epsx && abs(cy - y) <= epsy && abs(cz - z) <= epsz
            S = 0.0
            @simd for s in 1:n_sub
                dx = x - (subC[idx,s,1] + shiftx)
                dy = y - (subC[idx,s,2] + shifty)
                dz = z - (subC[idx,s,3] + shiftz)
                S += (1 + cos(pi*dx/epsx))*(1 + cos(pi*dy/epsy))*(1 + cos(pi*dz/epsz)) * scale
            end
            w = triAreas[idx] * S * inv_n_sub
            sx += w * eleGma[idx,1]
            sy += w * eleGma[idx,2]
            sz += w * eleGma[idx,3]
        end
    end
    return sx, sy, sz
end

function peskin_add_nearby_kernel!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix,
                                   triC::AbstractMatrix, subC, triAreas,
                                   coord::NTuple{3,Float64}, kernel::KernelType,
                                   eps::NTuple{3,Float64},
                                   shiftx::Float64, shifty::Float64, shiftz::Float64)
    sx, sy, sz = sum
    x, y, z = coord
    epsx, epsy, epsz = eps
    delr = kernel_support_radius(kernel)
    hx, hy, hz = epsx/delr, epsy/delr, epsz/delr
    n_sub = size(subC, 2)
    inv_n_sub = 1.0 / n_sub

    @inbounds for idx in 1:size(triC, 1)
        cx = triC[idx,1] + shiftx
        cy = triC[idx,2] + shifty
        cz = triC[idx,3] + shiftz
        if abs(cx - x) <= epsx && abs(cy - y) <= epsy && abs(cz - z) <= epsz
            S = 0.0
            @simd for s in 1:n_sub
                dx = x - (subC[idx,s,1] + shiftx)
                dy = y - (subC[idx,s,2] + shifty)
                dz = z - (subC[idx,s,3] + shiftz)
                S += kernel_function(kernel, dx, dy, dz, hx, hy, hz)
            end
            w = triAreas[idx] * S * inv_n_sub
            sx += w * eleGma[idx,1]
            sy += w * eleGma[idx,2]
            sz += w * eleGma[idx,3]
        end
    end
    return sx, sy, sz
end

# Accumulate vorticity at a coordinate from periodic tiles (9 tiles in xy like python)
function peskin_grid_sum(eleGma, triC, subC, coord, ds, triAreas; delr=4.0, 
                        domain::DomainSpec=default_domain())
    eps = (delr*ds[1], delr*ds[2], delr*ds[3]) # ds tuple indexing (dx,dy,dz)
    (sx,sy,sz) = (0.0,0.0,0.0)
    for (shiftx, shifty, shiftz) in periodic_shifts(domain)
        (sx, sy, sz) = peskin_add_nearby!((sx,sy,sz), eleGma, triC, subC, triAreas, coord,
                                          delr, eps, shiftx, shifty, shiftz)
    end
    return sx, sy, sz
end

# Enhanced grid sum with kernel selection
function peskin_grid_sum_kernel(eleGma, triC, subC, coord, ds, triAreas, 
                        kernel::KernelType; domain::DomainSpec=default_domain())
    delr = kernel_support_radius(kernel)
    eps = (delr*ds[1], delr*ds[2], delr*ds[3])
    (sx,sy,sz) = (0.0,0.0,0.0)
    for (shiftx, shifty, shiftz) in periodic_shifts(domain)
        (sx, sy, sz) = peskin_add_nearby_kernel!((sx,sy,sz), eleGma, triC, subC, triAreas, coord,
                                                 kernel, eps, shiftx, shifty, shiftz)
    end
    return sx, sy, sz
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
    VorX = reshape(view(global_buf,:,1), nz, ny, nx)
    VorY = reshape(view(global_buf,:,2), nz, ny, nx)
    VorZ = reshape(view(global_buf,:,3), nz, ny, nx)

    # Periodic wrap
    VorX[end, :, :] .= VorX[1, :, :]
    VorY[end, :, :] .= VorY[1, :, :]
    VorZ[end, :, :] .= VorZ[1, :, :]

    VorX[:, end, :] .= VorX[:, 1, :]
    VorY[:, end, :] .= VorY[:, 1, :]
    VorZ[:, end, :] .= VorZ[:, 1, :]

    VorX[:, :, end] .= VorX[:, :, 1]
    VorY[:, :, end] .= VorY[:, :, 1]
    VorZ[:, :, end] .= VorZ[:, :, 1]

    return VorX, VorY, VorZ
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
    VorX = reshape(view(global_buf,:,1), nz, ny, nx)
    VorY = reshape(view(global_buf,:,2), nz, ny, nx)
    VorZ = reshape(view(global_buf,:,3), nz, ny, nx)

    # Periodic wrap like python
    VorX[end, :, :] .= VorX[1, :, :]
    VorY[end, :, :] .= VorY[1, :, :]
    VorZ[end, :, :] .= VorZ[1, :, :]

    VorX[:, end, :] .= VorX[:, 1, :]
    VorY[:, end, :] .= VorY[:, 1, :]
    VorZ[:, end, :] .= VorZ[:, 1, :]

    VorX[:, :, end] .= VorX[:, :, 1]
    VorY[:, :, end] .= VorY[:, :, 1]
    VorZ[:, :, end] .= VorZ[:, :, 1]

    return VorX, VorY, VorZ
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
                          spread_vorticity_to_grid_mpi, interpolate_node_velocity_mpi,
                          spread_vorticity_to_grid_kernel_mpi, interpolate_node_velocity_kernel_mpi
