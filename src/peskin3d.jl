# 3D Peskin spreading and interpolation with MPI

module Peskin3D

using ..DomainImpl
using ..Kernels
using MPI

export init_mpi!, finalize_mpi!,
       triangle_centroids, triangle_areas,
       subtriangle_centroids4,
       spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_kernel_mpi,
       interpolate_node_velocity_mpi, interpolate_node_velocity_kernel_mpi

init_mpi!() = (MPI.Initialized() || MPI.Init(); nothing)
finalize_mpi!() = (MPI.Finalized() || MPI.Finalize(); nothing)

# Basic geometry helpers
triangle_centroids(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix) = begin
    nt = size(triXC, 1)
    C = zeros(Float64, nt, 3)
    @inbounds for t in 1:nt
        C[t, 1] = (triXC[t, 1] + triXC[t, 2] + triXC[t, 3]) / 3
        C[t, 2] = (triYC[t, 1] + triYC[t, 2] + triYC[t, 3]) / 3
        C[t, 3] = (triZC[t, 1] + triZC[t, 2] + triZC[t, 3]) / 3
    end
    C
end

function triangle_areas(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    nt = size(triXC, 1)
    A = zeros(Float64, nt)
    @inbounds for t in 1:nt
        p1 = (triXC[t, 1], triYC[t, 1], triZC[t, 1])
        p2 = (triXC[t, 2], triYC[t, 2], triZC[t, 2])
        p3 = (triXC[t, 3], triYC[t, 3], triZC[t, 3])
        a = hypot(hypot(p1[1]-p2[1], p1[2]-p2[2]), p1[3]-p2[3])
        b = hypot(hypot(p2[1]-p3[1], p2[2]-p3[2]), p2[3]-p3[3])
        c = hypot(hypot(p3[1]-p1[1], p3[2]-p1[2]), p3[3]-p1[3])
        s = (a+b+c)/2
        A[t] = sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
    end
    A
end

# Split a triangle into 4 and return centroids (to mirror python _centroid_4subTriangle_)
function subtriangle_centroids4(p1::NTuple{3,Float64}, p2::NTuple{3,Float64}, p3::NTuple{3,Float64})

    ds12 = ((p2[1]-p1[1])/2, (p2[2]-p1[2])/2, (p2[3]-p1[3])/2)
    ds23 = ((p3[1]-p2[1])/2, (p3[2]-p2[2])/2, (p3[3]-p2[3])/2)
    ds31 = ((p1[1]-p3[1])/2, (p1[2]-p3[2])/2, (p1[3]-p3[3])/2)

    # 4 sub-triangles (same as python)
    T = Array{Float64}(undef, 4, 3)
    
    # tri 1: p1, p1+ds12, p1-ds31
    v1 = p1
    v2 = (p1[1]+ds12[1], p1[2]+ds12[2], p1[3]+ds12[3])
    v3 = (p1[1]-ds31[1], p1[2]-ds31[2], p1[3]-ds31[3])
    T[1,:] = [(v1[1]+v2[1]+v3[1])/3, (v1[2]+v2[2]+v3[2])/3, (v1[3]+v2[3]+v3[3])/3]
    # tri 2: p1+ds12, p2+ds23, p1-ds31
    v1 = (p1[1]+ds12[1], p1[2]+ds12[2], p1[3]+ds12[3])
    v2 = (p2[1]+ds23[1], p2[2]+ds23[2], p2[3]+ds23[3])
    v3 = (p1[1]-ds31[1], p1[2]-ds31[2], p1[3]-ds31[3])
    T[2,:] = [(v1[1]+v2[1]+v3[1])/3, (v1[2]+v2[2]+v3[2])/3, (v1[3]+v2[3]+v3[3])/3]
    # tri 3: p2, p2+ds23, p1+ds12
    v1 = p2
    v2 = (p2[1]+ds23[1], p2[2]+ds23[2], p2[3]+ds23[3])
    v3 = (p1[1]+ds12[1], p1[2]+ds12[2], p1[3]+ds12[3])
    T[3,:] = [(v1[1]+v2[1]+v3[1])/3, (v1[2]+v2[2]+v3[2])/3, (v1[3]+v2[3]+v3[3])/3]
    # tri 4: p2+ds23, p3, p1-ds31
    v1 = (p2[1]+ds23[1], p2[2]+ds23[2], p2[3]+ds23[3])
    v2 = p3
    v3 = (p1[1]-ds31[1], p1[2]-ds31[2], p1[3]-ds31[3])
    T[4,:] = [(v1[1]+v2[1]+v3[1])/3, (v1[2]+v2[2]+v3[2])/3, (v1[3]+v2[3]+v3[3])/3]
    return T
end

# Build 4-subtriangle centroids for all triangles
function build_all_subcentroids(triXC, triYC, triZC)
    nt = size(triXC,1)
    C = Array{Float64}(undef, nt, 4, 3)
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        T = subtriangle_centroids4(p1,p2,p3) # 4 x 3
        C[t, :, 1] = view(T, :, 1)
        C[t, :, 2] = view(T, :, 2)
        C[t, :, 3] = view(T, :, 3)
    end
    C
end

# Find elements near a point within epsx,epsy,epsz (Python _find_elements_nearby_)
function find_elements_nearby(x,y,z, epsx,epsy,epsz, triC::AbstractMatrix)
    nt = size(triC,1)
    # simple vectorized test using broadcasting
    maskx = abs.(triC[:,1] .- x) .<= epsx
    masky = abs.(triC[:,2] .- y) .<= epsy
    maskz = abs.(triC[:,3] .- z) .<= epsz
    findall(maskx .& masky .& maskz)
end

# Periodic image helper: shift tri centroids and subcentroids by tile offsets
function shifted_centroids(triC, subC, dx::Tuple{Float64,Float64})
    Lx, Ly = dx
    triC0 = copy(triC)
    subC0 = copy(subC)
    triC0[:,1] .+= Lx
    subC0[:,:,1] .+= Lx
    return triC0, subC0
end

# Core Peskin sum for a list of triangles (grid vorticity accumulation)
function peskin_add_ele!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix, subC, triAreas, tri_list, coord, delr, eps)
    sx, sy, sz = sum
    (epsx,epsy,epsz) = eps
    x = coord
    @inbounds for idx in tri_list
        S = 0.0
        for s in 1:size(subC,2)
            dx = x[1] - subC[idx,s,1]
            dy = x[2] - subC[idx,s,2]
            dz = x[3] - subC[idx,s,3]
            tmp = (1 + cos(pi*dx/epsx))*(1 + cos(pi*dy/epsy))*(1 + cos(pi*dz/epsz)) / (8*delr^3)
            S += tmp
        end
        w = triAreas[idx] * (S/size(subC,2))
        sx += w * eleGma[idx,1]
        sy += w * eleGma[idx,2]
        sz += w * eleGma[idx,3]
    end
    return (sx,sy,sz)
end

# Accumulate vorticity at a coordinate from periodic tiles (9 tiles in xy like python)
function peskin_grid_sum(eleGma, triC, subC, coord, ds, triAreas; delr=4.0, dom::DomainSpec=default_domain())
    eps = (delr*ds[1], delr*ds[2], delr*ds[3]) # ds tuple indexing (dx,dy,dz)
    (sx,sy,sz) = (0.0,0.0,0.0)
    # Middle tile
    tri_list = find_elements_nearby(coord[1], coord[2], coord[3], eps... , triC)
    (sx,sy,sz) = peskin_add_ele!((sx,sy,sz), eleGma, subC, triAreas, tri_list, coord, delr, eps)
    # Neighbor tiles (E,W,N,S, and corners)
    Lx,Ly,Lz = dom.Lx, dom.Ly, dom.Lz
    function shifted(tile::Tuple{Float64,Float64})
        dx,dy = tile
        triC0 = copy(triC); subC0 = copy(subC)
        triC0[:,1] .+= dx; triC0[:,2] .+= dy
        subC0[:,:,1] .+= dx; subC0[:,:,2] .+= dy
        tl = find_elements_nearby(coord[1],coord[2],coord[3], eps... , triC0)
        peskin_add_ele!((sx,sy,sz), eleGma, subC0, triAreas, tl, coord, delr, eps)
    end
    # E, W, N, S
    (sx,sy,sz) = shifted((+Lx, 0.0)); (sx,sy,sz) = shifted((-Lx, 0.0))
    (sx,sy,sz) = shifted((0.0, +Ly)); (sx,sy,sz) = shifted((0.0, -Ly))
    # Corners
    (sx,sy,sz) = shifted((+Lx,+Ly)); (sx,sy,sz) = shifted((+Lx,-Ly))
    (sx,sy,sz) = shifted((-Lx,+Ly)); (sx,sy,sz) = shifted((-Lx,-Ly))
    return sx,sy,sz
end

# Enhanced grid sum with kernel selection
function peskin_grid_sum_kernel(eleGma, triC, subC, coord, ds, triAreas, kernel::KernelType; dom::DomainSpec=default_domain())
    delr = kernel_support_radius(kernel)
    eps = (delr*ds[1], delr*ds[2], delr*ds[3])
    (sx,sy,sz) = (0.0,0.0,0.0)
    
    # Middle tile
    tri_list = find_elements_nearby(coord[1], coord[2], coord[3], eps... , triC)
    (sx,sy,sz) = spread_element_kernel!((sx,sy,sz), eleGma, subC, triAreas, tri_list, coord, kernel, eps)
    
    # Neighbor tiles (E,W,N,S, and corners)
    Lx,Ly,Lz = dom.Lx, dom.Ly, dom.Lz
    function shifted(tile::Tuple{Float64,Float64})
        dx,dy = tile
        triC0 = copy(triC); subC0 = copy(subC)
        triC0[:,1] .+= dx; triC0[:,2] .+= dy
        subC0[:,:,1] .+= dx; subC0[:,:,2] .+= dy
        tl = find_elements_nearby(coord[1],coord[2],coord[3], eps... , triC0)
        spread_element_kernel!((sx,sy,sz), eleGma, subC0, triAreas, tl, coord, kernel, eps)
    end
    # E, W, N, S
    (sx,sy,sz) = shifted((+Lx, 0.0)); (sx,sy,sz) = shifted((-Lx, 0.0))
    (sx,sy,sz) = shifted((0.0, +Ly)); (sx,sy,sz) = shifted((0.0, -Ly))
    # Corners
    (sx,sy,sz) = shifted((+Lx,+Ly)); (sx,sy,sz) = shifted((+Lx,-Ly))
    (sx,sy,sz) = shifted((-Lx,+Ly)); (sx,sy,sz) = shifted((-Lx,-Ly))
    return sx,sy,sz
end

# MPI-parallel: spread element vorticity to grid with kernel selection
function spread_vorticity_to_grid_kernel_mpi(eleGma::AbstractMatrix,
                                            triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                            dom::DomainSpec, gr::GridSpec, kernel::KernelType=PeskinStandard())
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nt = size(triXC,1)
    triC = triangle_centroids(triXC, triYC, triZC)
    subC = build_all_subcentroids(triXC, triYC, triZC)
    areas = triangle_areas(triXC, triYC, triZC)

    x, y, z = grid_vectors(dom, gr)
    (dx,dy,dz) = grid_spacing(dom, gr)
    nx,ny,nz = gr.nx, gr.ny, gr.nz
    
    coords = Vector{NTuple{3,Float64}}(undef, nx*ny*nz)
    idx = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        coords[idx] = (x[i], y[j], z[k]); idx+=1
    end

    # Local buffers
    local = zeros(Float64, nx*ny*nz, 3)
    # strided work splitting
    @inbounds for idx in (rank+1):nprocs:length(coords)
        c = coords[idx]
        sx,sy,sz = peskin_grid_sum_kernel(eleGma, triC, subC, c, (dx,dy,dz), areas, kernel; dom=dom)
        # divide by cell volume
        local[idx,1] = sx/(dx*dy*dz)
        local[idx,2] = sy/(dx*dy*dz)
        local[idx,3] = sz/(dx*dy*dz)
    end

    # Reduce across ranks
    global = similar(local)
    MPI.Allreduce!(local, global, MPI.SUM, comm)

    # Reshape to (nz,ny,nx)
    VorX = reshape(view(global,:,1), nz, ny, nx)
    VorY = reshape(view(global,:,2), nz, ny, nx)
    VorZ = reshape(view(global,:,3), nz, ny, nx)

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
                                      dom::DomainSpec, gr::GridSpec)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nt = size(triXC,1)
    triC = triangle_centroids(triXC, triYC, triZC)
    subC = build_all_subcentroids(triXC, triYC, triZC)
    areas = triangle_areas(triXC, triYC, triZC)

    x, y, z = grid_vectors(dom, gr)
    (dx,dy,dz) = grid_spacing(dom, gr)
    # Flatten grid in order (nz,ny,nx) as in python final reshape
    nx,ny,nz = gr.nx, gr.ny, gr.nz
    # Build flattened coordinates in same order used later: we’ll use (k,j,i)
    coords = Vector{NTuple{3,Float64}}(undef, nx*ny*nz)
    idx = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        coords[idx] = (x[i], y[j], z[k]); idx+=1
    end

    # Local buffers
    local = zeros(Float64, nx*ny*nz, 3)
    # strided work splitting (round-robin like python)
    @inbounds for idx in (rank+1):nprocs:length(coords)
        c = coords[idx]
        sx,sy,sz = peskin_grid_sum(eleGma, triC, subC, c, (dx,dy,dz), areas; dom=dom)
        # divide by cell volume like python
        local[idx,1] = sx/(dx*dy*dz)
        local[idx,2] = sy/(dx*dy*dz)
        local[idx,3] = sz/(dx*dy*dz)
    end

    # Reduce across ranks (sum), even though strided fill is disjoint this is safe
    global = similar(local)
    MPI.Allreduce!(local, global, MPI.SUM, comm)

    # Reshape to (nz,ny,nx)
    VorX = reshape(view(global,:,1), nz, ny, nx)
    VorY = reshape(view(global,:,2), nz, ny, nx)
    VorZ = reshape(view(global,:,3), nz, ny, nx)

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
function interpolate_node_velocity_kernel_mpi(gridUx::Array{Float64,3}, gridUy::Array{Float64,3}, gridUz::Array{Float64,3},
                                              nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                                              dom::DomainSpec, gr::GridSpec, kernel::KernelType=PeskinStandard())
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nx,ny,nz = gr.nx, gr.ny, gr.nz
    (dx,dy,dz) = grid_spacing(dom, gr)
    # Build flattened grid arrays and coordinates
    x,y,z = grid_vectors(dom, gr)
    coords = Vector{NTuple{3,Float64}}(undef, nx*ny*nz)
    flatUx = Vector{Float64}(undef, nx*ny*nz)
    flatUy = similar(flatUx)
    flatUz = similar(flatUx)
    idx = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        coords[idx] = (x[i], y[j], z[k])
        flatUx[idx] = gridUx[k,j,i]
        flatUy[idx] = gridUy[k,j,i]
        flatUz[idx] = gridUz[k,j,i]
        idx+=1
    end

    delr = kernel_support_radius(kernel)
    epsx,epsy,epsz = delr*dx, delr*dy, delr*dz

    # local node buffers
    N = length(nodeX)
    local = zeros(Float64, N, 3)

    function nearby_grid(xc,yc,zc)
        maskx = abs.(getindex.(coords,1) .- xc) .<= epsx
        masky = abs.(getindex.(coords,2) .- yc) .<= epsy
        maskz = abs.(getindex.(coords,3) .- zc) .<= epsz
        findall(maskx .& masky .& maskz)
    end

    @inbounds for i in (rank+1):nprocs:N
        xc,yc,zc = nodeX[i], nodeY[i], nodeZ[i]
        sx=0.0; sy=0.0; sz=0.0
        # 9 tiles
        tiles = ((0.0,0.0),(+dom.Lx,0.0),(-dom.Lx,0.0),(0.0,+dom.Ly),(0.0,-dom.Ly),
                 (+dom.Lx,+dom.Ly),(+dom.Lx,-dom.Ly),(-dom.Lx,+dom.Ly),(-dom.Lx,-dom.Ly))
        for (dxL,dyL) in tiles
            xq = xc - dxL; yq = yc - dyL
            inds = nearby_grid(xq, yq, zc)
            for idxg in inds
                gx,gy,gz = coords[idxg]
                dxv = xq - gx
                dyv = yq - gy
                dzv = zc - gz
                w = interpolate_kernel_weight(kernel, dxv, dyv, dzv, dx, dy, dz)
                sx += flatUx[idxg]*w
                sy += flatUy[idxg]*w
                sz += flatUz[idxg]*w
            end
        end
        local[i,1]=sx; local[i,2]=sy; local[i,3]=sz
    end

    global = similar(local)
    MPI.Allreduce!(local, global, MPI.SUM, comm)
    return view(global,:,1), view(global,:,2), view(global,:,3)
end

# Interpolate node velocities from grid (MPI parallel over nodes) - original function
function interpolate_node_velocity_mpi(gridUx::Array{Float64,3}, gridUy::Array{Float64,3}, gridUz::Array{Float64,3},
                                       nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                                       dom::DomainSpec, gr::GridSpec)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nx,ny,nz = gr.nx, gr.ny, gr.nz
    (dx,dy,dz) = grid_spacing(dom, gr)
    # Build flattened grid arrays and coordinates in the same order used by spread (k,j,i)
    x,y,z = grid_vectors(dom, gr)
    coords = Vector{NTuple{3,Float64}}(undef, nx*ny*nz)
    flatUx = Vector{Float64}(undef, nx*ny*nz)
    flatUy = similar(flatUx)
    flatUz = similar(flatUx)
    idx = 1
    for k in 1:nz, j in 1:ny, i in 1:nx
        coords[idx] = (x[i], y[j], z[k])
        flatUx[idx] = gridUx[k,j,i]
        flatUy[idx] = gridUy[k,j,i]
        flatUz[idx] = gridUz[k,j,i]
        idx+=1
    end

    delr = 4.0
    epsx,epsy,epsz = delr*dx, delr*dy, delr*dz

    # local node buffers
    N = length(nodeX)
    local = zeros(Float64, N, 3)

    function nearby_grid(xc,yc,zc)
        maskx = abs.(getindex.(coords,1) .- xc) .<= epsx
        masky = abs.(getindex.(coords,2) .- yc) .<= epsy
        maskz = abs.(getindex.(coords,3) .- zc) .<= epsz
        findall(maskx .& masky .& maskz)
    end

    @inbounds for i in (rank+1):nprocs:N
        xc,yc,zc = nodeX[i], nodeY[i], nodeZ[i]
        sx=0.0; sy=0.0; sz=0.0
        # 9 tiles like python
        tiles = ((0.0,0.0),(+dom.Lx,0.0),(-dom.Lx,0.0),(0.0,+dom.Ly),(0.0,-dom.Ly),
                 (+dom.Lx,+dom.Ly),(+dom.Lx,-dom.Ly),(-dom.Lx,+dom.Ly),(-dom.Lx,-dom.Ly))
        for (dxL,dyL) in tiles
            # find neighbors relative to shifted tile by querying around (xc-dxL, yc-dyL, zc)
            xq = xc - dxL; yq = yc - dyL
            inds = nearby_grid(xq, yq, zc)
            for idxg in inds
                gx,gy,gz = coords[idxg]
                dxv = xq - gx
                dyv = yq - gy
                dzv = zc - gz
                w = (1 + cos(pi*dxv/epsx))*(1 + cos(pi*dyv/epsy))*(1 + cos(pi*dzv/epsz)) / (8*delr^3)
                sx += flatUx[idxg]*w
                sy += flatUy[idxg]*w
                sz += flatUz[idxg]*w
            end
        end
        local[i,1]=sx; local[i,2]=sy; local[i,3]=sz
    end

    global = similar(local)
    MPI.Allreduce!(local, global, MPI.SUM, comm)
    return view(global,:,1), view(global,:,2), view(global,:,3)
end

end # module

using .Peskin3D: init_mpi!, finalize_mpi!, triangle_centroids, triangle_areas, subtriangle_centroids4,
                          spread_vorticity_to_grid_mpi, interpolate_node_velocity_mpi,
                          spread_vorticity_to_grid_kernel_mpi, interpolate_node_velocity_kernel_mpi
