module Circulation

using MPI

export node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,
       triangle_normals, baroclinic_ele_gamma, TriangleGeometry, compute_triangle_geometry,
       node_circulation_from_ele_gamma_mpi, ele_gamma_from_node_circ_mpi,
       triangle_normals_mpi, baroclinic_ele_gamma_mpi, transport_ele_gamma_mpi

# Initialize MPI if not already initialized
init_mpi!() = (MPI.Initialized() || MPI.Init(); nothing)

# Cache for triangle geometry to avoid redundant calculations
struct TriangleGeometry{T<:AbstractFloat}
    areas::Vector{T}
    edge_vectors::Array{T,3}  # [triangle_id, edge_id, xyz]
    centroids::Matrix{T}  # [triangle_id, xyz]
end

# Constructor for triangle geometry cache
function TriangleGeometry(::Type{T}, nt::Int) where T<:AbstractFloat
    TriangleGeometry{T}(
        Vector{T}(undef, nt),
        Array{T}(undef, nt, 3, 3),  # 3 edges per triangle, 3 components each
        Matrix{T}(undef, nt, 3)
    )
end

TriangleGeometry(nt::Int) = TriangleGeometry(Float64, nt)

# Fast triangle area using cross product (more numerically stable than Heron's formula)
@inline function triangle_area_fast(p₁::NTuple{3,T},
                                    p₂::NTuple{3,T},
                                    p₃::NTuple{3,T}) where T
    # Area = 0.5 * ||(p₂-p₁) × (p₃-p₁)||
    v₁x, v₁y, v₁z = p₂[1] - p₁[1], p₂[2] - p₁[2], p₂[3] - p₁[3]
    v₂x, v₂y, v₂z = p₃[1] - p₁[1], p₃[2] - p₁[2], p₃[3] - p₁[3]

    cx = v₁y * v₂z - v₁z * v₂y
    cy = v₁z * v₂x - v₁x * v₂z
    cz = v₁x * v₂y - v₁y * v₂x

    return T(0.5) * sqrt(cx*cx + cy*cy + cz*cz)
end

# Compute and cache triangle geometry
function compute_triangle_geometry!(geom::TriangleGeometry{T},
                                triXC::AbstractMatrix,
                                triYC::AbstractMatrix,
                                triZC::AbstractMatrix) where T
    nt = size(triXC, 1)

    @inbounds for t in 1:nt
        p₁ = (T(triXC[t,1]), T(triYC[t,1]), T(triZC[t,1]))
        p₂ = (T(triXC[t,2]), T(triYC[t,2]), T(triZC[t,2]))
        p₃ = (T(triXC[t,3]), T(triYC[t,3]), T(triZC[t,3]))

        # Cache triangle area
        geom.areas[t] = triangle_area_fast(p₁, p₂, p₃)

        # Cache edge vectors
        geom.edge_vectors[t,1,1] = p₂[1] - p₁[1]  # X₁₂
        geom.edge_vectors[t,1,2] = p₂[2] - p₁[2]  # Y₁₂
        geom.edge_vectors[t,1,3] = p₂[3] - p₁[3]  # Z₁₂

        geom.edge_vectors[t,2,1] = p₃[1] - p₂[1]  # X₂₃
        geom.edge_vectors[t,2,2] = p₃[2] - p₂[2]  # Y₂₃
        geom.edge_vectors[t,2,3] = p₃[3] - p₂[3]  # Z₂₃

        geom.edge_vectors[t,3,1] = p₁[1] - p₃[1]  # X₃₁
        geom.edge_vectors[t,3,2] = p₁[2] - p₃[2]  # Y₃₁
        geom.edge_vectors[t,3,3] = p₁[3] - p₃[3]  # Z₃₁

        # Cache centroid
        geom.centroids[t,1] = (p₁[1] + p₂[1] + p₃[1]) / 3
        geom.centroids[t,2] = (p₁[2] + p₂[2] + p₃[2]) / 3
        geom.centroids[t,3] = (p₁[3] + p₂[3] + p₃[3]) / 3
    end

    return nothing
end

# Convenience constructor
function compute_triangle_geometry(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    nt = size(triXC, 1)
    geom = TriangleGeometry(nt)
    compute_triangle_geometry!(geom, triXC, triYC, triZC)
    return geom
end

#=============================================================================
  Serial implementations
=============================================================================#

# Optimized version using cached triangle geometry
function node_circulation_from_ele_gamma(geom::TriangleGeometry, element_gamma::AbstractMatrix)
    nt = length(geom.areas)
    τ = Matrix{Float64}(undef, nt, 3)

    @inbounds for t in 1:nt
        # Use cached edge vectors
        X₁₂, Y₁₂, Z₁₂ = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X₂₃, Y₂₃, Z₂₃ = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X₃₁, Y₃₁, Z₃₁ = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]

        M = [X₁₂ X₂₃ X₃₁;
             Y₁₂ Y₂₃ Y₃₁;
             Z₁₂ Z₂₃ Z₃₁;
             1.0 1.0 1.0]

        # Use cached area
        Aₜ = geom.areas[t]
        rhs = [Aₜ*element_gamma[t,1]; Aₜ*element_gamma[t,2]; Aₜ*element_gamma[t,3]; 0.0]
        aτ = M \ rhs
        τ[t,1] = aτ[1]; τ[t,2] = aτ[2]; τ[t,3] = aτ[3]
    end
    return τ
end

# Backward-compatible wrapper
function node_circulation_from_ele_gamma(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                         element_gamma::AbstractMatrix)
    geom = compute_triangle_geometry(triXC, triYC, triZC)
    return node_circulation_from_ele_gamma(geom, element_gamma)
end

# Optimized version using cached triangle geometry
function ele_gamma_from_node_circ(geom::TriangleGeometry, node_τ::AbstractMatrix)
    nt = length(geom.areas)
    eleGma = Matrix{Float64}(undef, nt, 3)

    @inbounds for t in 1:nt

        # Use cached edge vectors (X₃₁ = p₁ - p₃, etc.)
        X₁₂, Y₁₂, Z₁₂ = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X₂₃, Y₂₃, Z₂₃ = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X₃₁, Y₃₁, Z₃₁ = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]

        τ₁, τ₂, τ₃ = node_τ[t,1], node_τ[t,2], node_τ[t,3]

        # Use cached area
        inv_A = 1.0 / geom.areas[t]
        eleGma[t,1] = (τ₁*X₁₂ + τ₂*X₂₃ + τ₃*X₃₁) * inv_A
        eleGma[t,2] = (τ₁*Y₁₂ + τ₂*Y₂₃ + τ₃*Y₃₁) * inv_A
        eleGma[t,3] = (τ₁*Z₁₂ + τ₂*Z₂₃ + τ₃*Z₃₁) * inv_A
    end
    return eleGma
end

# Backward-compatible wrapper
function ele_gamma_from_node_circ(node_τ::AbstractMatrix,
                                  triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    geom = compute_triangle_geometry(triXC, triYC, triZC)
    return ele_gamma_from_node_circ(geom, node_τ)
end

# Transport element gamma from old triangles to new triangles by preserving node circulation
function transport_ele_gamma(eleGma_old::AbstractMatrix,
                             triXC_old::AbstractMatrix, triYC_old::AbstractMatrix, triZC_old::AbstractMatrix,
                             triXC_new::AbstractMatrix, triYC_new::AbstractMatrix, triZC_new::AbstractMatrix)

    τ = node_circulation_from_ele_gamma(triXC_old, triYC_old, triZC_old, eleGma_old)
    eleGma_new = ele_gamma_from_node_circ(τ, triXC_new, triYC_new, triZC_new)
    return eleGma_new
end

# Unit normals for each triangle; flip to ensure positive z-component like python helper
function triangle_normals(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    nt = size(triXC,1)
    N = zeros(Float64, nt, 3)

    @inbounds for t in 1:nt
        p₀ = (triXC[t,1], triYC[t,1], triZC[t,1])
        p₁ = (triXC[t,2], triYC[t,2], triZC[t,2])
        p₂ = (triXC[t,3], triYC[t,3], triZC[t,3])

        r₀₁ = (p₁[1]-p₀[1], p₁[2]-p₀[2], p₁[3]-p₀[3])
        r₁₂ = (p₂[1]-p₁[1], p₂[2]-p₁[2], p₂[3]-p₁[3])

        nₓ = r₀₁[2]*r₁₂[3] - r₀₁[3]*r₁₂[2]
        nᵧ = r₁₂[1]*r₀₁[3] - r₁₂[3]*r₀₁[1]
        nᵤ = r₀₁[1]*r₁₂[2] - r₀₁[2]*r₁₂[1]
        norm = sqrt(nₓ*nₓ + nᵧ*nᵧ + nᵤ*nᵤ)
        if norm == 0
            nₓ, nᵧ, nᵤ = 0.0, 0.0, 1.0
            norm = 1.0
        end

        nₓ /= norm; nᵧ /= norm; nᵤ /= norm
        if nᵤ < 0.0
            nₓ = -nₓ; nᵧ = -nᵧ; nᵤ = -nᵤ
        end

        N[t,1] = nₓ; N[t,2] = nᵧ; N[t,3] = nᵤ
    end
    return N
end

# Baroclinic contribution to element vorticity over dt: dγ = [+2At*nᵧ, -2At*nₓ, 0]*dt
# At is the Atwood number (dimensionless density ratio)
function baroclinic_ele_gamma(At::Float64, dt::Float64,
                            triXC::AbstractMatrix,
                            triYC::AbstractMatrix,
                            triZC::AbstractMatrix)

    N = triangle_normals(triXC, triYC, triZC)
    nt = size(triXC,1)
    dG = zeros(Float64, nt, 3)
    @inbounds for t in 1:nt
        nₓ = N[t,1]; nᵧ = N[t,2]
        dG[t,1] = +2*At*nᵧ*dt
        dG[t,2] = -2*At*nₓ*dt
        dG[t,3] = 0.0
    end
    return dG
end

#=============================================================================
  MPI-parallel implementations
=============================================================================#

"""
    node_circulation_from_ele_gamma_mpi(geom, element_gamma)

MPI-parallel version of node_circulation_from_ele_gamma.
Distributes triangle processing across MPI ranks using strided work splitting.
"""
function node_circulation_from_ele_gamma_mpi(geom::TriangleGeometry, element_gamma::AbstractMatrix)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nt = length(geom.areas)
    local_τ = zeros(Float64, nt, 3)

    # Strided work splitting across MPI ranks
    @inbounds for t in (rank+1):nprocs:nt
        X₁₂, Y₁₂, Z₁₂ = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X₂₃, Y₂₃, Z₂₃ = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X₃₁, Y₃₁, Z₃₁ = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]

        M = [X₁₂ X₂₃ X₃₁;
             Y₁₂ Y₂₃ Y₃₁;
             Z₁₂ Z₂₃ Z₃₁;
             1.0 1.0 1.0]

        Aₜ = geom.areas[t]
        rhs = [Aₜ*element_gamma[t,1]; Aₜ*element_gamma[t,2]; Aₜ*element_gamma[t,3]; 0.0]
        aτ = M \ rhs
        local_τ[t,1] = aτ[1]; local_τ[t,2] = aτ[2]; local_τ[t,3] = aτ[3]
    end

    # Reduce across all ranks
    global_τ = similar(local_τ)
    MPI.Allreduce!(local_τ, global_τ, MPI.SUM, comm)

    return global_τ
end

# Backward-compatible MPI wrapper
function node_circulation_from_ele_gamma_mpi(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                             element_gamma::AbstractMatrix)
    geom = compute_triangle_geometry(triXC, triYC, triZC)
    return node_circulation_from_ele_gamma_mpi(geom, element_gamma)
end

"""
    ele_gamma_from_node_circ_mpi(geom, node_τ)

MPI-parallel version of ele_gamma_from_node_circ.
Distributes triangle processing across MPI ranks using strided work splitting.
"""
function ele_gamma_from_node_circ_mpi(geom::TriangleGeometry, node_τ::AbstractMatrix)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nt = length(geom.areas)
    local_eleGma = zeros(Float64, nt, 3)

    # Strided work splitting across MPI ranks
    @inbounds for t in (rank+1):nprocs:nt
        X₁₂, Y₁₂, Z₁₂ = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X₂₃, Y₂₃, Z₂₃ = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X₃₁, Y₃₁, Z₃₁ = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]

        τ₁, τ₂, τ₃ = node_τ[t,1], node_τ[t,2], node_τ[t,3]

        inv_A = 1.0 / geom.areas[t]
        local_eleGma[t,1] = (τ₁*X₁₂ + τ₂*X₂₃ + τ₃*X₃₁) * inv_A
        local_eleGma[t,2] = (τ₁*Y₁₂ + τ₂*Y₂₃ + τ₃*Y₃₁) * inv_A
        local_eleGma[t,3] = (τ₁*Z₁₂ + τ₂*Z₂₃ + τ₃*Z₃₁) * inv_A
    end

    # Reduce across all ranks
    global_eleGma = similar(local_eleGma)
    MPI.Allreduce!(local_eleGma, global_eleGma, MPI.SUM, comm)

    return global_eleGma
end

# Backward-compatible MPI wrapper
function ele_gamma_from_node_circ_mpi(node_τ::AbstractMatrix,
                                      triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    geom = compute_triangle_geometry(triXC, triYC, triZC)
    return ele_gamma_from_node_circ_mpi(geom, node_τ)
end

"""
    transport_ele_gamma_mpi(eleGma_old, triXC_old, triYC_old, triZC_old, triXC_new, triYC_new, triZC_new)

MPI-parallel version of transport_ele_gamma.
"""
function transport_ele_gamma_mpi(eleGma_old::AbstractMatrix,
                                 triXC_old::AbstractMatrix, triYC_old::AbstractMatrix, triZC_old::AbstractMatrix,
                                 triXC_new::AbstractMatrix, triYC_new::AbstractMatrix, triZC_new::AbstractMatrix)

    τ = node_circulation_from_ele_gamma_mpi(triXC_old, triYC_old, triZC_old, eleGma_old)
    eleGma_new = ele_gamma_from_node_circ_mpi(τ, triXC_new, triYC_new, triZC_new)
    return eleGma_new
end

"""
    triangle_normals_mpi(triXC, triYC, triZC)

MPI-parallel version of triangle_normals.
Distributes triangle processing across MPI ranks using strided work splitting.
"""
function triangle_normals_mpi(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    nt = size(triXC, 1)
    local_N = zeros(Float64, nt, 3)

    # Strided work splitting across MPI ranks
    @inbounds for t in (rank+1):nprocs:nt
        p₀ = (triXC[t,1], triYC[t,1], triZC[t,1])
        p₁ = (triXC[t,2], triYC[t,2], triZC[t,2])
        p₂ = (triXC[t,3], triYC[t,3], triZC[t,3])

        r₀₁ = (p₁[1]-p₀[1], p₁[2]-p₀[2], p₁[3]-p₀[3])
        r₁₂ = (p₂[1]-p₁[1], p₂[2]-p₁[2], p₂[3]-p₁[3])

        nₓ = r₀₁[2]*r₁₂[3] - r₀₁[3]*r₁₂[2]
        nᵧ = r₁₂[1]*r₀₁[3] - r₁₂[3]*r₀₁[1]
        nᵤ = r₀₁[1]*r₁₂[2] - r₀₁[2]*r₁₂[1]
        norm = sqrt(nₓ*nₓ + nᵧ*nᵧ + nᵤ*nᵤ)
        if norm == 0
            nₓ, nᵧ, nᵤ = 0.0, 0.0, 1.0
            norm = 1.0
        end

        nₓ /= norm; nᵧ /= norm; nᵤ /= norm
        if nᵤ < 0.0
            nₓ = -nₓ; nᵧ = -nᵧ; nᵤ = -nᵤ
        end

        local_N[t,1] = nₓ; local_N[t,2] = nᵧ; local_N[t,3] = nᵤ
    end

    # Reduce across all ranks
    global_N = similar(local_N)
    MPI.Allreduce!(local_N, global_N, MPI.SUM, comm)

    return global_N
end

"""
    baroclinic_ele_gamma_mpi(At, dt, triXC, triYC, triZC)

MPI-parallel version of baroclinic_ele_gamma.
At is the Atwood number (dimensionless density ratio).
"""
function baroclinic_ele_gamma_mpi(At::Float64, dt::Float64,
                                  triXC::AbstractMatrix,
                                  triYC::AbstractMatrix,
                                  triZC::AbstractMatrix)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Compute normals in parallel
    N = triangle_normals_mpi(triXC, triYC, triZC)

    nt = size(triXC, 1)
    local_dG = zeros(Float64, nt, 3)

    # Strided work splitting across MPI ranks
    @inbounds for t in (rank+1):nprocs:nt
        nₓ = N[t,1]; nᵧ = N[t,2]
        local_dG[t,1] = +2*At*nᵧ*dt
        local_dG[t,2] = -2*At*nₓ*dt
        local_dG[t,3] = 0.0
    end

    # Reduce across all ranks
    global_dG = similar(local_dG)
    MPI.Allreduce!(local_dG, global_dG, MPI.SUM, comm)

    return global_dG
end

end # module

using .Circulation: node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,
                    triangle_normals, baroclinic_ele_gamma, TriangleGeometry, compute_triangle_geometry,
                    node_circulation_from_ele_gamma_mpi, ele_gamma_from_node_circ_mpi,
                    triangle_normals_mpi, baroclinic_ele_gamma_mpi, transport_ele_gamma_mpi
