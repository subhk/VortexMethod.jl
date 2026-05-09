module Circulation

using MPI
using StaticArrays
using ..DomainImpl

export node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,
       triangle_normals, baroclinic_ele_gamma, TriangleGeometry, compute_triangle_geometry,
       node_circulation_from_ele_gamma_mpi, ele_gamma_from_node_circ_mpi,
       triangle_normals_mpi, baroclinic_ele_gamma_mpi, transport_ele_gamma_mpi,
       node_circulation_from_ele_gamma!, ele_gamma_from_node_circ!

# Initialize MPI if not already initialized
init_mpi!() = (MPI.Initialized() || MPI.Init(); nothing)

# Cache for triangle geometry to avoid redundant calculations
struct TriangleGeometry{T<:AbstractFloat}
    areas::Vector{T}
    edge_vectors::Vector{SMatrix{3,3,T,9}}  # one per triangle; column k = edge k (12, 23, 31)
    centroids::Matrix{T}
end

# Constructor for triangle geometry cache
function TriangleGeometry(::Type{T}, nt::Int) where T<:AbstractFloat
    TriangleGeometry{T}(
        Vector{T}(undef, nt),
        Vector{SMatrix{3,3,T,9}}(undef, nt),
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

@inline function triangle_area_unwrapped(p₁::NTuple{3,T},
                                         p₂::NTuple{3,T},
                                         p₃::NTuple{3,T},
                                         domain::DomainSpec) where T
    return T(periodic_triangle_area(p₁, p₂, p₃, domain))
end

function triangle_edges_centroid(p₁::NTuple{3,T},
                                 p₂::NTuple{3,T},
                                 p₃::NTuple{3,T},
                                 domain::Union{Nothing,DomainSpec}) where T
    if domain === nothing
        X₁₂, Y₁₂, Z₁₂ = p₂[1] - p₁[1], p₂[2] - p₁[2], p₂[3] - p₁[3]
        X₂₃, Y₂₃, Z₂₃ = p₃[1] - p₂[1], p₃[2] - p₂[2], p₃[3] - p₂[3]
        X₃₁, Y₃₁, Z₃₁ = p₁[1] - p₃[1], p₁[2] - p₃[2], p₁[3] - p₃[3]
        cx, cy, cz = (p₁[1] + p₂[1] + p₃[1]) / 3,
                     (p₁[2] + p₂[2] + p₃[2]) / 3,
                     (p₁[3] + p₂[3] + p₃[3]) / 3
        return SVector{3,T}(X₁₂, Y₁₂, Z₁₂),
               SVector{3,T}(X₂₃, Y₂₃, Z₂₃),
               SVector{3,T}(X₃₁, Y₃₁, Z₃₁),
               SVector{3,T}(cx, cy, cz)
    end

    q₁, q₂, q₃ = unwrap_triangle(p₁, p₂, p₃, domain)
    X₁₂, Y₁₂, Z₁₂ = q₂[1] - q₁[1], q₂[2] - q₁[2], q₂[3] - q₁[3]
    X₂₃, Y₂₃, Z₂₃ = q₃[1] - q₂[1], q₃[2] - q₂[2], q₃[3] - q₂[3]
    X₃₁, Y₃₁, Z₃₁ = q₁[1] - q₃[1], q₁[2] - q₃[2], q₁[3] - q₃[3]
    cx, cy, cz = periodic_centroid(p₁, p₂, p₃, domain)
    return SVector{3,T}(X₁₂, Y₁₂, Z₁₂),
           SVector{3,T}(X₂₃, Y₂₃, Z₂₃),
           SVector{3,T}(X₃₁, Y₃₁, Z₃₁),
           SVector{3,T}(cx, cy, cz)
end

@inline function solve_circulation_weights(X₁₂::Float64, Y₁₂::Float64, Z₁₂::Float64,
                                           X₂₃::Float64, Y₂₃::Float64, Z₂₃::Float64,
                                           X₃₁::Float64, Y₃₁::Float64, Z₃₁::Float64,
                                           rhs₁::Float64, rhs₂::Float64, rhs₃::Float64)
    # Solve the 4x3 least-squares system used by node_circulation_from_ele_gamma
    # via expanded normal equations, avoiding per-triangle Matrix/Vector allocations.
    a₁₁ = X₁₂*X₁₂ + Y₁₂*Y₁₂ + Z₁₂*Z₁₂ + 1.0
    a₁₂ = X₁₂*X₂₃ + Y₁₂*Y₂₃ + Z₁₂*Z₂₃ + 1.0
    a₁₃ = X₁₂*X₃₁ + Y₁₂*Y₃₁ + Z₁₂*Z₃₁ + 1.0
    a₂₂ = X₂₃*X₂₃ + Y₂₃*Y₂₃ + Z₂₃*Z₂₃ + 1.0
    a₂₃ = X₂₃*X₃₁ + Y₂₃*Y₃₁ + Z₂₃*Z₃₁ + 1.0
    a₃₃ = X₃₁*X₃₁ + Y₃₁*Y₃₁ + Z₃₁*Z₃₁ + 1.0

    b₁ = X₁₂*rhs₁ + Y₁₂*rhs₂ + Z₁₂*rhs₃
    b₂ = X₂₃*rhs₁ + Y₂₃*rhs₂ + Z₂₃*rhs₃
    b₃ = X₃₁*rhs₁ + Y₃₁*rhs₂ + Z₃₁*rhs₃

    det = a₁₁*(a₂₂*a₃₃ - a₂₃*a₂₃) -
          a₁₂*(a₁₂*a₃₃ - a₂₃*a₁₃) +
          a₁₃*(a₁₂*a₂₃ - a₂₂*a₁₃)

    if abs(det) < 1e-15
        return 0.0, 0.0, 0.0
    end

    inv_det = 1.0 / det
    inv₁₁ = (a₂₂*a₃₃ - a₂₃*a₂₃) * inv_det
    inv₁₂ = (a₁₃*a₂₃ - a₁₂*a₃₃) * inv_det
    inv₁₃ = (a₁₂*a₂₃ - a₁₃*a₂₂) * inv_det
    inv₂₂ = (a₁₁*a₃₃ - a₁₃*a₁₃) * inv_det
    inv₂₃ = (a₁₂*a₁₃ - a₁₁*a₂₃) * inv_det
    inv₃₃ = (a₁₁*a₂₂ - a₁₂*a₁₂) * inv_det

    return inv₁₁*b₁ + inv₁₂*b₂ + inv₁₃*b₃,
           inv₁₂*b₁ + inv₂₂*b₂ + inv₂₃*b₃,
           inv₁₃*b₁ + inv₂₃*b₂ + inv₃₃*b₃
end

# Compute and cache triangle geometry
function compute_triangle_geometry!(geom::TriangleGeometry{T},
                                triXC::AbstractMatrix,
                                triYC::AbstractMatrix,
                                triZC::AbstractMatrix;
                                domain::Union{Nothing,DomainSpec}=nothing) where T
    nt = size(triXC, 1)

    @inbounds for t in 1:nt
        p₁ = (T(triXC[t,1]), T(triYC[t,1]), T(triZC[t,1]))
        p₂ = (T(triXC[t,2]), T(triYC[t,2]), T(triZC[t,2]))
        p₃ = (T(triXC[t,3]), T(triYC[t,3]), T(triZC[t,3]))

        e₁₂, e₂₃, e₃₁, centroid = triangle_edges_centroid(p₁, p₂, p₃, domain)

        # Cache triangle area
        geom.areas[t] = domain === nothing ? triangle_area_fast(p₁, p₂, p₃) :
                                             triangle_area_unwrapped(p₁, p₂, p₃, domain)

        # Cache edge vectors (column-major: column k = edge k)
        geom.edge_vectors[t] = SMatrix{3,3,T,9}(
            e₁₂[1], e₁₂[2], e₁₂[3],
            e₂₃[1], e₂₃[2], e₂₃[3],
            e₃₁[1], e₃₁[2], e₃₁[3]
        )

        # Cache centroid
        geom.centroids[t,1] = centroid[1]
        geom.centroids[t,2] = centroid[2]
        geom.centroids[t,3] = centroid[3]
    end

    return nothing
end

# Convenience constructor
function compute_triangle_geometry(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                                   domain::Union{Nothing,DomainSpec}=nothing)
    nt = size(triXC, 1)
    geom = TriangleGeometry(nt)
    compute_triangle_geometry!(geom, triXC, triYC, triZC; domain=domain)
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
        X₁₂ = geom.edge_vectors[t][1,1]; Y₁₂ = geom.edge_vectors[t][2,1]; Z₁₂ = geom.edge_vectors[t][3,1]
        X₂₃ = geom.edge_vectors[t][1,2]; Y₂₃ = geom.edge_vectors[t][2,2]; Z₂₃ = geom.edge_vectors[t][3,2]
        X₃₁ = geom.edge_vectors[t][1,3]; Y₃₁ = geom.edge_vectors[t][2,3]; Z₃₁ = geom.edge_vectors[t][3,3]

        # Use cached area
        Aₜ = geom.areas[t]
        τ₁, τ₂, τ₃ = solve_circulation_weights(X₁₂, Y₁₂, Z₁₂,
                                                X₂₃, Y₂₃, Z₂₃,
                                                X₃₁, Y₃₁, Z₃₁,
                                                Aₜ*element_gamma[t,1],
                                                Aₜ*element_gamma[t,2],
                                                Aₜ*element_gamma[t,3])
        τ[t,1] = τ₁; τ[t,2] = τ₂; τ[t,3] = τ₃
    end
    return τ
end

# Float64-only: solve_circulation_weights is Float64-pinned; generalize with T in Task 8
# In-place zero-allocation variant
function node_circulation_from_ele_gamma!(out::AbstractMatrix{Float64},
                                          geom::TriangleGeometry,
                                          element_gamma::AbstractMatrix)
    nt = length(geom.areas)
    @inbounds for t in 1:nt
        X₁₂ = geom.edge_vectors[t][1,1]; Y₁₂ = geom.edge_vectors[t][2,1]; Z₁₂ = geom.edge_vectors[t][3,1]
        X₂₃ = geom.edge_vectors[t][1,2]; Y₂₃ = geom.edge_vectors[t][2,2]; Z₂₃ = geom.edge_vectors[t][3,2]
        X₃₁ = geom.edge_vectors[t][1,3]; Y₃₁ = geom.edge_vectors[t][2,3]; Z₃₁ = geom.edge_vectors[t][3,3]
        Aₜ = geom.areas[t]
        τ₁, τ₂, τ₃ = solve_circulation_weights(X₁₂, Y₁₂, Z₁₂, X₂₃, Y₂₃, Z₂₃, X₃₁, Y₃₁, Z₃₁,
                                                Aₜ*element_gamma[t,1],
                                                Aₜ*element_gamma[t,2],
                                                Aₜ*element_gamma[t,3])
        out[t,1] = τ₁; out[t,2] = τ₂; out[t,3] = τ₃
    end
    return out
end

# Backward-compatible wrapper
function node_circulation_from_ele_gamma(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                         element_gamma::AbstractMatrix;
                                         domain::Union{Nothing,DomainSpec}=nothing)
    geom = compute_triangle_geometry(triXC, triYC, triZC; domain=domain)
    return node_circulation_from_ele_gamma(geom, element_gamma)
end

# Optimized version using cached triangle geometry
function ele_gamma_from_node_circ(geom::TriangleGeometry, node_τ::AbstractMatrix)
    nt = length(geom.areas)
    eleGma = Matrix{Float64}(undef, nt, 3)

    @inbounds for t in 1:nt

        # Use cached edge vectors (X₃₁ = p₁ - p₃, etc.)
        X₁₂ = geom.edge_vectors[t][1,1]; Y₁₂ = geom.edge_vectors[t][2,1]; Z₁₂ = geom.edge_vectors[t][3,1]
        X₂₃ = geom.edge_vectors[t][1,2]; Y₂₃ = geom.edge_vectors[t][2,2]; Z₂₃ = geom.edge_vectors[t][3,2]
        X₃₁ = geom.edge_vectors[t][1,3]; Y₃₁ = geom.edge_vectors[t][2,3]; Z₃₁ = geom.edge_vectors[t][3,3]

        τ₁, τ₂, τ₃ = node_τ[t,1], node_τ[t,2], node_τ[t,3]

        # Use cached area
        inv_A = 1.0 / geom.areas[t]
        eleGma[t,1] = (τ₁*X₁₂ + τ₂*X₂₃ + τ₃*X₃₁) * inv_A
        eleGma[t,2] = (τ₁*Y₁₂ + τ₂*Y₂₃ + τ₃*Y₃₁) * inv_A
        eleGma[t,3] = (τ₁*Z₁₂ + τ₂*Z₂₃ + τ₃*Z₃₁) * inv_A
    end
    return eleGma
end

# Float64-only: inv_A and dot products are Float64; generalize with T in Task 8
# In-place zero-allocation variant
function ele_gamma_from_node_circ!(out::AbstractMatrix{Float64},
                                   geom::TriangleGeometry,
                                   node_τ::AbstractMatrix)
    nt = length(geom.areas)
    @inbounds for t in 1:nt
        X₁₂ = geom.edge_vectors[t][1,1]; Y₁₂ = geom.edge_vectors[t][2,1]; Z₁₂ = geom.edge_vectors[t][3,1]
        X₂₃ = geom.edge_vectors[t][1,2]; Y₂₃ = geom.edge_vectors[t][2,2]; Z₂₃ = geom.edge_vectors[t][3,2]
        X₃₁ = geom.edge_vectors[t][1,3]; Y₃₁ = geom.edge_vectors[t][2,3]; Z₃₁ = geom.edge_vectors[t][3,3]
        τ₁, τ₂, τ₃ = node_τ[t,1], node_τ[t,2], node_τ[t,3]
        inv_A = 1.0 / geom.areas[t]
        out[t,1] = (τ₁*X₁₂ + τ₂*X₂₃ + τ₃*X₃₁) * inv_A
        out[t,2] = (τ₁*Y₁₂ + τ₂*Y₂₃ + τ₃*Y₃₁) * inv_A
        out[t,3] = (τ₁*Z₁₂ + τ₂*Z₂₃ + τ₃*Z₃₁) * inv_A
    end
    return out
end

# Backward-compatible wrapper
function ele_gamma_from_node_circ(node_τ::AbstractMatrix,
                                  triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                                  domain::Union{Nothing,DomainSpec}=nothing)
    geom = compute_triangle_geometry(triXC, triYC, triZC; domain=domain)
    return ele_gamma_from_node_circ(geom, node_τ)
end

# Transport element gamma between unchanged-topology triangle sets by preserving node circulation.
# Topology-changing remesh paths must update circulation locally as they split/merge.
function transport_ele_gamma(eleGma_old::AbstractMatrix,
                             triXC_old::AbstractMatrix, triYC_old::AbstractMatrix, triZC_old::AbstractMatrix,
                             triXC_new::AbstractMatrix, triYC_new::AbstractMatrix, triZC_new::AbstractMatrix;
                             domain::Union{Nothing,DomainSpec}=nothing,
                             method::Symbol=:node)

    geom_old = compute_triangle_geometry(triXC_old, triYC_old, triZC_old; domain=domain)
    geom_new = compute_triangle_geometry(triXC_new, triYC_new, triZC_new; domain=domain)

    if method == :auto
        method = :node
    end

    method == :node ||
        throw(ArgumentError("Unsupported circulation transport method: $method; topology-changing transport must use remesh-specific circulation handling"))
    length(geom_old.areas) == length(geom_new.areas) ||
        throw(DimensionMismatch("node-circulation transport requires the same number of elements; topology-changing remesh must update circulation locally"))

    τ = node_circulation_from_ele_gamma(geom_old, eleGma_old)
    return ele_gamma_from_node_circ(geom_new, τ)
end

function require_same_transport_topology(triXC_old::AbstractMatrix, triXC_new::AbstractMatrix)
    size(triXC_old, 1) == size(triXC_new, 1) ||
        throw(DimensionMismatch("node-circulation transport requires the same number of elements; topology-changing remesh must update circulation locally"))
    return nothing
end

function require_transport_method(method::Symbol)
    if method == :auto || method == :node
        return nothing
    end
    throw(ArgumentError("Unsupported circulation transport method: $method; topology-changing transport must use remesh-specific circulation handling"))
end

# Unit normals for each triangle; flip to ensure positive z-component like python helper
function triangle_normals(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                          domain::Union{Nothing,DomainSpec}=nothing)
    nt = size(triXC,1)
    N = zeros(Float64, nt, 3)

    @inbounds for t in 1:nt
        p₀ = (triXC[t,1], triYC[t,1], triZC[t,1])
        p₁ = (triXC[t,2], triYC[t,2], triZC[t,2])
        p₂ = (triXC[t,3], triYC[t,3], triZC[t,3])

        e₀₁, e₁₂, _, _ = triangle_edges_centroid(p₀, p₁, p₂, domain)
        r₀₁ = e₀₁
        r₁₂ = e₁₂

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
function atwood_value(At::Number, ::Int, ::Int)
    return Float64(At)
end

function atwood_value(At::AbstractVector, t::Int, nt::Int)
    length(At) == nt || throw(DimensionMismatch("At vector length $(length(At)) does not match element count $nt"))
    return Float64(At[t])
end

function has_baroclinicity(At::Number)
    return At != 0
end

function has_baroclinicity(At::AbstractVector)
    return any(!iszero, At)
end

function baroclinic_ele_gamma(At, dt::Float64,
                            triXC::AbstractMatrix,
                            triYC::AbstractMatrix,
                            triZC::AbstractMatrix;
                            domain::Union{Nothing,DomainSpec}=nothing)

    N = triangle_normals(triXC, triYC, triZC; domain=domain)
    nt = size(triXC,1)
    dG = zeros(Float64, nt, 3)
    @inbounds for t in 1:nt
        nₓ = N[t,1]; nᵧ = N[t,2]
        At_t = atwood_value(At, t, nt)
        dG[t,1] = +2*At_t*nᵧ*dt
        dG[t,2] = -2*At_t*nₓ*dt
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
        X₁₂ = geom.edge_vectors[t][1,1]; Y₁₂ = geom.edge_vectors[t][2,1]; Z₁₂ = geom.edge_vectors[t][3,1]
        X₂₃ = geom.edge_vectors[t][1,2]; Y₂₃ = geom.edge_vectors[t][2,2]; Z₂₃ = geom.edge_vectors[t][3,2]
        X₃₁ = geom.edge_vectors[t][1,3]; Y₃₁ = geom.edge_vectors[t][2,3]; Z₃₁ = geom.edge_vectors[t][3,3]

        Aₜ = geom.areas[t]
        τ₁, τ₂, τ₃ = solve_circulation_weights(X₁₂, Y₁₂, Z₁₂,
                                                X₂₃, Y₂₃, Z₂₃,
                                                X₃₁, Y₃₁, Z₃₁,
                                                Aₜ*element_gamma[t,1],
                                                Aₜ*element_gamma[t,2],
                                                Aₜ*element_gamma[t,3])
        local_τ[t,1] = τ₁; local_τ[t,2] = τ₂; local_τ[t,3] = τ₃
    end

    # Reduce across all ranks
    global_τ = similar(local_τ)
    MPI.Allreduce!(local_τ, global_τ, MPI.SUM, comm)

    return global_τ
end

# Backward-compatible MPI wrapper
function node_circulation_from_ele_gamma_mpi(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                             element_gamma::AbstractMatrix;
                                             domain::Union{Nothing,DomainSpec}=nothing)
    geom = compute_triangle_geometry(triXC, triYC, triZC; domain=domain)
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
        X₁₂ = geom.edge_vectors[t][1,1]; Y₁₂ = geom.edge_vectors[t][2,1]; Z₁₂ = geom.edge_vectors[t][3,1]
        X₂₃ = geom.edge_vectors[t][1,2]; Y₂₃ = geom.edge_vectors[t][2,2]; Z₂₃ = geom.edge_vectors[t][3,2]
        X₃₁ = geom.edge_vectors[t][1,3]; Y₃₁ = geom.edge_vectors[t][2,3]; Z₃₁ = geom.edge_vectors[t][3,3]

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
                                      triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                                      domain::Union{Nothing,DomainSpec}=nothing)
    geom = compute_triangle_geometry(triXC, triYC, triZC; domain=domain)
    return ele_gamma_from_node_circ_mpi(geom, node_τ)
end

"""
    transport_ele_gamma_mpi(eleGma_old, triXC_old, triYC_old, triZC_old, triXC_new, triYC_new, triZC_new)

MPI-parallel version of transport_ele_gamma.
"""
function transport_ele_gamma_mpi(eleGma_old::AbstractMatrix,
                                 triXC_old::AbstractMatrix, triYC_old::AbstractMatrix, triZC_old::AbstractMatrix,
                                 triXC_new::AbstractMatrix, triYC_new::AbstractMatrix, triZC_new::AbstractMatrix;
                                 domain::Union{Nothing,DomainSpec}=nothing,
                                 method::Symbol=:node)
    require_transport_method(method)
    require_same_transport_topology(triXC_old, triXC_new)

    τ = node_circulation_from_ele_gamma_mpi(triXC_old, triYC_old, triZC_old, eleGma_old; domain=domain)
    return ele_gamma_from_node_circ_mpi(τ, triXC_new, triYC_new, triZC_new; domain=domain)
end

"""
    triangle_normals_mpi(triXC, triYC, triZC)

MPI-parallel version of triangle_normals.
Distributes triangle processing across MPI ranks using strided work splitting.
"""
function triangle_normals_mpi(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix;
                              domain::Union{Nothing,DomainSpec}=nothing)
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

        e₀₁, e₁₂, _, _ = triangle_edges_centroid(p₀, p₁, p₂, domain)
        r₀₁ = e₀₁
        r₁₂ = e₁₂

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
function baroclinic_ele_gamma_mpi(At, dt::Float64,
                                  triXC::AbstractMatrix,
                                  triYC::AbstractMatrix,
                                  triZC::AbstractMatrix;
                                  domain::Union{Nothing,DomainSpec}=nothing)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Compute normals in parallel
    N = triangle_normals_mpi(triXC, triYC, triZC; domain=domain)

    nt = size(triXC, 1)
    local_dG = zeros(Float64, nt, 3)

    # Strided work splitting across MPI ranks
    @inbounds for t in (rank+1):nprocs:nt
        nₓ = N[t,1]; nᵧ = N[t,2]
        At_t = atwood_value(At, t, nt)
        local_dG[t,1] = +2*At_t*nᵧ*dt
        local_dG[t,2] = -2*At_t*nₓ*dt
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
                    triangle_normals_mpi, baroclinic_ele_gamma_mpi, transport_ele_gamma_mpi,
                    node_circulation_from_ele_gamma!, ele_gamma_from_node_circ!
