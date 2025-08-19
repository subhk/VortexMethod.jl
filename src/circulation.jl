module Circulation

export node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,
       triangle_normals, baroclinic_ele_gamma, TriangleGeometry, compute_triangle_geometry

# Cache for triangle geometry to avoid redundant calculations
struct TriangleGeometry{T<:AbstractFloat}
    areas::Vector{T}
    edge_vectors::Array{T,3}  # [triangle_id, edge_id, xyz]
    inverse_matrices::Array{T,3}  # Pre-computed M^{-1} for each triangle
    centroids::Matrix{T}  # [triangle_id, xyz]
end

# Constructor for triangle geometry cache
function TriangleGeometry(::Type{T}, nt::Int) where T<:AbstractFloat
    TriangleGeometry{T}(
        Vector{T}(undef, nt),
        Array{T}(undef, nt, 3, 3),  # 3 edges per triangle, 3 components each
        Array{T}(undef, nt, 4, 3),  # 4x3 inverse matrices
        Matrix{T}(undef, nt, 3)
    )
end

TriangleGeometry(nt::Int) = TriangleGeometry(Float64, nt)

# Fast triangle area using cross product (more numerically stable than Heron's formula)
@inline function triangle_area_fast(p1::NTuple{3,T}, p2::NTuple{3,T}, p3::NTuple{3,T}) where T
    # Area = 0.5 * ||(p2-p1) × (p3-p1)||
    v1x, v1y, v1z = p2[1] - p1[1], p2[2] - p1[2], p2[3] - p1[3]
    v2x, v2y, v2z = p3[1] - p1[1], p3[2] - p1[2], p3[3] - p1[3]
    
    cx = v1y * v2z - v1z * v2y
    cy = v1z * v2x - v1x * v2z  
    cz = v1x * v2y - v1y * v2x
    
    return T(0.5) * sqrt(cx*cx + cy*cy + cz*cz)
end

# Compute and cache triangle geometry
function compute_triangle_geometry!(geom::TriangleGeometry{T}, 
                                  triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix) where T
    nt = size(triXC, 1)
    
    @inbounds for t in 1:nt
        p1 = (T(triXC[t,1]), T(triYC[t,1]), T(triZC[t,1]))
        p2 = (T(triXC[t,2]), T(triYC[t,2]), T(triZC[t,2]))
        p3 = (T(triXC[t,3]), T(triYC[t,3]), T(triZC[t,3]))
        
        # Cache triangle area
        geom.areas[t] = triangle_area_fast(p1, p2, p3)
        
        # Cache edge vectors
        geom.edge_vectors[t,1,1] = p2[1] - p1[1]  # X12
        geom.edge_vectors[t,1,2] = p2[2] - p1[2]  # Y12  
        geom.edge_vectors[t,1,3] = p2[3] - p1[3]  # Z12
        
        geom.edge_vectors[t,2,1] = p3[1] - p2[1]  # X23
        geom.edge_vectors[t,2,2] = p3[2] - p2[2]  # Y23
        geom.edge_vectors[t,2,3] = p3[3] - p2[3]  # Z23
        
        geom.edge_vectors[t,3,1] = p1[1] - p3[1]  # X31
        geom.edge_vectors[t,3,2] = p1[2] - p3[2]  # Y31
        geom.edge_vectors[t,3,3] = p1[3] - p3[3]  # Z31
        
        # Cache centroid
        geom.centroids[t,1] = (p1[1] + p2[1] + p3[1]) / 3
        geom.centroids[t,2] = (p1[2] + p2[2] + p3[2]) / 3
        geom.centroids[t,3] = (p1[3] + p2[3] + p3[3]) / 3
        
        # Pre-compute and cache inverse matrix M^{-1}
        # M = [X12 X23 X31; Y12 Y23 Y31; Z12 Z23 Z31; 1 1 1]
        # This is expensive, so we use a more efficient approach
        X12, Y12, Z12 = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X23, Y23, Z23 = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X31, Y31, Z31 = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]
        
        # Store matrix elements for later inversion (4x3 system)
        M = [X12 X23 X31; Y12 Y23 Y31; Z12 Z23 Z31; 1.0 1.0 1.0]
        # For efficiency, we'll solve the system each time rather than pre-computing inverse
        # geom.inverse_matrices[t,:,:] .= pinv(M)  # Too expensive to pre-compute
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

# Optimized version using cached triangle geometry
function node_circulation_from_ele_gamma(geom::TriangleGeometry, element_gamma::AbstractMatrix)
    nt = length(geom.areas)
    τ = Matrix{Float64}(undef, nt, 3)
    
    @inbounds for t in 1:nt
        # Use cached edge vectors
        X12, Y12, Z12 = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X23, Y23, Z23 = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X31, Y31, Z31 = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]
        
        M = [X12 X23 X31;
             Y12 Y23 Y31;
             Z12 Z23 Z31;
             1.0 1.0 1.0]
        
        # Use cached area
        A_t = geom.areas[t]
        rhs = [A_t*element_gamma[t,1]; A_t*element_gamma[t,2]; A_t*element_gamma[t,3]; 0.0]
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
        # Use cached edge vectors
        X12, Y12, Z12 = geom.edge_vectors[t,1,1], geom.edge_vectors[t,1,2], geom.edge_vectors[t,1,3]
        X23, Y23, Z23 = geom.edge_vectors[t,2,1], geom.edge_vectors[t,2,2], geom.edge_vectors[t,2,3]
        X13, Y13, Z13 = geom.edge_vectors[t,3,1], geom.edge_vectors[t,3,2], geom.edge_vectors[t,3,3]
        
        τ1, τ2, τ3 = node_τ[t,1], node_τ[t,2], node_τ[t,3]
        
        # Use cached area  
        inv_A = 1.0 / geom.areas[t]
        eleGma[t,1] = (τ1*X12 + τ2*X23 + τ3*X13) * inv_A
        eleGma[t,2] = (τ1*Y12 + τ2*Y23 + τ3*Y13) * inv_A
        eleGma[t,3] = (τ1*Z12 + τ2*Z23 + τ3*Z13) * inv_A
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
        p0 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p1 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p2 = (triXC[t,3], triYC[t,3], triZC[t,3])
        r01 = (p1[1]-p0[1], p1[2]-p0[2], p1[3]-p0[3])
        r12 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
        nx = r01[2]*r12[3] - r01[3]*r12[2]
        ny = r12[1]*r01[3] - r12[3]*r01[1]
        nz = r01[1]*r12[2] - r01[2]*r12[1]
        norm = sqrt(nx*nx + ny*ny + nz*nz)
        if norm == 0
            nx,ny,nz = 0.0,0.0,1.0
            norm = 1.0
        end
        nx/=norm; ny/=norm; nz/=norm
        if nz < 0.0
            nx = -nx; ny = -ny; nz = -nz
        end
        N[t,1]=nx; N[t,2]=ny; N[t,3]=nz
    end
    return N
end

# Baroclinic contribution to element vorticity over dt: dγ = [+2At*ny, -2At*nx, 0]*dt
function baroclinic_ele_gamma(At::Float64, dt::Float64, triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    N = triangle_normals(triXC, triYC, triZC)
    nt = size(triXC,1)
    dG = zeros(Float64, nt, 3)
    @inbounds for t in 1:nt
        nx = N[t,1]; ny = N[t,2]
        dG[t,1] = +2*At*ny*dt
        dG[t,2] = -2*At*nx*dt
        dG[t,3] = 0.0
    end
    return dG
end

end # module
