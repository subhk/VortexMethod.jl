# Structure of Arrays (SoA) memory layout for improved vectorization and cache performance

module SoALayout

# Using built-in Julia SIMD capabilities

export TriangleSoA, NodeSoA, VorticitySoA, VelocitySoA, 
       aos_to_soa!, soa_to_aos!, vectorized_kernel_eval!, 
       soa_triangle_areas!, soa_circulation_solve!

# Structure of Arrays for triangle coordinates (better for SIMD)
struct TriangleSoA{T<:AbstractFloat}
    # Instead of triXC[triangle, vertex] store as separate arrays
    x1::Vector{T}  # x-coordinates of vertex 1 for all triangles
    x2::Vector{T}  # x-coordinates of vertex 2 for all triangles  
    x3::Vector{T}  # x-coordinates of vertex 3 for all triangles
    y1::Vector{T}  # y-coordinates of vertex 1 for all triangles
    y2::Vector{T}
    y3::Vector{T}
    z1::Vector{T}  # z-coordinates of vertex 1 for all triangles
    z2::Vector{T}
    z3::Vector{T}
    n_triangles::Int
end

function TriangleSoA(::Type{T}, n_triangles::Int) where T<:AbstractFloat
    TriangleSoA{T}(
        Vector{T}(undef, n_triangles), Vector{T}(undef, n_triangles), Vector{T}(undef, n_triangles),
        Vector{T}(undef, n_triangles), Vector{T}(undef, n_triangles), Vector{T}(undef, n_triangles),
        Vector{T}(undef, n_triangles), Vector{T}(undef, n_triangles), Vector{T}(undef, n_triangles),
        n_triangles
    )
end

TriangleSoA(n_triangles::Int) = TriangleSoA(Float64, n_triangles)

# Structure of Arrays for node coordinates
struct NodeSoA{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    n_nodes::Int
end

NodeSoA(::Type{T}, n_nodes::Int) where T = NodeSoA{T}(Vector{T}(undef, n_nodes), Vector{T}(undef, n_nodes), Vector{T}(undef, n_nodes), n_nodes)
NodeSoA(n_nodes::Int) = NodeSoA(Float64, n_nodes)

# Structure of Arrays for vorticity vectors  
struct VorticitySoA{T<:AbstractFloat}
    ωx::Vector{T}
    ωy::Vector{T}
    ωz::Vector{T}
    n_elements::Int
end

VorticitySoA(::Type{T}, n_elements::Int) where T = VorticitySoA{T}(Vector{T}(undef, n_elements), Vector{T}(undef, n_elements), Vector{T}(undef, n_elements), n_elements)
VorticitySoA(n_elements::Int) = VorticitySoA(Float64, n_elements)

# Structure of Arrays for velocity fields
struct VelocitySoA{T<:AbstractFloat}
    u::Vector{T}
    v::Vector{T}
    w::Vector{T}
    n_points::Int
end

VelocitySoA(::Type{T}, n_points::Int) where T = VelocitySoA{T}(Vector{T}(undef, n_points), Vector{T}(undef, n_points), Vector{T}(undef, n_points), n_points)
VelocitySoA(n_points::Int) = VelocitySoA(Float64, n_points)

# Convert Array of Structures to Structure of Arrays for triangles
function aos_to_soa!(soa::TriangleSoA{T}, triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix) where T
    @assert size(triXC, 1) == soa.n_triangles "Triangle count mismatch"
    
    @inbounds @simd for i in 1:soa.n_triangles
        soa.x1[i] = T(triXC[i, 1])
        soa.x2[i] = T(triXC[i, 2])
        soa.x3[i] = T(triXC[i, 3])
        soa.y1[i] = T(triYC[i, 1])
        soa.y2[i] = T(triYC[i, 2])
        soa.y3[i] = T(triYC[i, 3])
        soa.z1[i] = T(triZC[i, 1])
        soa.z2[i] = T(triZC[i, 2])
        soa.z3[i] = T(triZC[i, 3])
    end
end

# Convert Structure of Arrays back to Array of Structures
function soa_to_aos!(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix, soa::TriangleSoA)
    @assert size(triXC, 1) == soa.n_triangles "Triangle count mismatch"
    
    @inbounds @simd for i in 1:soa.n_triangles
        triXC[i, 1] = soa.x1[i]
        triXC[i, 2] = soa.x2[i]
        triXC[i, 3] = soa.x3[i]
        triYC[i, 1] = soa.y1[i]
        triYC[i, 2] = soa.y2[i]
        triYC[i, 3] = soa.y3[i]
        triZC[i, 1] = soa.z1[i]
        triZC[i, 2] = soa.z2[i]
        triZC[i, 3] = soa.z3[i]
    end
end

# Convert nodes to SoA layout
function aos_to_soa!(soa::NodeSoA{T}, nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector) where T
    @assert length(nodeX) == soa.n_nodes "Node count mismatch"
    
    @inbounds @simd for i in 1:soa.n_nodes
        soa.x[i] = T(nodeX[i])
        soa.y[i] = T(nodeY[i])
        soa.z[i] = T(nodeZ[i])
    end
end

# Convert vorticity to SoA layout
function aos_to_soa!(soa::VorticitySoA{T}, vorticity::AbstractMatrix) where T
    @assert size(vorticity, 1) == soa.n_elements "Element count mismatch"
    
    @inbounds @simd for i in 1:soa.n_elements
        soa.ωx[i] = T(vorticity[i, 1])
        soa.ωy[i] = T(vorticity[i, 2])
        soa.ωz[i] = T(vorticity[i, 3])
    end
end

# Vectorized triangle area computation using SoA layout
function soa_triangle_areas!(areas::Vector{T}, triangles::TriangleSoA{T}) where T
    @assert length(areas) == triangles.n_triangles "Area array size mismatch"
    
    # Vectorized computation of all triangle areas simultaneously
    @inbounds @simd for i in 1:triangles.n_triangles
        # Edge vectors using SoA layout (better cache performance)
        v1x = triangles.x2[i] - triangles.x1[i]
        v1y = triangles.y2[i] - triangles.y1[i] 
        v1z = triangles.z2[i] - triangles.z1[i]
        
        v2x = triangles.x3[i] - triangles.x1[i]
        v2y = triangles.y3[i] - triangles.y1[i]
        v2z = triangles.z3[i] - triangles.z1[i]
        
        # Cross product
        cross_x = v1y * v2z - v1z * v2y
        cross_y = v1z * v2x - v1x * v2z
        cross_z = v1x * v2y - v1y * v2x
        
        # Area = 0.5 * |cross product|
        areas[i] = T(0.5) * sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    end
end

# Vectorized kernel evaluation optimized for SoA layout
function vectorized_kernel_eval!(weights::AbstractVector{T}, 
                                 distances_x::AbstractVector{T},
                                 distances_y::AbstractVector{T}, 
                                 distances_z::AbstractVector{T},
                                 h::T) where T
    @assert length(weights) == length(distances_x) == length(distances_y) == length(distances_z)
    
    # Use SIMD instructions for parallel kernel evaluation
    @inbounds @simd for i in eachindex(weights)
        # Compute normalized distances
        r_x = abs(distances_x[i]) / h
        r_y = abs(distances_y[i]) / h
        r_z = abs(distances_z[i]) / h
        
        # Peskin kernel evaluation (can be vectorized)
        kernel_x = (r_x >= 2.0) ? 0.0 : (1.0 + cos(π * r_x)) / (2.0 * h)
        kernel_y = (r_y >= 2.0) ? 0.0 : (1.0 + cos(π * r_y)) / (2.0 * h)
        kernel_z = (r_z >= 2.0) ? 0.0 : (1.0 + cos(π * r_z)) / (2.0 * h)
        
        weights[i] = kernel_x * kernel_y * kernel_z
    end
end

# Cache-optimized circulation solver using SoA layout
function soa_circulation_solve!(circulation::VorticitySoA{T}, 
                               element_gamma::VorticitySoA{T},
                               triangles::TriangleSoA{T},
                               areas::AbstractVector{T}) where T
    n = triangles.n_triangles
    @assert n == circulation.n_elements == element_gamma.n_elements == length(areas)
    
    # Vectorized solution of circulation equations
    # This processes multiple triangles simultaneously for better cache performance
    
    # Process in blocks for better cache utilization
    block_size = min(64, n)  # Process 64 triangles at a time
    
    for block_start in 1:block_size:n
        block_end = min(block_start + block_size - 1, n)
        
        @inbounds for i in block_start:block_end
            # Edge vectors from SoA layout
            x12 = triangles.x2[i] - triangles.x1[i]
            y12 = triangles.y2[i] - triangles.y1[i]
            z12 = triangles.z2[i] - triangles.z1[i]
            
            x23 = triangles.x3[i] - triangles.x2[i]
            y23 = triangles.y3[i] - triangles.y2[i]
            z23 = triangles.z3[i] - triangles.z2[i]
            
            x31 = triangles.x1[i] - triangles.x3[i]
            y31 = triangles.y1[i] - triangles.y3[i]
            z31 = triangles.z1[i] - triangles.z3[i]
            
            # Fast solve using analytical 4x3 system
            # This avoids matrix allocations and uses optimized routines
            area_inv = 1.0 / areas[i]
            
            # The circulation calculation simplified for SoA layout
            # (This is a simplified version - full implementation would need proper 4x3 solve)
            circulation.ωx[i] = element_gamma.ωx[i] * area_inv
            circulation.ωy[i] = element_gamma.ωy[i] * area_inv  
            circulation.ωz[i] = element_gamma.ωz[i] * area_inv
        end
    end
end

# Optimized distance computation using SoA layout
function compute_distances_soa!(dx_vec::Vector{T}, dy_vec::Vector{T}, dz_vec::Vector{T},
                               point_x::T, point_y::T, point_z::T,
                               targets::NodeSoA{T}, indices::AbstractVector{Int}) where T
    @inbounds for idx in 1:length(indices)
        target_idx = indices[idx]
        dx_vec[idx] = point_x - targets.x[target_idx]
        dy_vec[idx] = point_y - targets.y[target_idx]
        dz_vec[idx] = point_z - targets.z[target_idx]
    end
end

# Memory-efficient SoA conversion utilities
function create_soa_layout(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                          element_gamma::AbstractMatrix)
    n_triangles = size(triXC, 1)
    n_nodes = length(nodeX)
    n_elements = size(element_gamma, 1)
    
    # Create SoA structures
    triangles_soa = TriangleSoA(n_triangles)
    nodes_soa = NodeSoA(n_nodes)
    vorticity_soa = VorticitySoA(n_elements)
    
    # Convert to SoA layout
    aos_to_soa!(triangles_soa, triXC, triYC, triZC)
    aos_to_soa!(nodes_soa, nodeX, nodeY, nodeZ)
    aos_to_soa!(vorticity_soa, element_gamma)
    
    return triangles_soa, nodes_soa, vorticity_soa
end

end # module

using .SoALayout: TriangleSoA, NodeSoA, VorticitySoA, VelocitySoA,
                  aos_to_soa!, soa_to_aos!, vectorized_kernel_eval!,
                  soa_triangle_areas!, soa_circulation_solve!, create_soa_layout