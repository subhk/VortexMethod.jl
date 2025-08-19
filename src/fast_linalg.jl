# Fast linear algebra routines optimized for small matrices in vortex methods

module FastLinAlg

using LinearAlgebra
using StaticArrays

export solve_3x3!, solve_4x3!, fast_inv_3x3!, fast_det_3x3, fast_cross_product!,
       batch_solve_3x3!, TriangleMatrix3x3, EdgeVectorCache,
       fast_triangle_area, fast_matrix_vector_3x3!

# Optimized 3x3 matrix operations using StaticArrays for better performance
const Matrix3x3 = SMatrix{3,3,Float64,9}
const Vector3 = SVector{3,Float64}

# Fast 3x3 determinant using explicit formula (faster than generic det)
@inline function fast_det_3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    return a11 * (a22 * a33 - a23 * a32) - 
           a12 * (a21 * a33 - a23 * a31) + 
           a13 * (a21 * a32 - a22 * a31)
end

@inline function fast_det_3x3(A::AbstractMatrix)
    @inbounds fast_det_3x3(A[1,1], A[1,2], A[1,3], A[2,1], A[2,2], A[2,3], A[3,1], A[3,2], A[3,3])
end

# Fast 3x3 matrix inversion using analytical formulas
function fast_inv_3x3!(A_inv::AbstractMatrix, A::AbstractMatrix)
    @inbounds begin
        a11, a12, a13 = A[1,1], A[1,2], A[1,3]
        a21, a22, a23 = A[2,1], A[2,2], A[2,3]
        a31, a32, a33 = A[3,1], A[3,2], A[3,3]
        
        # Compute determinant
        det_A = fast_det_3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33)
        
        if abs(det_A) < 1e-15
            fill!(A_inv, 0.0)  # Singular matrix
            return false
        end
        
        inv_det = 1.0 / det_A
        
        # Compute adjugate matrix elements
        A_inv[1,1] = (a22 * a33 - a23 * a32) * inv_det
        A_inv[1,2] = (a13 * a32 - a12 * a33) * inv_det
        A_inv[1,3] = (a12 * a23 - a13 * a22) * inv_det
        
        A_inv[2,1] = (a23 * a31 - a21 * a33) * inv_det
        A_inv[2,2] = (a11 * a33 - a13 * a31) * inv_det
        A_inv[2,3] = (a13 * a21 - a11 * a23) * inv_det
        
        A_inv[3,1] = (a21 * a32 - a22 * a31) * inv_det
        A_inv[3,2] = (a12 * a31 - a11 * a32) * inv_det
        A_inv[3,3] = (a11 * a22 - a12 * a21) * inv_det
    end
    
    return true
end

# Specialized solver for circulation 4x3 overdetermined systems
# Solves M * x = rhs where M is 4x3 (used in circulation calculations)
function solve_4x3!(x::AbstractVector, M::AbstractMatrix, rhs::AbstractVector)
    @assert size(M) == (4, 3) "Matrix must be 4x3"
    @assert length(x) == 3 "Solution vector must have length 3"
    @assert length(rhs) == 4 "RHS vector must have length 4"
    
    # Use normal equations: M^T M x = M^T rhs
    # This is more efficient than QR for small matrices
    @inbounds begin
        # Compute M^T M (3x3 symmetric)
        MTM11 = M[1,1]*M[1,1] + M[2,1]*M[2,1] + M[3,1]*M[3,1] + M[4,1]*M[4,1]
        MTM12 = M[1,1]*M[1,2] + M[2,1]*M[2,2] + M[3,1]*M[3,2] + M[4,1]*M[4,2]
        MTM13 = M[1,1]*M[1,3] + M[2,1]*M[2,3] + M[3,1]*M[3,3] + M[4,1]*M[4,3]
        MTM22 = M[1,2]*M[1,2] + M[2,2]*M[2,2] + M[3,2]*M[3,2] + M[4,2]*M[4,2]
        MTM23 = M[1,2]*M[1,3] + M[2,2]*M[2,3] + M[3,2]*M[3,3] + M[4,2]*M[4,3]
        MTM33 = M[1,3]*M[1,3] + M[2,3]*M[2,3] + M[3,3]*M[3,3] + M[4,3]*M[4,3]
        
        # Compute M^T rhs (3x1)
        MTrhs1 = M[1,1]*rhs[1] + M[2,1]*rhs[2] + M[3,1]*rhs[3] + M[4,1]*rhs[4]
        MTrhs2 = M[1,2]*rhs[1] + M[2,2]*rhs[2] + M[3,2]*rhs[3] + M[4,2]*rhs[4]
        MTrhs3 = M[1,3]*rhs[1] + M[2,3]*rhs[2] + M[3,3]*rhs[3] + M[4,3]*rhs[4]
        
        # Solve 3x3 system using fast inversion
        det = fast_det_3x3(MTM11, MTM12, MTM13, MTM12, MTM22, MTM23, MTM13, MTM23, MTM33)
        
        if abs(det) < 1e-15
            fill!(x, 0.0)
            return false
        end
        
        inv_det = 1.0 / det
        
        # Compute inverse manually for 3x3 symmetric matrix
        inv11 = (MTM22 * MTM33 - MTM23 * MTM23) * inv_det
        inv12 = (MTM13 * MTM23 - MTM12 * MTM33) * inv_det
        inv13 = (MTM12 * MTM23 - MTM13 * MTM22) * inv_det
        inv22 = (MTM11 * MTM33 - MTM13 * MTM13) * inv_det
        inv23 = (MTM12 * MTM13 - MTM11 * MTM23) * inv_det
        inv33 = (MTM11 * MTM22 - MTM12 * MTM12) * inv_det
        
        # Compute solution
        x[1] = inv11 * MTrhs1 + inv12 * MTrhs2 + inv13 * MTrhs3
        x[2] = inv12 * MTrhs1 + inv22 * MTrhs2 + inv23 * MTrhs3
        x[3] = inv13 * MTrhs1 + inv23 * MTrhs2 + inv33 * MTrhs3
    end
    
    return true
end

# Batch operations for multiple small matrices (better cache utilization)
function batch_solve_3x3!(X::AbstractMatrix, A_batch::AbstractArray{T,3}, b_batch::AbstractMatrix) where T
    n_matrices = size(A_batch, 3)
    @assert size(X, 2) == n_matrices "Output matrix must have same number of columns as batch size"
    @assert size(A_batch, 1) == size(A_batch, 2) == 3 "Each matrix must be 3x3"
    
    # Process matrices in batches for better cache performance
    batch_size = min(64, n_matrices)  # Process 64 matrices at a time
    
    for batch_start in 1:batch_size:n_matrices
        batch_end = min(batch_start + batch_size - 1, n_matrices)
        
        @inbounds for i in batch_start:batch_end
            # Extract 3x3 matrix and 3x1 vector
            A = @view A_batch[:, :, i]
            b = @view b_batch[:, i]
            x = @view X[:, i]
            
            # Fast solve using analytical inverse
            solve_3x3!(x, A, b)
        end
    end
end

function solve_3x3!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    @inbounds begin
        # Direct solve using Cramer's rule (fastest for 3x3)
        det_A = fast_det_3x3(A)
        
        if abs(det_A) < 1e-15
            fill!(x, 0.0)
            return false
        end
        
        inv_det = 1.0 / det_A
        
        # Cramer's rule for each component
        x[1] = fast_det_3x3(b[1], A[1,2], A[1,3], b[2], A[2,2], A[2,3], b[3], A[3,2], A[3,3]) * inv_det
        x[2] = fast_det_3x3(A[1,1], b[1], A[1,3], A[2,1], b[2], A[2,3], A[3,1], b[3], A[3,3]) * inv_det  
        x[3] = fast_det_3x3(A[1,1], A[1,2], b[1], A[2,1], A[2,2], b[2], A[3,1], A[3,2], b[3]) * inv_det
    end
    
    return true
end

# Fast cross product for triangle normal computation
@inline function fast_cross_product!(result::AbstractVector, a::AbstractVector, b::AbstractVector)
    @inbounds begin
        result[1] = a[2] * b[3] - a[3] * b[2]
        result[2] = a[3] * b[1] - a[1] * b[3]
        result[3] = a[1] * b[2] - a[2] * b[1]
    end
end

# Optimized triangle area computation using cross product
@inline function fast_triangle_area(p1::NTuple{3,T}, p2::NTuple{3,T}, p3::NTuple{3,T}) where T
    # Vectors from p1 to p2 and p1 to p3
    v1_x, v1_y, v1_z = p2[1] - p1[1], p2[2] - p1[2], p2[3] - p1[3]
    v2_x, v2_y, v2_z = p3[1] - p1[1], p3[2] - p1[2], p3[3] - p1[3]
    
    # Cross product
    cross_x = v1_y * v2_z - v1_z * v2_y
    cross_y = v1_z * v2_x - v1_x * v2_z
    cross_z = v1_x * v2_y - v1_y * v2_x
    
    # Area = 0.5 * |cross|
    return T(0.5) * sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
end

# Fast 3x3 matrix-vector multiplication
@inline function fast_matrix_vector_3x3!(result::AbstractVector, A::AbstractMatrix, x::AbstractVector)
    @inbounds begin
        result[1] = A[1,1] * x[1] + A[1,2] * x[2] + A[1,3] * x[3]
        result[2] = A[2,1] * x[1] + A[2,2] * x[2] + A[2,3] * x[3]
        result[3] = A[3,1] * x[1] + A[3,2] * x[2] + A[3,3] * x[3]
    end
end

# Cache-optimized data structures for repeated computations
struct TriangleMatrix3x3
    matrices::Array{Float64, 3}  # Pre-computed 3x3 matrices for each triangle
    valid::Vector{Bool}           # Whether each matrix is valid (non-singular)
end

function TriangleMatrix3x3(n_triangles::Int)
    TriangleMatrix3x3(
        Array{Float64}(undef, 3, 3, n_triangles),
        Vector{Bool}(undef, n_triangles)
    )
end

# Edge vector cache for circulation calculations
struct EdgeVectorCache{T<:AbstractFloat}
    edge_vectors::Array{T, 3}    # [triangle, edge(1-3), component(x,y,z)]
    edge_lengths::Matrix{T}      # [triangle, edge]  
    triangle_areas::Vector{T}    # Pre-computed areas
    matrix_cache::TriangleMatrix3x3  # Pre-computed circulation matrices
end

function EdgeVectorCache(::Type{T}, n_triangles::Int) where T
    EdgeVectorCache{T}(
        Array{T}(undef, n_triangles, 3, 3),
        Matrix{T}(undef, n_triangles, 3),
        Vector{T}(undef, n_triangles),
        TriangleMatrix3x3(n_triangles)
    )
end

EdgeVectorCache(n_triangles::Int) = EdgeVectorCache(Float64, n_triangles)

end # module

using .FastLinAlg: solve_3x3!, solve_4x3!, fast_inv_3x3!, fast_det_3x3, fast_cross_product!,
                   batch_solve_3x3!, TriangleMatrix3x3, EdgeVectorCache,
                   fast_triangle_area, fast_matrix_vector_3x3!