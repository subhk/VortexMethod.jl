# Multiple interpolation kernels for vortex methods
# Implements various spreading/interpolation schemes described in the thesis

module Kernels

using StaticArrays

export KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,
       kernel_function, kernel_support_radius, kernel_function_vec!,
       compute_distances!, KernelWorkspace, spread_element_kernel!,
       spread_element_kernel_workspace!, interpolate_kernel_weight

abstract type KernelType end

# Standard Peskin kernel (current implementation)
struct PeskinStandard <: KernelType
    delr::Float64
    PeskinStandard(delr=4.0) = new(delr)
end

# Cosine-based Peskin kernel 
struct PeskinCosine <: KernelType
    delr::Float64
    PeskinCosine(delr=4.0) = new(delr)
end

# M4' kernel (4th order with compact support)
struct M4Prime <: KernelType
    delr::Float64
    M4Prime(delr=2.0) = new(delr)
end

# Area-weighted distribution (thesis Chapter 2.3.2)
struct AreaWeighting <: KernelType
    delr::Float64
    AreaWeighting(delr=1.0) = new(delr)
end

# Kernel support radius
@inline kernel_support_radius(k::PeskinStandard) = k.delr
@inline kernel_support_radius(k::PeskinCosine)   = k.delr
@inline kernel_support_radius(k::M4Prime)        = k.delr
@inline kernel_support_radius(k::AreaWeighting)  = k.delr

# 1D kernel functions
@inline function kernel_1d(::PeskinStandard, r::T, h::T)::T where T<:AbstractFloat
    # Standard 4-point Peskin discrete delta function
    x = abs(r) / h
    if x >= T(2)
        return zero(T)
    elseif x <= one(T)
        return (T(3) - T(2)*x + sqrt(one(T) + T(4)*x - T(4)*x^2)) / (T(8)*h)
    else
        return (T(5) - T(2)*x - sqrt(-T(7) + T(12)*x - T(4)*x^2)) / (T(8)*h)
    end
end

@inline function kernel_1d(::PeskinCosine, r::T, h::T)::T where T<:AbstractFloat
    # Improved cosine kernel with better smoothness properties
    x = abs(r) / h
    if x >= T(1.5)
        return zero(T)
    else
        return (one(T) + cos(T(2π)*x/T(3))) / (T(3)*h)
    end
end

@inline function kernel_1d(::M4Prime, r::T, h::T)::T where T<:AbstractFloat
    # 4th order accurate kernel with compact support
    x = abs(r) / h
    if x >= T(2)
        return zero(T)
    elseif x <= T(0.5)
        return (T(3) - T(2)*x + sqrt(one(T) + T(4)*x - T(4)*x^2)) / (T(8)*h)
    elseif x <= T(1.5)
        return (T(5) - T(2)*x - sqrt(-T(7) + T(12)*x - T(4)*x^2)) / (T(8)*h)
    else
        return zero(T)
    end
end

@inline function kernel_1d(::AreaWeighting, r::T, h::T)::T where T<:AbstractFloat
    # Area-weighted distribution (hat function)
    x = abs(r) / h
    if x >= one(T)
        return zero(T)
    else
        return (one(T) - x) / h
    end
end

# 3D kernel function
@inline function kernel_function(k::KernelType, dx::T, dy::T, dz::T,
                                hx::T, hy::T, hz::T)::T where T<:AbstractFloat
    return kernel_1d(k, dx, hx) * kernel_1d(k, dy, hy) * kernel_1d(k, dz, hz)
end

@inline function kernel_function(k::KernelType, d::SVector{3,T}, h::SVector{3,T})::T where T<:AbstractFloat
    return kernel_function(k, d[1], d[2], d[3], h[1], h[2], h[3])
end

# Optimized vectorized kernel evaluation for multiple points
@inline function kernel_function_vec!(weights::AbstractVector{T},
                                     k::KernelType,
                                     dx_vec::AbstractVector{T},
                                     dy_vec::AbstractVector{T},
                                     dz_vec::AbstractVector{T},
                                     hx::T, hy::T, hz::T) where T<:AbstractFloat
    @inbounds @simd for i in eachindex(weights)
        weights[i] = kernel_1d(k, dx_vec[i], hx) * 
                    kernel_1d(k, dy_vec[i], hy) * 
                    kernel_1d(k, dz_vec[i], hz)
    end
    return nothing
end

# Fast distance computation with SIMD
@inline function compute_distances!(dx_vec::AbstractVector{T},
                                   dy_vec::AbstractVector{T},
                                   dz_vec::AbstractVector{T},
                                   coord::SVector{3,T},
                                   subC::AbstractArray{T,3},
                                   idx::Int) where T<:AbstractFloat
    @inbounds @simd for s in 1:size(subC,2)
        dx_vec[s] = coord[1] - subC[idx,s,1]
        dy_vec[s] = coord[2] - subC[idx,s,2]
        dz_vec[s] = coord[3] - subC[idx,s,3]
    end
    return nothing
end

@inline function compute_distances!(dx_vec::AbstractVector{T},
                                   dy_vec::AbstractVector{T},
                                   dz_vec::AbstractVector{T},
                                   coord,
                                   subC::AbstractArray{T,3},
                                   idx::Int) where T<:AbstractFloat
    sv_coord = SVector{3,T}(T(coord[1]), T(coord[2]), T(coord[3]))
    return compute_distances!(dx_vec, dy_vec, dz_vec, sv_coord, subC, idx)
end

# Enhanced spreading function with kernel selection and vectorization
function spread_element_kernel!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix, 
                               subC, triAreas, tri_list, coord, kernel::KernelType, 
                               eps::NTuple{3,Float64})
    sx, sy, sz = sum
    (epsx, epsy, epsz) = eps
    x = SVector{3,Float64}(Float64(coord[1]), Float64(coord[2]), Float64(coord[3]))
    delr = kernel_support_radius(kernel)
    hx, hy, hz = epsx/delr, epsy/delr, epsz/delr
    
    # Pre-allocate temporary arrays for vectorized operations
    n_sub = size(subC, 2)
    dx_vec = Vector{Float64}(undef, n_sub)
    dy_vec = Vector{Float64}(undef, n_sub)
    dz_vec = Vector{Float64}(undef, n_sub)
    weights = Vector{Float64}(undef, n_sub)
    
    @inbounds for idx in tri_list
        # Vectorized distance computation
        compute_distances!(dx_vec, dy_vec, dz_vec, x, subC, idx)
        
        # Vectorized kernel evaluation
        kernel_function_vec!(weights, kernel, dx_vec, dy_vec, dz_vec, hx, hy, hz)
        
        # Sum weights
        S = 0.0
        @simd for s in 1:n_sub
            S += weights[s]
        end
        
        weight = triAreas[idx] * (S / n_sub)
        sx += weight * eleGma[idx,1]
        sy += weight * eleGma[idx,2]
        sz += weight * eleGma[idx,3]
    end
    return (sx, sy, sz)
end

# Memory-efficient version that reuses workspace
struct KernelWorkspace{T<:AbstractFloat}
    dx_vec::Vector{T}
    dy_vec::Vector{T}
    dz_vec::Vector{T}
    weights::Vector{T}
end

KernelWorkspace(::Type{T}, n::Int) where T = KernelWorkspace{T}(
    Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n), Vector{T}(undef, n)
)
KernelWorkspace(n::Int) = KernelWorkspace(Float64, n)

function spread_element_kernel_workspace!(workspace::KernelWorkspace, 
                                        sum::NTuple{3,Float64}, eleGma::AbstractMatrix, 
                                        subC, triAreas, tri_list, coord, kernel::KernelType, 
                                        eps::NTuple{3,Float64})
    sx, sy, sz = sum
    (epsx, epsy, epsz) = eps
    x = SVector{3,Float64}(Float64(coord[1]), Float64(coord[2]), Float64(coord[3]))
    delr = kernel_support_radius(kernel)
    hx, hy, hz = epsx/delr, epsy/delr, epsz/delr
    
    # Reuse workspace arrays
    dx_vec, dy_vec, dz_vec, weights = workspace.dx_vec, workspace.dy_vec, workspace.dz_vec, workspace.weights
    n_sub = length(dx_vec)
    
    @inbounds for idx in tri_list
        # Vectorized distance computation
        compute_distances!(dx_vec, dy_vec, dz_vec, x, subC, idx)
        
        # Vectorized kernel evaluation
        kernel_function_vec!(weights, kernel, dx_vec, dy_vec, dz_vec, hx, hy, hz)
        
        # Sum weights
        S = 0.0
        @simd for s in 1:n_sub
            S += weights[s]
        end
        
        weight = triAreas[idx] * (S / n_sub)
        sx += weight * eleGma[idx,1]
        sy += weight * eleGma[idx,2]
        sz += weight * eleGma[idx,3]
    end
    return (sx, sy, sz)
end

# Enhanced interpolation function with kernel selection
@inline function interpolate_kernel_weight(kernel::KernelType, dx::T, dy::T, dz::T,
                                           hx::T, hy::T, hz::T)::T where T<:AbstractFloat
    delr = T(kernel_support_radius(kernel))
    return kernel_function(kernel, dx, dy, dz, hx/delr, hy/delr, hz/delr)
end

end # module

using .Kernels: KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,
                kernel_function, kernel_support_radius, spread_element_kernel!,
                interpolate_kernel_weight, kernel_function_vec!, compute_distances!,
                KernelWorkspace, spread_element_kernel_workspace!
