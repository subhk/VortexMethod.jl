# Multiple interpolation kernels for vortex methods
# Implements various spreading/interpolation schemes described in the thesis

module Kernels

export KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,
       kernel_function, kernel_support_radius

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
kernel_support_radius(k::PeskinStandard) = k.delr
kernel_support_radius(k::PeskinCosine)   = k.delr
kernel_support_radius(k::M4Prime)        = k.delr
kernel_support_radius(k::AreaWeighting)  = k.delr

# 1D kernel functions
function kernel_1d(::PeskinStandard, r::Float64, h::Float64)::Float64
    # Standard discrete delta function
    x = abs(r) / h
    if x >= 2.0
        return 0.0
    elseif x <= 1.0
        return (1 + cos(π*x)) / (2*h)
    else
        return (1 + cos(π*x)) / (2*h)
    end
end

function kernel_1d(::PeskinCosine, r::Float64, h::Float64)::Float64
    # Improved cosine kernel with better smoothness properties
    x = abs(r) / h
    if x >= 1.5
        return 0.0
    else
        return (1 + cos(2*π*x/3)) / (3*h)
    end
end

function kernel_1d(::M4Prime, r::Float64, h::Float64)::Float64
    # 4th order accurate kernel with compact support
    x = abs(r) / h
    if x >= 2.0
        return 0.0
    elseif x <= 0.5
        return (3 - 2*x + sqrt(1 + 4*x - 4*x^2)) / (8*h)
    elseif x <= 1.5
        return (5 - 2*x - sqrt(-7 + 12*x - 4*x^2)) / (8*h)
    else
        return 0.0
    end
end

function kernel_1d(::AreaWeighting, r::Float64, h::Float64)::Float64
    # Area-weighted distribution (hat function)
    x = abs(r) / h
    if x >= 1.0
        return 0.0
    else
        return (1 - x) / h
    end
end

# 3D kernel function
function kernel_function(k::KernelType, dx::Float64, dy::Float64, dz::Float64, 
                        hx::Float64, hy::Float64, hz::Float64)::Float64
    return kernel_1d(k, dx, hx) * kernel_1d(k, dy, hy) * kernel_1d(k, dz, hz)
end

# Optimized vectorized kernel evaluation for multiple points
@inline function kernel_function_vec!(weights::AbstractVector{Float64}, 
                                     k::KernelType,
                                     dx_vec::AbstractVector{Float64}, 
                                     dy_vec::AbstractVector{Float64}, 
                                     dz_vec::AbstractVector{Float64},
                                     hx::Float64, hy::Float64, hz::Float64)
    @inbounds @simd for i in eachindex(weights)
        weights[i] = kernel_1d(k, dx_vec[i], hx) * 
                    kernel_1d(k, dy_vec[i], hy) * 
                    kernel_1d(k, dz_vec[i], hz)
    end
    return nothing
end

# Fast distance computation with SIMD
@inline function compute_distances!(dx_vec::AbstractVector{Float64},
                                   dy_vec::AbstractVector{Float64}, 
                                   dz_vec::AbstractVector{Float64},
                                   coord::AbstractVector{Float64},
                                   subC::AbstractArray{Float64,3},
                                   idx::Int)
    @inbounds @simd for s in 1:size(subC,2)
        dx_vec[s] = coord[1] - subC[idx,s,1]
        dy_vec[s] = coord[2] - subC[idx,s,2]
        dz_vec[s] = coord[3] - subC[idx,s,3]
    end
    return nothing
end

# Enhanced spreading function with kernel selection and vectorization
function spread_element_kernel!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix, 
                               subC, triAreas, tri_list, coord, kernel::KernelType, 
                               eps::NTuple{3,Float64})
    sx, sy, sz = sum
    (epsx, epsy, epsz) = eps
    x = coord
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
    x = coord
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
function interpolate_kernel_weight(kernel::KernelType, dx::Float64, dy::Float64, dz::Float64,
                                 hx::Float64, hy::Float64, hz::Float64)::Float64
    delr = kernel_support_radius(kernel)
    return kernel_function(kernel, dx, dy, dz, hx/delr, hy/delr, hz/delr)
end

end # module

using .Kernels: KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,
                kernel_function, kernel_support_radius, spread_element_kernel!,
                interpolate_kernel_weight