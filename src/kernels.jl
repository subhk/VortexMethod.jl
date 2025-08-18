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
function kernel_1d(::PeskinStandard, r::Float64, h::Float64)
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

function kernel_1d(::PeskinCosine, r::Float64, h::Float64)
    # Improved cosine kernel with better smoothness properties
    x = abs(r) / h
    if x >= 1.5
        return 0.0
    else
        return (1 + cos(2*π*x/3)) / (3*h)
    end
end

function kernel_1d(::M4Prime, r::Float64, h::Float64)
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

function kernel_1d(::AreaWeighting, r::Float64, h::Float64)
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
                        hx::Float64, hy::Float64, hz::Float64)
    return kernel_1d(k, dx, hx) * kernel_1d(k, dy, hy) * kernel_1d(k, dz, hz)
end

# Enhanced spreading function with kernel selection
function spread_element_kernel!(sum::NTuple{3,Float64}, eleGma::AbstractMatrix, 
                               subC, triAreas, tri_list, coord, kernel::KernelType, 
                               eps::NTuple{3,Float64})
    sx, sy, sz = sum
    (epsx, epsy, epsz) = eps
    x = coord
    delr = kernel_support_radius(kernel)
    
    @inbounds for idx in tri_list
        S = 0.0
        for s in 1:size(subC,2)
            dx = x[1] - subC[idx,s,1]
            dy = x[2] - subC[idx,s,2]
            dz = x[3] - subC[idx,s,3]
            
            # Use selected kernel function
            w = kernel_function(kernel, dx, dy, dz, epsx/delr, epsy/delr, epsz/delr)
            S += w
        end
        weight = triAreas[idx] * (S/size(subC,2))
        sx += weight * eleGma[idx,1]
        sy += weight * eleGma[idx,2]
        sz += weight * eleGma[idx,3]
    end
    return (sx, sy, sz)
end

# Enhanced interpolation function with kernel selection
function interpolate_kernel_weight(kernel::KernelType, dx::Float64, dy::Float64, dz::Float64,
                                 hx::Float64, hy::Float64, hz::Float64)
    delr = kernel_support_radius(kernel)
    return kernel_function(kernel, dx, dy, dz, hx/delr, hy/delr, hz/delr)
end

end # module

using .Kernels: KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,
                kernel_function, kernel_support_radius, spread_element_kernel!,
                interpolate_kernel_weight