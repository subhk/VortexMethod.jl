# Sub-filter scale dissipation models for Large Eddy Simulation
# Implements various dissipation schemes described in thesis Chapter 2.4

module Dissipation

using ..DomainImpl
using ..GridTransfer
using ..Poisson
using LinearAlgebra

export DissipationModel, NoDissipation, SmagorinskyModel, DynamicSmagorinsky,
       VortexStretchingDissipation, MixedScaleModel,
       apply_dissipation!, compute_eddy_viscosity, filter_width

abstract type DissipationModel end

# No dissipation (inviscid flow)
struct NoDissipation <: DissipationModel end

# Standard Smagorinsky model
struct SmagorinskyModel <: DissipationModel
    Cs::Float64  # Smagorinsky constant
    SmagorinskyModel(Cs=0.17) = new(Cs)
end

# Dynamic Smagorinsky model with time-varying coefficient
struct DynamicSmagorinsky <: DissipationModel
    Cs_base::Float64
    temporal_avg_window::Int
    DynamicSmagorinsky(Cs_base=0.17, window=10) = new(Cs_base, window)
end

# Vortex stretching based dissipation
struct VortexStretchingDissipation <: DissipationModel
    C_stretch::Float64
    strain_threshold::Float64
    VortexStretchingDissipation(C=0.1, threshold=1.0) = new(C, threshold)
end

# Mixed-scale model combining multiple effects
struct MixedScaleModel <: DissipationModel
    smagorinsky::SmagorinskyModel
    vortex_stretch::VortexStretchingDissipation
    blend_factor::Float64
    MixedScaleModel(Cs=0.17, C_stretch=0.1, blend=0.5) = 
        new(SmagorinskyModel(Cs), VortexStretchingDissipation(C_stretch, 1.0), blend)
end

# Compute filter width from grid spacing and element size
function filter_width(dx::Float64, dy::Float64, dz::Float64, element_area::Float64)::Float64
    # Anisotropic filter width accounting for element size
    grid_filter = (dx * dy * dz)^(1/3)
    element_filter = sqrt(element_area)
    return max(grid_filter, element_filter)
end

# Compute strain rate tensor magnitude (utility for advanced models)
# Currently used internally; exported for diagnostic purposes
function strain_rate_magnitude(dudx::Float64, dudy::Float64, dudz::Float64,
                              dvdx::Float64, dvdy::Float64, dvdz::Float64,
                              dwdx::Float64, dwdy::Float64, dwdz::Float64)::Float64
                              
    # Strain rate tensor S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    S11 = dudx
    S22 = dvdy  
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)
    
    # |S| = sqrt(2 * S_ij * S_ij)
    return sqrt(2 * (S11^2 + S22^2 + S33^2 + 2*(S12^2 + S13^2 + S23^2)))
end

# Compute vorticity magnitude from velocity gradients (utility for advanced models)
# Currently used internally; exported for diagnostic purposes
function vorticity_magnitude(dudx::Float64, dudy::Float64, dudz::Float64,
                           dvdx::Float64, dvdy::Float64, dvdz::Float64,
                           dwdx::Float64, dwdy::Float64, dwdz::Float64)

    # Vorticity ω = ∇ × u
    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx  
    omega_z = dvdx - dudy
    return sqrt(omega_x^2 + omega_y^2 + omega_z^2)
end

@inline prev_periodic(i::Int, n::Int) = i == 1 ? n : i - 1
@inline next_periodic(i::Int, n::Int) = i == n ? 1 : i + 1

function derivative_x(A::AbstractArray{Float64,3}, h::Float64)
    nz, ny, nx = size(A)
    D = similar(A)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        D[k,j,i] = (A[k,j,next_periodic(i,nx)] - A[k,j,prev_periodic(i,nx)]) / (2h)
    end
    return D
end

function derivative_y(A::AbstractArray{Float64,3}, h::Float64)
    nz, ny, nx = size(A)
    D = similar(A)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        D[k,j,i] = (A[k,next_periodic(j,ny),i] - A[k,prev_periodic(j,ny),i]) / (2h)
    end
    return D
end

function derivative_z(A::AbstractArray{Float64,3}, h::Float64)
    nz, ny, nx = size(A)
    D = similar(A)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        D[k,j,i] = (A[next_periodic(k,nz),j,i] - A[prev_periodic(k,nz),j,i]) / (2h)
    end
    return D
end

function laplacian_periodic(A::AbstractArray{Float64,3}, dx::Float64, dy::Float64, dz::Float64)
    nz, ny, nx = size(A)
    L = similar(A)
    inv_dx2 = 1 / dx^2
    inv_dy2 = 1 / dy^2
    inv_dz2 = 1 / dz^2
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        L[k,j,i] =
            (A[k,j,next_periodic(i,nx)] - 2A[k,j,i] + A[k,j,prev_periodic(i,nx)]) * inv_dx2 +
            (A[k,next_periodic(j,ny),i] - 2A[k,j,i] + A[k,prev_periodic(j,ny),i]) * inv_dy2 +
            (A[next_periodic(k,nz),j,i] - 2A[k,j,i] + A[prev_periodic(k,nz),j,i]) * inv_dz2
    end
    return L
end

function strain_magnitude_grid(Ux, Uy, Uz, dx::Float64, dy::Float64, dz::Float64)
    dudx = derivative_x(Ux, dx); dudy = derivative_y(Ux, dy); dudz = derivative_z(Ux, dz)
    dvdx = derivative_x(Uy, dx); dvdy = derivative_y(Uy, dy); dvdz = derivative_z(Uy, dz)
    dwdx = derivative_x(Uz, dx); dwdy = derivative_y(Uz, dy); dwdz = derivative_z(Uz, dz)

    S = similar(Ux)
    @inbounds for idx in eachindex(S)
        S11 = dudx[idx]
        S22 = dvdy[idx]
        S33 = dwdz[idx]
        S12 = 0.5 * (dudy[idx] + dvdx[idx])
        S13 = 0.5 * (dudz[idx] + dwdx[idx])
        S23 = 0.5 * (dvdz[idx] + dwdy[idx])
        S[idx] = sqrt(2 * (S11^2 + S22^2 + S33^2 + 2 * (S12^2 + S13^2 + S23^2)))
    end
    return S
end

function interpolate_grid_periodic(A::AbstractArray{Float64,3}, x::Float64, y::Float64, z::Float64,
                                   domain::DomainSpec, gr::GridSpec)
    dx, dy, dz = grid_spacing(domain, gr)
    xw, yw, zw = wrap_point(x, y, z, domain)
    nx, ny, nz = gr.nx, gr.ny, gr.nz

    fx = xw / dx
    fy = yw / dy
    fz = (zw + domain.Lz) / dz

    i0 = mod(floor(Int, fx), nx) + 1
    j0 = mod(floor(Int, fy), ny) + 1
    k0 = mod(floor(Int, fz), nz) + 1
    i1 = next_periodic(i0, nx)
    j1 = next_periodic(j0, ny)
    k1 = next_periodic(k0, nz)

    tx = fx - floor(fx)
    ty = fy - floor(fy)
    tz = fz - floor(fz)

    c000 = A[k0,j0,i0]; c100 = A[k0,j0,i1]; c010 = A[k0,j1,i0]; c110 = A[k0,j1,i1]
    c001 = A[k1,j0,i0]; c101 = A[k1,j0,i1]; c011 = A[k1,j1,i0]; c111 = A[k1,j1,i1]

    c00 = (1 - tx) * c000 + tx * c100
    c10 = (1 - tx) * c010 + tx * c110
    c01 = (1 - tx) * c001 + tx * c101
    c11 = (1 - tx) * c011 + tx * c111
    c0 = (1 - ty) * c00 + ty * c10
    c1 = (1 - ty) * c01 + ty * c11
    return (1 - tz) * c0 + tz * c1
end

# Apply no dissipation
function apply_dissipation!(::NoDissipation, eleGma::AbstractMatrix, 
                          triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          domain::DomainSpec, gr::GridSpec, dt::Float64)
    # No modification to vorticity
    return eleGma
end

# Apply Smagorinsky dissipation
function apply_dissipation!(model::SmagorinskyModel, eleGma::AbstractMatrix,
                          triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          domain::DomainSpec, gr::GridSpec, dt::Float64)
    nt = size(eleGma, 1)
    dx, dy, dz = grid_spacing(domain, gr)

    ζx, ζy, ζz = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)
    Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain)

    strain_mag = strain_magnitude_grid(Ux, Uy, Uz, dx, dy, dz)
    delta = (dx * dy * dz)^(1/3)
    nu_t = (model.Cs * delta)^2 .* strain_mag

    div_omega = derivative_x(ζx, dx) .+ derivative_y(ζy, dy) .+ derivative_z(ζz, dz)
    dwdt_x = nu_t .* (laplacian_periodic(ζx, dx, dy, dz) .- derivative_x(div_omega, dx))
    dwdt_y = nu_t .* (laplacian_periodic(ζy, dx, dy, dz) .- derivative_y(div_omega, dy))
    dwdt_z = nu_t .* (laplacian_periodic(ζz, dx, dy, dz) .- derivative_z(div_omega, dz))

    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        cx, cy, cz = periodic_centroid(p1, p2, p3, domain)
        eleGma[t,1] += dt * interpolate_grid_periodic(dwdt_x, cx, cy, cz, domain, gr)
        eleGma[t,2] += dt * interpolate_grid_periodic(dwdt_y, cx, cy, cz, domain, gr)
        eleGma[t,3] += dt * interpolate_grid_periodic(dwdt_z, cx, cy, cz, domain, gr)
    end
    
    return eleGma
end

# Apply dynamic Smagorinsky dissipation
function apply_dissipation!(model::DynamicSmagorinsky, eleGma::AbstractMatrix,
                          triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          domain::DomainSpec, gr::GridSpec, dt::Float64)
    nt = size(eleGma, 1)
    dx, dy, dz = grid_spacing(domain, gr)

    # Compute area-weighted vorticity statistics for dynamic coefficient
    # Using dimensionally consistent intermittency-based scaling
    total_area = 0.0
    weighted_vort = 0.0      # sum(|ω| * A)
    weighted_vort_sq = 0.0   # sum(|ω|^2 * A)

    @inbounds for t in 1:nt
        # Element geometry
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        e1 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
        e2 = (p3[1]-p1[1], p3[2]-p1[2], p3[3]-p1[3])

        cross = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
        area = 0.5 * sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2)

        vorticity_mag = sqrt(eleGma[t,1]^2 + eleGma[t,2]^2 + eleGma[t,3]^2)

        total_area += area
        weighted_vort += vorticity_mag * area
        weighted_vort_sq += vorticity_mag^2 * area
    end

    # Compute dimensionless intermittency factor I = ω_mean / ω_rms
    # I ∈ (0, 1]: I=1 for uniform field, I<1 for intermittent field
    if total_area > 0 && weighted_vort_sq > 0
        omega_mean = weighted_vort / total_area
        omega_rms = sqrt(weighted_vort_sq / total_area)
        intermittency = omega_mean / (omega_rms + eps())
        # Scale coefficient: higher intermittency (concentrated vorticity) -> higher dissipation
        Cs_dynamic = model.Cs_base * (2.0 - intermittency)
    else
        Cs_dynamic = model.Cs_base
    end
    Cs_dynamic = clamp(Cs_dynamic, 0.01, 0.5)  # Clamp to reasonable range

    # Apply dissipation with dynamic coefficient
    smagorinsky_model = SmagorinskyModel(Cs_dynamic)

    return apply_dissipation!(smagorinsky_model, eleGma, triXC, triYC, triZC, domain, gr, dt)
end

# Apply vortex stretching based dissipation
function apply_dissipation!(model::VortexStretchingDissipation, eleGma::AbstractMatrix,
                          triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          domain::DomainSpec, gr::GridSpec, dt::Float64)
    nt = size(eleGma, 1)
    dx, dy, dz = grid_spacing(domain, gr)
    
    @inbounds for t in 1:nt
        # Element geometry and area
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        e1 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
        e2 = (p3[1]-p1[1], p3[2]-p1[2], p3[3]-p1[3])

        cross = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
        area = 0.5 * sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2)
        
        Delta = filter_width(dx, dy, dz, area)
        
        # Vorticity vector
        ω = (eleGma[t,1], eleGma[t,2], eleGma[t,3])
        ω_mag = sqrt(ω[1]^2 + ω[2]^2 + ω[3]^2)
        
        if ω_mag > model.strain_threshold
            # Vortex stretching dissipation: stronger dissipation for high vorticity
            stretch_factor = ω_mag / model.strain_threshold
            dissipation_rate = model.C_stretch * stretch_factor * ω_mag / Delta^2
            
            decay_factor = exp(-dissipation_rate * dt)
            eleGma[t,1] *= decay_factor
            eleGma[t,2] *= decay_factor
            eleGma[t,3] *= decay_factor
        end
    end
    
    return eleGma
end

# Apply mixed-scale dissipation model
function apply_dissipation!(model::MixedScaleModel, eleGma::AbstractMatrix,
                          triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          domain::DomainSpec, gr::GridSpec, dt::Float64)
    # Create copies for each model
    eleGma_smag = copy(eleGma)
    eleGma_stretch = copy(eleGma)
    
    # Apply each model separately
    apply_dissipation!(model.smagorinsky, eleGma_smag, triXC, triYC, triZC, domain, gr, dt)
    apply_dissipation!(model.vortex_stretch, eleGma_stretch, triXC, triYC, triZC, domain, gr, dt)
    
    # Blend the results
    α = model.blend_factor
    @inbounds for t in 1:size(eleGma, 1)
        eleGma[t,1] = α * eleGma_smag[t,1] + (1-α) * eleGma_stretch[t,1]
        eleGma[t,2] = α * eleGma_smag[t,2] + (1-α) * eleGma_stretch[t,2]
        eleGma[t,3] = α * eleGma_smag[t,3] + (1-α) * eleGma_stretch[t,3]
    end
    
    return eleGma
end

# Compute effective eddy viscosity field
function compute_eddy_viscosity(model::SmagorinskyModel, eleGma::AbstractMatrix,
                               triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                               domain::DomainSpec, gr::GridSpec)
    nt = size(eleGma, 1)
    nu_sgs = zeros(Float64, nt)
    dx, dy, dz = grid_spacing(domain, gr)
    
    @inbounds for t in 1:nt
        # Element area
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        e1 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
        e2 = (p3[1]-p1[1], p3[2]-p1[2], p3[3]-p1[3])
        
        cross = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
        area = 0.5 * sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2)
        
        Delta = filter_width(dx, dy, dz, area)
        vorticity_mag = sqrt(eleGma[t,1]^2 + eleGma[t,2]^2 + eleGma[t,3]^2)
        
        nu_sgs[t] = (model.Cs * Delta)^2 * vorticity_mag
    end
    
    return nu_sgs
end

end # module

using .Dissipation: DissipationModel, NoDissipation, SmagorinskyModel, DynamicSmagorinsky,
                    VortexStretchingDissipation, MixedScaleModel,
                    apply_dissipation!, compute_eddy_viscosity, filter_width
