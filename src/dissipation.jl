# Sub-filter scale dissipation models for Large Eddy Simulation
# Implements various dissipation schemes described in thesis Chapter 2.4

module Dissipation

using ..DomainImpl
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
    
    @inbounds for t in 1:nt
        # Element centroid
        cx = (triXC[t,1] + triXC[t,2] + triXC[t,3]) / 3
        cy = (triYC[t,1] + triYC[t,2] + triYC[t,3]) / 3
        cz = (triZC[t,1] + triZC[t,2] + triZC[t,3]) / 3
        
        # Element area
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        e1 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
        e2 = (p3[1]-p1[1], p3[2]-p1[2], p3[3]-p1[3])

        cross = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
        area = 0.5 * sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2)
        
        # Filter width
        Delta = filter_width(dx, dy, dz, area)
        
        # Estimate local strain rate (simplified approach using vorticity magnitude)
        vorticity_mag = sqrt(eleGma[t,1]^2 + eleGma[t,2]^2 + eleGma[t,3]^2)
        strain_mag = vorticity_mag  # Simplified: |S| ≈ |ω| for developed turbulence
        
        # Smagorinsky eddy viscosity
        nu_sgs = (model.Cs * Delta)^2 * strain_mag
        
        # Dissipation rate: dγ/dt = -ν_sgs * γ / Delta^2
        dissipation_rate = nu_sgs / Delta^2
        
        # Apply exponential decay
        decay_factor = exp(-dissipation_rate * dt)
        eleGma[t,1] *= decay_factor
        eleGma[t,2] *= decay_factor  
        eleGma[t,3] *= decay_factor
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
