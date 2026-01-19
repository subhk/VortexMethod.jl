# Baroclinic Vorticity Generation

This page describes the treatment of density discontinuities and baroclinic vorticity generation, based on [Stock (2006), §3.7](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837).

## Physical Background

!!! info "Baroclinic Vorticity Generation"
    When a density interface exists in a flow under acceleration (gravity or otherwise), vorticity is generated at the interface. This **baroclinic** mechanism drives important instabilities like Rayleigh–Taylor and Richtmyer–Meshkov.

When a density interface exists in a flow under acceleration (gravity or otherwise), vorticity is generated at the interface. This **baroclinic** mechanism drives important instabilities:

- **Rayleigh–Taylor instability:** Heavy fluid over light fluid under gravity
- **Richtmyer–Meshkov instability:** Shock-accelerated density interface
- **Kelvin–Helmholtz with stratification:** Shear layer with stabilizing/destabilizing buoyancy

## Boussinesq Approximation

!!! note "Small Density Differences"
    The Boussinesq approximation assumes density differences are small (``|A| \ll 1``), so inertia is dominated by the mean density while buoyancy depends on the density difference.

In the Boussinesq limit (small density differences), the baroclinic source term in the vortex sheet strength equation is:

```math
\left(\frac{D\boldsymbol{\gamma}}{Dt}\right)_{\text{baroclinic}} = 2A\,\hat{\mathbf{n}} \times (\bar{\mathbf{a}} - \mathbf{g})
```

where:
- ``A = (\rho_2 - \rho_1)/(\rho_1 + \rho_2)`` is the **Atwood number**
- ``\hat{\mathbf{n}}`` is the interface normal (pointing from fluid 1 to fluid 2)
- ``\bar{\mathbf{a}}`` is the average interface acceleration
- ``\mathbf{g}`` is gravitational acceleration

The corresponding circulation equation:

```math
\frac{D\Gamma}{Dt} = 2A\left(\frac{D\phi}{Dt} - \frac{1}{2}|\mathbf{u}|^2 + \frac{1}{8}|\Delta\mathbf{u}|^2 - \mathbf{g}\cdot\mathbf{x}\right)
```

## Implementation

### Element Data Structure

Each triangular element stores its own local Atwood number. All elements in an interconnected sheet share the same value, though separate sheets may have different values.

### Vorticity Update

For each element ``i`` at the density interface:

```math
\Delta\boldsymbol{\alpha}_i = \Delta t \cdot 2\,a_i\,\theta_i\,\hat{\mathbf{n}}_i \times \hat{\mathbf{g}}
```

where:
- ``a_i`` is the element area
- ``\theta_i = A \cdot g`` is the Boussinesq coefficient (Atwood number × gravity)
- ``\hat{\mathbf{n}}_i`` is the element normal
- ``\hat{\mathbf{g}}`` is the gravity direction

This vector change in total vorticity is converted to edge circulations using the standard procedure (see [Theory](theory.md#circulation-assignment)).

### Integration with Time Stepping

Baroclinic updates are applied at every sub-step of the time integration:

```julia
function rk2_step_with_baroclinic!(nodes, tri, eleGma, domain, gr, dt; atwood=0.0, gravity=[0,0,-1])
    # Stage 1: Compute velocities
    vel1 = node_velocities(eleGma, tri, nodes..., domain, gr)

    # Update positions to midpoint
    nodes_mid = nodes .+ 0.5 * dt .* vel1

    # Baroclinic source at midpoint
    if atwood != 0
        apply_baroclinic_source!(eleGma, tri, nodes_mid, atwood, gravity, 0.5*dt)
    end

    # Stage 2: Compute velocities at midpoint
    vel2 = node_velocities(eleGma, tri, nodes_mid..., domain, gr)

    # Full step
    nodes .+= dt .* vel2

    # Baroclinic source for full step
    if atwood != 0
        apply_baroclinic_source!(eleGma, tri, nodes, atwood, gravity, 0.5*dt)
    end
end
```

## Validation: Linear Theory Comparison

### Stable Oscillation (Light over Heavy)

For ``A > 0`` (lighter fluid on top), a sinusoidally-perturbed interface oscillates:

```math
f(x,t) = F_0 \exp(ikx + it\sqrt{|{\theta}|k})
```

with period:

```math
\tau = \frac{2\pi}{\sqrt{|\theta|k}}
```

### Unstable Growth (Heavy over Light)

For ``A < 0`` (heavier fluid on top, or heavy over light with gravity reversed), the amplitude grows exponentially:

```math
f_{\max} = F_0 \cosh(t\sqrt{|{\theta}|k})
```

### Test Setup

```julia
# Computational domain [0:1]×[0:1]×[-4:4]
# Periodic in x,y; wall in z
domain = DomainSpec(1.0, 1.0, 8.0)
gr = GridSpec(15, 15, 120)  # or finer

# Initial perturbation: z = 0.01 sin(2πx) → k = 2π
# Boussinesq coefficient θ = 1

# Expected period (stable case):
τ_theory = 2π / √(2π)  # ≈ 2.507

# Expected growth (unstable case):
magnification(t) = cosh(t * √(2π))
```

### Convergence

!!! tip "Kernel Selection"
    The M4' kernel provides the lowest errors for baroclinic validation. Area-weighting and Peskin kernels show approximately first-order convergence.

The method shows approximately first-order convergence to analytical limits, with the M4' kernel providing lowest errors. Regularization causes:
- Longer oscillation periods than theory (smoother = slower)
- Smaller unstable growth rates than theory (regularization damps peaks)

## Multi-Sheet Simulations

Simulations can contain multiple separate vortex sheets, each with its own Atwood number:

```julia
# Example: Two-layer stratification
sheets = [
    (tri=tri_upper, eleGma=eleGma_upper, atwood=0.1),   # Upper interface
    (tri=tri_lower, eleGma=eleGma_lower, atwood=-0.1),  # Lower interface
]

for sheet in sheets
    apply_baroclinic_source!(sheet.eleGma, sheet.tri, nodes, sheet.atwood, gravity, dt)
end
```

## Kelvin–Helmholtz with Stratification

Combining shear (initial vortex sheet strength) with stratification:

### Linear Growth Rate

From Chandrasekhar (1961):

```math
n = \frac{ik\,\Delta u\,(\rho_2 - \rho_1)}{2(\rho_1 + \rho_2)} + \sqrt{\frac{k^2(\Delta u)^2\rho_1\rho_2}{(\rho_1+\rho_2)^2} - \frac{gk(\rho_2-\rho_1)}{\rho_1+\rho_2}}
```

### Stability Criterion

!!! warning "Richardson Number"
    At the critical Richardson number ``Ri_c = 1/4``, the shear layer transitions from stable to unstable. Below this value, shear dominates buoyancy.

The interface is unstable when:

```math
k(\Delta u)^2 > \frac{g(\rho_2 - \rho_1)(\rho_1 + \rho_2)}{\rho_1\rho_2}
```

At the critical Richardson number, ``Ri_c = 1/4``, the shear layer is marginally stable.

## API Reference

```julia
# Apply baroclinic vorticity source
apply_baroclinic_source!(eleGma, tri, nodeX, nodeY, nodeZ, atwood, gravity, dt)

# Time stepping with baroclinic terms
rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt;
          baroclinic=true, atwood=0.1, gravity=[0.0, 0.0, -1.0])
```

## References

- Chandrasekhar, S. (1961). *Hydrodynamic and Hydromagnetic Stability*. Oxford University Press.
- Stock, M. J. (2006). *A regularized inviscid vortex sheet method for three dimensional flows with density interfaces*. Ph.D. Thesis, California Institute of Technology.
- Zalosh, R. G. (1976). Discretized simulation of vortex sheet evolution with buoyancy and surface tension effects. *AIAA Journal*, 14(11), 1517–1523.
