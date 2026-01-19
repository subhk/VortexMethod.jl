# Boundary Conditions

This page describes the treatment of boundary conditions for the vortex-in-cell method, based on [Stock (2006), §3.3.2–3.3.5](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837).

## Overview

!!! info "Boundary Condition Categories"
    Flow simulations fall under two categories based on their boundaries: **internal** flows (bounded by walls) and **external** flows (periodic or extending to infinity).

Flow simulations fall under two categories based on their boundaries:

| Type | Boundary Conditions | Applications |
|------|---------------------|--------------|
| **Internal** | Wall (slip or no-slip) | Channel flows, cavity flows |
| **External** | Periodic or free-space (open) | Mixing layers, jets, wakes |

The Poisson solver (HW3CRT/FFT) supports Dirichlet, Neumann, or periodic boundary conditions, enabling uniform inlet, wall, periodic, or open boundaries.

## Periodic Boundaries

Periodic boundary conditions are commonly used to mimic free-space or to study spatially-developing instabilities.

### Element Treatment

!!! warning "Straddling Elements"
    Because we use triangular elements with connectivity, extra care is required when elements "straddle" a periodic boundary. Failing to handle these correctly leads to incorrect vorticity spreading and interpolation.

Because we use triangular elements with connectivity, extra care is required when elements "straddle" a periodic boundary:

1. **Straddling Detection:** For each element, check if any two nodes are separated by more than ``2.5\Delta x`` in a periodic direction
2. **Correction:** Temporarily move nodes near the upper boundary down by one domain length
3. **Operation:** Perform calculations (area, normal, center, spreading)
4. **Restoration:** Return nodes to their "natural" positions

### Vorticity Field Creation

When spreading vorticity near periodic boundaries:

- Grid nodes are indexed ``[0..N]``, with index ``N`` representing index ``0`` of the next domain
- Vorticity at node ``i_{\text{node}} < 0`` is added to ``i_{\text{node}} + N``
- Vorticity at node ``i_{\text{node}} > N-1`` is added to ``i_{\text{node}} - N``
- After spreading, copy vorticity at index 0 to index ``N`` for consistency

```julia
# Periodic wrap for vorticity spreading
function periodic_wrap_index(inode, N)
    if inode < 0
        return inode + N
    elseif inode > N - 1
        return inode - N
    else
        return inode
    end
end
```

### Usage

```julia
using VortexMethod

# Create domain with periodic x and y
domain = DomainSpec(1.0, 1.0, 4.0)  # Lx, Ly, Lz
gr = GridSpec(64, 64, 256)

# Wrap nodes after any modification
wrap_nodes!(nodeX, nodeY, nodeZ, domain)
```

## Open (Free-Space) Boundaries

!!! note "Complexity"
    Open boundaries allow vorticity to influence the flow beyond the computational domain. Implementation is more complex than periodic boundaries because boundary values must be computed from the interior vorticity field.

Open boundaries allow vorticity to influence the flow beyond the computational domain. Implementation is more complex than periodic boundaries.

### Dirichlet Boundary Values

When no vorticity is near open boundaries, use Dirichlet conditions with boundary velocities computed via direct Biot-Savart summation:

```math
\mathbf{u}(\mathbf{x}_{\text{boundary}}) = \sum_{\text{interior cells}} \frac{\boldsymbol{\omega} \times (\mathbf{x}_{\text{boundary}} - \mathbf{x}_{\text{cell}})}{4\pi|\mathbf{x}_{\text{boundary}} - \mathbf{x}_{\text{cell}}|^3}
```

**Optimization:** Integrate over every 4th boundary node and interpolate the rest. Only sum over cells with non-zero vorticity.

### Kinetic Energy Calculation

For open boundaries, kinetic energy requires special treatment because velocity is only computed inside the domain. Using the divergence theorem, assuming potential flow outside:

```math
E = \frac{1}{2}\left[\int_V |\mathbf{u}|^2\,d\mathbf{x} + \int_S \phi\,(\mathbf{u}\cdot\mathbf{n})\,dA\right]
```

where ``\phi`` is a scalar velocity potential computed on the boundary.

### Element Entry/Exit

Elements crossing open boundaries require special handling—either removing them or using absorbing layers. Current simulations are designed to avoid this situation.

## Wall Boundaries (Slip)

!!! tip "High-Reynolds-Number Approximation"
    For high-Reynolds-number flows, wall-generated vorticity stays within a thin boundary layer, allowing slip-wall treatment. This avoids resolving the viscous sublayer while maintaining correct inviscid behavior.

For high-Reynolds-number flows, wall-generated vorticity stays within a thin boundary layer, allowing slip-wall treatment.

### Vorticity Near Walls

When elements are within the regularization distance of a wall:

**On the wall (``i_{\text{node}} = 0`` or ``N``):**
- Double the wall-normal vorticity component
- Zero the wall-tangential components

**Beyond the wall (``i_{\text{node}} < 0`` or ``> N``):**
- Reflect to ``i_{\text{node}} \to -i_{\text{node}}`` or ``2N - i_{\text{node}}``
- Zero the normal component
- Negate the tangential components (image vorticity)

This ensures near-wall vortex lines are smoothed as if they and their reflections existed.

### Boundary Conditions for Poisson Solve

- **Normal velocity:** ``u_n = 0`` (no penetration)
- **Tangential velocity derivative:** ``\partial u_t / \partial n = 0`` (free slip)

### Element-Wall Interaction

Two types of vortex sheet–wall interactions:

1. **Edge attached to wall:** The edge must stay on the wall; nodes within ``\epsilon = 10^{-5}`` of the wall are constrained to remain there
2. **Sheet approaching wall:** Handled by the image vorticity treatment described above

## Special Boundary Conditions

### Inlet/Outlet (Jets, Mixing Layers)

For circular jets or mixing layers:
- Modify normal velocity component at inlet
- Insert shedding edges manually (see [Stock (2006), §3.6])

### Traveling Window

For problems like propagating vortex rings, the computational domain can "follow" the structure:

```julia
# Domain moves with sheet centroid
domain_z_min = sheet_centroid_z - 2.0
domain_z_max = sheet_centroid_z + 2.0
```

## Boundary Condition Selection Guide

| Flow Type | Recommended BCs | Notes |
|-----------|-----------------|-------|
| Periodic instability (KH, RT) | Periodic in x,y; wall in z | Most common setup |
| Free vortex ring | Open in all directions | Use traveling window |
| Channel flow | Wall in y; periodic in x,z | Standard internal flow |
| Jet in crossflow | Inlet + open | Complex setup |
| Mixing layer | Periodic in x,y; open in z | Matches experiment |

## References

- Stock, M. J. (2006). *A regularized inviscid vortex sheet method for three dimensional flows with density interfaces*. Ph.D. Thesis, California Institute of Technology.
- Swarztrauber, P., & Sweet, R. (1975). *Efficient FORTRAN subprograms for the solution of elliptic partial differential equations*. NCAR Technical Note.
