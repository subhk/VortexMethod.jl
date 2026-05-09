# Enhanced vortex sheet tracking and evolution
# Implements advanced sheet tracking methods from thesis Chapter 3.3

module Sheets

using ..DomainImpl
using ..Kernels
using ..Remeshing: element_quality_metrics_periodic
using LinearAlgebra

export VortexSheet, SheetEvolution, LagrangianSheet, EulerianSheet, 
       HybridSheet, evolve_sheet!, track_sheet_interface!, 
       compute_sheet_curvature, detect_sheet_rollup, check_sheet_reconnection!,
       reconnect_sheet_nodes!, adaptive_sheet_tracking!, compute_mesh_quality_sheet

abstract type VortexSheet end
abstract type SheetEvolution end

# Lagrangian sheet tracking
struct LagrangianSheet <: VortexSheet
    nodes::Matrix{Float64}  # N×3 node positions
    connectivity::Matrix{Int}  # triangular connectivity
    strength::Matrix{Float64}  # N×3 vorticity vectors
    age::Vector{Float64}  # node ages for tracking
    interface_markers::Vector{Bool}  # marks sheet interface
end

# Eulerian sheet tracking with level sets
struct EulerianSheet <: VortexSheet
    level_set::Array{Float64,3}  # distance function
    velocity::Array{Float64,3}  # interface velocity
    strength_field::Array{Float64,4}  # 3D vorticity field (nx×ny×nz×3)
    grid::GridSpec
    domain::DomainSpec
end

# Hybrid Lagrangian-Eulerian approach
struct HybridSheet <: VortexSheet
    lagrangian::LagrangianSheet
    eulerian::EulerianSheet
    coupling_strength::Float64  # 0=pure Lagrangian, 1=pure Eulerian
end

# Sheet evolution algorithms
struct ClassicalEvolution <: SheetEvolution end

struct AdaptiveEvolution <: SheetEvolution 
    curvature_threshold::Float64
    reconnection_distance::Float64
end

struct HighOrderEvolution <: SheetEvolution
    order::Int  # RK order
    adaptive_timestep::Bool
end

# Initialize a vortex sheet from triangular mesh
function VortexSheet(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                    tri::Matrix{Int}, eleGma::Matrix{Float64})
    nt = size(tri, 1)
    nodes = hcat(nodeX, nodeY, nodeZ)
    
    # Detect interface nodes (simplified: all boundary nodes)
    interface_markers = detect_interface_nodes(tri, length(nodeX))
    
    # Initialize ages
    ages = zeros(Float64, length(nodeX))
    
    return LagrangianSheet(nodes, tri, eleGma, ages, interface_markers)
end


# Detect interface nodes
function detect_interface_nodes(tri::Matrix{Int}, num_nodes::Int)
    edge_count = Dict{Tuple{Int,Int}, Int}()
    
    # Count edges
    for t in 1:size(tri, 1)
        v1, v2, v3 = tri[t, 1], tri[t, 2], tri[t, 3]
        for (a, b) in [(v1,v2), (v2,v3), (v3,v1)]
            edge = a < b ? (a, b) : (b, a)
            edge_count[edge] = get(edge_count, edge, 0) + 1
        end
    end
    
    # Boundary nodes have edges with count = 1
    boundary_nodes = Set{Int}()
    for ((a, b), count) in edge_count
        if count == 1
            push!(boundary_nodes, a)
            push!(boundary_nodes, b)
        end
    end
    
    interface = falses(num_nodes)
    for node in boundary_nodes
        interface[node] = true
    end
    
    return interface
end


# Evolve vortex sheet using classical method
function evolve_sheet!(sheet::LagrangianSheet, evolution::ClassicalEvolution, 
                      velocity_field, dt::Float64, domain::DomainSpec)
    n_nodes = size(sheet.nodes, 1)
    
    # Simple Euler step for node positions
    for i in 1:n_nodes
        x, y, z = sheet.nodes[i, 1], sheet.nodes[i, 2], sheet.nodes[i, 3]
        u, v, w = velocity_field(x, y, z)
        
        # Update position
        sheet.nodes[i, 1] += dt * u
        sheet.nodes[i, 2] += dt * v
        sheet.nodes[i, 3] += dt * w
        
        # Apply periodic boundary conditions
        sheet.nodes[i, 1] = mod(sheet.nodes[i, 1], domain.Lx)
        sheet.nodes[i, 2] = mod(sheet.nodes[i, 2], domain.Ly)
        sheet.nodes[i, 3] = mod(sheet.nodes[i, 3] + domain.Lz, 2*domain.Lz) - domain.Lz
        
        # Update age
        sheet.age[i] += dt
    end
    
    return sheet
end

# Advanced adaptive evolution
function evolve_sheet!(sheet::LagrangianSheet, evolution::AdaptiveEvolution,
                      velocity_field, dt::Float64, domain::DomainSpec)
    # First, evolve using classical method
    evolve_sheet!(sheet, ClassicalEvolution(), velocity_field, dt, domain)
    
    # Then apply adaptive corrections
    curvatures = compute_sheet_curvature(sheet)
    
    # Detect high curvature regions
    high_curvature_nodes = findall(c -> c > evolution.curvature_threshold, curvatures)
    
    # Apply curvature smoothing
    for node_idx in high_curvature_nodes
        smooth_local_curvature!(sheet, node_idx, 0.1 * dt, domain)
    end
    
    # Check for reconnection events
    check_sheet_reconnection!(sheet, evolution.reconnection_distance, domain)
    
    return sheet
end

# High-order Runge-Kutta evolution
function evolve_sheet!(sheet::LagrangianSheet, evolution::HighOrderEvolution,
                      velocity_field, dt::Float64, domain::DomainSpec)
    if evolution.order == 2
        # RK2 evolution
        evolve_sheet_rk2!(sheet, velocity_field, dt, domain)
    elseif evolution.order == 4
        # RK4 evolution
        evolve_sheet_rk4!(sheet, velocity_field, dt, domain)
    else
        # Fall back to Euler
        evolve_sheet!(sheet, ClassicalEvolution(), velocity_field, dt, domain)
    end
    
    return sheet
end

# RK2 evolution for sheet
function evolve_sheet_rk2!(sheet::LagrangianSheet, velocity_field, dt::Float64, domain::DomainSpec)
    n_nodes = size(sheet.nodes, 1)
    
    @inbounds for i in 1:n_nodes
        x0 = sheet.nodes[i, 1]
        y0 = sheet.nodes[i, 2]
        z0 = sheet.nodes[i, 3]
        u1, v1, w1 = velocity_field(x0, y0, z0)
        xm = x0 + 0.5 * dt * u1
        ym = y0 + 0.5 * dt * v1
        zm = z0 + 0.5 * dt * w1
        u2, v2, w2 = velocity_field(xm, ym, zm)
        
        sheet.nodes[i, 1] = x0 + dt * u2
        sheet.nodes[i, 2] = y0 + dt * v2
        sheet.nodes[i, 3] = z0 + dt * w2
        
        # Apply periodic BC
        sheet.nodes[i, 1] = mod(sheet.nodes[i, 1], domain.Lx)
        sheet.nodes[i, 2] = mod(sheet.nodes[i, 2], domain.Ly)
        sheet.nodes[i, 3] = mod(sheet.nodes[i, 3] + domain.Lz, 2*domain.Lz) - domain.Lz
        
        sheet.age[i] += dt
    end

    return sheet
end

# RK4 evolution for sheet
function evolve_sheet_rk4!(sheet::LagrangianSheet, velocity_field, dt::Float64, domain::DomainSpec)
    n_nodes = size(sheet.nodes, 1)

    @inbounds for i in 1:n_nodes
        x0 = sheet.nodes[i, 1]
        y0 = sheet.nodes[i, 2]
        z0 = sheet.nodes[i, 3]

        k1x, k1y, k1z = velocity_field(x0, y0, z0)
        k2x, k2y, k2z = velocity_field(
            x0 + 0.5 * dt * k1x,
            y0 + 0.5 * dt * k1y,
            z0 + 0.5 * dt * k1z,
        )
        k3x, k3y, k3z = velocity_field(
            x0 + 0.5 * dt * k2x,
            y0 + 0.5 * dt * k2y,
            z0 + 0.5 * dt * k2z,
        )
        k4x, k4y, k4z = velocity_field(
            x0 + dt * k3x,
            y0 + dt * k3y,
            z0 + dt * k3z,
        )

        sheet.nodes[i, 1] = mod(x0 + (dt / 6) * (k1x + 2*k2x + 2*k3x + k4x), domain.Lx)
        sheet.nodes[i, 2] = mod(y0 + (dt / 6) * (k1y + 2*k2y + 2*k3y + k4y), domain.Ly)
        sheet.nodes[i, 3] = mod(
            z0 + (dt / 6) * (k1z + 2*k2z + 2*k3z + k4z) + domain.Lz,
            2*domain.Lz,
        ) - domain.Lz
        sheet.age[i] += dt
    end

    return sheet
end

@inline function triangle_unit_normal(nodes::AbstractMatrix{Float64}, connectivity::AbstractMatrix{Int}, t::Int)
    v1 = connectivity[t, 1]
    v2 = connectivity[t, 2]
    v3 = connectivity[t, 3]

    p1x = nodes[v1, 1]; p1y = nodes[v1, 2]; p1z = nodes[v1, 3]
    e1x = nodes[v2, 1] - p1x
    e1y = nodes[v2, 2] - p1y
    e1z = nodes[v2, 3] - p1z
    e2x = nodes[v3, 1] - p1x
    e2y = nodes[v3, 2] - p1y
    e2z = nodes[v3, 3] - p1z

    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    inv_norm = inv(sqrt(nx*nx + ny*ny + nz*nz) + eps(Float64))
    return nx * inv_norm, ny * inv_norm, nz * inv_norm
end

# Compute curvature at each node
function compute_sheet_curvature(sheet::LagrangianSheet)
    n_nodes = size(sheet.nodes, 1)
    n_triangles = size(sheet.connectivity, 1)
    curvatures = zeros(Float64, n_nodes)

    counts = zeros(Int, n_nodes)
    @inbounds for t in 1:n_triangles
        counts[sheet.connectivity[t, 1]] += 1
        counts[sheet.connectivity[t, 2]] += 1
        counts[sheet.connectivity[t, 3]] += 1
    end

    offsets = Vector{Int}(undef, n_nodes + 1)
    offsets[1] = 1
    @inbounds for i in 1:n_nodes
        offsets[i + 1] = offsets[i] + counts[i]
    end

    cursor = copy(offsets)
    node_triangles = Vector{Int}(undef, 3 * n_triangles)
    @inbounds for t in 1:n_triangles
        node1 = sheet.connectivity[t, 1]
        node2 = sheet.connectivity[t, 2]
        node3 = sheet.connectivity[t, 3]
        node_triangles[cursor[node1]] = t
        cursor[node1] += 1
        node_triangles[cursor[node2]] = t
        cursor[node2] += 1
        node_triangles[cursor[node3]] = t
        cursor[node3] += 1
    end

    @inbounds for i in 1:n_nodes
        first_triangle = offsets[i]
        last_triangle = offsets[i + 1] - 1
        if last_triangle > first_triangle
            n1x, n1y, n1z = triangle_unit_normal(sheet.nodes, sheet.connectivity, node_triangles[first_triangle])
            curvature = 0.0
            for pos in (first_triangle + 1):last_triangle
                nx, ny, nz = triangle_unit_normal(sheet.nodes, sheet.connectivity, node_triangles[pos])
                curvature += acos(clamp(n1x*nx + n1y*ny + n1z*nz, -1.0, 1.0))
            end
            curvatures[i] = curvature / (last_triangle - first_triangle)
        end
    end

    return curvatures
end

# Detect sheet rollup regions
function detect_sheet_rollup(sheet::LagrangianSheet; vorticity_threshold::Float64=1.0)
    n_triangles = size(sheet.connectivity, 1)
    rollup_regions = Vector{Bool}(undef, n_triangles)
    
    @inbounds for t in 1:n_triangles
        gx = sheet.strength[t, 1]
        gy = sheet.strength[t, 2]
        gz = sheet.strength[t, 3]
        rollup_regions[t] = sqrt(gx*gx + gy*gy + gz*gz) > vorticity_threshold
    end
    
    return rollup_regions
end

# Smooth local curvature
function smooth_local_curvature!(sheet::LagrangianSheet, node_idx::Int,
                            smoothing_factor::Float64, domain::Union{DomainSpec,Nothing}=nothing)
    # Find neighboring nodes
    neighbors = find_node_neighbors(sheet, node_idx)
    
    if !isempty(neighbors)
        # Average position with neighbors (periodic-aware if domain provided)
        center = sheet.nodes[node_idx, :]
        avg_pos = zeros(3)
        for neighbor in neighbors
            p = sheet.nodes[neighbor, :]
            if domain === nothing
                avg_pos += p
            else
                dx = p[1] - center[1]; dy = p[2] - center[2]; dz = p[3] - center[3]
                if domain.Lx > 0; dx -= domain.Lx*round(dx/domain.Lx); end
                if domain.Ly > 0; dy -= domain.Ly*round(dy/domain.Ly); end
                if domain.Lz > 0; dz -= 2*domain.Lz*round(dz/(2*domain.Lz)); end
                
                avg_pos[1] += center[1] + dx
                avg_pos[2] += center[2] + dy
                avg_pos[3] += center[3] + dz
            end
        end
        avg_pos /= length(neighbors)
        
        # Apply smoothing
        newp = (1 - smoothing_factor) * sheet.nodes[node_idx, :] + smoothing_factor * avg_pos
        if domain === nothing
            sheet.nodes[node_idx, :] = newp
        else
            xw, yw, zw = wrap_point(newp[1], newp[2], newp[3], domain)
            sheet.nodes[node_idx, 1] = xw; sheet.nodes[node_idx, 2] = yw; sheet.nodes[node_idx, 3] = zw
        end
    end
end

# Find neighboring nodes
function find_node_neighbors(sheet::LagrangianSheet, node_idx::Int)
    neighbors = Set{Int}()
    
    # Find all triangles containing this node
    for t in 1:size(sheet.connectivity, 1)
        triangle = sheet.connectivity[t, :]
        if node_idx in triangle
            # Add other nodes in this triangle as neighbors
            for node in triangle
                if node != node_idx
                    push!(neighbors, node)
                end
            end
        end
    end
    
    return collect(neighbors)
end

# Check for sheet reconnection events
function check_sheet_reconnection!(sheet::LagrangianSheet, reconnection_distance::Float64, domain::DomainSpec)
    interface_nodes = findall(sheet.interface_markers)
    
    for i in 1:length(interface_nodes)
        for j in i+1:length(interface_nodes)
            node1 = interface_nodes[i]
            node2 = interface_nodes[j]
            
            # Check minimum-image distance between interface nodes
            p1 = sheet.nodes[node1, :]; p2 = sheet.nodes[node2, :]
            dx = p1[1] - p2[1]; dy = p1[2] - p2[2]; dz = p1[3] - p2[3]
            if domain.Lx > 0; dx -= domain.Lx*round(dx/domain.Lx); end
            if domain.Ly > 0; dy -= domain.Ly*round(dy/domain.Ly); end
            if domain.Lz > 0; dz -= 2*domain.Lz*round(dz/(2*domain.Lz)); end
            dist = sqrt(dx^2 + dy^2 + dz^2)
            
            if dist < reconnection_distance
                # Perform reconnection
                reconnect_sheet_nodes!(sheet, node1, node2, domain)
            end
        end
    end
end

# Perform sheet reconnection
function reconnect_sheet_nodes!(sheet::LagrangianSheet, node1::Int, node2::Int, domain::Union{DomainSpec,Nothing}=nothing)
    # Simple reconnection: merge the two nodes by moving node1 to midpoint
    # Note: strength is element-based (per triangle), not node-based, so we don't merge it here
    if domain === nothing
        merge_pos = 0.5 * (sheet.nodes[node1, :] + sheet.nodes[node2, :])
    else
        p1 = sheet.nodes[node1, :]; p2 = sheet.nodes[node2, :]
        dx = p2[1] - p1[1]; dy = p2[2] - p1[2]; dz = p2[3] - p1[3]
        if domain.Lx > 0; dx -= domain.Lx*round(dx/domain.Lx); end
        if domain.Ly > 0; dy -= domain.Ly*round(dy/domain.Ly); end
        if domain.Lz > 0; dz -= 2*domain.Lz*round(dz/(2*domain.Lz)); end
        mx, my, mz = p1[1] + 0.5*dx, p1[2] + 0.5*dy, p1[3] + 0.5*dz
        xw, yw, zw = wrap_point(mx, my, mz, domain)
        merge_pos = [xw, yw, zw]
    end
    merge_age = 0.5 * (sheet.age[node1] + sheet.age[node2])

    # Update first node with merged position and age
    sheet.nodes[node1, :] = merge_pos
    sheet.age[node1] = merge_age

    # Remap connectivity: replace all references to node2 with node1
    for t in 1:size(sheet.connectivity, 1)
        for k in 1:3
            if sheet.connectivity[t, k] == node2
                sheet.connectivity[t, k] = node1
            end
        end
    end

    # Mark second node for removal (simplified approach)
    sheet.interface_markers[node2] = false
end

# Track sheet interface using level sets (for Eulerian approach)
function track_sheet_interface!(sheet::EulerianSheet, velocity_field, dt::Float64)
    # Evolve level set using velocity field
    nz, ny, nx = size(sheet.level_set)
    new_level_set = copy(sheet.level_set)
    
    dx = sheet.domain.Lx / (nx - 1)
    dy = sheet.domain.Ly / (ny - 1)
    dz = 2 * sheet.domain.Lz / (nz - 1)
    
    # Upwind scheme for level set evolution
    for k in 2:nz-1, j in 2:ny-1, i in 2:nx-1
        # Get velocity at grid point
        x = (i-1) * dx
        y = (j-1) * dy
        z = (k-1) * dz - sheet.domain.Lz
        u, v, w = velocity_field(x, y, z)
        
        # Compute spatial derivatives using upwind
        if u > 0
            dphi_dx = (sheet.level_set[k,j,i] - sheet.level_set[k,j,i-1]) / dx
        else
            dphi_dx = (sheet.level_set[k,j,i+1] - sheet.level_set[k,j,i]) / dx
        end
        
        if v > 0
            dphi_dy = (sheet.level_set[k,j,i] - sheet.level_set[k,j-1,i]) / dy
        else
            dphi_dy = (sheet.level_set[k,j+1,i] - sheet.level_set[k,j,i]) / dy
        end
        
        if w > 0
            dphi_dz = (sheet.level_set[k,j,i] - sheet.level_set[k-1,j,i]) / dz
        else
            dphi_dz = (sheet.level_set[k+1,j,i] - sheet.level_set[k,j,i]) / dz
        end
        
        # Update level set
        new_level_set[k,j,i] = sheet.level_set[k,j,i] - dt * (u*dphi_dx + v*dphi_dy + w*dphi_dz)
    end
    
    sheet.level_set .= new_level_set
    return sheet
end

# Adaptive sheet tracking combining multiple methods
function adaptive_sheet_tracking!(sheet::LagrangianSheet,
                                velocity_field,
                                dt::Float64,
                                domain::DomainSpec;
                                curvature_threshold::Float64=1.0, 
                                quality_threshold::Float64=0.3)
    # Compute mesh quality metrics
    qualities = compute_mesh_quality_sheet(sheet, domain)
    
    # Determine evolution method based on local conditions
    if maximum(qualities) < quality_threshold
        # Use high-order method for good quality regions
        evolution = HighOrderEvolution(4, true)
    else
        # Use adaptive method for poor quality regions
        evolution = AdaptiveEvolution(curvature_threshold, 0.1)
    end
    
    # Evolve sheet
    evolve_sheet!(sheet, evolution, velocity_field, dt, domain)
    
    return sheet
end

# Compute mesh quality for sheet triangles
function compute_mesh_quality_sheet(sheet::LagrangianSheet, domain::DomainSpec)
    n_triangles = size(sheet.connectivity, 1)
    qualities = zeros(Float64, n_triangles)
    
    for t in 1:n_triangles
        v1, v2, v3 = sheet.connectivity[t, 1], sheet.connectivity[t, 2], sheet.connectivity[t, 3]
        p1 = (sheet.nodes[v1, 1], sheet.nodes[v1, 2], sheet.nodes[v1, 3])
        p2 = (sheet.nodes[v2, 1], sheet.nodes[v2, 2], sheet.nodes[v2, 3])
        p3 = (sheet.nodes[v3, 1], sheet.nodes[v3, 2], sheet.nodes[v3, 3])
        
        # Use periodic minimum-image quality metrics from Remeshing module
        quality = element_quality_metrics_periodic(p1, p2, p3, domain)
        qualities[t] = quality.jacobian_quality
    end
    
    return qualities
end

end # module

using .Sheets: VortexSheet, SheetEvolution, LagrangianSheet, EulerianSheet,
               HybridSheet, evolve_sheet!, track_sheet_interface!,
               compute_sheet_curvature, detect_sheet_rollup, check_sheet_reconnection!,
               reconnect_sheet_nodes!, adaptive_sheet_tracking!, compute_mesh_quality_sheet
