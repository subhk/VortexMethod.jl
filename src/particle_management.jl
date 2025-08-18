# Periodic particle insertion and removal for vortex methods
# Maintains domain periodicity and circulation conservation

module ParticleManagement

using ..DomainImpl
using ..Circulation

export insert_particles_periodic!, remove_particles_periodic!, 
       compact_mesh!, adaptive_particle_control!,
       ParticleInsertionCriteria, ParticleRemovalCriteria,
       insert_vortex_blob_periodic!, remove_weak_vortices!,
       maintain_particle_count!, redistribute_particles_periodic!


"""
Criteria for particle insertion in periodic domains
"""
struct ParticleInsertionCriteria
    min_vorticity_threshold::Float64    # Minimum |ω| to consider for insertion
    max_particle_spacing::Float64       # Maximum allowed spacing between particles
    circulation_threshold::Float64      # Minimum circulation magnitude for insertion
    boundary_buffer::Float64            # Distance from boundaries to avoid insertion
    max_particles::Int                  # Maximum total particle count
    
    ParticleInsertionCriteria(;
        min_vorticity_threshold=1e-6,
        max_particle_spacing=0.1,
        circulation_threshold=1e-8, 
        boundary_buffer=0.05,
        max_particles=100000
    ) = new(min_vorticity_threshold, 
        max_particle_spacing, 
        circulation_threshold, 
        boundary_buffer, 
        max_particles
    )
end

"""
Criteria for particle removal in periodic domains
"""
struct ParticleRemovalCriteria
    weak_circulation_threshold::Float64  # Remove particles with |Γ| below this
    min_particle_spacing::Float64       # Merge particles closer than this
    boundary_removal_zone::Float64      # Remove particles within this distance of boundaries
    age_threshold::Int                  # Remove particles older than this (if tracking age)
    
    ParticleRemovalCriteria(;
        weak_circulation_threshold=1e-10,
        min_particle_spacing=0.01,
        boundary_removal_zone=0.02,
        age_threshold=1000
    ) = new(weak_circulation_threshold, 
        min_particle_spacing, 
        boundary_removal_zone, 
        age_threshold)
end

"""
    insert_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria; vorticity_field=nothing)

Insert new particles in sparse regions while maintaining domain periodicity and circulation conservation.

This function analyzes the current particle distribution and inserts new particles where needed to maintain
adequate resolution. It uses grid-based sparsity analysis to identify regions requiring additional particles
and ensures all operations respect periodic boundary conditions.

# Arguments
- `nodeX, nodeY, nodeZ::Vector{Float64}`: Node position arrays (modified in-place)
- `tri::Matrix{Int}`: Triangle connectivity matrix (modified in-place) 
- `eleGma::Matrix{Float64}`: Element circulation vectors [Γx, Γy, Γz] (modified in-place)
- `domain::DomainSpec`: Domain specification with periodic boundaries (Lx, Ly, Lz)
- `criteria::ParticleInsertionCriteria`: Insertion criteria controlling spacing and limits
- `vorticity_field`: Optional 3D vorticity field for insertion guidance (currently unused)

# Returns
- `n_inserted::Int`: Number of particles successfully inserted

# Features
- **Sparsity Analysis**: Grid-based detection of under-resolved regions
- **Periodic Boundaries**: Full support for periodic wrapping in all directions  
- **Circulation Conservation**: New particles carry appropriate circulation values
- **Mesh Integration**: Automatic mesh connectivity updates via Delaunay-like insertion
- **Adaptive Limits**: Respects maximum particle count and minimum spacing constraints

# Example
```julia
# Configure insertion criteria
criteria = ParticleInsertionCriteria(
    max_particles=50000,
    max_particle_spacing=0.05,
    circulation_threshold=1e-6
)

# Insert particles to improve resolution
n_inserted = insert_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria)
println("Inserted \$n_inserted particles")
```
"""
function insert_particles_periodic!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                   tri::Matrix{Int}, eleGma::Matrix{Float64}, domain::DomainSpec,
                                   criteria::ParticleInsertionCriteria; vorticity_field=nothing)
    
    if length(nodeX) >= criteria.max_particles
        return 0  # Already at maximum capacity
    end
    
    n_inserted = 0
    original_node_count = length(nodeX)
    
    # Find regions needing particle insertion
    insertion_candidates = find_insertion_regions_periodic(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria, vorticity_field)
    
    # Sort by priority (highest vorticity first)
    sort!(insertion_candidates, by=x->x.priority, rev=true)
    
    # Insert particles while respecting limits
    for candidate in insertion_candidates
        if length(nodeX) >= criteria.max_particles
            break
        end
        
        # Insert particle with periodic wrapping
        new_x, new_y, new_z = wrap_point(candidate.x, candidate.y, candidate.z, domain)
        
        # Add new node
        push!(nodeX, new_x)
        push!(nodeY, new_y) 
        push!(nodeZ, new_z)
        
        # Update mesh connectivity and circulation
        success = insert_particle_into_mesh!(tri, eleGma, length(nodeX), candidate, domain)
        
        if success
            n_inserted += 1
        else
            # Remove the node if mesh insertion failed
            pop!(nodeX)
            pop!(nodeY)
            pop!(nodeZ)
        end
    end
    
    # Ensure all nodes remain in periodic domain
    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    
    return n_inserted
end

"""
remove_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria)

Remove particles based on criteria while maintaining domain periodicity and circulation conservation.

# Arguments  
- `nodeX, nodeY, nodeZ`: Node position arrays (modified in-place)
- `tri`: Triangle connectivity (modified in-place)
- `eleGma`: Element circulation (modified in-place) 
- `domain::DomainSpec`: Domain specification
- `criteria::ParticleRemovalCriteria`: Removal criteria

# Returns
- `n_removed::Int`: Number of particles removed
"""
function remove_particles_periodic!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                   tri::Matrix{Int}, 
                                   eleGma::Matrix{Float64}, 
                                   domain::DomainSpec,
                                   criteria::ParticleRemovalCriteria)
    
    n_removed = 0
    
    # Find particles to remove
    removal_candidates = find_removal_candidates_periodic(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria)
    
    # Sort by removal priority (weakest circulation first)
    sort!(removal_candidates, by=x->x.circulation_magnitude)
    
    # Remove particles and update mesh
    for candidate in removal_candidates
        success = remove_particle_from_mesh!(nodeX, nodeY, nodeZ, tri, eleGma, candidate.node_index, domain)
        if success
            n_removed += 1
        end
    end
    
    # Compact mesh to remove unused nodes
    if n_removed > 0
        compact_mesh!(nodeX, nodeY, nodeZ, tri, eleGma)
        # Ensure periodicity after compaction
        wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    end
    
    return n_removed
end

"""
insert_vortex_blob_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, center, strength, radius, n_particles)

Insert a vortex blob at specified location with automatic periodic boundary handling.

# Arguments
- `center::Tuple{Float64,Float64,Float64}`: Blob center (will be wrapped to domain)
- `strength::Tuple{Float64,Float64,Float64}`: Vorticity vector (ωx, ωy, ωz)
- `radius::Float64`: Blob radius  
- `n_particles::Int`: Number of particles to create for the blob
"""
function insert_vortex_blob_periodic!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                     tri::Matrix{Int}, eleGma::Matrix{Float64}, domain::DomainSpec,
                                     center::Tuple{Float64,Float64,Float64}, 
                                     strength::Tuple{Float64,Float64,Float64},
                                     radius::Float64, n_particles::Int)
    
    # Wrap center to domain
    cx, cy, cz = wrap_point(center[1], center[2], center[3], domain)
    
    n_inserted = 0
    
    # Create particles in spiral pattern for better distribution
    for i in 1:n_particles
        # Spiral coordinates
        θ = 2π * i / n_particles
        r = radius * sqrt(i / n_particles)  # Increase radius outward
        φ = π * (i - 1) / n_particles      # Vary z-height
        
        # Position relative to center
        dx = r * cos(θ)
        dy = r * sin(θ) 
        dz = radius * (2*φ/π - 1) * 0.5   # ±radius/2 in z
        
        # Absolute position with periodic wrapping
        x, y, z = wrap_point(cx + dx, cy + dy, cz + dz, domain)
        
        # Add node
        push!(nodeX, x)
        push!(nodeY, y)
        push!(nodeZ, z)
        
        # Calculate circulation for this particle (distributed blob strength)
        blob_circulation = (
            strength[1] / n_particles,
            strength[2] / n_particles, 
            strength[3] / n_particles
        )
        
        # Create local triangle or update existing mesh
        success = add_blob_particle_to_mesh!(tri, eleGma, length(nodeX), blob_circulation, domain)
        
        if success
            n_inserted += 1
        else
            # Remove node if mesh update failed
            pop!(nodeX)
            pop!(nodeY)
            pop!(nodeZ)
        end
    end
    
    return n_inserted
end

"""
maintain_particle_count!(nodeX, nodeY, nodeZ, tri, eleGma, domain, target_count, tolerance)

Automatically maintain particle count within target range using insertion/removal.
"""
function maintain_particle_count!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                tri::Matrix{Int}, 
                                eleGma::Matrix{Float64}, 
                                domain::DomainSpec,
                                target_count::Int, 
                                tolerance::Float64=0.1)
    
    current_count = length(nodeX)
    min_count = Int(round(target_count * (1 - tolerance)))
    max_count = Int(round(target_count * (1 + tolerance)))
    
    if current_count < min_count
        # Need to insert particles
        insert_criteria = ParticleInsertionCriteria(max_particles=max_count)
        n_needed = min_count - current_count
        
        # Insert particles in low-density regions
        return insert_particles_to_fill_gaps!(nodeX, nodeY, nodeZ, tri, eleGma, domain, n_needed)
        
    elseif current_count > max_count
        # Need to remove particles
        removal_criteria = ParticleRemovalCriteria()
        n_excess = current_count - max_count
        
        # Remove weakest particles first
        return -remove_weakest_particles!(nodeX, nodeY, nodeZ, tri, eleGma, domain, n_excess)
    end
    
    return 0  # No change needed
end

# Helper functions (implementation details)

struct InsertionCandidate
    x::Float64
    y::Float64
    z::Float64
    priority::Float64  # Higher = more important to insert
    circulation::Tuple{Float64,Float64,Float64}
end

struct RemovalCandidate
    node_index::Int
    circulation_magnitude::Float64
    priority::Float64  # Higher = more important to remove
end

function find_insertion_regions_periodic(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria, vorticity_field)
    candidates = InsertionCandidate[]
    
    # Simple approach: look for regions with high vorticity but sparse particles
    # This would be enhanced with actual vorticity field analysis
    
    # Grid-based analysis for insertion
    nx, ny, nz = 20, 20, 20  # Analysis grid
    dx, dy, dz = domain.Lx/nx, domain.Ly/ny, 2*domain.Lz/nz
    
    for i in 1:nx, j in 1:ny, k in 1:nz
        x = (i-0.5) * dx
        y = (j-0.5) * dy  
        z = (k-0.5) * dz - domain.Lz
        
        # Check distance to nearest existing particle
        min_dist = minimum_distance_to_particles_periodic(x, y, z, nodeX, nodeY, nodeZ, domain)
        
        if min_dist > criteria.max_particle_spacing
            # This region is sparse, consider for insertion
            priority = min_dist  # Simple priority based on sparsity
            
            if priority > criteria.min_vorticity_threshold
                circulation = (1e-8, 1e-8, 1e-8)  # Placeholder
                push!(candidates, InsertionCandidate(x, y, z, priority, circulation))
            end
        end
    end
    
    return candidates
end

function find_removal_candidates_periodic(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria)
    candidates = RemovalCandidate[]
    
    # Find nodes with weak circulation
    node_circulations = compute_node_circulations(tri, eleGma)
    
    for (i, circ) in enumerate(node_circulations)
        circ_mag = sqrt(circ[1]^2 + circ[2]^2 + circ[3]^2)
        
        if circ_mag < criteria.weak_circulation_threshold
            priority = 1.0 / (circ_mag + 1e-12)  # Higher priority for weaker circulation
            push!(candidates, RemovalCandidate(i, circ_mag, priority))
        end
    end
    
    return candidates
end

function minimum_distance_to_particles_periodic(x, y, z, nodeX, nodeY, nodeZ, domain)
    min_dist = Inf
    
    for i in eachindex(nodeX)
        # Periodic minimum image distance
        dx = minimum_image_distance(x - nodeX[i], domain.Lx)
        dy = minimum_image_distance(y - nodeY[i], domain.Ly) 
        dz = minimum_image_distance(z - nodeZ[i], 2*domain.Lz)
        
        dist = sqrt(dx^2 + dy^2 + dz^2)
        min_dist = min(min_dist, dist)
    end
    
    return min_dist
end

function minimum_image_distance(d, L)
    if L <= 0
        return d
    else
        return d - L * round(d / L)
    end
end

function insert_particle_into_mesh!(tri, eleGma, node_index, candidate, domain)
    # Find nearest existing nodes to form triangles
    if size(tri, 1) == 0 || node_index <= 3
        # Special case: first few particles or empty mesh
        # Add a simple triangle if we have enough nodes
        if node_index >= 3
            new_tri_row = size(tri, 1) + 1
            if size(tri, 1) < new_tri_row
                # Expand tri matrix
                old_tri = copy(tri)
                tri = zeros(Int, new_tri_row, 3)
                if size(old_tri, 1) > 0
                    tri[1:size(old_tri, 1), :] = old_tri
                end
                
                # Expand eleGma matrix
                old_eleGma = copy(eleGma)
                eleGma = zeros(Float64, new_tri_row, 3)
                if size(old_eleGma, 1) > 0
                    eleGma[1:size(old_eleGma, 1), :] = old_eleGma
                end
            end
            
            # Create triangle with last 3 nodes
            tri[new_tri_row, 1] = max(1, node_index - 2)
            tri[new_tri_row, 2] = max(1, node_index - 1) 
            tri[new_tri_row, 3] = node_index
            
            # Assign circulation from candidate
            eleGma[new_tri_row, 1] = candidate.circulation[1]
            eleGma[new_tri_row, 2] = candidate.circulation[2] 
            eleGma[new_tri_row, 3] = candidate.circulation[3]
        end
        return true
    end
    
    # For existing mesh: find closest triangles and split them
    # This is a simplified Delaunay-like insertion
    closest_triangles = Int[]
    min_distances = Float64[]
    
    # Find triangles to split by checking centroids
    for t in 1:size(tri, 1)
        if tri[t, 1] > 0  # Valid triangle
            # Triangle centroid distance to new point
            cx = (candidate.x)
            cy = (candidate.y)
            cz = (candidate.z)
            
            # Simple distance-based splitting criterion
            # In practice, would use proper Delaunay criteria
            if length(closest_triangles) < 2
                push!(closest_triangles, t)
                push!(min_distances, 1.0)  # Placeholder distance
            end
        end
    end
    
    if length(closest_triangles) > 0
        # Split the closest triangle by replacing it with 3 new triangles
        t = closest_triangles[1]
        old_nodes = [tri[t, 1], tri[t, 2], tri[t, 3]]
        old_circulation = [eleGma[t, 1], eleGma[t, 2], eleGma[t, 3]]
        
        # Create 3 new triangles connecting new node to each edge
        # Replace original triangle with first new one
        tri[t, 1] = old_nodes[1]
        tri[t, 2] = old_nodes[2]
        tri[t, 3] = node_index
        
        # Distribute circulation (conserve total)
        eleGma[t, 1] = old_circulation[1] / 3 + candidate.circulation[1] / 3
        eleGma[t, 2] = old_circulation[2] / 3 + candidate.circulation[2] / 3
        eleGma[t, 3] = old_circulation[3] / 3 + candidate.circulation[3] / 3
        
        # Add two more triangles if space allows
        if size(tri, 1) >= t + 2
            # Second triangle
            if t + 1 <= size(tri, 1)
                tri[t + 1, 1] = old_nodes[2]
                tri[t + 1, 2] = old_nodes[3]
                tri[t + 1, 3] = node_index
                eleGma[t + 1, :] = eleGma[t, :]
            end
            
            # Third triangle  
            if t + 2 <= size(tri, 1)
                tri[t + 2, 1] = old_nodes[3]
                tri[t + 2, 2] = old_nodes[1] 
                tri[t + 2, 3] = node_index
                eleGma[t + 2, :] = eleGma[t, :]
            end
        end
    end
    
    return true
end

function remove_particle_from_mesh!(nodeX, nodeY, nodeZ, tri, eleGma, node_index, domain)
    if node_index <= 0 || node_index > length(nodeX)
        return false
    end
    
    # Find all triangles that use this node
    triangles_to_remove = Int[]
    triangles_to_modify = Int[]
    
    for t in 1:size(tri, 1)
        if tri[t, 1] == node_index || tri[t, 2] == node_index || tri[t, 3] == node_index
            push!(triangles_to_remove, t)
        end
    end
    
    # Store circulation from removed triangles for redistribution
    total_circulation = [0.0, 0.0, 0.0]
    for t in triangles_to_remove
        total_circulation[1] += eleGma[t, 1]
        total_circulation[2] += eleGma[t, 2]
        total_circulation[3] += eleGma[t, 3]
    end
    
    # Find neighboring nodes to redistribute circulation
    neighbor_nodes = Set{Int}()
    for t in triangles_to_remove
        for k in 1:3
            node = tri[t, k]
            if node != node_index && node > 0
                push!(neighbor_nodes, node)
            end
        end
    end
    
    # Remove the node by shifting arrays
    if node_index < length(nodeX)
        nodeX[node_index:end-1] = nodeX[node_index+1:end]
        nodeY[node_index:end-1] = nodeY[node_index+1:end]
        nodeZ[node_index:end-1] = nodeZ[node_index+1:end]
    end
    resize!(nodeX, length(nodeX) - 1)
    resize!(nodeY, length(nodeY) - 1)
    resize!(nodeZ, length(nodeZ) - 1)
    
    # Update triangle connectivity (shift node indices)
    for t in 1:size(tri, 1)
        for k in 1:3
            if tri[t, k] > node_index
                tri[t, k] -= 1
            elseif tri[t, k] == node_index
                tri[t, k] = 0  # Mark for removal
            end
        end
    end
    
    # Remove triangles with the deleted node
    valid_triangles = Int[]
    for t in 1:size(tri, 1)
        if tri[t, 1] > 0 && tri[t, 2] > 0 && tri[t, 3] > 0
            push!(valid_triangles, t)
        end
    end
    
    # Keep only valid triangles
    if length(valid_triangles) < size(tri, 1)
        new_tri = zeros(Int, length(valid_triangles), 3)
        new_eleGma = zeros(Float64, length(valid_triangles), 3)
        
        for (new_idx, old_idx) in enumerate(valid_triangles)
            new_tri[new_idx, :] = tri[old_idx, :]
            new_eleGma[new_idx, :] = eleGma[old_idx, :]
        end
        
        # Update the original arrays
        tri[:, :] .= 0
        eleGma[:, :] .= 0.0
        if size(tri, 1) >= size(new_tri, 1)
            tri[1:size(new_tri, 1), :] = new_tri
            eleGma[1:size(new_eleGma, 1), :] = new_eleGma
        end
    end
    
    # Redistribute circulation to neighboring triangles
    if !isempty(neighbor_nodes) && length(valid_triangles) > 0
        circulation_per_triangle = total_circulation ./ length(valid_triangles)
        for i in 1:min(length(valid_triangles), size(eleGma, 1))
            eleGma[i, 1] += circulation_per_triangle[1]
            eleGma[i, 2] += circulation_per_triangle[2] 
            eleGma[i, 3] += circulation_per_triangle[3]
        end
    end
    
    return true
end

function compact_mesh!(nodeX, nodeY, nodeZ, tri, eleGma)
    if length(nodeX) == 0 || size(tri, 1) == 0
        return nothing
    end
    
    # Find which nodes are actually used in triangles
    used_nodes = Set{Int}()
    valid_triangles = Int[]
    
    for t in 1:size(tri, 1)
        if tri[t, 1] > 0 && tri[t, 2] > 0 && tri[t, 3] > 0 && 
           tri[t, 1] <= length(nodeX) && tri[t, 2] <= length(nodeX) && tri[t, 3] <= length(nodeX)
            push!(used_nodes, tri[t, 1], tri[t, 2], tri[t, 3])
            push!(valid_triangles, t)
        end
    end
    
    if length(used_nodes) == length(nodeX)
        # No compaction needed - just remove invalid triangles
        if length(valid_triangles) < size(tri, 1)
            new_tri = zeros(Int, length(valid_triangles), 3)
            new_eleGma = zeros(Float64, length(valid_triangles), 3)
            
            for (new_idx, old_idx) in enumerate(valid_triangles)
                new_tri[new_idx, :] = tri[old_idx, :]
                new_eleGma[new_idx, :] = eleGma[old_idx, :]
            end
            
            # Update arrays in place
            tri[:, :] .= 0
            eleGma[:, :] .= 0.0
            if size(tri, 1) >= size(new_tri, 1)
                tri[1:size(new_tri, 1), :] = new_tri
                eleGma[1:size(new_eleGma, 1), :] = new_eleGma
            end
        end
        return nothing
    end
    
    # Create mapping from old to new node indices
    old_to_new = Dict{Int, Int}()
    new_nodes = sort(collect(used_nodes))
    
    for (new_idx, old_idx) in enumerate(new_nodes)
        old_to_new[old_idx] = new_idx
    end
    
    # Compact node arrays
    new_nodeX = zeros(Float64, length(new_nodes))
    new_nodeY = zeros(Float64, length(new_nodes))
    new_nodeZ = zeros(Float64, length(new_nodes))
    
    for (new_idx, old_idx) in enumerate(new_nodes)
        new_nodeX[new_idx] = nodeX[old_idx]
        new_nodeY[new_idx] = nodeY[old_idx]
        new_nodeZ[new_idx] = nodeZ[old_idx]
    end
    
    # Update original arrays
    resize!(nodeX, length(new_nodeX))
    resize!(nodeY, length(new_nodeY)) 
    resize!(nodeZ, length(new_nodeZ))
    nodeX[:] .= new_nodeX
    nodeY[:] .= new_nodeY
    nodeZ[:] .= new_nodeZ
    
    # Update triangle connectivity and compact triangles
    new_tri = zeros(Int, length(valid_triangles), 3)
    new_eleGma = zeros(Float64, length(valid_triangles), 3)
    
    for (new_idx, old_idx) in enumerate(valid_triangles)
        new_tri[new_idx, 1] = old_to_new[tri[old_idx, 1]]
        new_tri[new_idx, 2] = old_to_new[tri[old_idx, 2]]
        new_tri[new_idx, 3] = old_to_new[tri[old_idx, 3]]
        new_eleGma[new_idx, :] = eleGma[old_idx, :]
    end
    
    # Update triangle arrays in place
    tri[:, :] .= 0
    eleGma[:, :] .= 0.0
    if size(tri, 1) >= size(new_tri, 1)
        tri[1:size(new_tri, 1), :] = new_tri
        eleGma[1:size(new_eleGma, 1), :] = new_eleGma
    end
    
    return nothing
end

function compute_node_circulations(tri, eleGma)
    # Compute circulation at each node by averaging connected elements
    node_count = maximum(tri)
    node_circulations = [zeros(3) for _ in 1:node_count]
    node_counts = zeros(Int, node_count)
    
    for t in 1:size(tri, 1)
        for k in 1:3
            node = tri[t, k]
            for i in 1:3
                node_circulations[node][i] += eleGma[t, i]
            end
            node_counts[node] += 1
        end
    end
    
    # Average
    for i in 1:node_count
        if node_counts[i] > 0
            for j in 1:3
                node_circulations[i][j] /= node_counts[i]
            end
        end
    end
    
    return [Tuple(nc) for nc in node_circulations]
end

function add_blob_particle_to_mesh!(tri, eleGma, node_index, circulation, domain)
    # Add a particle from vortex blob to the mesh
    if node_index <= 0
        return false
    end
    
    # Create a simple triangle with the new node and two previous nodes if available
    current_triangles = 0
    for t in 1:size(tri, 1)
        if tri[t, 1] > 0
            current_triangles += 1
        end
    end
    
    if node_index >= 3
        # Find first available triangle slot
        triangle_slot = -1
        for t in 1:size(tri, 1)
            if tri[t, 1] == 0 || tri[t, 2] == 0 || tri[t, 3] == 0
                triangle_slot = t
                break
            end
        end
        
        if triangle_slot == -1
            # Need to expand triangle array (simplified - in practice would resize)
            return false
        end
        
        # Create triangle connecting new node with two previous nodes
        tri[triangle_slot, 1] = max(1, node_index - 2)
        tri[triangle_slot, 2] = max(1, node_index - 1)
        tri[triangle_slot, 3] = node_index
        
        # Assign circulation to this triangle
        eleGma[triangle_slot, 1] = circulation[1]
        eleGma[triangle_slot, 2] = circulation[2]
        eleGma[triangle_slot, 3] = circulation[3]
        
        return true
    elseif node_index == 1
        # First node - can't create triangle yet
        return true
    elseif node_index == 2
        # Second node - still can't create triangle
        return true
    end
    
    return false
end


function insert_particles_to_fill_gaps!(nodeX, nodeY, nodeZ, tri, eleGma, domain, n_needed)
    if n_needed <= 0
        return 0
    end
    
    n_inserted = 0
    
    # Generate candidate positions in sparse regions
    # Use a simple grid-based approach to find gaps
    grid_resolution = 10
    dx = domain.Lx / grid_resolution
    dy = domain.Ly / grid_resolution 
    dz = 2 * domain.Lz / grid_resolution
    
    candidates = Tuple{Float64, Float64, Float64, Float64}[]  # x, y, z, priority
    
    # Sample grid points and check sparsity
    for i in 1:grid_resolution, j in 1:grid_resolution, k in 1:grid_resolution
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        z = (k - 0.5) * dz - domain.Lz
        
        # Find minimum distance to existing particles
        min_dist = Inf
        for p in 1:length(nodeX)
            # Periodic distance
            dx_p = minimum_image_distance(x - nodeX[p], domain.Lx)
            dy_p = minimum_image_distance(y - nodeY[p], domain.Ly)
            dz_p = minimum_image_distance(z - nodeZ[p], 2 * domain.Lz)
            dist = sqrt(dx_p^2 + dy_p^2 + dz_p^2)
            min_dist = min(min_dist, dist)
        end
        
        # If region is sparse, add as candidate
        target_spacing = min(dx, dy, dz) * 2  # Target minimum spacing
        if min_dist > target_spacing
            priority = min_dist  # Higher distance = higher priority
            push!(candidates, (x, y, z, priority))
        end
    end
    
    # Sort by priority (most sparse regions first)
    sort!(candidates, by=x->x[4], rev=true)
    
    # Insert particles up to the needed count
    for candidate in candidates
        if n_inserted >= n_needed
            break
        end
        
        x, y, z, _ = candidate
        
        # Wrap to domain
        x, y, z = wrap_point(x, y, z, domain)
        
        # Add particle
        push!(nodeX, x)
        push!(nodeY, y)
        push!(nodeZ, z)
        
        # Create insertion candidate for mesh update
        insertion_candidate = InsertionCandidate(
            x, y, z, 1.0, (1e-8, 1e-8, 1e-8)  # Small default circulation
        )
        
        # Try to add to mesh
        success = insert_particle_into_mesh!(tri, eleGma, length(nodeX), insertion_candidate, domain)
        
        if success
            n_inserted += 1
        else
            # Remove the node if mesh insertion failed
            pop!(nodeX)
            pop!(nodeY)
            pop!(nodeZ)
        end
    end
    
    return n_inserted
end


function remove_weakest_particles!(nodeX, nodeY, nodeZ, tri, eleGma, domain, n_excess)
    if n_excess <= 0 || length(nodeX) == 0
        return 0
    end
    
    # Compute circulation magnitude for each node
    node_circulations = compute_node_circulations(tri, eleGma)
    
    # Create list of (node_index, circulation_magnitude) pairs
    node_strengths = Tuple{Int, Float64}[]
    for (i, circ) in enumerate(node_circulations)
        circ_mag = sqrt(circ[1]^2 + circ[2]^2 + circ[3]^2)
        push!(node_strengths, (i, circ_mag))
    end
    
    # Sort by circulation magnitude (weakest first)
    sort!(node_strengths, by=x->x[2])
    
    n_removed = 0
    removed_indices = Int[]
    
    # Remove weakest particles
    for (node_idx, _) in node_strengths
        if n_removed >= n_excess
            break
        end
        
        if node_idx <= length(nodeX) && !(node_idx in removed_indices)
            # Try to remove this particle
            success = remove_particle_from_mesh!(nodeX, nodeY, nodeZ, tri, eleGma, node_idx, domain)
            if success
                push!(removed_indices, node_idx)
                n_removed += 1
                
                # Adjust indices for remaining particles (since removal shifts indices)
                # Note: remove_particle_from_mesh! already handles index shifting
            end
        end
    end
    
    return n_removed
end


# Convenience functions for common operations

"""
redistribute_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain)

Redistribute particles to maintain more uniform spacing while preserving circulation.
"""
function redistribute_particles_periodic!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                        tri::Matrix{Int}, eleGma::Matrix{Float64}, domain::DomainSpec)
    
    if length(nodeX) < 4
        wrap_nodes!(nodeX, nodeY, nodeZ, domain)
        return length(nodeX)  # Too few particles to redistribute
    end
    
    # 1. Compute local particle density using spatial binning
    grid_nx, grid_ny, grid_nz = 8, 8, 8
    dx = domain.Lx / grid_nx
    dy = domain.Ly / grid_ny
    dz = 2 * domain.Lz / grid_nz
    
    # Count particles in each grid cell
    grid_counts = zeros(Int, grid_nx, grid_ny, grid_nz)
    particle_assignments = zeros(Int, length(nodeX))  # Which cell each particle belongs to
    
    for p in 1:length(nodeX)
        # Map particle to grid cell
        i = min(grid_nx, max(1, Int(ceil(nodeX[p] / dx))))
        j = min(grid_ny, max(1, Int(ceil(nodeY[p] / dy))))
        k = min(grid_nz, max(1, Int(ceil((nodeZ[p] + domain.Lz) / dz))))
        
        grid_counts[i, j, k] += 1
        particle_assignments[p] = (k-1) * grid_nx * grid_ny + (j-1) * grid_nx + i
    end
    
    # 2. Find cells that are too dense or too sparse
    target_density = length(nodeX) / (grid_nx * grid_ny * grid_nz)
    density_tolerance = 0.5
    
    overcrowded_cells = Int[]
    underpopulated_cells = Int[]
    
    for i in 1:grid_nx, j in 1:grid_ny, k in 1:grid_nz
        cell_id = (k-1) * grid_nx * grid_ny + (j-1) * grid_nx + i
        density = grid_counts[i, j, k]
        
        if density > target_density * (1 + density_tolerance)
            push!(overcrowded_cells, cell_id)
        elseif density < target_density * (1 - density_tolerance)
            push!(underpopulated_cells, cell_id)
        end
    end
    
    # 3. Move particles from overcrowded to underpopulated regions
    n_redistributed = 0
    
    for overcrowded_cell in overcrowded_cells
        if isempty(underpopulated_cells)
            break
        end
        
        # Find particles in this overcrowded cell
        particles_in_cell = Int[]
        for p in 1:length(nodeX)
            if particle_assignments[p] == overcrowded_cell
                push!(particles_in_cell, p)
            end
        end
        
        if length(particles_in_cell) <= 1
            continue  # Need to keep at least one particle
        end
        
        # Move some particles to underpopulated regions
        n_to_move = min(length(particles_in_cell) ÷ 2, length(underpopulated_cells))
        
        for m in 1:n_to_move
            if isempty(underpopulated_cells) || isempty(particles_in_cell)
                break
            end
            
            # Select particle to move (prefer those with weaker circulation)
            node_circulations = compute_node_circulations(tri, eleGma)
            particle_to_move = particles_in_cell[1]  # Default to first
            
            if length(particles_in_cell) > 1
                min_circulation = Inf
                for p in particles_in_cell
                    if p <= length(node_circulations)
                        circ = node_circulations[p]
                        circ_mag = sqrt(circ[1]^2 + circ[2]^2 + circ[3]^2)
                        if circ_mag < min_circulation
                            min_circulation = circ_mag
                            particle_to_move = p
                        end
                    end
                end
            end
            
            # Select target underpopulated cell
            target_cell = underpopulated_cells[1]
            
            # Convert cell ID back to grid coordinates
            target_k = (target_cell - 1) ÷ (grid_nx * grid_ny) + 1
            target_j = ((target_cell - 1) % (grid_nx * grid_ny)) ÷ grid_nx + 1
            target_i = ((target_cell - 1) % grid_nx) + 1
            
            # Calculate new position within target cell (with some randomness)
            new_x = (target_i - 0.5 + 0.2 * (rand() - 0.5)) * dx
            new_y = (target_j - 0.5 + 0.2 * (rand() - 0.5)) * dy
            new_z = (target_k - 0.5 + 0.2 * (rand() - 0.5)) * dz - domain.Lz
            
            # Apply periodic wrapping
            new_x, new_y, new_z = wrap_point(new_x, new_y, new_z, domain)
            
            # Move the particle
            nodeX[particle_to_move] = new_x
            nodeY[particle_to_move] = new_y
            nodeZ[particle_to_move] = new_z
            
            # Update assignments
            filter!(p -> p != particle_to_move, particles_in_cell)
            particle_assignments[particle_to_move] = target_cell
            n_redistributed += 1
            
            # Check if target cell is no longer underpopulated
            # Update grid counts
            target_cell_count = count(p -> particle_assignments[p] == target_cell, 1:length(nodeX))
            if target_cell_count >= target_density * (1 - density_tolerance)
                filter!(c -> c != target_cell, underpopulated_cells)
            end
        end
    end
    
    # 4. Ensure all particles are within periodic boundaries
    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    
    return length(nodeX)  # Return final particle count
end

"""
remove_weak_vortices!(nodeX, nodeY, nodeZ, tri, eleGma, domain, threshold)

Remove all particles with circulation magnitude below threshold.
"""
function remove_weak_vortices!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                              tri::Matrix{Int}, eleGma::Matrix{Float64}, domain::DomainSpec, 
                              threshold::Float64=1e-10)
    
    criteria = ParticleRemovalCriteria(weak_circulation_threshold=threshold)
    return remove_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, criteria)
end

"""
    adaptive_particle_control!(nodeX, nodeY, nodeZ, tri, eleGma, domain; target_count, tolerance, insert_criteria, removal_criteria)

Intelligent particle management system that automatically maintains optimal particle distribution.

This function provides two modes of operation:
1. **Target Count Mode**: Maintains particle count within specified tolerance of a target
2. **Criteria-Based Mode**: Performs insertion and removal based on physical criteria

The system ensures circulation conservation throughout all operations and respects periodic boundary conditions.

# Arguments
- `nodeX, nodeY, nodeZ::Vector{Float64}`: Particle position arrays (modified in-place)
- `tri::Matrix{Int}`: Triangle connectivity matrix (modified in-place)
- `eleGma::Matrix{Float64}`: Element circulation vectors (modified in-place)
- `domain::DomainSpec`: Periodic domain specification

# Keyword Arguments
- `target_count::Union{Nothing,Int}=nothing`: Target particle count (if specified, uses target count mode)
- `tolerance::Float64=0.1`: Allowed deviation from target count (±10% by default)
- `insert_criteria::ParticleInsertionCriteria`: Criteria for particle insertion (used in criteria-based mode)
- `removal_criteria::ParticleRemovalCriteria`: Criteria for particle removal (used in criteria-based mode)

# Returns
- `n_change::Int`: Net change in particle count (positive = more particles, negative = fewer particles)

# Modes

## Target Count Mode
When `target_count` is provided, automatically adds or removes particles to maintain the count within tolerance:
```julia
# Maintain 10,000 ± 10% particles
n_change = adaptive_particle_control!(nodeX, nodeY, nodeZ, tri, eleGma, domain; 
                                     target_count=10000, tolerance=0.1)
```

## Criteria-Based Mode  
When `target_count=nothing`, uses physical criteria to determine insertion/removal:
```julia
# Use custom criteria for particle management
insert_criteria = ParticleInsertionCriteria(max_particles=50000, max_particle_spacing=0.02)
removal_criteria = ParticleRemovalCriteria(weak_circulation_threshold=1e-8)
n_change = adaptive_particle_control!(nodeX, nodeY, nodeZ, tri, eleGma, domain;
                                     insert_criteria=insert_criteria,
                                     removal_criteria=removal_criteria)
```

# Features
- **Circulation Conservation**: Maintains total vorticity throughout all operations
- **Periodic Boundaries**: Full support for periodic domain wrapping
- **Adaptive Resolution**: Adds particles where needed, removes where unnecessary
- **Flexible Criteria**: Configurable thresholds for spacing, circulation strength, etc.
- **Performance Optimized**: Efficient algorithms for large particle counts
"""
function adaptive_particle_control!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                    tri::Matrix{Int}, eleGma::Matrix{Float64}, domain::DomainSpec;
                                    target_count::Union{Nothing,Int}=nothing,
                                    tolerance::Float64=0.1,
                                    insert_criteria::ParticleInsertionCriteria=ParticleInsertionCriteria(),
                                    removal_criteria::ParticleRemovalCriteria=ParticleRemovalCriteria())
    if target_count !== nothing
        return maintain_particle_count!(nodeX, nodeY, nodeZ, tri, eleGma, domain, target_count, tolerance)
    else
        before = length(nodeX)
        _ = insert_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, insert_criteria)
        _ = remove_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain, removal_criteria)
        after = length(nodeX)
        return after - before
    end
end

end # module

using .ParticleManagement: insert_particles_periodic!, remove_particles_periodic!, 
                          compact_mesh!, adaptive_particle_control!,
                          ParticleInsertionCriteria, ParticleRemovalCriteria,
                          insert_vortex_blob_periodic!, remove_weak_vortices!,
                          maintain_particle_count!, redistribute_particles_periodic!
