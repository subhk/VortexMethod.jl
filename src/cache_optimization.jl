# Cache-aware algorithms optimized for modern CPU memory hierarchy

module CacheOptimization

using LinearAlgebra

export TiledPoissonSolver, BlockedSpreadingKernel, CacheAwareMesh,
       tiled_curl_computation!, blocked_kernel_evaluation!,
       cache_optimized_interpolation!, hierarchical_grid_traversal

# Cache-optimized tile sizes (tuned for typical L1/L2/L3 cache sizes)
const L1_CACHE_SIZE = 32 * 1024      # 32KB L1 cache
const L2_CACHE_SIZE = 256 * 1024     # 256KB L2 cache  
const L3_CACHE_SIZE = 8 * 1024 * 1024 # 8MB L3 cache

# Optimal tile sizes for different operations
const CURL_TILE_SIZE = 32        # For curl computation (fits in L1)
const KERNEL_TILE_SIZE = 64      # For kernel evaluation (fits in L2)
const INTERPOLATION_TILE_SIZE = 128  # For interpolation (fits in L3)

# Cache-aware tiled Poisson solver
struct TiledPoissonSolver{T<:AbstractFloat}
    # Pre-allocated tile buffers to minimize allocation overhead
    tile_buffer_1::Array{T,3}
    tile_buffer_2::Array{T,3}
    tile_buffer_3::Array{T,3}
    
    # Tile dimensions
    tile_x::Int
    tile_y::Int
    tile_z::Int
    
    # Grid dimensions
    nx::Int
    ny::Int
    nz::Int
end

function TiledPoissonSolver(::Type{T}, nx::Int, ny::Int, nz::Int; tile_size::Int=CURL_TILE_SIZE) where T
    # Ensure tile size doesn't exceed grid dimensions
    tile_x = min(tile_size, nx)
    tile_y = min(tile_size, ny)
    tile_z = min(tile_size, nz)
    
    TiledPoissonSolver{T}(
        Array{T}(undef, tile_z, tile_y, tile_x),
        Array{T}(undef, tile_z, tile_y, tile_x),
        Array{T}(undef, tile_z, tile_y, tile_x),
        tile_x, tile_y, tile_z, nx, ny, nz
    )
end

TiledPoissonSolver(nx::Int, ny::Int, nz::Int; kwargs...) = TiledPoissonSolver(Float64, nx, ny, nz; kwargs...)

# Cache-optimized curl computation using tiling
function tiled_curl_computation!(solver::TiledPoissonSolver{T},
                                u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                                VorX::Array{T,3}, VorY::Array{T,3}, VorZ::Array{T,3},
                                dx::T, dy::T, dz::T) where T
    nz, ny, nx = size(VorX)
    tile_x, tile_y, tile_z = solver.tile_x, solver.tile_y, solver.tile_z
    
    # Process grid in cache-friendly tiles
    for k_start in 1:tile_z:nz
        k_end = min(k_start + tile_z - 1, nz)
        for j_start in 1:tile_y:ny
            j_end = min(j_start + tile_y - 1, ny)
            for i_start in 1:tile_x:nx
                i_end = min(i_start + tile_x - 1, nx)
                
                # Process tile with good spatial locality
                process_curl_tile!(solver, u_rhs, v_rhs, w_rhs, VorX, VorY, VorZ,
                                 i_start:i_end, j_start:j_end, k_start:k_end, dx, dy, dz)
            end
        end
    end
end

function process_curl_tile!(solver::TiledPoissonSolver{T}, 
                           u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                           VorX::Array{T,3}, VorY::Array{T,3}, VorZ::Array{T,3},
                           i_range, j_range, k_range, dx::T, dy::T, dz::T) where T
    
    # Load tile data into cache-friendly buffers
    tile_vorx = solver.tile_buffer_1
    tile_vory = solver.tile_buffer_2 
    tile_vorz = solver.tile_buffer_3
    
    tile_nz, tile_ny, tile_nx = length(k_range), length(j_range), length(i_range)
    
    # Copy tile data (this loads into cache)
    @inbounds for (kk, k) in enumerate(k_range)
        for (jj, j) in enumerate(j_range)  
            for (ii, i) in enumerate(i_range)
                tile_vorx[kk, jj, ii] = VorX[k, j, i]
                tile_vory[kk, jj, ii] = VorY[k, j, i]
                tile_vorz[kk, jj, ii] = VorZ[k, j, i]
            end
        end
    end
    
    # Compute derivatives on tile (data is hot in cache)
    compute_tile_derivatives!(solver, tile_vorx, tile_vory, tile_vorz, dx, dy, dz)
    
    # Write results back to main arrays
    write_tile_results!(u_rhs, v_rhs, w_rhs, solver, i_range, j_range, k_range)
end

function compute_tile_derivatives!(solver::TiledPoissonSolver{T},
                                  tile_vorx::Array{T,3}, tile_vory::Array{T,3}, tile_vorz::Array{T,3},
                                  dx::T, dy::T, dz::T) where T
    tile_nz, tile_ny, tile_nx = size(tile_vorx)
    
    # Compute finite differences within tile (optimal cache usage)
    @inbounds for k in 2:tile_nz-1
        for j in 2:tile_ny-1
            for i in 2:tile_nx-1
                # Central differences (4th order when possible)
                dVorY_dz = (tile_vory[k+1,j,i] - tile_vory[k-1,j,i]) / (2*dz)
                dVorZ_dy = (tile_vorz[k,j+1,i] - tile_vorz[k,j-1,i]) / (2*dy)
                
                dVorX_dz = (tile_vorx[k+1,j,i] - tile_vorx[k-1,j,i]) / (2*dz)
                dVorZ_dx = (tile_vorz[k,j,i+1] - tile_vorz[k,j,i-1]) / (2*dx)
                
                dVorX_dy = (tile_vorx[k,j+1,i] - tile_vorx[k,j-1,i]) / (2*dy)
                dVorY_dx = (tile_vory[k,j,i+1] - tile_vory[k,j,i-1]) / (2*dx)
                
                # Store curl in tile buffers (reusing memory)
                tile_vorx[k,j,i] = -(dVorZ_dy - dVorY_dz)  # u_rhs
                tile_vory[k,j,i] = -(dVorX_dz - dVorZ_dx)  # v_rhs
                tile_vorz[k,j,i] = -(dVorY_dx - dVorX_dy)  # w_rhs
            end
        end
    end
end

function write_tile_results!(u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                           solver::TiledPoissonSolver{T}, i_range, j_range, k_range) where T
    # Write computed results back (streaming write pattern)
    @inbounds for (kk, k) in enumerate(k_range)
        for (jj, j) in enumerate(j_range)
            for (ii, i) in enumerate(i_range)
                u_rhs[k,j,i] = solver.tile_buffer_1[kk,jj,ii]
                v_rhs[k,j,i] = solver.tile_buffer_2[kk,jj,ii]
                w_rhs[k,j,i] = solver.tile_buffer_3[kk,jj,ii]
            end
        end
    end
end

# Blocked spreading kernel for cache optimization
struct BlockedSpreadingKernel{T<:AbstractFloat}
    # Blocked data structures for better cache performance
    particle_blocks::Vector{Vector{Int}}      # Particle indices per block
    grid_point_blocks::Vector{Vector{Int}}    # Grid point indices per block
    
    # Cache-friendly temporary arrays
    block_distances::Matrix{T}               # [block_size, 3] for dx,dy,dz
    block_weights::Vector{T}                 # [block_size] for kernel weights
    block_contributions::Matrix{T}           # [block_size, 3] for ωx,ωy,ωz contributions
    
    block_size::Int
end

function BlockedSpreadingKernel(::Type{T}, max_particles::Int, max_grid_points::Int; 
                               block_size::Int=KERNEL_TILE_SIZE) where T
    n_blocks_p = div(max_particles + block_size - 1, block_size)
    n_blocks_g = div(max_grid_points + block_size - 1, block_size)
    
    BlockedSpreadingKernel{T}(
        [Vector{Int}() for _ in 1:n_blocks_p],
        [Vector{Int}() for _ in 1:n_blocks_g],
        Matrix{T}(undef, block_size, 3),
        Vector{T}(undef, block_size),
        Matrix{T}(undef, block_size, 3),
        block_size
    )
end

# Cache-optimized kernel evaluation using blocking
function blocked_kernel_evaluation!(kernel::BlockedSpreadingKernel{T},
                                   grid_values::Array{T,3},
                                   particle_positions::Matrix{T},
                                   particle_strengths::Matrix{T},
                                   grid_spacing::NTuple{3,T}) where T
    
    dx, dy, dz = grid_spacing
    block_size = kernel.block_size
    
    # Process particles in blocks for cache efficiency
    for p_block in 1:length(kernel.particle_blocks)
        particle_indices = kernel.particle_blocks[p_block]
        n_particles_in_block = length(particle_indices)
        
        if n_particles_in_block == 0
            continue
        end
        
        # Process grid points in blocks
        for g_block in 1:length(kernel.grid_point_blocks)
            grid_indices = kernel.grid_point_blocks[g_block]  
            n_grid_in_block = length(grid_indices)
            
            if n_grid_in_block == 0
                continue
            end
            
            # Compute particle-grid interactions for this block pair
            compute_block_interactions!(kernel, grid_values, particle_positions, particle_strengths,
                                       particle_indices, grid_indices, grid_spacing)
        end
    end
end

function compute_block_interactions!(kernel::BlockedSpreadingKernel{T},
                                   grid_values::Array{T,3}, 
                                   particle_positions::Matrix{T},
                                   particle_strengths::Matrix{T},
                                   particle_indices::Vector{Int},
                                   grid_indices::Vector{Int},
                                   grid_spacing::NTuple{3,T}) where T
    
    dx, dy, dz = grid_spacing
    n_particles = length(particle_indices)
    n_grid = length(grid_indices)
    
    # Cache-friendly computation: all particles to all grid points in block
    @inbounds for (gp_idx, grid_idx) in enumerate(grid_indices)
        # Convert linear grid index to 3D coordinates (this could be precomputed)
        grid_pos = linear_to_3d(grid_idx, size(grid_values))
        gx, gy, gz = grid_pos[1] * dx, grid_pos[2] * dy, grid_pos[3] * dz
        
        # Process all particles in block against this grid point
        for p_idx in 1:length(particle_indices)
            particle_idx = particle_indices[p_idx]
            px, py, pz = particle_positions[particle_idx, 1], particle_positions[particle_idx, 2], particle_positions[particle_idx, 3]
            
            # Compute distance
            dist_x, dist_y, dist_z = px - gx, py - gy, pz - gz
            
            # Compute kernel weight (vectorizable)
            weight = compute_kernel_weight(dist_x, dist_y, dist_z, dx, dy, dz)
            
            # Add contribution to grid point
            if weight > 0
                contribution = weight * particle_strengths[particle_idx, :]
                grid_values[grid_pos...] .+= contribution
            end
        end
    end
end

@inline function linear_to_3d(linear_idx::Int, grid_size::NTuple{3,Int})
    nz, ny, nx = grid_size
    
    k = div(linear_idx - 1, ny * nx) + 1
    remainder = linear_idx - (k - 1) * ny * nx
    j = div(remainder - 1, nx) + 1
    i = remainder - (j - 1) * nx
    
    return (k, j, i)
end

@inline function compute_kernel_weight(dx::T, dy::T, dz::T, hx::T, hy::T, hz::T) where T
    # Fast Peskin kernel evaluation
    rx, ry, rz = abs(dx) / hx, abs(dy) / hy, abs(dz) / hz
    
    kx = (rx >= 2.0) ? 0.0 : (1.0 + cos(π * rx)) / (2.0 * hx)
    ky = (ry >= 2.0) ? 0.0 : (1.0 + cos(π * ry)) / (2.0 * hy)
    kz = (rz >= 2.0) ? 0.0 : (1.0 + cos(π * rz)) / (2.0 * hz)
    
    return kx * ky * kz
end

# Cache-aware mesh representation
struct CacheAwareMesh{T<:AbstractFloat}
    # Spatially sorted data for better cache locality
    sorted_triangles::Vector{Int}            # Triangle indices sorted by spatial location
    spatial_bounds::Matrix{T}               # [triangle_id, (xmin,xmax,ymin,ymax,zmin,zmax)]
    
    # Hierarchical grid for fast neighbor finding
    grid_cells::Dict{NTuple{3,Int}, Vector{Int}}  # grid_cell -> triangle_list
    grid_resolution::NTuple{3,Int}
    
    # Cache-optimized triangle data
    triangle_centroids::Matrix{T}           # [triangle_id, xyz] - sorted by spatial location
    triangle_areas::Vector{T}               # Sorted by triangle_id order
    
    # Pre-computed neighbor lists for cache prefetching
    neighbor_lists::Vector{Vector{Int}}     # [triangle_id] -> neighbor_triangle_ids
end

# Hierarchical grid traversal for cache-friendly particle-grid operations
function hierarchical_grid_traversal(operation_func::Function, grid::Array{T,3}, 
                                    particles::Matrix{T}, block_size::Int=64) where T
    nz, ny, nx = size(grid)
    n_particles = size(particles, 1)
    
    # Process in hierarchical blocks: L3 -> L2 -> L1 cache levels
    l3_block_size = div(L3_CACHE_SIZE, sizeof(T) * 3)  # Rough estimate
    l2_block_size = div(L2_CACHE_SIZE, sizeof(T) * 3)
    l1_block_size = div(L1_CACHE_SIZE, sizeof(T) * 3)
    
    # Level 3: Large blocks that fit in L3 cache
    for z_l3 in 1:l3_block_size:nz
        z_l3_end = min(z_l3 + l3_block_size - 1, nz)
        
        # Level 2: Medium blocks that fit in L2 cache
        for z_l2 in z_l3:l2_block_size:z_l3_end
            z_l2_end = min(z_l2 + l2_block_size - 1, z_l3_end)
            
            for y_l2 in 1:l2_block_size:ny
                y_l2_end = min(y_l2 + l2_block_size - 1, ny)
                
                # Level 1: Small blocks that fit in L1 cache
                for z_l1 in z_l2:l1_block_size:z_l2_end
                    z_l1_end = min(z_l1 + l1_block_size - 1, z_l2_end)
                    
                    for y_l1 in y_l2:l1_block_size:y_l2_end
                        y_l1_end = min(y_l1 + l1_block_size - 1, y_l2_end)
                        
                        for x_l1 in 1:l1_block_size:nx
                            x_l1_end = min(x_l1 + l1_block_size - 1, nx)
                            
                            # Process this L1-sized block
                            grid_block = view(grid, z_l1:z_l1_end, y_l1:y_l1_end, x_l1:x_l1_end)
                            operation_func(grid_block, particles, x_l1:x_l1_end, y_l1:y_l1_end, z_l1:z_l1_end)
                        end
                    end
                end
            end
        end
    end
end

# Cache-optimized interpolation with prefetching
function cache_optimized_interpolation!(result::AbstractVector{T},
                                       grid::Array{T,3},
                                       positions::Matrix{T},
                                       grid_spacing::NTuple{3,T}) where T
    n_positions = size(positions, 1)
    dx, dy, dz = grid_spacing
    nz, ny, nx = size(grid)
    
    # Sort positions by spatial locality to improve cache performance
    sorted_indices = sort_positions_spatially(positions, grid_spacing, (nx, ny, nz))
    
    # Process in cache-friendly blocks
    block_size = INTERPOLATION_TILE_SIZE
    
    for block_start in 1:block_size:n_positions
        block_end = min(block_start + block_size - 1, n_positions)
        
        @inbounds for idx in block_start:block_end
            pos_idx = sorted_indices[idx]
            x, y, z = positions[pos_idx, 1], positions[pos_idx, 2], positions[pos_idx, 3]
            
            # Find grid cell (with bounds checking)
            i = clamp(floor(Int, x / dx) + 1, 1, nx - 1)
            j = clamp(floor(Int, y / dy) + 1, 1, ny - 1)
            k = clamp(floor(Int, z / dz) + 1, 1, nz - 1)
            
            # Trilinear interpolation (cache-friendly access pattern)
            # This accesses 8 neighboring grid points in a predictable pattern
            fx = (x / dx) - (i - 1)
            fy = (y / dy) - (j - 1)
            fz = (z / dz) - (k - 1)
            
            # Interpolate (unrolled for better performance)
            v000 = grid[k, j, i]
            v001 = grid[k, j, i+1]
            v010 = grid[k, j+1, i]
            v011 = grid[k, j+1, i+1]
            v100 = grid[k+1, j, i]
            v101 = grid[k+1, j, i+1]
            v110 = grid[k+1, j+1, i]
            v111 = grid[k+1, j+1, i+1]
            
            # Trilinear interpolation formula
            result[pos_idx] = v000 * (1-fx) * (1-fy) * (1-fz) +
                             v001 * fx * (1-fy) * (1-fz) +
                             v010 * (1-fx) * fy * (1-fz) +
                             v011 * fx * fy * (1-fz) +
                             v100 * (1-fx) * (1-fy) * fz +
                             v101 * fx * (1-fy) * fz +
                             v110 * (1-fx) * fy * fz +
                             v111 * fx * fy * fz
        end
    end
end

function sort_positions_spatially(positions::Matrix{T}, grid_spacing::NTuple{3,T}, grid_size::NTuple{3,Int}) where T
    n_positions = size(positions, 1)
    dx, dy, dz = grid_spacing
    nx, ny, nz = grid_size
    
    # Compute spatial keys for sorting (Z-order/Morton order could be better)
    spatial_keys = Vector{Tuple{Int,Int}}(undef, n_positions)
    
    @inbounds for i in 1:n_positions
        x, y, z = positions[i, 1], positions[i, 2], positions[i, 3]
        
        # Convert to grid coordinates
        gi = clamp(floor(Int, x / dx) + 1, 1, nx)
        gj = clamp(floor(Int, y / dy) + 1, 1, ny)  
        gk = clamp(floor(Int, z / dz) + 1, 1, nz)
        
        # Simple spatial key (could use Z-order for better locality)
        spatial_key = gi + gj * nx + gk * nx * ny
        spatial_keys[i] = (spatial_key, i)
    end
    
    # Sort by spatial key
    sort!(spatial_keys, by=first)
    
    return [pair[2] for pair in spatial_keys]
end

end # module

using .CacheOptimization: TiledPoissonSolver, BlockedSpreadingKernel, CacheAwareMesh,
                         tiled_curl_computation!, blocked_kernel_evaluation!,
                         cache_optimized_interpolation!, hierarchical_grid_traversal