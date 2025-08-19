# Lightweight domain/grid config and helpers

module DomainImpl

export DomainSpec, GridSpec, default_domain, default_grid,
       grid_vectors, grid_spacing, grid_mesh, kvec,
       wrap_point, wrap_nodes!

struct DomainSpec
    Lx::Float64
    Ly::Float64
    Lz::Float64
end

struct GridSpec
    nx::Int
    ny::Int
    nz::Int
end

default_domain() = DomainSpec(1.0, 1.0, 1.0)
default_grid()   = GridSpec(25, 50, 2*50-1) # mirrors python utility_3d_paral._SmoothGrid_Generation_

function grid_vectors(domain::DomainSpec, gr::GridSpec)
    # For periodic FFT, grid should go from 0 to L*(1-1/N), not 0 to L
    x = range(0.0, domain.Lx * (1 - 1/gr.nx); length=gr.nx) |> collect
    y = range(0.0, domain.Ly * (1 - 1/gr.ny); length=gr.ny) |> collect
    # z is symmetric [-Lz, Lz] with periodic endpoints  
    z = range(-domain.Lz, domain.Lz * (1 - 1/gr.nz); length=gr.nz) |> collect
    return x, y, z
end

grid_spacing(domain::DomainSpec, gr::GridSpec) = begin
    x, y, z = grid_vectors(domain, gr)
    (abs(x[2]-x[1]), abs(y[2]-y[1]), abs(z[2]-z[1]))
end

function grid_mesh(domain::DomainSpec, gr::GridSpec)
    x, y, z = grid_vectors(domain, gr)
    # match python ordering: y3d, z3d, x3d = np.meshgrid(y, z, x)
    # We return x3d, y3d, z3d flattened to 1D for convenience
    x3d = Array{Float64}(undef, gr.ny, gr.nz, gr.nx)
    y3d = Array{Float64}(undef, gr.ny, gr.nz, gr.nx)
    z3d = Array{Float64}(undef, gr.ny, gr.nz, gr.nx)

    @inbounds for j in 1:gr.ny, k in 1:gr.nz, i in 1:gr.nx
        x3d[j,k,i] = x[i]
        y3d[j,k,i] = y[j]
        z3d[j,k,i] = z[k]
    end
    
    return vec(permutedims(x3d, (3,1,2))),
           vec(permutedims(y3d, (3,1,2))),
           vec(permutedims(z3d, (3,1,2)))
end

# Spectral wavenumber vector for periodic FFT Poisson solve
function kvec(n::Int, L::Float64)
    k = similar(zeros(Float64), n)
    n2 = fld(n, 2)
    @inbounds for i in 1:n
        m = i-1
        if m <= n2
            k[i] = 2π * m / L
        else
            k[i] = -2π * (n - m) / L
        end
    end
    return k
end

# Wrap a point into periodic domain ranges
function wrap_point(x::Float64, y::Float64, z::Float64, domain::DomainSpec)
    # Use mod to ensure results lie within [0,L) and [-Lz, Lz]
    xw = mod(x, domain.Lx)
    yw = mod(y, domain.Ly)
    zw = mod(z + domain.Lz, 2 * domain.Lz) - domain.Lz
    return xw, yw, zw
end

# In-place wrapping of node arrays
function wrap_nodes!(node_x::AbstractVector{<:Real}, 
                    node_y::AbstractVector{<:Real}, 
                    node_z::AbstractVector{<:Real}, 
                    domain::DomainSpec)

    @inbounds for i in eachindex(node_x)
        xw, yw, zw = wrap_point(Float64(node_x[i]), Float64(node_y[i]), Float64(node_z[i]), domain)
        node_x[i] = xw; node_y[i] = yw; node_z[i] = zw
    end

    return nothing
end

end # module

using .DomainImpl: DomainSpec, GridSpec, default_domain, default_grid, grid_vectors, grid_spacing, grid_mesh, kvec, wrap_point, wrap_nodes!
