# Lightweight domain/grid config and helpers

module DomainImpl

export DomainSpec, GridSpec, default_domain, default_grid,
       grid_vectors, grid_spacing, grid_mesh, kvec

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

function grid_vectors(dom::DomainSpec, gr::GridSpec)
    x = range(0.0, dom.Lx; length=gr.nx) |> collect
    y = range(0.0, dom.Ly; length=gr.ny) |> collect
    # z is symmetric [-Lz, Lz]
    z = range(-dom.Lz, dom.Lz; length=gr.nz) |> collect
    return x, y, z
end

grid_spacing(dom::DomainSpec, gr::GridSpec) = begin
    x, y, z = grid_vectors(dom, gr)
    (abs(x[2]-x[1]), abs(y[2]-y[1]), abs(z[2]-z[1]))
end

function grid_mesh(dom::DomainSpec, gr::GridSpec)
    x, y, z = grid_vectors(dom, gr)
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
            k[i] = 2pi * m / L
        else
            k[i] = -2pi * (n - m) / L
        end
    end
    return k
end

end # module

using .DomainImpl: DomainSpec, GridSpec, default_domain, default_grid, grid_vectors, grid_spacing, grid_mesh, kvec

