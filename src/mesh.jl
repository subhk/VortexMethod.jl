module Mesh

using ..DomainImpl

export structured_mesh

# Build a structured periodic mesh (Nx by Ny points) with sinusoidal z-perturbation
function structured_mesh(Nx::Int, Ny::Int; dom::DomainSpec=default_domain(),
                         amp::Float64=1e-2)
    x = range(0.0, dom.Lx; length=Nx) |> collect
    y = range(0.0, dom.Ly; length=Ny) |> collect
    # perturb x slightly like python (_xgrid = x + 0.01*sin(2πx))
    X = Array{Float64}(undef, Ny, Nx)
    Y = Array{Float64}(undef, Ny, Nx)
    Z = Array{Float64}(undef, Ny, Nx)
    @inbounds for j in 1:Ny, i in 1:Nx
        X[j,i] = x[i] + 0.01*sin(2pi*x[i])
        Y[j,i] = y[j]
        Z[j,i] = 0.01*sin(2pi*x[i]) + 0.01*sin(4pi*y[j])
    end
    # nodes flattened row-major (j fast or i fast? Use j major consistent with python meshgrid)
    nodeX = vec(permutedims(X, (2,1)))
    nodeY = vec(permutedims(Y, (2,1)))
    nodeZ = vec(permutedims(Z, (2,1)))

    # connectivity: each cell -> two triangles
    ntri = 2*(Nx-1)*(Ny-1)
    tri = Array{Int}(undef, ntri, 3)
    t = 1
    # node index function (i=1..Nx, j=1..Ny) mapped to linear with i first
    idx(i,j) = (j-1)*Nx + (i-1) + 1
    for j in 1:Ny-1, i in 1:Nx-1
        n00 = idx(i, j)
        n10 = idx(i+1, j)
        n01 = idx(i, j+1)
        n11 = idx(i+1, j+1)
        # split quad into two tris: (n00,n10,n01) and (n10,n11,n01)
        tri[t,1]=n00; tri[t,2]=n10; tri[t,3]=n01; t+=1
        tri[t,1]=n10; tri[t,2]=n11; tri[t,3]=n01; t+=1
    end
    # build triangle coordinate matrices
    nt = size(tri,1)
    triXC = Array{Float64}(undef, nt, 3)
    triYC = Array{Float64}(undef, nt, 3)
    triZC = Array{Float64}(undef, nt, 3)
    @inbounds for k in 1:3, t in 1:nt
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    return nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC
end

end # module

using .Mesh: structured_mesh

