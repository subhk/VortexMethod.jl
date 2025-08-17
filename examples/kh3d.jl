# Kelvin–Helmholtz-like 3D vortex sheet example with RK2 stepping

using VortexMethod
using MPI

init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

dom = default_domain()
gr = default_grid()

# Mesh resolution (structured)
Nx = 64
Ny = 64

nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; dom=dom)

nt = size(tri,1)
eleGma = zeros(Float64, nt, 3)
eleGma[:,2] .= 1.0 # initial vortex strength aligned with y

dt = 1e-3
nsteps = 10

if rank == 0
    println("KH 3D: Nx=$(Nx) Ny=$(Ny) nt=$(nt) dt=$(dt) steps=$(nsteps)")
end

for it in 1:nsteps
    rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt)
    # recompute tri coords for next step (connectivity unchanged)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    if rank == 0 && it % 5 == 0
        println("step $it: x=[", minimum(nodeX), ",", maximum(nodeX), "] y=[", minimum(nodeY), ",", maximum(nodeY), "] z=[", minimum(nodeZ), ",", maximum(nodeZ), "]")
    end
end

finalize_mpi!()

