# Kelvin–Helmholtz-like 3D vortex sheet example with RK2 stepping

using VortexMethod
using MPI
using Printf

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

# Atwood*gravity for baroclinic source (set >0 to activate)
Atg = 0.0

if rank == 0
    println("KH 3D: Nx=$(Nx) Ny=$(Ny) nt=$(nt) dt=$(dt) steps=$(nsteps)")
end

for it in 1:nsteps
    rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt; At=Atg, adaptive=true, CFL=0.5, poisson_mode=:fd)
    # recompute tri coords for next step (connectivity unchanged)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    # simple edge-length based diagnostics; thresholds from grid spacing
    dx,dy,dz = grid_spacing(dom, gr)
    ds_max = 0.80*max(dx,dy)
    ds_min = 0.05*max(dx,dy)
    tmax, maxedge = VortexMethod.Remesh.detect_max_edge_length(triXC, triYC, triZC, ds_max)
    tmin, minedge = VortexMethod.Remesh.detect_min_edge_length(triXC, triYC, triZC, ds_min)
    # perform simple remeshing: split one too-long tri or flip edge for too-short
    nodeCirc = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma)
    tri, didsplit = VortexMethod.Remesh.remesh_pass!(nodeX, nodeY, nodeZ, tri, ds_max, ds_min)
    if didsplit
        nt = size(tri,1)
        triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
        @inbounds for k in 1:3, t in 1:nt
            v = tri[t,k]
            triXC[t,k] = nodeX[v]; triYC[t,k] = nodeY[v]; triZC[t,k] = nodeZ[v]
        end
        eleGma = ele_gamma_from_node_circ(nodeCirc, triXC, triYC, triZC)
    end
    if rank == 0 && it % 5 == 0
        println("step $it: x=[", minimum(nodeX), ",", maximum(nodeX), "] y=[", minimum(nodeY), ",", maximum(nodeY), "] z=[", minimum(nodeZ), ",", maximum(nodeZ), "]")
        println("  max edge len: ", @sprintf("%.3e", maxedge), " (tri=", tmax, ")  min edge len: ", @sprintf("%.3e", minedge), " (tri=", tmin, ")")
    end
end

finalize_mpi!()
