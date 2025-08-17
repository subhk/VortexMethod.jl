using VortexMethod
using MPI
using Printf

init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

dom = default_domain()
gr = default_grid()

# Load the latest checkpoint from directory
ckpt_dir = get(ENV, "CKPT_DIR", "checkpoints")
nodeX, nodeY, nodeZ, tri, eleGma = load_latest_checkpoint(ckpt_dir)

nt = size(tri,1)
triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
@inbounds for k in 1:3, t in 1:nt
    v = tri[t,k]
    triXC[t,k] = nodeX[v]; triYC[t,k] = nodeY[v]; triZC[t,k] = nodeZ[v]
end

# Parameters
dt = 1e-3
nsteps = 50
Atg = 0.0
remesh_every = 1
checkpoint_every = 10
ar_max = 4.0

if rank == 0
    println("Resuming: nt=$(nt), nodes=$(length(nodeX))")
end

for it in 1:nsteps
    rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt; At=Atg, adaptive=true, CFL=0.5, poisson_mode=:fd)
    # rebuild tris
    nt = size(tri,1)
    triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
    @inbounds for k in 1:3, t in 1:nt
        v = tri[t,k]
        triXC[t,k] = nodeX[v]; triYC[t,k] = nodeY[v]; triZC[t,k] = nodeZ[v]
    end
    dx,dy,dz = grid_spacing(dom, gr)
    ds_max = 0.80*max(dx,dy)
    ds_min = 0.05*max(dx,dy)
    if it % remesh_every == 0
        nodeCirc = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma)
        tri, changed = VortexMethod.Remesh.remesh_pass!(nodeX, nodeY, nodeZ, tri, ds_max, ds_min; dom=dom, ar_max=ar_max)
        if changed
            nt = size(tri,1)
            triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
            @inbounds for k in 1:3, t in 1:nt
                v = tri[t,k]
                triXC[t,k] = nodeX[v]; triYC[t,k] = nodeY[v]; triZC[t,k] = nodeZ[v]
            end
            eleGma = ele_gamma_from_node_circ(nodeCirc, triXC, triYC, triZC)
        end
    end
    if rank == 0 && (it % checkpoint_every == 0)
        base = save_checkpoint_mat!(ckpt_dir, it, nodeX, nodeY, nodeZ, tri, eleGma)
        println("  checkpoint saved: ", base)
    end
end

finalize_mpi!()

