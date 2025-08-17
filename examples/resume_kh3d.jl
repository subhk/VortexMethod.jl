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
ck = load_latest_jld2(ckpt_dir)
nodeX, nodeY, nodeZ, tri, eleGma = ck.nodeX, ck.nodeY, ck.nodeZ, ck.tri, ck.eleGma

# Restore domain/grid if stored
if ck.dom !== nothing
    d = ck.dom
    dom = VortexMethod.DomainSpec(d["Lx"], d["Ly"], d["Lz"])
end
if ck.grid !== nothing
    g = ck.grid
    gr = VortexMethod.GridSpec(g["nx"], g["ny"], g["nz"])
end

nt = size(tri,1)
triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
@inbounds for k in 1:3, t in 1:nt
    v = tri[t,k]
    triXC[t,k] = nodeX[v]; triYC[t,k] = nodeY[v]; triZC[t,k] = nodeZ[v]
end

# Parameters
dt = 1e-3
nsteps = 50
Atg = get(ck.params, :Atg, 0.0)
remesh_every = get(ck.params, :remesh_every, 1)
save_interval = get(ck.params, :save_interval, 0.1)
ar_max = get(ck.params, :ar_max, 4.0)
time = ck.time
next_save_t = save_interval
ke_stride = 5
save_series = true
series_file = joinpath(ckpt_dir, "series.jld2")

if rank == 0
    println("Resuming: nt=$(nt), nodes=$(length(nodeX))")
end

for it in 1:nsteps
    dt_used = rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt; At=Atg, adaptive=true, CFL=0.5, poisson_mode=:fd)
    time += dt_used
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
    if rank == 0 && (time >= next_save_t)
        # compute KE with stride
        save_count = Int(floor(time / save_interval))
        KE = nothing
        if save_count % ke_stride == 0
            nt = size(tri,1)
            triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
            @inbounds for k in 1:3, t in 1:nt
                v = tri[t,k]
                triXC[t,k] = nodeX[v]; triYC[t,k] = nodeY[v]; triZC[t,k] = nodeZ[v]
            end
            KE = gamma_ke(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=:fd)
        end
        if save_series
            base = save_state_timeseries!(series_file, time, nodeX, nodeY, nodeZ, tri, eleGma;
                                          dom=dom, grid=gr, dt=dt_used, CFL=0.5, adaptive=true,
                                          poisson_mode=:fd, remesh_every=remesh_every, save_interval=save_interval,
                                          ar_max=ar_max, step=it,
                                          params_extra=(; Atg=Atg, KE=KE))
        else
            base = save_state!(ckpt_dir, time, nodeX, nodeY, nodeZ, tri, eleGma;
                               dom=dom, grid=gr, dt=dt_used, CFL=0.5, adaptive=true,
                               poisson_mode=:fd, remesh_every=remesh_every, save_interval=save_interval,
                               ar_max=ar_max, step=it,
                               params_extra=(; Atg=Atg, KE=KE))
        end
        println("  checkpoint saved (t=$(round(time,digits=4))): ", save_series ? series_file : base)
        next_save_t += save_interval
    end
end

finalize_mpi!()
