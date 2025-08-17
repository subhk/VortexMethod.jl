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
nsteps = 50

# Atwood*gravity for baroclinic source (set >0 to activate)
Atg = 0.0

# Remeshing controls
remesh_every = 1         # perform remesh pass every N steps
save_interval = 0.1      # physical-time saving interval
ar_max = 4.0             # aspect ratio threshold (longest/shortest edge)

time = 0.0
next_save_t = save_interval

if rank == 0
    println("KH 3D: Nx=$(Nx) Ny=$(Ny) nt=$(nt) dt=$(dt) steps=$(nsteps)")
end

for it in 1:nsteps
    dt_used = rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt; At=Atg, adaptive=true, CFL=0.5, poisson_mode=:fd)
    time += dt_used
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
    # aspect ratio diagnostic (max over all triangles)
    function max_aspect_ratio(triXC, triYC, triZC)
        nt = size(triXC,1)
        armax = 0.0
        @inbounds for t in 1:nt
            p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
            p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
            p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
            l12 = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2)
            l23 = sqrt((p2[1]-p3[1])^2 + (p2[2]-p3[2])^2 + (p2[3]-p3[3])^2)
            l31 = sqrt((p3[1]-p1[1])^2 + (p3[2]-p1[2])^2 + (p3[3]-p1[3])^2)
            lmin = min(l12, min(l23, l31)); lmax = max(l12, max(l23, l31))
            if lmin > 0
                ar = lmax / lmin
                armax = max(armax, ar)
            end
        end
        return armax
    end
    ARmax = max_aspect_ratio(triXC, triYC, triZC)
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
    if rank == 0 && it % 5 == 0
        println("step $it: x=[", minimum(nodeX), ",", maximum(nodeX), "] y=[", minimum(nodeY), ",", maximum(nodeY), "] z=[", minimum(nodeZ), ",", maximum(nodeZ), "]")
        println("  max edge len: ", @sprintf("%.3e", maxedge), " (tri=", tmax, ")  min edge len: ", @sprintf("%.3e", minedge), " (tri=", tmin, ")  ARmax=", @sprintf("%.2f", ARmax))
    end
    if rank == 0 && (time >= next_save_t)
        params = (Atg=Atg, CFL=0.5, adaptive=true, poisson_mode=:fd,
                  remesh_every=remesh_every, save_interval=save_interval, ar_max=ar_max,
                  Nx=Nx, Ny=Ny)
        base = save_checkpoint_jld2!("checkpoints", time, nodeX, nodeY, nodeZ, tri, eleGma;
                                     dom=dom, grid=gr, params=params, step=it)
        println("  checkpoint saved (t=$(round(time,digits=4))): ", base)
        next_save_t += save_interval
    end
end

finalize_mpi!()
