using VortexMethod
using MPI

# Minimal MPI functional check: spread → curl RHS → Poisson (rank-0 broadcast) → interpolate

VortexMethod.init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

dom = VortexMethod.default_domain()
gr  = VortexMethod.GridSpec(8,8,9)  # small grid to keep CI fast

# Tiny structured sheet
Nx, Ny = 4, 4
nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = VortexMethod.structured_mesh(Nx, Ny; dom=dom)
nt = size(tri,1)
eleGma = zeros(Float64, nt, 3)
eleGma[:,2] .= 1.0  # simple nonzero gamma to exercise kernels

# Spread to grid (MPI) and run Poisson
VorX, VorY, VorZ = VortexMethod.spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)
dx,dy,dz = VortexMethod.grid_spacing(dom, gr)
u_rhs, v_rhs, w_rhs = VortexMethod.curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
Ux, Uy, Uz = VortexMethod.poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=:spectral)

# Interpolate back on a handful of nodes (MPI)
sel = 1:min(5, length(nodeX))
u, v, w = VortexMethod.interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX[sel], nodeY[sel], nodeZ[sel], dom, gr)

# Basic sanity assertions on rank 0
if rank == 0
    @assert size(VorX) == (gr.nz, gr.ny, gr.nx)
    @assert size(Ux)   == (gr.nz, gr.ny, gr.nx)
    @assert length(u)  == length(sel)
end

VortexMethod.finalize_mpi!()

