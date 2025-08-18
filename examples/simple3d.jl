# Minimal example: uniform vortex sheet on a lifted square, periodic domain

using VortexMethod
using MPI
using Printf

domain = default_domain()
gr = default_grid()

init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Build a simple two-triangle square in 3D at z=0
triXC = [0.25 0.75 0.25;
         0.75 0.75 0.25]
triYC = [0.25 0.25 0.75;
         0.25 0.75 0.75]
triZC = [0.00 0.00 0.00;
         0.00 0.00 0.00]

nt = size(triXC,1)
eleGma = zeros(Float64, nt, 3)
eleGma[:,2] .= 1.0 # vorticity aligned with y like python init

VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)

dx,dy,dz = grid_spacing(domain, gr)
u_rhs, v_rhs, w_rhs = VortexMethod.curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
Ux, Uy, Uz = poisson_velocity_fft(u_rhs, v_rhs, w_rhs, domain)

# interpolate on the triangle nodes (6 nodes for two triangles; we just pick vertices used above)
nodesX = vec(unique([triXC...]))[1:6]
nodesY = vec(unique([triYC...]))[1:6]
nodesZ = zeros(Float64, length(nodesX))
u, v, w = interpolate_node_velocity_mpi(Ux, Uy, Uz, nodesX, nodesY, nodesZ, domain, gr)

if rank == 0
    println("Computed node velocities (first 3):")
    for i in 1:min(3, length(u))
        println(lpad(string(i),3), ": (", @sprintf("%.4e", u[i]), ", ", @sprintf("%.4e", v[i]), ", ", @sprintf("%.4e", w[i]), ")")
    end
end

finalize_mpi!()
