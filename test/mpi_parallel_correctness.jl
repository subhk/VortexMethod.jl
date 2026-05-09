using Test
using MPI
using VortexMethod

VortexMethod.init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

const Peskin = VortexMethod.GridTransfer

function serial_spread_reference(eleGma, triXC, triYC, triZC, domain, gr)
    x, y, z = VortexMethod.grid_vectors(domain, gr)
    dx, dy, dz = VortexMethod.grid_spacing(domain, gr)
    segments = Peskin.subtriangle_segment_count(triXC, triYC, triZC, (dx, dy, dz);
                                                domain=domain, max_segments=8)
    triC = Peskin.triangle_centroids(triXC, triYC, triZC; domain=domain)
    subC = Peskin.build_all_subcentroids(triXC, triYC, triZC; domain=domain,
                                         subsegments=segments)
    areas = Peskin.triangle_areas(triXC, triYC, triZC; domain=domain)
    ζx = zeros(Float64, gr.nz, gr.ny, gr.nx)
    ζy = similar(ζx)
    ζz = similar(ζx)
    inv_cell_volume = 1.0 / (dx * dy * dz)

    @inbounds for i in 1:gr.nx, j in 1:gr.ny, k in 1:gr.nz
        sx, sy, sz = Peskin.peskin_grid_sum(eleGma, triC, subC, (x[i], y[j], z[k]),
                                            (dx, dy, dz), areas; domain=domain)
        ζx[k, j, i] = sx * inv_cell_volume
        ζy[k, j, i] = sy * inv_cell_volume
        ζz[k, j, i] = sz * inv_cell_volume
    end

    Peskin.copy_periodic_boundaries!(ζx)
    Peskin.copy_periodic_boundaries!(ζy)
    Peskin.copy_periodic_boundaries!(ζz)
    return ζx, ζy, ζz
end

function serial_interpolate_reference(gridUx, gridUy, gridUz, nodeX, nodeY, nodeZ, domain, gr)
    x, y, z = VortexMethod.grid_vectors(domain, gr)
    dx, dy, dz = VortexMethod.grid_spacing(domain, gr)
    delr = 4.0
    epsx, epsy, epsz = delr * dx, delr * dy, delr * dz
    tiles = VortexMethod.periodic_shifts(domain)
    u = zeros(Float64, length(nodeX))
    v = similar(u)
    w = similar(u)

    @inbounds for n in eachindex(nodeX)
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for (dxL, dyL, dzL) in tiles
            xq = nodeX[n] - dxL
            yq = nodeY[n] - dyL
            zq = nodeZ[n] - dzL
            for k in 1:gr.nz, j in 1:gr.ny, i in 1:gr.nx
                dxv = xq - x[i]
                dyv = yq - y[j]
                dzv = zq - z[k]
                if abs(dxv) <= epsx && abs(dyv) <= epsy && abs(dzv) <= epsz
                    weight = (1 + cos(pi * dxv / epsx)) *
                             (1 + cos(pi * dyv / epsy)) *
                             (1 + cos(pi * dzv / epsz)) / (8 * delr^3)
                    sx += gridUx[k, j, i] * weight
                    sy += gridUy[k, j, i] * weight
                    sz += gridUz[k, j, i] * weight
                end
            end
        end
        u[n] = sx
        v[n] = sy
        w[n] = sz
    end
    return u, v, w
end

function sample_rhs(domain, gr)
    u = zeros(Float64, gr.nz, gr.ny, gr.nx)
    v = similar(u)
    w = similar(u)
    @inbounds for k in 1:gr.nz, j in 1:gr.ny, i in 1:gr.nx
        x = (i - 1) / gr.nx
        y = (j - 1) / gr.ny
        z = (k - 1) / gr.nz
        u[k, j, i] = sin(2pi * x) + 0.2 * cos(2pi * y)
        v[k, j, i] = cos(2pi * z) - 0.1 * sin(2pi * x)
        w[k, j, i] = sin(2pi * y) + 0.3 * cos(2pi * z)
    end
    return u, v, w
end

@testset "MPI parallel correctness" begin
    @test nprocs > 1

    domain = VortexMethod.default_domain()
    gr = VortexMethod.GridSpec(8, 6, 7)
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC =
        VortexMethod.structured_mesh(4, 4; domain=domain, amp=0.01)
    eleGma = zeros(Float64, size(tri, 1), 3)
    @inbounds for t in axes(eleGma, 1)
        eleGma[t, 1] = 0.1 * sin(t)
        eleGma[t, 2] = 1.0 + 0.05 * cos(t)
        eleGma[t, 3] = 0.02 * sin(2t)
    end

    spread_ref = serial_spread_reference(eleGma, triXC, triYC, triZC, domain, gr)
    spread_mpi = VortexMethod.spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC,
                                                           domain, gr)
    @test all(isapprox.(spread_mpi, spread_ref; rtol=1e-12, atol=1e-12))
    ws = VortexMethod.VortexWorkspace(Float64, length(nodeX), size(tri, 1),
                                      gr.nx, gr.ny, gr.nz, domain)
    VortexMethod.spread_vorticity_to_grid_mpi!(ws, eleGma, triXC, triYC, triZC,
                                               domain, gr)
    @test ws.ζx ≈ spread_ref[1] rtol=1e-12 atol=1e-12
    @test ws.ζy ≈ spread_ref[2] rtol=1e-12 atol=1e-12
    @test ws.ζz ≈ spread_ref[3] rtol=1e-12 atol=1e-12

    Ux, Uy, Uz = sample_rhs(domain, gr)
    interp_ref = serial_interpolate_reference(Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr)
    interp_mpi = VortexMethod.interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ,
                                                            domain, gr)
    @test all(isapprox.(map(collect, interp_mpi), interp_ref; rtol=1e-12, atol=1e-12))
    VortexMethod.interpolate_node_velocity_mpi!(ws, Ux, Uy, Uz, nodeX, nodeY, nodeZ,
                                                domain, gr)
    @test ws.u1 ≈ interp_ref[1] rtol=1e-12 atol=1e-12
    @test ws.v1 ≈ interp_ref[2] rtol=1e-12 atol=1e-12
    @test ws.w1 ≈ interp_ref[3] rtol=1e-12 atol=1e-12

    τ_ref = VortexMethod.node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma;
                                                         domain=domain)
    τ_mpi = VortexMethod.node_circulation_from_ele_gamma_mpi(triXC, triYC, triZC, eleGma;
                                                             domain=domain)
    @test τ_mpi ≈ τ_ref rtol=1e-12 atol=1e-12
    @test VortexMethod.ele_gamma_from_node_circ_mpi(τ_ref, triXC, triYC, triZC;
                                                    domain=domain) ≈
          VortexMethod.ele_gamma_from_node_circ(τ_ref, triXC, triYC, triZC;
                                                domain=domain) rtol=1e-12 atol=1e-12

    u_rhs, v_rhs, w_rhs = sample_rhs(domain, gr)
    for mode in (:spectral, :fd)
        serial_poisson = VortexMethod.poisson_velocity_fft(u_rhs, v_rhs, w_rhs,
                                                           domain; mode=mode)
        broadcast_poisson = VortexMethod.poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs,
                                                                  domain; mode=mode)
        @test all(isapprox.(broadcast_poisson, serial_poisson; rtol=1e-12, atol=1e-12))
        VortexMethod.poisson_velocity_fft_mpi!(ws.gridUx, ws.gridUy, ws.gridUz,
                                               ws.fft_x, ws.fft_y, ws.fft_z,
                                               u_rhs, v_rhs, w_rhs, domain; mode=mode)
        @test ws.gridUx ≈ serial_poisson[1] rtol=1e-12 atol=1e-12
        @test ws.gridUy ≈ serial_poisson[2] rtol=1e-12 atol=1e-12
        @test ws.gridUz ≈ serial_poisson[3] rtol=1e-12 atol=1e-12

        pencil_poisson = VortexMethod.poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs,
                                                                  domain; mode=mode)
        @test all(isapprox.(pencil_poisson, serial_poisson; rtol=1e-10, atol=1e-10))

        grid_broadcast = VortexMethod.grid_velocity(eleGma, triXC, triYC, triZC,
                                                    domain, gr; poisson_mode=mode,
                                                    parallel_fft=false)
        grid_pencil = VortexMethod.grid_velocity(eleGma, triXC, triYC, triZC,
                                                 domain, gr; poisson_mode=mode,
                                                 parallel_fft=true)
        @test all(isapprox.(grid_pencil, grid_broadcast; rtol=1e-10, atol=1e-10))
        ws.geom_dirty[] = true
        grid_workspace = VortexMethod.grid_velocity!(ws, eleGma, triXC, triYC, triZC,
                                                     domain, gr; poisson_mode=mode,
                                                     parallel_fft=false)
        @test all(isapprox.(grid_workspace, grid_broadcast; rtol=1e-12, atol=1e-12))
        ws.geom_dirty[] = true
        grid_workspace_pencil = VortexMethod.grid_velocity!(ws, eleGma, triXC, triYC, triZC,
                                                            domain, gr; poisson_mode=mode,
                                                            parallel_fft=true)
        @test all(isapprox.(grid_workspace_pencil, grid_broadcast; rtol=1e-10, atol=1e-10))
        GC.gc()
        ws.geom_dirty[] = true
        @test @allocated(VortexMethod.grid_velocity!(ws, eleGma, triXC, triYC, triZC,
                                                     domain, gr; poisson_mode=mode,
                                                     parallel_fft=true)) < 60_000

        node_broadcast = VortexMethod.node_velocities(eleGma, triXC, triYC, triZC,
                                                      nodeX, nodeY, nodeZ, domain, gr;
                                                      poisson_mode=mode,
                                                      parallel_fft=false)
        node_pencil = VortexMethod.node_velocities(eleGma, triXC, triYC, triZC,
                                                   nodeX, nodeY, nodeZ, domain, gr;
                                                   poisson_mode=mode,
                                                   parallel_fft=true)
        @test all(isapprox.(node_pencil, node_broadcast; rtol=1e-10, atol=1e-10))
        ws.geom_dirty[] = true
        node_workspace = VortexMethod.node_velocities!(ws, ws.u1, ws.v1, ws.w1,
                                                       eleGma, triXC, triYC, triZC,
                                                       nodeX, nodeY, nodeZ, domain, gr;
                                                       poisson_mode=mode,
                                                       parallel_fft=false)
        @test all(isapprox.(node_workspace, node_broadcast; rtol=1e-12, atol=1e-12))
        ws.geom_dirty[] = true
        node_workspace_pencil = VortexMethod.node_velocities!(ws, ws.u1, ws.v1, ws.w1,
                                                              eleGma, triXC, triYC, triZC,
                                                              nodeX, nodeY, nodeZ, domain, gr;
                                                              poisson_mode=mode,
                                                              parallel_fft=true)
        @test all(isapprox.(node_workspace_pencil, node_broadcast; rtol=1e-10, atol=1e-10))
    end
end

rank == 0 && println("MPI parallel correctness checks passed on $nprocs ranks.")
