@testset "Checkpoint JLD2" begin
    domain = VortexMethod.default_domain()
    gr = VortexMethod.default_grid()
    Nx, Ny = 8, 8
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = VortexMethod.structured_mesh(Nx, Ny; domain=domain)
    eleGma = zeros(Float64, size(tri,1), 3)
    tmpfile = joinpath("checkpoints", "test_series.jld2")
    
    # Clean up any existing test file to ensure a fresh start
    isfile(tmpfile) && rm(tmpfile)
    isdir(dirname(tmpfile)) || mkpath(dirname(tmpfile))
    base1 = VortexMethod.save_state_timeseries!(tmpfile, 0.0, nodeX, nodeY, nodeZ, tri, eleGma;
        domain=domain, grid=gr, step=1)
    base2 = VortexMethod.save_state_timeseries!(tmpfile, 0.1, nodeX, nodeY, nodeZ, tri, eleGma;
        domain=domain, grid=gr, step=2)
    times, steps, count = VortexMethod.series_times(tmpfile)
    @test count == 2
    @test steps == [1,2]
    idx, snap = VortexMethod.load_series_nearest_time(tmpfile, 0.09)
    @test idx == 2
    @test size(snap.tri,2) == 3
    
    # Clean up test file
    isfile(tmpfile) && rm(tmpfile)
end
