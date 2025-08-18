using VortexMethod
using Plots
using Printf

series_file = get(ENV, "SERIES_FILE", "checkpoints/series.jld2")
times, steps, count = series_times(series_file)
if count == 0
    error("No snapshots in series file: $series_file")
end

println("Found $count snapshots in $series_file")
KEs = Vector{Float64}(undef, count)
for i in 1:count
    snap = load_series_snapshot(series_file, i)
    # read KE if provided in params; else compute
    ke_val = try
        get(snap.params, :KE, nothing)
    catch
        nothing
    end
    if ke_val === nothing
        # reconstruct tri coords
        nt = size(snap.tri,1)
        triXC = Array{Float64}(undef, nt, 3); triYC = similar(triXC); triZC = similar(triXC)
        @inbounds for k in 1:3, t in 1:nt
            v = snap.tri[t,k]
            triXC[t,k] = snap.nodeX[v]; triYC[t,k] = snap.nodeY[v]; triZC[t,k] = snap.nodeZ[v]
        end
        # restore domain/grid if present, else defaults
        dom = default_domain(); gr = default_grid()
        if snap.dom !== nothing
            d = snap.dom; dom = VortexMethod.DomainSpec(d["Lx"], d["Ly"], d["Lz"])
        end
        if snap.grid !== nothing
            g = snap.grid; gr = VortexMethod.GridSpec(g["nx"], g["ny"], g["nz"])
        end
        ke_val = gamma_ke(snap.eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=:fd)
    end
    KEs[i] = ke_val
end

plt = plot(times, KEs; xlabel="time", ylabel="KE", lw=2, marker=:circle, title="Kinetic Energy vs Time")

# Optional: save plot if OUTPUT_PNG is set
outpath = get(ENV, "OUTPUT_PNG", "")
if !isempty(outpath)
    mkpath(dirname(outpath))
    png(plt, outpath)
    @info "KE plot saved" outpath
else
    display(plt)
end
