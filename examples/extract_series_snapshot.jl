using VortexMethod
using Printf

function usage()
    println("Usage: julia --project examples/extract_series_snapshot.jl [INDEX|TIME] [OUT_DIR]")
    println("  INDEX: integer 1-based snapshot index in the series")
    println("  TIME:  real-valued target time; nearest snapshot is selected")
    println("  OUT_DIR: directory to write standalone JLD2 (default: checkpoints/extracted)")
end

series_file = get(ENV, "SERIES_FILE", "checkpoints/series.jld2")
args = copy(ARGS)
if length(args) == 0
    usage(); exit(1)
end

out_dir = length(args) >= 2 ? args[2] : "checkpoints/extracted"
mkpath(out_dir)

idx = nothing
snap = nothing
if occursin(r"^[0-9]+$", args[1])
    idx = parse(Int, args[1])
    snap = load_series_snapshot(series_file, idx)
else
    t = parse(Float64, args[1])
    idx, snap = load_series_nearest_time(series_file, t)
end

println(@sprintf("Extracting snapshot #%d at t=%.6f", idx, snap.time))

# Save standalone JLD2
domain = default_domain(); gr = default_grid()
if snap.domain !== nothing
    d = snap.domain; domain = VortexMethod.DomainSpec(d["Lx"], d["Ly"], d["Lz"])
end
if snap.grid !== nothing
    g = snap.grid; gr = VortexMethod.GridSpec(g["nx"], g["ny"], g["nz"])
end

base = save_state!(out_dir, snap.time, snap.nodeX, snap.nodeY, snap.nodeZ, snap.tri, snap.eleGma;
                   domain=domain, grid=gr, step=idx, params_extra=(; extracted_from=series_file))
println("Wrote: ", base)
