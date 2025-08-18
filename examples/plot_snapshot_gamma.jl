using VortexMethod
using Plots
using Printf

series_file = get(ENV, "SERIES_FILE", "checkpoints/advanced_series.jld2")
snap_idx = parse(Int, get(ENV, "SNAP_INDEX", "1"))
outpath = get(ENV, "OUTPUT_PNG", "")

snap = load_series_snapshot(series_file, snap_idx)

nt = size(snap.tri,1)
# Triangle centroids and |gamma|
cx = zeros(Float64, nt); cy = similar(cx); cz = similar(cx); gm = similar(cx)
@inbounds for t in 1:nt
    v1, v2, v3 = snap.tri[t,1], snap.tri[t,2], snap.tri[t,3]
    x1,y1,z1 = snap.nodeX[v1], snap.nodeY[v1], snap.nodeZ[v1]
    x2,y2,z2 = snap.nodeX[v2], snap.nodeY[v2], snap.nodeZ[v2]
    x3,y3,z3 = snap.nodeX[v3], snap.nodeY[v3], snap.nodeZ[v3]
    cx[t] = (x1+x2+x3)/3
    cy[t] = (y1+y2+y3)/3
    cz[t] = (z1+z2+z3)/3
    gm[t] = sqrt(snap.eleGma[t,1]^2 + snap.eleGma[t,2]^2 + snap.eleGma[t,3]^2)
end

plt = scatter(cx, cy; zcolor=gm, ms=3, c=:viridis, colorbar_title="|γ|",
              xlabel="x", ylabel="y", title=@sprintf("Gamma magnitude at t=%.3f (idx %d)", snap.time, snap_idx), legend=false)

if !isempty(outpath)
    mkpath(dirname(outpath))
    png(plt, outpath)
    @info "Snapshot plot saved" outpath
else
    display(plt)
end

