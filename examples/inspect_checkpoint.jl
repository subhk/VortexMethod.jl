using VortexMethod
using Printf

ckpt_dir = get(ENV, "CKPT_DIR", "checkpoints")
ck = load_latest_jld2(ckpt_dir)

println("Loaded checkpoint from ", ckpt_dir)
println("time: ", ck.time)
println("nodes: ", length(ck.nodeX), "  triangles: ", size(ck.tri,1))

if ck.domain !== nothing
    d = ck.domain
    @printf("domain: Lx=%.3f Ly=%.3f Lz=%.3f\n", d["Lx"], d["Ly"], d["Lz"])
end
if ck.grid !== nothing
    g = ck.grid
    println("grid: nx=$(g[\"nx\"]) ny=$(g[\"ny\"]) nz=$(g[\"nz\"]) ")
end
if ck.params !== nothing
    println("params:")
    for (k,v) in pairs(ck.params)
        println("  ", k, ": ", v)
    end
end
if ck.stats !== nothing
    s = ck.stats
    println("stats:")
    for (k,v) in pairs(s)
        println("  ", k, ": ", v)
    end
end
