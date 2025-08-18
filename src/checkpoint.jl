module Checkpoint

export save_checkpoint!, load_latest_checkpoint,
       save_checkpoint_jld2!, load_latest_jld2, load_checkpoint_jld2,
       save_state!, mesh_stats, save_state_timeseries!,
       series_times, load_series_snapshot, load_series_nearest_time

using DelimitedFiles
using Dates
# MAT dependency removed - JLD2 only
using JLD2
using ..DomainImpl

function save_checkpoint!(dir::AbstractString, step::Integer,
                          nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                          tri::Array{Int,2}, eleGma::AbstractMatrix)
    # Default to JLD2 format, using step as time approximation
    return save_checkpoint_jld2!(dir, Float64(step), nodeX, nodeY, nodeZ, tri, eleGma; step=step)
end

# MAT format support removed - use save_checkpoint_jld2! instead

function load_latest_checkpoint(dir::AbstractString)
    isdir(dir) || error("Checkpoint directory not found: \"$dir\"")
    files = filter(f->occursin("chkpt_", f), readdir(dir))
    isempty(files) && error("No checkpoints in \"$dir\"")
    # prefer JLD2 files if present, fallback to CSV
    jld2s = sort(filter(f->endswith(f, ".jld2"), files))
    csvs = sort(filter(f->endswith(f, ".csv"), files))
    if !isempty(jld2s)
        ck = load_checkpoint_jld2(joinpath(dir, jld2s[end]))
        return ck.nodeX, ck.nodeY, ck.nodeZ, ck.tri, ck.eleGma
    else
        # reconstruct base from latest _tri.csv
        tri_files = sort(filter(f->endswith(f, "_tri.csv"), files))
        fname = replace(tri_files[end], "_tri.csv"=>"")
        base = joinpath(dir, fname)
        nodeX = vec(readdlm(base*"_nodes_x.csv", ','))
        nodeY = vec(readdlm(base*"_nodes_y.csv", ','))
        nodeZ = vec(readdlm(base*"_nodes_z.csv", ','))
        tri   = Array{Int}(readdlm(base*"_tri.csv", ','))
        eleGma= Array{Float64}(readdlm(base*"_gamma.csv", ','))
        return nodeX, nodeY, nodeZ, tri, eleGma
end

function save_checkpoint_jld2!(dir::AbstractString, time::Real,
                               nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                               tri::Array{Int,2}, eleGma::AbstractMatrix;
                               dom=nothing, grid=nothing, params=nothing, attrs...)
    mkpath(dir)
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    base = joinpath(dir, @sprintf("chkpt_t%010.6f_%s.jld2", float(time), ts))
    data = Dict{String,Any}("nodeX"=>nodeX, "nodeY"=>nodeY, "nodeZ"=>nodeZ,
                            "tri"=>tri, "gamma"=>eleGma, "time"=>float(time))
    if dom !== nothing
        data["domain"] = Dict("Lx"=>dom.Lx, "Ly"=>dom.Ly, "Lz"=>dom.Lz)
    end
    if grid !== nothing
        data["grid"] = Dict("nx"=>grid.nx, "ny"=>grid.ny, "nz"=>grid.nz)
    end
    if params !== nothing
        data["params"] = params
    end
    # extra attributes
    for (k,v) in pairs(Dict(attrs))
        data[String(k)] = v
    end
    jldsave(base; data...)
    return base
end

function load_latest_jld2(dir::AbstractString)
    isdir(dir) || error("Checkpoint directory not found: \"$dir\"")
    files = sort(filter(f->endswith(f, ".jld2"), readdir(dir)))
    isempty(files) && error("No JLD2 checkpoints in \"$dir\"")
    return load_checkpoint_jld2(joinpath(dir, files[end]))
end

function load_checkpoint_jld2(path::AbstractString)
    data = JLD2.jldopen(path, "r") do f
        Dict{String,Any}(name=>read(f,name) for name in keys(f))
    end
    nodeX = data["nodeX"]; nodeY = data["nodeY"]; nodeZ = data["nodeZ"]
    tri   = data["tri"];   eleGma = data["gamma"]
    time  = get(data, "time", 0.0)
    dom   = get(data, "domain", nothing)
    grid  = get(data, "grid", nothing)
    params= get(data, "params", nothing)
    stats = get(data, "stats", nothing)
    return (; nodeX, nodeY, nodeZ, tri, eleGma, time, dom, grid, params, stats)
end

# -------- Convenience helpers --------

function mesh_stats(nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector, tri::Array{Int,2})
    nv = length(nodeX)
    nt = size(tri,1)
    xmin = minimum(nodeX); xmax = maximum(nodeX)
    ymin = minimum(nodeY); ymax = maximum(nodeY)
    zmin = minimum(nodeZ); zmax = maximum(nodeZ)
    # simple ARmax (Euclidean edges)
    armax = 0.0
    @inbounds for t in 1:nt
        v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
        l12 = sqrt((nodeX[v1]-nodeX[v2])^2 + (nodeY[v1]-nodeY[v2])^2 + (nodeZ[v1]-nodeZ[v2])^2)
        l23 = sqrt((nodeX[v2]-nodeX[v3])^2 + (nodeY[v2]-nodeY[v3])^2 + (nodeZ[v2]-nodeZ[v3])^2)
        l31 = sqrt((nodeX[v3]-nodeX[v1])^2 + (nodeY[v3]-nodeY[v1])^2 + (nodeZ[v3]-nodeZ[v1])^2)
        lmin = min(l12, min(l23, l31)); lmax = max(l12, max(l23, l31))
        if lmin > 0
            armax = max(armax, lmax/lmin)
        end
    end
    return (; n_nodes=nv, n_tris=nt, xmin, xmax, ymin, ymax, zmin, zmax, ARmax=armax)
end

"""
Periodic mesh stats using minimum-image edge lengths.
"""
function mesh_stats(nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                    tri::Array{Int,2}, dom::DomainSpec)
    nv = length(nodeX)
    nt = size(tri,1)
    xmin = minimum(nodeX); xmax = maximum(nodeX)
    ymin = minimum(nodeY); ymax = maximum(nodeY)
    zmin = minimum(nodeZ); zmax = maximum(nodeZ)
    Lx, Ly, Lz2 = dom.Lx, dom.Ly, 2*dom.Lz
    _mi(d,L) = (L<=0 ? d : (d - L*round(d/L)))
    armax = 0.0
    @inbounds for t in 1:nt
        v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
        dx12 = _mi(nodeX[v1]-nodeX[v2], Lx); dy12 = _mi(nodeY[v1]-nodeY[v2], Ly); dz12 = _mi(nodeZ[v1]-nodeZ[v2], Lz2)
        dx23 = _mi(nodeX[v2]-nodeX[v3], Lx); dy23 = _mi(nodeY[v2]-nodeY[v3], Ly); dz23 = _mi(nodeZ[v2]-nodeZ[v3], Lz2)
        dx31 = _mi(nodeX[v3]-nodeX[v1], Lx); dy31 = _mi(nodeY[v3]-nodeY[v1], Ly); dz31 = _mi(nodeZ[v3]-nodeZ[v1], Lz2)
        l12 = sqrt(dx12^2 + dy12^2 + dz12^2)
        l23 = sqrt(dx23^2 + dy23^2 + dz23^2)
        l31 = sqrt(dx31^2 + dy31^2 + dz31^2)
        lmin = min(l12, min(l23, l31)); lmax = max(l12, max(l23, l31))
        if lmin > 0
            armax = max(armax, lmax/lmin)
        end
    end
    return (; n_nodes=nv, n_tris=nt, xmin, xmax, ymin, ymax, zmin, zmax, ARmax=armax)
end

"""
save_state!(dir, time, nodeX,nodeY,nodeZ, tri, eleGma; dom, grid, dt, CFL, adaptive,
            poisson_mode, remesh_every, save_interval, ar_max, step, params_extra)

Bundles common fields into a single JLD2 checkpoint call and adds mesh stats.
"""
function save_state!(dir::AbstractString, time::Real,
                     nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                     tri::Array{Int,2}, eleGma::AbstractMatrix;
                     dom=nothing, grid=nothing, dt=nothing, CFL=nothing, adaptive=nothing,
                     poisson_mode=nothing, remesh_every=nothing, save_interval=nothing, ar_max=nothing,
                     step=nothing, params_extra=NamedTuple())
    # assemble params NamedTuple
    params = (;)
    if dt !== nothing;            params = merge(params, (; dt=dt)); end
    if CFL !== nothing;           params = merge(params, (; CFL=CFL)); end
    if adaptive !== nothing;      params = merge(params, (; adaptive=adaptive)); end
    if poisson_mode !== nothing;  params = merge(params, (; poisson_mode=poisson_mode)); end
    if remesh_every !== nothing;  params = merge(params, (; remesh_every=remesh_every)); end
    if save_interval !== nothing; params = merge(params, (; save_interval=save_interval)); end
    if ar_max !== nothing;        params = merge(params, (; ar_max=ar_max)); end
    params = merge(params, params_extra)

    stats = dom === nothing ? mesh_stats(nodeX, nodeY, nodeZ, tri) : mesh_stats(nodeX, nodeY, nodeZ, tri, dom)
    base = save_checkpoint_jld2!(dir, time, nodeX, nodeY, nodeZ, tri, eleGma;
                                 dom=dom, grid=grid, params=params, stats=stats, step=step)
    return base
end

"""
Append snapshot into a single time-series JLD2 file.

save_state_timeseries!(file, time, nodeX,nodeY,nodeZ, tri, eleGma; dom, grid,
                       dt, CFL, adaptive, poisson_mode, remesh_every, save_interval,
                       ar_max, step, params_extra)

Creates group paths under `snapshots/NNNNNN/` with fields nodeX,nodeY,nodeZ,tri,gamma,time,
and updates top-level arrays `times` and `steps`. Mesh stats stored at `snapshots/NNNNNN/stats`.
"""
function save_state_timeseries!(file::AbstractString, time::Real,
                                nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                                tri::Array{Int,2}, eleGma::AbstractMatrix;
                                dom=nothing, grid=nothing, dt=nothing, CFL=nothing, adaptive=nothing,
                                poisson_mode=nothing, remesh_every=nothing, save_interval=nothing, ar_max=nothing,
                                step=nothing, params_extra=NamedTuple())
    mkpath(dirname(file))
    # assemble params
    params = (;)
    if dt !== nothing;            params = merge(params, (; dt=dt)); end
    if CFL !== nothing;           params = merge(params, (; CFL=CFL)); end
    if adaptive !== nothing;      params = merge(params, (; adaptive=adaptive)); end
    if poisson_mode !== nothing;  params = merge(params, (; poisson_mode=poisson_mode)); end
    if remesh_every !== nothing;  params = merge(params, (; remesh_every=remesh_every)); end
    if save_interval !== nothing; params = merge(params, (; save_interval=save_interval)); end
    if ar_max !== nothing;        params = merge(params, (; ar_max=ar_max)); end
    params = merge(params, params_extra)

    stats = dom === nothing ? mesh_stats(nodeX, nodeY, nodeZ, tri) : mesh_stats(nodeX, nodeY, nodeZ, tri, dom)

    JLD2.jldopen(file, "a+") do f
        # determine next snapshot id
        count = haskey(f, "count") ? read(f, "count")::Int : 0
        count += 1
        write(f, "count", count)
        # update index arrays
        times = haskey(f, "times") ? read(f, "times")::Vector{Float64} : Float64[]
        steps = haskey(f, "steps") ? read(f, "steps")::Vector{Int}     : Int[]
        push!(times, float(time)); push!(steps, Int(step === nothing ? count : step))
        write(f, "times", times); write(f, "steps", steps)

        key = @sprintf("snapshots/%06d/", count)
        write(f, key*"time", float(time))
        write(f, key*"nodeX", nodeX); write(f, key*"nodeY", nodeY); write(f, key*"nodeZ", nodeZ)
        write(f, key*"tri", tri);      write(f, key*"gamma", eleGma)
        if dom !== nothing
            write(f, key*"domain", Dict("Lx"=>dom.Lx, "Ly"=>dom.Ly, "Lz"=>dom.Lz))
        end
        if grid !== nothing
            write(f, key*"grid", Dict("nx"=>grid.nx, "ny"=>grid.ny, "nz"=>grid.nz))
        end
        write(f, key*"params", params)
        write(f, key*"stats", stats)
    end
    return file
end

"""
series_times(file) -> times::Vector{Float64}, steps::Vector{Int}, count::Int

Reads top-level times/steps/count from a series JLD2 file.
"""
function series_times(file::AbstractString)
    return JLD2.jldopen(file, "r") do f
        times = haskey(f, "times") ? read(f, "times")::Vector{Float64} : Float64[]
        steps = haskey(f, "steps") ? read(f, "steps")::Vector{Int} : Int[]
        count = haskey(f, "count") ? read(f, "count")::Int : length(times)
        (times, steps, count)
    end
end

"""
load_series_snapshot(file, idx) -> NamedTuple

Loads snapshot at 1-based index `idx` from the series file.
Returns (; nodeX, nodeY, nodeZ, tri, eleGma, time, dom, grid, params, stats)
"""
function load_series_snapshot(file::AbstractString, idx::Integer)
    return JLD2.jldopen(file, "r") do f
        key = @sprintf("snapshots/%06d/", idx)
        nodeX = read(f, key*"nodeX"); nodeY = read(f, key*"nodeY"); nodeZ = read(f, key*"nodeZ")
        tri   = read(f, key*"tri");    eleGma = read(f, key*"gamma")
        time  = read(f, key*"time")
        dom   = haskey(f, key*"domain") ? read(f, key*"domain") : nothing
        grid  = haskey(f, key*"grid")   ? read(f, key*"grid")   : nothing
        params= haskey(f, key*"params") ? read(f, key*"params") : nothing
        stats = haskey(f, key*"stats")  ? read(f, key*"stats")  : nothing
        (; nodeX, nodeY, nodeZ, tri, eleGma, time, dom, grid, params, stats)
    end
end

"""
load_series_nearest_time(file, t) -> (idx, snapshot)

Finds snapshot index nearest to time `t` and returns (idx, NamedTuple snapshot).
"""
function load_series_nearest_time(file::AbstractString, t::Real)
    times, steps, count = series_times(file)
    isempty(times) && error("No snapshots in series: $file")
    # find index of closest time
    idx = argmin(abs.(times .- float(t)))
    return idx, load_series_snapshot(file, idx)
end

end # module
