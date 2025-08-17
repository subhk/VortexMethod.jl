module Checkpoint

export save_checkpoint!, save_checkpoint_mat!, load_latest_checkpoint,
       save_checkpoint_jld2!, load_latest_jld2, load_checkpoint_jld2

using DelimitedFiles
using Dates
using MAT
using JLD2

function save_checkpoint!(dir::AbstractString, step::Integer,
                          nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                          tri::Array{Int,2}, eleGma::AbstractMatrix)
    mkpath(dir)
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    base = joinpath(dir, @sprintf("chkpt_%06d_%s", step, ts))
    # Nodes
    writedlm(base * "_nodes_x.csv", nodeX, ',')
    writedlm(base * "_nodes_y.csv", nodeY, ',')
    writedlm(base * "_nodes_z.csv", nodeZ, ',')
    # Triangles (indices)
    writedlm(base * "_tri.csv", tri, ',')
    # Element vorticity
    writedlm(base * "_gamma.csv", eleGma, ',')
    return base
end

function save_checkpoint_mat!(dir::AbstractString, step::Integer,
                              nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                              tri::Array{Int,2}, eleGma::AbstractMatrix)
    mkpath(dir)
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    base = joinpath(dir, @sprintf("chkpt_%06d_%s.mat", step, ts))
    file = matopen(base, "w")
    try
        write(file, "nodeX", nodeX)
        write(file, "nodeY", nodeY)
        write(file, "nodeZ", nodeZ)
        write(file, "tri", tri)
        write(file, "gamma", eleGma)
        write(file, "step", step)
    finally
        close(file)
    end
    return base
end

function load_latest_checkpoint(dir::AbstractString)
    isdir(dir) || error("Checkpoint directory not found: \"$dir\"")
    files = filter(f->occursin("chkpt_", f), readdir(dir))
    isempty(files) && error("No checkpoints in \"$dir\"")
    # prefer MAT files if present
    mats = sort(filter(f->endswith(f, ".mat"), files))
    csvs = sort(filter(f->endswith(f, ".csv"), files))
    if !isempty(mats)
        fname = joinpath(dir, mats[end])
        file = matopen(fname)
        try
            nodeX = read(file, "nodeX"); nodeY = read(file, "nodeY"); nodeZ = read(file, "nodeZ")
            tri   = read(file, "tri");    eleGma = read(file, "gamma")
        finally
            close(file)
        end
        return nodeX, nodeY, nodeZ, tri, eleGma
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
    return (; nodeX, nodeY, nodeZ, tri, eleGma, time, dom, grid, params)
end

end # module
