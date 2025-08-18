module Checkpoint

export save_checkpoint!, load_latest_checkpoint,
       save_checkpoint_jld2!, load_latest_jld2, load_checkpoint_jld2,
       save_checkpoint_jld2, load_latest_checkpoint_jld2, load_checkpoint,
       save_state!, mesh_stats, save_state_timeseries!,
       series_times, load_series_snapshot, load_series_nearest_time,
       find_series_files, get_series_info

using Dates
using Printf
using JLD2
using ..DomainImpl

function save_checkpoint!(dir::AbstractString, step::Integer,
                          nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                          tri::Array{Int,2}, eleGma::AbstractMatrix)

    # Default to JLD2 format, using step as time approximation
    return save_checkpoint_jld2!(dir, Float64(step), nodeX, nodeY, nodeZ, tri, eleGma; step=step)
end

# All checkpoints use JLD2 format exclusively for better performance and metadata support

function load_latest_checkpoint(dir::AbstractString)
    isdir(dir) || error("Checkpoint directory not found: \"$dir\"")
    files = filter(f->occursin("chkpt_", f) && endswith(f, ".jld2"), readdir(dir))
    isempty(files) && error("No JLD2 checkpoints in \"$dir\"")

    # Sort files by step number (extract number from chkpt_N.jld2)
    file_numbers = Tuple{Int, String}[]
    for f in files
        m = match(r"chkpt_(\d+)\.jld2", f)
        if m !== nothing
            step_num = parse(Int, m.captures[1])
            push!(file_numbers, (step_num, f))
        end
    end
    
    # Load the file with the highest step number
    if isempty(file_numbers)
        error("No valid checkpoint files found in \"$dir\"")
    end
    
    sort!(file_numbers, by=x->x[1])  # Sort by step number
    latest_file = file_numbers[end][2]  # Get filename with highest step
    
    ck = load_checkpoint_jld2(joinpath(dir, latest_file))
    return ck.nodeX, ck.nodeY, ck.nodeZ, ck.tri, ck.eleGma
end

function save_checkpoint_jld2!(dir::AbstractString, time::Real,
                               nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                               tri::Array{Int,2}, eleGma::AbstractMatrix;
                               domain=nothing, grid=nothing, params=nothing, step=nothing, attrs...)
    mkpath(dir)
    
    # Use step-based naming if step is provided, otherwise use time-based
    if step !== nothing
        base = joinpath(dir, @sprintf("chkpt_%d.jld2", Int(step)))
    else
        # If no step provided, try to auto-increment based on existing files
        existing_files = filter(f -> startswith(f, "chkpt_") && endswith(f, ".jld2"), readdir(dir))
        if isempty(existing_files)
            next_num = 1
        else
            # Extract numbers from existing files and find the next one
            numbers = Int[]
            for f in existing_files
                m = match(r"chkpt_(\d+)\.jld2", f)
                if m !== nothing
                    push!(numbers, parse(Int, m.captures[1]))
                end
            end
            next_num = isempty(numbers) ? 1 : maximum(numbers) + 1
        end
        base = joinpath(dir, @sprintf("chkpt_%d.jld2", next_num))
    end
    data = Dict{String,Any}("nodeX"=>nodeX, "nodeY"=>nodeY, "nodeZ"=>nodeZ,
                            "tri"=>tri, "gamma"=>eleGma, "time"=>float(time))

    if domain !== nothing
        data["domain"] = Dict("Lx"=>domain.Lx, "Ly"=>domain.Ly, "Lz"=>domain.Lz)
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
    JLD2.jldopen(base, "w") do f
        for (k, v) in data
            write(f, k, v)
        end
    end
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
    
    nodeX = data["nodeX"]; 
    nodeY = data["nodeY"]; 
    nodeZ = data["nodeZ"];

    tri   = data["tri"];   
    eleGma = data["gamma"]
    time  = get(data, "time", 0.0)

    domain   = get(data, "domain", nothing)
    grid  = get(data, "grid", nothing)
    params= get(data, "params", nothing)
    stats = get(data, "stats", nothing)

    return (; nodeX, nodeY, nodeZ, tri, eleGma, time, domain, grid, params, stats)
end

# Convenience aliases for cleaner API (defined after implementations)
const save_checkpoint_jld2 = save_checkpoint_jld2!
const load_latest_checkpoint_jld2 = load_latest_jld2
const load_checkpoint = load_checkpoint_jld2

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
                    tri::Array{Int,2}, domain::DomainSpec)
    nv = length(nodeX)
    nt = size(tri,1)
    xmin = minimum(nodeX); xmax = maximum(nodeX)
    ymin = minimum(nodeY); ymax = maximum(nodeY)
    zmin = minimum(nodeZ); zmax = maximum(nodeZ)
    Lx, Ly, Lz2 = domain.Lx, domain.Ly, 2*domain.Lz
    _mi(d,L) = (L<=0 ? d : (d - L*round(d/L)))
    armax = 0.0
    @inbounds for t in 1:nt
        v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
        dx12 = _mi(nodeX[v1]-nodeX[v2], Lx); 
        dy12 = _mi(nodeY[v1]-nodeY[v2], Ly); 
        dz12 = _mi(nodeZ[v1]-nodeZ[v2], Lz2)

        dx23 = _mi(nodeX[v2]-nodeX[v3], Lx); 
        dy23 = _mi(nodeY[v2]-nodeY[v3], Ly); 
        dz23 = _mi(nodeZ[v2]-nodeZ[v3], Lz2)

        dx31 = _mi(nodeX[v3]-nodeX[v1], Lx); 
        dy31 = _mi(nodeY[v3]-nodeY[v1], Ly); 
        dz31 = _mi(nodeZ[v3]-nodeZ[v1], Lz2)

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
save_state!(dir, time, nodeX,nodeY,nodeZ, tri, eleGma; domain, grid, dt, CFL, adaptive,
            poisson_mode, remesh_every, save_interval, ar_max, step, params_extra)

Bundles common fields into a single JLD2 checkpoint call and adds mesh stats.
"""
function save_state!(dir::AbstractString, time::Real,
                     nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                     tri::Array{Int,2}, eleGma::AbstractMatrix;
                     domain=nothing, grid=nothing, dt=nothing, CFL=nothing, adaptive=nothing,
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

    stats = domain === nothing ? mesh_stats(nodeX, nodeY, nodeZ, tri) : mesh_stats(nodeX, nodeY, nodeZ, tri, domain)

    base = save_checkpoint_jld2!(dir, time, nodeX, nodeY, nodeZ, tri, eleGma;
                            domain=domain, grid=grid, params=params, stats=stats, step=step)

    return base
end

"""
    save_state_timeseries!(file, time, nodeX, nodeY, nodeZ, tri, eleGma; domain, grid, step, params...)

Efficiently store simulation snapshots in JLD2 time-series files with random access and configurable limits.

This function appends simulation state to a JLD2 file, creating a searchable time series where individual
snapshots can be accessed by index or time. When the snapshot limit is reached, it automatically creates
a new file with incremented numbering (e.g., `series_001.jld2`, `series_002.jld2`).

# Arguments
- `file::String`: Base path to JLD2 time-series file (created if it doesn't exist)
- `time::Real`: Simulation time for this snapshot
- `nodeX, nodeY, nodeZ::Vector{Float64}`: Particle positions
- `tri::Matrix{Int}`: Triangle connectivity matrix
- `eleGma::Matrix{Float64}`: Element circulation vectors [Γx, Γy, Γz]

# Keyword Arguments
- `domain::DomainSpec`: Domain specification (Lx, Ly, Lz)
- `grid::GridSpec`: Grid specification (nx, ny, nz)
- `step::Int`: Simulation step number (for clean organization)
- `max_snapshots::Int=1000`: Maximum snapshots per file (automatic rollover when exceeded)
- `dt, CFL, adaptive, poisson_mode, remesh_every, save_interval, ar_max`: Simulation parameters
- `params_extra::NamedTuple`: Additional custom parameters to store

# Returns
- `file::String`: Path to the updated JLD2 file (may differ from input if rollover occurred)

# Features
- **Configurable File Size**: Set maximum snapshots per file to control file sizes
- **Automatic Rollover**: Creates new files when snapshot limit is reached
- **Single File Storage**: All snapshots in one JLD2 file with efficient compression
- **Random Access**: Load any snapshot by index or nearest time
- **Complete Metadata**: Domain, grid, parameters, and mesh statistics automatically stored
- **Incremental Updates**: Efficient appending without rewriting existing data
- **Cross-Platform**: Consistent binary format across different systems

# File Structure
```
series_001.jld2         # First 1000 snapshots
├── count=1000
├── times               # Array of snapshot times  
├── steps               # Array of step numbers
└── snapshots/
    ├── 000001/ to 001000/

series_002.jld2         # Next 1000 snapshots
├── count=1000
└── snapshots/
    ├── 000001/ to 001000/
```

# Example
```julia
# Save snapshots with custom limits
for step in 1:5000
    # ... time stepping ...
    if step % 10 == 0
        file = save_state_timeseries!("simulation.jld2", step*dt, nodeX, nodeY, nodeZ, tri, eleGma;
                                      domain=domain, grid=grid, step=step, max_snapshots=500)
        # Automatically creates simulation_001.jld2, simulation_002.jld2, etc.
    end
end

# Load from specific file
times, steps, count = series_times("simulation_002.jld2")  
idx, snapshot = load_series_nearest_time("simulation_002.jld2", 5.0)
```
"""
function save_state_timeseries!(file::AbstractString, time::Real,
                                nodeX::AbstractVector, nodeY::AbstractVector, nodeZ::AbstractVector,
                                tri::Array{Int,2}, eleGma::AbstractMatrix;
                                domain=nothing, grid=nothing, dt=nothing, CFL=nothing, adaptive=nothing,
                                poisson_mode=nothing, remesh_every=nothing, save_interval=nothing, ar_max=nothing,
                                step=nothing, max_snapshots::Int=1000, params_extra=NamedTuple())
    mkpath(dirname(file))
    
    # Determine the correct file to use with rollover logic
    actual_file = determine_series_file(file, max_snapshots)
    
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

    stats = domain === nothing ? mesh_stats(nodeX, nodeY, nodeZ, tri) : mesh_stats(nodeX, nodeY, nodeZ, tri, domain)

    JLD2.jldopen(actual_file, "a+") do f
        # determine next snapshot id
        count = haskey(f, "count") ? read(f, "count")::Int : 0
        count += 1
        if haskey(f, "count"); delete!(f, "count"); end
        write(f, "count", count)
        # update index arrays
        times = haskey(f, "times") ? read(f, "times")::Vector{Float64} : Float64[]
        steps = haskey(f, "steps") ? read(f, "steps")::Vector{Int}     : Int[]
        
        push!(times, float(time)); 
        push!(steps, Int(step === nothing ? count : step))

        if haskey(f, "times"); delete!(f, "times"); end
        if haskey(f, "steps"); delete!(f, "steps"); end
        write(f, "times", times); 
        write(f, "steps", steps)

        key = @sprintf("snapshots/%06d/", count)
        write(f, key*"time", float(time))
        write(f, key*"nodeX", nodeX); 
        write(f, key*"nodeY", nodeY); 
        write(f, key*"nodeZ", nodeZ)
        write(f, key*"tri", tri);      
        write(f, key*"gamma", eleGma)

        if domain !== nothing
            write(f, key*"domain", Dict("Lx"=>domain.Lx, "Ly"=>domain.Ly, "Lz"=>domain.Lz))
        end
        if grid !== nothing
            write(f, key*"grid", Dict("nx"=>grid.nx, "ny"=>grid.ny, "nz"=>grid.nz))
        end
        write(f, key*"params", params)
        write(f, key*"stats", stats)
    end
    return actual_file
end

"""
    determine_series_file(base_file, max_snapshots) -> actual_file

Determines which file to use for storing the next snapshot, creating new files when the limit is reached.

This function implements automatic file rollover:
- If no numbered files exist, uses the base filename
- If the latest file has reached max_snapshots, creates a new numbered file
- Numbered files follow the pattern: `base_001.jld2`, `base_002.jld2`, etc.
"""
function determine_series_file(base_file::AbstractString, max_snapshots::Int)
    dir = dirname(base_file)
    base_name = splitext(basename(base_file))[1]
    ext = splitext(base_file)[2]
    
    # Find existing series files
    if isdir(dir)
        pattern = Regex("^$(escape_string(base_name))_(\\d+)\\.jld2\$")
        existing_files = filter(readdir(dir)) do f
            occursin(pattern, f)
        end
        
        if !isempty(existing_files)
            # Extract numbers and find the latest file
            file_numbers = Int[]
            for f in existing_files
                m = match(pattern, f)
                if m !== nothing
                    push!(file_numbers, parse(Int, m.captures[1]))
                end
            end
            
            if !isempty(file_numbers)
                latest_num = maximum(file_numbers)
                latest_file = joinpath(dir, "$(base_name)_$(lpad(latest_num, 3, '0')).jld2")
                
                # Check if latest file has reached its limit
                if isfile(latest_file)
                    try
                        count = JLD2.jldopen(latest_file, "r") do f
                            haskey(f, "count") ? read(f, "count")::Int : 0
                        end
                        
                        if count >= max_snapshots
                            # Create new file with next number
                            next_num = latest_num + 1
                            return joinpath(dir, "$(base_name)_$(lpad(next_num, 3, '0')).jld2")
                        else
                            # Use existing latest file
                            return latest_file
                        end
                    catch
                        # If there's an error reading the file, create a new one
                        next_num = latest_num + 1
                        return joinpath(dir, "$(base_name)_$(lpad(next_num, 3, '0')).jld2")
                    end
                else
                    return latest_file
                end
            end
        end
    end
    
    # Check if the base file itself exists and has reached limit
    if isfile(base_file)
        try
            count = JLD2.jldopen(base_file, "r") do f
                haskey(f, "count") ? read(f, "count")::Int : 0
            end
            
            if count >= max_snapshots
                # Create first numbered file
                return joinpath(dir, "$(base_name)_001.jld2")
            end
        catch
            # If there's an error reading the file, create a numbered file
            return joinpath(dir, "$(base_name)_001.jld2")
        end
    end
    
    # Use the original base file
    return base_file
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
Returns (; nodeX, nodeY, nodeZ, tri, eleGma, time, domain, grid, params, stats)
"""
function load_series_snapshot(file::AbstractString, idx::Integer)
    return JLD2.jldopen(file, "r") do f
        key = @sprintf("snapshots/%06d/", idx)
        nodeX = read(f, key*"nodeX"); 
        nodeY = read(f, key*"nodeY"); 
        nodeZ = read(f, key*"nodeZ")

        tri   = read(f, key*"tri");    
        eleGma = read(f, key*"gamma")
        time  = read(f, key*"time")

        domain   = haskey(f, key*"domain") ? read(f, key*"domain") : nothing
        grid  = haskey(f, key*"grid")   ? read(f, key*"grid")   : nothing
        params= haskey(f, key*"params") ? read(f, key*"params") : nothing
        stats = haskey(f, key*"stats")  ? read(f, key*"stats")  : nothing
        
        (; nodeX, nodeY, nodeZ, tri, eleGma, time, domain, grid, params, stats)
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

"""
    find_series_files(base_file) -> Vector{String}

Find all series files that match the base filename pattern.

Returns a vector of filenames sorted by their sequence number (e.g., ["sim.jld2", "sim_001.jld2", "sim_002.jld2"]).
"""
function find_series_files(base_file::AbstractString)
    dir = dirname(base_file)
    base_name = splitext(basename(base_file))[1]
    
    if !isdir(dir)
        return String[]
    end
    
    files = String[]
    
    # Check for the base file
    if isfile(base_file)
        push!(files, base_file)
    end
    
    # Find numbered files
    pattern = Regex("^$(escape_string(base_name))_(\\d+)\\.jld2\$")
    numbered_files = Tuple{Int, String}[]
    
    for f in readdir(dir)
        m = match(pattern, f)
        if m !== nothing
            num = parse(Int, m.captures[1])
            push!(numbered_files, (num, joinpath(dir, f)))
        end
    end
    
    # Sort by number and add to files list
    sort!(numbered_files, by=x->x[1])
    append!(files, [f[2] for f in numbered_files])
    
    return files
end

"""
    get_series_info(base_file) -> NamedTuple

Get comprehensive information about all files in a series.

Returns:
- `files`: Vector of all series file paths
- `total_snapshots`: Total number of snapshots across all files
- `file_counts`: Number of snapshots in each file
- `time_ranges`: Time range (min, max) for each file
"""
function get_series_info(base_file::AbstractString)
    files = find_series_files(base_file)
    total_snapshots = 0
    file_counts = Int[]
    time_ranges = Tuple{Float64, Float64}[]
    
    for file in files
        try
            times, steps, count = series_times(file)
            total_snapshots += count
            push!(file_counts, count)
            
            if !isempty(times)
                push!(time_ranges, (minimum(times), maximum(times)))
            else
                push!(time_ranges, (NaN, NaN))
            end
        catch e
            push!(file_counts, 0)
            push!(time_ranges, (NaN, NaN))
        end
    end
    
    return (files=files, total_snapshots=total_snapshots, 
            file_counts=file_counts, time_ranges=time_ranges)
end

end # module

using .Checkpoint: save_checkpoint!, load_latest_checkpoint,
                   save_checkpoint_jld2!, load_latest_jld2, load_checkpoint_jld2,
                   save_checkpoint_jld2, load_latest_checkpoint_jld2, load_checkpoint,
                   save_state!, mesh_stats, save_state_timeseries!,
                   series_times, load_series_snapshot, load_series_nearest_time
