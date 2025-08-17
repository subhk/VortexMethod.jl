module Checkpoint

export save_checkpoint!

using DelimitedFiles
using Dates

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

end # module

