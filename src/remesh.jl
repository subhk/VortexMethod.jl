module Remesh

export detect_max_edge_length, detect_min_edge_length,
       element_splitting, element_merging

function detect_max_edge_length(triXC, triYC, triZC, ds_max::Float64)
    return -1, 0.0
end

function detect_min_edge_length(triXC, triYC, triZC, ds_min::Float64)
    return -1, 0.0
end

function element_splitting(triXC, triYC, triZC, ele_idx, max_edge,
                           nodeX, nodeY, nodeZ, areas, eleGma, ds_max,
                           bnd_nodes...)
    return triXC, triYC, triZC, nothing, nodeX, nodeY, nodeZ, eleGma, bnd_nodes...
end

function element_merging(triXC, triYC, triZC, ele_idx, min_edge,
                         nodeX, nodeY, nodeZ, areas, eleGma, ds_min,
                         bnd_nodes...)
    return triXC, triYC, triZC, nothing, nodeX, nodeY, nodeZ, eleGma, bnd_nodes...
end

end # module

