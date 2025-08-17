module Remesh

       element_splitting, element_merging

function detect_max_edge_length(triXC, triYC, triZC, ds_max::Float64)
    nt = size(triXC,1)
    maxlen = 0.0
    maxidx = -1
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        e12 = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2)
        e23 = sqrt((p2[1]-p3[1])^2 + (p2[2]-p3[2])^2 + (p2[3]-p3[3])^2)
        e31 = sqrt((p3[1]-p1[1])^2 + (p3[2]-p1[2])^2 + (p3[3]-p1[3])^2)
        el = max(e12, e23, e31)
        if el > maxlen
            maxlen = el; maxidx = t
        end
    end
    if maxlen > ds_max
        return maxidx, maxlen
    else
        return -1, maxlen
    end
end

function detect_min_edge_length(triXC, triYC, triZC, ds_min::Float64)
    nt = size(triXC,1)
    minlen = Inf
    minidx = -1
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        e12 = sqrt((p1[1]-p2[1])^2 + (p1[2]-p2[2])^2 + (p1[3]-p2[3])^2)
        e23 = sqrt((p2[1]-p3[1])^2 + (p2[2]-p3[2])^2 + (p2[3]-p3[3])^2)
        e31 = sqrt((p3[1]-p1[1])^2 + (p3[2]-p1[2])^2 + (p3[3]-p1[3])^2)
        el = min(e12, min(e23, e31))
        if el < minlen
            minlen = el; minidx = t
        end
    end
    if minlen < ds_min
        return minidx, minlen
    else
        return -1, minlen
    end
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
