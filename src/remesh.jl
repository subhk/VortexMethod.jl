module Remesh

export detect_max_edge_length, detect_min_edge_length,
       element_splitting!, edge_flip_small_edge!

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

function element_splitting!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                            tri::Array{Int,2}, ele_idx::Int)
    v1, v2, v3 = tri[ele_idx,1], tri[ele_idx,2], tri[ele_idx,3]
    p1 = (nodeX[v1], nodeY[v1], nodeZ[v1])
    p2 = (nodeX[v2], nodeY[v2], nodeZ[v2])
    p3 = (nodeX[v3], nodeY[v3], nodeZ[v3])
    push!(nodeX, (p1[1]+p2[1])/2); push!(nodeY, (p1[2]+p2[2])/2); push!(nodeZ, (p1[3]+p2[3])/2); m12 = length(nodeX)
    push!(nodeX, (p2[1]+p3[1])/2); push!(nodeY, (p2[2]+p3[2])/2); push!(nodeZ, (p2[3]+p3[3])/2); m23 = length(nodeX)
    push!(nodeX, (p3[1]+p1[1])/2); push!(nodeY, (p3[2]+p1[2])/2); push!(nodeZ, (p3[3]+p1[3])/2); m31 = length(nodeX)
    tri[ele_idx, :] .= (v1, m12, m31)
    tri_new = Array{Int}(undef, 3, 3)
    tri_new[1,:] = (m12, v2, m23)
    tri_new[2,:] = (m31, m23, v3)
    tri_new[3,:] = (m12, m23, m31)
    tri = vcat(tri, tri_new)
    return nodeX, nodeY, nodeZ, tri
end

function edge_flip_small_edge!(tri::Array{Int,2}, ele_idx::Int)
    v = tri[ele_idx, :]
    edges = [(v[1],v[2]), (v[2],v[3]), (v[3],v[1])]
    function find_neighbor(a::Int,b::Int)
        for t in 1:size(tri,1)
            if t==ele_idx; continue; end
            w = tri[t,:]
            if (a in w) && (b in w)
                return t
            end
        end
        return -1
    end
    for (a,b) in edges
        tnb = find_neighbor(a,b)
        if tnb != -1
            c = setdiff(v, (a,b))[1]
            w = tri[tnb,:]
            d = setdiff(w, (a,b))[1]
            tri[ele_idx,:] .= (c,a,d)
            tri[tnb,   : ] .= (c,d,b)
            return tri
        end
    end
    return tri
end

end # module
