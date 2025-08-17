module Remesh

export detect_max_edge_length, detect_min_edge_length,
       element_splitting!, edge_flip_small_edge!, remesh_pass!

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

# Build edge map: (i,j) with i<j -> list of incident triangle indices
function edge_map(tri::Array{Int,2})
    m = Dict{Tuple{Int,Int}, Vector{Int}}()
    @inbounds for t in 1:size(tri,1)
        v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
        for (a,b) in ((v1,v2),(v2,v3),(v3,v1))
            e = a<b ? (a,b) : (b,a)
            if haskey(m,e)
                push!(m[e], t)
            else
                m[e] = [t]
            end
        end
    end
    return m
end

@inline function edge_length(nodeX,nodeY,nodeZ, a::Int,b::Int)
    dx = nodeX[a]-nodeX[b]; dy = nodeY[a]-nodeY[b]; dz = nodeZ[a]-nodeZ[b]
    return sqrt(dx*dx+dy*dy+dz*dz)
end

# Global remeshing pass: propagate 1->4 splits on long edges, then attempt a few edge flips for short edges
function remesh_pass!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                      tri::Array{Int,2}, ds_max::Float64, ds_min::Float64; max_splits::Int=1000, max_flips::Int=1000)
    changed = false
    # 1) Long-edge refinement: mark all edges of any tri with any edge > ds_max
    emap = edge_map(tri)
    long_edges = Set{Tuple{Int,Int}}()
    tris_to_split = Set{Int}()
    @inbounds for t in 1:size(tri,1)
        v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
        if edge_length(nodeX,nodeY,nodeZ,v1,v2) > ds_max ||
           edge_length(nodeX,nodeY,nodeZ,v2,v3) > ds_max ||
           edge_length(nodeX,nodeY,nodeZ,v3,v1) > ds_max
            push!(tris_to_split, t)
            for (a,b) in ((v1,v2),(v2,v3),(v3,v1))
                e = a<b ? (a,b) : (b,a)
                push!(long_edges, e)
                # also mark neighbor triangle sharing this edge to split (to avoid hanging nodes)
                if haskey(emap,e)
                    for tnb in emap[e]
                        push!(tris_to_split, tnb)
                    end
                end
            end
        end
    end
    if !isempty(tris_to_split)
        changed = true
        # create midpoints for all long_edges
        midpoint = Dict{Tuple{Int,Int}, Int}()
        for e in long_edges
            a,b = e
            mx = 0.5*(nodeX[a]+nodeX[b])
            my = 0.5*(nodeY[a]+nodeY[b])
            mz = 0.5*(nodeZ[a]+nodeZ[b])
            push!(nodeX, mx); push!(nodeY, my); push!(nodeZ, mz)
            midpoint[e] = length(nodeX)
        end
        # rebuild triangle list with splits
        newtris = Vector{NTuple{3,Int}}()
        @inbounds for t in 1:size(tri,1)
            v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
            if t in tris_to_split
                e12 = (min(v1,v2), max(v1,v2)); m12 = get(midpoint, e12, 0)
                e23 = (min(v2,v3), max(v2,v3)); m23 = get(midpoint, e23, 0)
                e31 = (min(v3,v1), max(v3,v1)); m31 = get(midpoint, e31, 0)
                # ensure midpoints for all three edges (if edge not marked long, still create consistent midpoint)
                if m12==0
                    push!(nodeX, 0.5*(nodeX[v1]+nodeX[v2])); push!(nodeY, 0.5*(nodeY[v1]+nodeY[v2])); push!(nodeZ, 0.5*(nodeZ[v1]+nodeZ[v2]))
                    m12 = length(nodeX)
                end
                if m23==0
                    push!(nodeX, 0.5*(nodeX[v2]+nodeX[v3])); push!(nodeY, 0.5*(nodeY[v2]+nodeY[v3])); push!(nodeZ, 0.5*(nodeZ[v2]+nodeZ[v3]))
                    m23 = length(nodeX)
                end
                if m31==0
                    push!(nodeX, 0.5*(nodeX[v3]+nodeX[v1])); push!(nodeY, 0.5*(nodeY[v3]+nodeY[v1])); push!(nodeZ, 0.5*(nodeZ[v3]+nodeZ[v1]))
                    m31 = length(nodeX)
                end
                push!(newtris, (v1, m12, m31))
                push!(newtris, (m12, v2, m23))
                push!(newtris, (m31, m23, v3))
                push!(newtris, (m12, m23, m31))
            else
                push!(newtris, (v1,v2,v3))
            end
        end
        tri = reshape(collect(Iterators.flatten(newtris)), (3, length(newtris)))' # n×3
    end
    # 2) Edge flips for short edges (conservative stabilization)
    flips = 0
    while flips < max_flips
        emap = edge_map(tri)
        didflip = false
        for (e, tlst) in emap
            if length(tlst) == 2
                a,b = e
                if edge_length(nodeX,nodeY,nodeZ,a,b) < ds_min
                    tri = edge_flip_small_edge!(tri, tlst[1])
                    flips += 1
                    didflip = true
                    break
                end
            end
        end
        didflip || break
    end
    return tri, changed
end

end # module
