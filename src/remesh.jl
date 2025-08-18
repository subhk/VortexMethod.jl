module Remesh

export detect_max_edge_length, detect_min_edge_length,
       element_splitting!, edge_flip_small_edge!, remesh_pass!

using ..DomainImpl

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
                            tri::Array{Int,2}, ele_idx::Int; domain::DomainSpec=default_domain())

    v1, v2, v3 = tri[ele_idx,1], tri[ele_idx,2], tri[ele_idx,3]
    p1 = (nodeX[v1], nodeY[v1], nodeZ[v1])
    p2 = (nodeX[v2], nodeY[v2], nodeZ[v2])
    p3 = (nodeX[v3], nodeY[v3], nodeZ[v3])

    mx,my,mz = midpoint_periodic(p1[1],p1[2],p1[3], p2[1],p2[2],p2[3], domain)
    push!(nodeX, mx); 
    push!(nodeY, my); 
    push!(nodeZ, mz); 
    m12 = length(nodeX)

    mx,my,mz = midpoint_periodic(p2[1],p2[2],p2[3], p3[1],p3[2],p3[3], domain)
    push!(nodeX, mx); 
    push!(nodeY, my); 
    push!(nodeZ, mz); 
    m23 = length(nodeX)

    mx,my,mz = midpoint_periodic(p3[1],p3[2],p3[3], p1[1],p1[2],p1[3], domain)
    push!(nodeX, mx); 
    push!(nodeY, my); 
    push!(nodeZ, mz); 
    m31 = length(nodeX)

    tri[ele_idx, :] .= (v1, m12, m31)
    
    tri_new = Array{Int}(undef, 3, 3)
    tri_new[1,:] = (m12, v2, m23)
    tri_new[2,:] = (m31, m23, v3)
    tri_new[3,:] = (m12, m23, m31)
    tri = vcat(tri, tri_new)
    # enforce periodic wrap (safety)
    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    
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

@inline function edge_length(nodeX, nodeY, nodeZ, a::Int, b::Int)
    dx = nodeX[a] - nodeX[b]; 
    dy = nodeY[a] - nodeY[b]; 
    dz = nodeZ[a] - nodeZ[b]
    return sqrt(dx*dx+dy*dy+dz*dz)
end

@inline min_image(d::Float64, L::Float64) = begin
    if L <= 0; return d; end
    d1 = d
    if d1 >  L/2; d1 -= L; end
    if d1 < -L/2; d1 += L; end
    d1
end

function periodic_edge_length(nodeX, nodeY, nodeZ, a::Int,b::Int, domain::DomainSpec)
    Lx = domain.Lx; 
    Ly = domain.Ly; 
    Lz2 = 2*domain.Lz

    dx = min_image(nodeX[a]-nodeX[b], Lx)
    dy = min_image(nodeY[a]-nodeY[b], Ly)
    dz = min_image(nodeZ[a]-nodeZ[b], Lz2)

    return sqrt(dx*dx + dy*dy + dz*dz)
end

function midpoint_periodic(x1, y1, z1, x2, y2, z2, domain::DomainSpec)
    Lx = domain.Lx; 
    Ly = domain.Ly; 
    Lz2 = 2*domain.Lz
    # unwrap x2,y2,z2 near x1,y1,z1 using minimum image
    dx = min_image(x2 - x1, Lx)
    dy = min_image(y2 - y1, Ly)
    dz = min_image(z2 - z1, Lz2)

    xm = x1 + 0.5*dx
    ym = y1 + 0.5*dy
    zm = z1 + 0.5*dz
    
    # wrap back to domain ranges: x in [0,Lx), y in [0,Ly), z in [-Lz,+Lz]
    xm = (xm % Lx)
    ym = (ym % Ly)
    
    zshift = zm + domain.Lz
    zshift = (zshift % (2*domain.Lz))
    zm = zshift - domain.Lz
    
    return xm, ym, zm
end

# Global remeshing pass: propagate 1->4 splits on long edges, then attempt a few edge flips for short edges
function remesh_pass!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                      tri::Array{Int,2}, ds_max::Float64, ds_min::Float64;
                      max_splits::Int=1000, max_flips::Int=1000, max_merges::Int=1000,
                      domain::DomainSpec=default_domain(), compact::Bool=true,
                      ar_max::Float64=Inf)

    changed = false
    # 1) Long-edge refinement: mark all edges of any tri with any edge > ds_max
    emap = edge_map(tri)
    long_edges = Set{Tuple{Int,Int}}()
    tris_to_split = Set{Int}()

    @inbounds for t in 1:size(tri,1)
        v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
        # edge lengths
        l12 = periodic_edge_length(nodeX, nodeY, nodeZ, v1, v2, domain)
        l23 = periodic_edge_length(nodeX, nodeY, nodeZ, v2, v3, domain)
        l31 = periodic_edge_length(nodeX, nodeY, nodeZ, v3, v1, domain)

        # aspect ratio as longest/shortest
        ar = maximum((l12,l23,l31)) / max(eps(), minimum((l12,l23,l31)))
        if (l12 > ds_max || l23 > ds_max || l31 > ds_max) || (ar > ar_max)
            push!(tris_to_split, t)
            for (a, b) in ((v1, v2), (v2, v3), (v3, v1))
                e = a < b ? (a, b) : (b, a)
                push!(long_edges, e)
                # also mark neighbor triangle sharing this edge to split (to avoid hanging nodes)
                if haskey(emap, e)
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
            a, b = e
            mx, my, mz = midpoint_periodic(nodeX[a], nodeY[a], nodeZ[a], nodeX[b], nodeY[b], nodeZ[b], domain)
            push!(nodeX, mx);
            push!(nodeY, my);
            push!(nodeZ, mz);
            midpoint[e] = length(nodeX);
        end
        # rebuild triangle list with splits
        newtris = Vector{NTuple{3,Int}}()
        @inbounds for t in 1:size(tri,1)
            v1, v2, v3 = tri[t,1], tri[t,2], tri[t,3]
            if t in tris_to_split
                e12 = (min(v1, v2), max(v1, v2)); 
                m12 = get(midpoint, e12, 0)
                
                e23 = (min(v2, v3), max(v2, v3)); 
                m23 = get(midpoint, e23, 0)
                
                e31 = (min(v3, v1), max(v3, v1)); 
                m31 = get(midpoint, e31, 0)

                # ensure midpoints for all three edges (if edge not marked long, still create consistent midpoint)
                if m12==0
                    mx, my, mz = midpoint_periodic(nodeX[v1], nodeY[v1], nodeZ[v1], 
                                                nodeX[v2], nodeY[v2], nodeZ[v2], domain)
                    push!(nodeX, mx); 
                    push!(nodeY, my); 
                    push!(nodeZ, mz)
                    m12 = length(nodeX)
                end

                if m23==0
                    mx, my, mz = midpoint_periodic(nodeX[v2], nodeY[v2], nodeZ[v2], 
                                                nodeX[v3], nodeY[v3], nodeZ[v3], domain)
                    push!(nodeX, mx); 
                    push!(nodeY, my); 
                    push!(nodeZ, mz)
                    m23 = length(nodeX)
                end
                
                if m31==0
                    mx, my, mz = midpoint_periodic(nodeX[v3], nodeY[v3], nodeZ[v3], 
                                                nodeX[v1], nodeY[v1], nodeZ[v1], domain)
                    push!(nodeX, mx); 
                    push!(nodeY, my); 
                    push!(nodeZ, mz)
                    m31 = length(nodeX)
                end
                push!(newtris, (v1, m12, m31))
                push!(newtris, (m12, v2, m23))
                push!(newtris, (m31, m23, v3))
                push!(newtris, (m12, m23, m31))
            else
                push!(newtris, (v1, v2, v3))
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
                if periodic_edge_length(nodeX,nodeY,nodeZ,a,b, domain) < ds_min
                    tri = edge_flip_small_edge!(tri, tlst[1])
                    flips += 1
                    didflip = true
                    break
                end
            end
        end
        didflip || break
    end
    # 3) Edge collapses for persistent short edges
    merges = 0
    while merges < max_merges
        emap = edge_map(tri)
        collapsed = false
        for (e, tlst) in emap
            a,b = e
                if periodic_edge_length(nodeX,nodeY,nodeZ,a,b, domain) < ds_min
                # midpoint and new node
                mx,my,mz = midpoint_periodic(nodeX[a],nodeY[a],nodeZ[a], nodeX[b],nodeY[b],nodeZ[b], domain)
                push!(nodeX, mx); push!(nodeY, my); push!(nodeZ, mz)
                m = length(nodeX)
                # replace all occurrences of a or b with m
                @inbounds for t in 1:size(tri,1), k in 1:3
                    v = tri[t,k]
                    if v==a || v==b
                        tri[t,k] = m
                    end
                end
                # remove degenerate triangles (duplicate vertices)
                keep = trues(size(tri,1))
                @inbounds for t in 1:size(tri,1)
                    v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
                    if v1==v2 || v2==v3 || v3==v1
                        keep[t] = false
                    end
                end
                tri = tri[keep, :]
                merges += 1
                collapsed = true
                break
            end
        end
        collapsed || break
    end
    # 4) Optional compaction: remove unused node indices and remap tri connectivity
    if compact
        used = falses(length(nodeX))
        @inbounds for t in 1:size(tri,1), k in 1:3
            used[tri[t,k]] = true
        end
        # build map old->new indices
        old2new = Dict{Int,Int}()
        nodeXnew = Float64[]; nodeYnew = Float64[]; nodeZnew = Float64[]
        newidx = 0
        for i in 1:length(nodeX)
            if used[i]
                newidx += 1
                push!(nodeXnew, nodeX[i]); push!(nodeYnew, nodeY[i]); push!(nodeZnew, nodeZ[i])
                old2new[i] = newidx
            end
        end
        # remap tri
        @inbounds for t in 1:size(tri,1), k in 1:3
            tri[t,k] = old2new[tri[t,k]]
        end
        nodeX[:] = nodeXnew; nodeY[:] = nodeYnew; nodeZ[:] = nodeZnew
    end
    # Ensure all nodes are wrapped at the end
    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    return tri, changed
end

end # module
