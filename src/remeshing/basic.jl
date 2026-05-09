# Minimum image convention for periodic boundaries
@inline min_image(d::Float64, L::Float64) = begin
    if L <= 0; return d; end
    d1 = d
    if d1 >  L/2; d1 -= L; end
    if d1 < -L/2; d1 += L; end
    d1
end

function triangle_coordinate_matrices(nodeX, nodeY, nodeZ, tri)
    triXC = Matrix{Float64}(undef, size(tri, 1), 3)
    triYC = similar(triXC)
    triZC = similar(triXC)
    @inbounds for t in 1:size(tri, 1), k in 1:3
        v = tri[t, k]
        triXC[t, k] = nodeX[v]
        triYC[t, k] = nodeY[v]
        triZC[t, k] = nodeZ[v]
    end
    return triXC, triYC, triZC
end

function detect_max_edge_length(triXC, triYC, triZC, ds_max::Float64; domain::DomainSpec=default_domain())
    nt = size(triXC,1)
    maxlen = 0.0
    maxidx = -1
    Lx = domain.Lx
    Ly = domain.Ly
    Lz2 = 2*domain.Lz
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        # Use minimum image convention for periodic boundaries
        dx12 = min_image(p1[1]-p2[1], Lx); dy12 = min_image(p1[2]-p2[2], Ly); dz12 = min_image(p1[3]-p2[3], Lz2)
        dx23 = min_image(p2[1]-p3[1], Lx); dy23 = min_image(p2[2]-p3[2], Ly); dz23 = min_image(p2[3]-p3[3], Lz2)
        dx31 = min_image(p3[1]-p1[1], Lx); dy31 = min_image(p3[2]-p1[2], Ly); dz31 = min_image(p3[3]-p1[3], Lz2)

        e12 = sqrt(dx12*dx12 + dy12*dy12 + dz12*dz12)
        e23 = sqrt(dx23*dx23 + dy23*dy23 + dz23*dz23)
        e31 = sqrt(dx31*dx31 + dy31*dy31 + dz31*dz31)

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

function detect_min_edge_length(triXC, triYC, triZC, ds_min::Float64; domain::DomainSpec=default_domain())
    nt = size(triXC,1)
    minlen = Inf
    minidx = -1
    Lx = domain.Lx
    Ly = domain.Ly
    Lz2 = 2*domain.Lz
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])

        # Use minimum image convention for periodic boundaries
        dx12 = min_image(p1[1]-p2[1], Lx); dy12 = min_image(p1[2]-p2[2], Ly); dz12 = min_image(p1[3]-p2[3], Lz2)
        dx23 = min_image(p2[1]-p3[1], Lx); dy23 = min_image(p2[2]-p3[2], Ly); dz23 = min_image(p2[3]-p3[3], Lz2)
        dx31 = min_image(p3[1]-p1[1], Lx); dy31 = min_image(p3[2]-p1[2], Ly); dz31 = min_image(p3[3]-p1[3], Lz2)

        e12 = sqrt(dx12*dx12 + dy12*dy12 + dz12*dz12)
        e23 = sqrt(dx23*dx23 + dy23*dy23 + dz23*dz23)
        e31 = sqrt(dx31*dx31 + dy31*dy31 + dz31*dz31)

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
    xm = mod(xm, Lx)
    ym = mod(ym, Ly)
    
    zshift = zm + domain.Lz
    zshift = mod(zshift, 2*domain.Lz)
    zm = zshift - domain.Lz
    
    return xm, ym, zm
end

function tris_to_matrix(rows::Vector{NTuple{3,Int}})
    tri = Array{Int}(undef, length(rows), 3)
    @inbounds for (t, row) in pairs(rows)
        tri[t, 1] = row[1]
        tri[t, 2] = row[2]
        tri[t, 3] = row[3]
    end
    return tri
end

function gammas_to_matrix(rows::Vector{NTuple{3,Float64}})
    eleGma = Array{Float64}(undef, length(rows), 3)
    @inbounds for (t, row) in pairs(rows)
        eleGma[t, 1] = row[1]
        eleGma[t, 2] = row[2]
        eleGma[t, 3] = row[3]
    end
    return eleGma
end

@inline gamma_row(eleGma::AbstractMatrix, t::Int) =
    (Float64(eleGma[t, 1]), Float64(eleGma[t, 2]), Float64(eleGma[t, 3]))

function triangle_area(nodeX, nodeY, nodeZ, v1::Int, v2::Int, v3::Int, domain::DomainSpec)
    p1 = (nodeX[v1], nodeY[v1], nodeZ[v1])
    p2 = (nodeX[v2], nodeY[v2], nodeZ[v2])
    p3 = (nodeX[v3], nodeY[v3], nodeZ[v3])
    return periodic_triangle_area(p1, p2, p3, domain)
end

function triangle_total_vorticity(nodeX, nodeY, nodeZ, tri, eleGma, t::Int, domain::DomainSpec)
    area = triangle_area(nodeX, nodeY, nodeZ, tri[t, 1], tri[t, 2], tri[t, 3], domain)
    return (area * eleGma[t, 1], area * eleGma[t, 2], area * eleGma[t, 3])
end

function triangle_key(v1::Int, v2::Int, v3::Int)
    s = sort([v1, v2, v3])
    return (s[1], s[2], s[3])
end

function collapse_edge_with_circulation!(nodeX::Vector{Float64},
                                         nodeY::Vector{Float64},
                                         nodeZ::Vector{Float64},
                                         tri::Array{Int,2},
                                         eleGma::AbstractMatrix,
                                         a::Int, b::Int,
                                         domain::DomainSpec)
    old_tri = copy(tri)
    old_eleGma = Matrix{Float64}(eleGma)

    mx, my, mz = midpoint_periodic(nodeX[a], nodeY[a], nodeZ[a],
                                   nodeX[b], nodeY[b], nodeZ[b], domain)
    push!(nodeX, mx)
    push!(nodeY, my)
    push!(nodeZ, mz)
    m = length(nodeX)

    @inbounds for t in 1:size(tri, 1), k in 1:3
        v = tri[t, k]
        if v == a || v == b
            tri[t, k] = m
        end
    end

    keep = trues(size(tri, 1))
    seen = Dict{NTuple{3,Int}, Int}()
    @inbounds for t in 1:size(tri, 1)
        v1, v2, v3 = tri[t, 1], tri[t, 2], tri[t, 3]
        if v1 == v2 || v2 == v3 || v3 == v1
            keep[t] = false
            continue
        end

        key = triangle_key(v1, v2, v3)
        if haskey(seen, key)
            keep[t] = false
        else
            seen[key] = t
        end
    end

    old_total = (0.0, 0.0, 0.0)
    @inbounds for t in 1:size(old_tri, 1)
        tx, ty, tz = triangle_total_vorticity(nodeX, nodeY, nodeZ,
                                              old_tri, old_eleGma, t, domain)
        old_total = (old_total[1] + tx,
                     old_total[2] + ty,
                     old_total[3] + tz)
    end

    tri_new = tri[keep, :]
    eleGma_new = Matrix{Float64}(old_eleGma[keep, :])
    remaining_area = 0.0
    survivor_total = (0.0, 0.0, 0.0)
    @inbounds for t in 1:size(tri_new, 1)
        area = triangle_area(nodeX, nodeY, nodeZ,
                             tri_new[t, 1], tri_new[t, 2], tri_new[t, 3],
                             domain)
        remaining_area += area
        survivor_total = (survivor_total[1] + area * eleGma_new[t, 1],
                          survivor_total[2] + area * eleGma_new[t, 2],
                          survivor_total[3] + area * eleGma_new[t, 3])
    end

    if remaining_area > eps(Float64)
        correction = ((old_total[1] - survivor_total[1]) / remaining_area,
                      (old_total[2] - survivor_total[2]) / remaining_area,
                      (old_total[3] - survivor_total[3]) / remaining_area)
        @inbounds for t in 1:size(eleGma_new, 1)
            eleGma_new[t, 1] += correction[1]
            eleGma_new[t, 2] += correction[2]
            eleGma_new[t, 3] += correction[3]
        end
    end

    return tri_new, eleGma_new
end

# Circulation-aware remeshing pass: local 1->4 split carryover, node-circulation
# edge flips, and circulation-conserving edge collapse.
function remesh_pass!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                      tri::Array{Int,2}, eleGma::AbstractMatrix,
                      ds_max::Float64, ds_min::Float64;
                      max_splits::Int=1000, max_flips::Int=1000, max_merges::Int=1000,
                      domain::DomainSpec=default_domain(), compact::Bool=true,
                      ar_max::Float64=Inf)
    size(eleGma, 1) == size(tri, 1) ||
        throw(DimensionMismatch("eleGma row count $(size(eleGma, 1)) does not match triangle count $(size(tri, 1))"))

    changed = false
    eleGma_work = Matrix{Float64}(eleGma)

    emap = edge_map(tri)
    long_edges = Set{Tuple{Int,Int}}()
    tris_to_split = Set{Int}()

    @inbounds for t in 1:size(tri, 1)
        v1, v2, v3 = tri[t, 1], tri[t, 2], tri[t, 3]
        l12 = periodic_edge_length(nodeX, nodeY, nodeZ, v1, v2, domain)
        l23 = periodic_edge_length(nodeX, nodeY, nodeZ, v2, v3, domain)
        l31 = periodic_edge_length(nodeX, nodeY, nodeZ, v3, v1, domain)
        ar = maximum((l12, l23, l31)) / max(eps(), minimum((l12, l23, l31)))
        if (l12 > ds_max || l23 > ds_max || l31 > ds_max) || (ar > ar_max)
            push!(tris_to_split, t)
            for (a, b) in ((v1, v2), (v2, v3), (v3, v1))
                e = a < b ? (a, b) : (b, a)
                push!(long_edges, e)
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
        midpoint = Dict{Tuple{Int,Int}, Int}()
        for e in long_edges
            a, b = e
            mx, my, mz = midpoint_periodic(nodeX[a], nodeY[a], nodeZ[a],
                                           nodeX[b], nodeY[b], nodeZ[b], domain)
            push!(nodeX, mx)
            push!(nodeY, my)
            push!(nodeZ, mz)
            midpoint[e] = length(nodeX)
        end

        newtris = Vector{NTuple{3,Int}}()
        newgamma = Vector{NTuple{3,Float64}}()
        @inbounds for t in 1:size(tri, 1)
            v1, v2, v3 = tri[t, 1], tri[t, 2], tri[t, 3]
            γ = gamma_row(eleGma_work, t)
            if t in tris_to_split
                e12 = (min(v1, v2), max(v1, v2))
                e23 = (min(v2, v3), max(v2, v3))
                e31 = (min(v3, v1), max(v3, v1))
                m12 = get(midpoint, e12, 0)
                m23 = get(midpoint, e23, 0)
                m31 = get(midpoint, e31, 0)

                if m12 == 0
                    mx, my, mz = midpoint_periodic(nodeX[v1], nodeY[v1], nodeZ[v1],
                                                   nodeX[v2], nodeY[v2], nodeZ[v2],
                                                   domain)
                    push!(nodeX, mx)
                    push!(nodeY, my)
                    push!(nodeZ, mz)
                    m12 = length(nodeX)
                end
                if m23 == 0
                    mx, my, mz = midpoint_periodic(nodeX[v2], nodeY[v2], nodeZ[v2],
                                                   nodeX[v3], nodeY[v3], nodeZ[v3],
                                                   domain)
                    push!(nodeX, mx)
                    push!(nodeY, my)
                    push!(nodeZ, mz)
                    m23 = length(nodeX)
                end
                if m31 == 0
                    mx, my, mz = midpoint_periodic(nodeX[v3], nodeY[v3], nodeZ[v3],
                                                   nodeX[v1], nodeY[v1], nodeZ[v1],
                                                   domain)
                    push!(nodeX, mx)
                    push!(nodeY, my)
                    push!(nodeZ, mz)
                    m31 = length(nodeX)
                end

                append!(newtris, ((v1, m12, m31),
                                  (m12, v2, m23),
                                  (m31, m23, v3),
                                  (m12, m23, m31)))
                append!(newgamma, (γ, γ, γ, γ))
            else
                push!(newtris, (v1, v2, v3))
                push!(newgamma, γ)
            end
        end
        tri = tris_to_matrix(newtris)
        eleGma_work = gammas_to_matrix(newgamma)
    end

    flips = 0
    while flips < max_flips
        emap = edge_map(tri)
        didflip = false
        for (e, tlst) in emap
            if length(tlst) == 2
                a, b = e
                if periodic_edge_length(nodeX, nodeY, nodeZ, a, b, domain) < ds_min
                    triXC_old, triYC_old, triZC_old = triangle_coordinate_matrices(nodeX, nodeY, nodeZ, tri)
                    tri = edge_flip_small_edge!(tri, tlst[1])
                    triXC_new, triYC_new, triZC_new = triangle_coordinate_matrices(nodeX, nodeY, nodeZ, tri)
                    eleGma_work = transport_ele_gamma(eleGma_work,
                                                      triXC_old, triYC_old, triZC_old,
                                                      triXC_new, triYC_new, triZC_new;
                                                      domain=domain, method=:node)
                    flips += 1
                    didflip = true
                    changed = true
                    break
                end
            end
        end
        didflip || break
    end

    merges = 0
    while merges < max_merges
        emap = edge_map(tri)
        collapsed = false
        for (e, _) in emap
            a, b = e
            if periodic_edge_length(nodeX, nodeY, nodeZ, a, b, domain) < ds_min
                tri, eleGma_work = collapse_edge_with_circulation!(nodeX, nodeY, nodeZ,
                                                                   tri, eleGma_work,
                                                                   a, b, domain)
                merges += 1
                collapsed = true
                changed = true
                break
            end
        end
        collapsed || break
    end

    if compact
        used = falses(length(nodeX))
        @inbounds for t in 1:size(tri, 1), k in 1:3
            used[tri[t, k]] = true
        end
        old2new = Dict{Int,Int}()
        nodeXnew = Float64[]
        nodeYnew = Float64[]
        nodeZnew = Float64[]
        newidx = 0
        for i in 1:length(nodeX)
            if used[i]
                newidx += 1
                push!(nodeXnew, nodeX[i])
                push!(nodeYnew, nodeY[i])
                push!(nodeZnew, nodeZ[i])
                old2new[i] = newidx
            end
        end
        @inbounds for t in 1:size(tri, 1), k in 1:3
            tri[t, k] = old2new[tri[t, k]]
        end
        resize!(nodeX, length(nodeXnew))
        resize!(nodeY, length(nodeYnew))
        resize!(nodeZ, length(nodeZnew))
        nodeX .= nodeXnew
        nodeY .= nodeYnew
        nodeZ .= nodeZnew
    end

    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    return tri, eleGma_work, changed
end
