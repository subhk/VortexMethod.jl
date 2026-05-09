# Advanced remeshing with sophisticated quality metrics and splitting criteria
# Based on thesis Chapter 3.4 - Element Remeshing

module RemeshAdvanced

using ..DomainImpl

export MeshQuality, compute_mesh_quality, quality_based_remesh!,
       element_quality_metrics, element_quality_metrics_periodic, anisotropic_remesh!,
       curvature_based_remesh!, flow_adaptive_remesh!

@inline function smart_midpoint(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                a::Int, b::Int, domain::DomainSpec)
    dx = nodeX[a] - nodeX[b]
    dy = nodeY[a] - nodeY[b]
    dz = nodeZ[a] - nodeZ[b]

    if domain.Lx > 0; dx = dx - domain.Lx * round(dx/domain.Lx); end
    if domain.Ly > 0; dy = dy - domain.Ly * round(dy/domain.Ly); end
    if domain.Lz > 0; dz = dz - 2*domain.Lz * round(dz/(2*domain.Lz)); end

    mx = mod(nodeX[a] - 0.5 * dx, domain.Lx)
    my = mod(nodeY[a] - 0.5 * dy, domain.Ly)
    mz = mod(nodeZ[a] - 0.5 * dz + domain.Lz, 2*domain.Lz) - domain.Lz
    return mx, my, mz
end

function push_midpoint!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                        a::Int, b::Int, domain::DomainSpec)
    mx, my, mz = smart_midpoint(nodeX, nodeY, nodeZ, a, b, domain)
    push!(nodeX, mx)
    push!(nodeY, my)
    push!(nodeZ, mz)
    return length(nodeX)
end

function split_refinements_batch!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                  tri::Array{Int,2}, eleGma::AbstractMatrix,
                                  elements_to_refine::Vector{Int}, split_count::Int,
                                  domain::DomainSpec)
    nt = size(tri, 1)
    gamma_cols = size(eleGma, 2)
    tri_work = Array{Int}(undef, nt + 3 * split_count, 3)
    eleGma_work = Array{Float64}(undef, nt + 3 * split_count, gamma_cols)
    @views tri_work[1:nt, :] .= tri
    @views eleGma_work[1:nt, :] .= eleGma
    sizehint!(nodeX, length(nodeX) + 3 * split_count)
    sizehint!(nodeY, length(nodeY) + 3 * split_count)
    sizehint!(nodeZ, length(nodeZ) + 3 * split_count)

    append_row = nt
    @inbounds for i in 1:split_count
        ele_idx = elements_to_refine[i]
        v1, v2, v3 = tri_work[ele_idx, 1], tri_work[ele_idx, 2], tri_work[ele_idx, 3]

        mid12 = push_midpoint!(nodeX, nodeY, nodeZ, v1, v2, domain)
        mid23 = push_midpoint!(nodeX, nodeY, nodeZ, v2, v3, domain)
        mid31 = push_midpoint!(nodeX, nodeY, nodeZ, v3, v1, domain)

        tri_work[ele_idx, 1] = v1
        tri_work[ele_idx, 2] = mid12
        tri_work[ele_idx, 3] = mid31

        tri_work[append_row + 1, 1] = mid12
        tri_work[append_row + 1, 2] = v2
        tri_work[append_row + 1, 3] = mid23
        tri_work[append_row + 2, 1] = mid31
        tri_work[append_row + 2, 2] = mid23
        tri_work[append_row + 2, 3] = v3
        tri_work[append_row + 3, 1] = mid12
        tri_work[append_row + 3, 2] = mid23
        tri_work[append_row + 3, 3] = mid31

        for k in 1:gamma_cols
            γ = eleGma_work[ele_idx, k]
            eleGma_work[append_row + 1, k] = γ
            eleGma_work[append_row + 2, k] = γ
            eleGma_work[append_row + 3, k] = γ
        end

        append_row += 3
    end

    return tri_work, eleGma_work
end

function apply_split_refinements!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                  tri::Array{Int,2}, eleGma::AbstractMatrix,
                                  elements_to_refine::Vector{Int},
                                  domain::DomainSpec, max_elements::Int)
    size(eleGma, 1) == size(tri, 1) ||
        throw(DimensionMismatch("eleGma row count $(size(eleGma, 1)) does not match triangle count $(size(tri, 1))"))

    if isempty(elements_to_refine) || size(tri, 1) + 3 > max_elements
        wrap_nodes!(nodeX, nodeY, nodeZ, domain)
        return tri, eleGma, false
    end

    sort!(elements_to_refine, rev=true)
    split_count = min(length(elements_to_refine), max(0, (max_elements - size(tri, 1)) ÷ 3))
    split_count > 0 || return tri, eleGma, false
    tri_work, eleGma_work = split_refinements_batch!(
        nodeX, nodeY, nodeZ, tri, eleGma, elements_to_refine, split_count, domain,
    )
    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    return tri_work, eleGma_work, true
end

# Mesh quality metrics structure
struct MeshQuality
    aspect_ratio::Float64
    skewness::Float64
    area_ratio::Float64
    angle_quality::Float64
    edge_length_ratio::Float64
    jacobian_quality::Float64
end

# Compute comprehensive quality metrics for a triangle
function element_quality_metrics(p1::NTuple{3,Float64}, p2::NTuple{3,Float64}, p3::NTuple{3,Float64})
    # Edge vectors
    e1 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
    e2 = (p3[1]-p2[1], p3[2]-p2[2], p3[3]-p2[3])
    e3 = (p1[1]-p3[1], p1[2]-p3[2], p1[3]-p3[3])
    
    # Edge lengths
    l1 = sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2)
    l2 = sqrt(e2[1]^2 + e2[2]^2 + e2[3]^2)
    l3 = sqrt(e3[1]^2 + e3[2]^2 + e3[3]^2)
    
    # Triangle area using cross product
    cross = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
    area = 0.5 * sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2)
    
    # Aspect ratio (longest/shortest edge)
    lmax = max(l1, l2, l3)
    lmin = min(l1, l2, l3)
    aspect_ratio = lmax / max(lmin, eps())
    
    # Skewness (deviation from equilateral)
    perimeter = l1 + l2 + l3
    equilateral_area = perimeter^2 / (12 * sqrt(3))
    skewness = abs(area - equilateral_area) / max(equilateral_area, eps())
    
    # Area ratio (compared to circumradius-based ideal)
    circumradius = (l1 * l2 * l3) / (4 * area + eps())
    ideal_area = 3 * sqrt(3) / 4 * (perimeter/3)^2
    area_ratio = area / max(ideal_area, eps())
    
    # Angle quality (minimum angle / 60°)
    # Using dot products to find angles
    cos_a = (l2^2 + l3^2 - l1^2) / (2 * l2 * l3 + eps())
    cos_b = (l1^2 + l3^2 - l2^2) / (2 * l1 * l3 + eps())
    cos_c = (l1^2 + l2^2 - l3^2) / (2 * l1 * l2 + eps())
    
    angle_a = acos(clamp(cos_a, -1.0, 1.0))
    angle_b = acos(clamp(cos_b, -1.0, 1.0))
    angle_c = acos(clamp(cos_c, -1.0, 1.0))
    
    min_angle = min(angle_a, angle_b, angle_c)
    angle_quality = min_angle / (π/3)  # normalized by 60°
    
    # Edge length ratio uniformity
    edge_length_ratio = lmin / lmax
    
    # Jacobian quality (shape regularity)
    jacobian_quality = 4 * sqrt(3) * area / (l1^2 + l2^2 + l3^2 + eps())
    
    return MeshQuality(aspect_ratio, skewness, area_ratio, angle_quality, 
                      edge_length_ratio, jacobian_quality)
end

# Periodic minimum-image quality metrics (uses domain lengths)
@inline _minimg(d::Float64, L::Float64) = (L <= 0 ? d : (d - L * round(d/L)))
function element_quality_metrics_periodic(p1::NTuple{3,Float64}, p2::NTuple{3,Float64}, p3::NTuple{3,Float64}, domain::DomainSpec)
    # Edge vectors with minimum image
    dx12 = _minimg(p2[1]-p1[1], domain.Lx); dy12 = _minimg(p2[2]-p1[2], domain.Ly); dz12 = _minimg(p2[3]-p1[3], 2*domain.Lz)
    dx23 = _minimg(p3[1]-p2[1], domain.Lx); dy23 = _minimg(p3[2]-p2[2], domain.Ly); dz23 = _minimg(p3[3]-p2[3], 2*domain.Lz)
    dx31 = _minimg(p1[1]-p3[1], domain.Lx); dy31 = _minimg(p1[2]-p3[2], domain.Ly); dz31 = _minimg(p1[3]-p3[3], 2*domain.Lz)

    e1 = (dx12, dy12, dz12)
    e2 = (dx23, dy23, dz23)
    e3 = (dx31, dy31, dz31)

    # Edge lengths
    l1 = sqrt(e1[1]^2 + e1[2]^2 + e1[3]^2)
    l2 = sqrt(e2[1]^2 + e2[2]^2 + e2[3]^2)
    l3 = sqrt(e3[1]^2 + e3[2]^2 + e3[3]^2)

    # Triangle area using cross product
    cross = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
    area = 0.5 * sqrt(cross[1]^2 + cross[2]^2 + cross[3]^2)

    # Aspect ratio (longest/shortest edge)
    lmax = max(l1, l2, l3)
    lmin = min(l1, l2, l3)
    aspect_ratio = lmax / max(lmin, eps())

    # Skewness (deviation from equilateral)
    perimeter = l1 + l2 + l3
    equilateral_area = perimeter^2 / (12 * sqrt(3))
    skewness = abs(area - equilateral_area) / max(equilateral_area, eps())

    # Area ratio (compared to circumradius-based ideal)
    circumradius = (l1 * l2 * l3) / (4 * area + eps())
    ideal_area = 3 * sqrt(3) / 4 * (perimeter/3)^2
    area_ratio = area / max(ideal_area, eps())

    # Angle quality (minimum angle / 60°)
    cos_a = (l2^2 + l3^2 - l1^2) / (2 * l2 * l3 + eps())
    cos_b = (l1^2 + l3^2 - l2^2) / (2 * l1 * l3 + eps())
    cos_c = (l1^2 + l2^2 - l3^2) / (2 * l1 * l2 + eps())
    angle_a = acos(clamp(cos_a, -1.0, 1.0))
    angle_b = acos(clamp(cos_b, -1.0, 1.0))
    angle_c = acos(clamp(cos_c, -1.0, 1.0))
    min_angle = min(angle_a, angle_b, angle_c)
    angle_quality = min_angle / (π/3)

    # Edge length ratio uniformity
    edge_length_ratio = lmin / lmax

    # Jacobian quality (shape regularity)
    jacobian_quality = 4 * sqrt(3) * area / (l1^2 + l2^2 + l3^2 + eps())

    return MeshQuality(aspect_ratio, skewness, area_ratio, angle_quality,
                       edge_length_ratio, jacobian_quality)
end

# Compute quality metrics for entire mesh
function compute_mesh_quality(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    nt = size(triXC, 1)
    qualities = Vector{MeshQuality}(undef, nt)
    
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        qualities[t] = element_quality_metrics(p1, p2, p3)
    end
    
    return qualities
end

# Periodic variant
function compute_mesh_quality(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix, domain::DomainSpec)
    nt = size(triXC, 1)
    qualities = Vector{MeshQuality}(undef, nt)
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        qualities[t] = element_quality_metrics_periodic(p1, p2, p3, domain)
    end
    return qualities
end

# Quality-based remeshing decision
function should_refine_element(quality::MeshQuality;
                              max_aspect_ratio::Float64=4.0,
                              max_skewness::Float64=0.8,
                              min_angle_quality::Float64=0.3,
                              min_jacobian_quality::Float64=0.3)
    return (quality.aspect_ratio > max_aspect_ratio ||
            quality.skewness > max_skewness ||
            quality.angle_quality < min_angle_quality ||
            quality.jacobian_quality < min_jacobian_quality)
end

function quality_based_remesh!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                               tri::Array{Int,2}, eleGma::AbstractMatrix, domain::DomainSpec;
                               max_aspect_ratio::Float64=4.0,
                               max_skewness::Float64=0.8,
                               min_angle_quality::Float64=0.3,
                               min_jacobian_quality::Float64=0.3,
                               max_elements::Int=50000)
    nt = size(tri, 1)
    triXC = Array{Float64}(undef, nt, 3)
    triYC = Array{Float64}(undef, nt, 3)
    triZC = Array{Float64}(undef, nt, 3)
    @inbounds for k in 1:3, t in 1:nt
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end

    qualities = compute_mesh_quality(triXC, triYC, triZC, domain)
    elements_to_refine = Int[]
    @inbounds for t in 1:nt
        if should_refine_element(qualities[t];
                                 max_aspect_ratio=max_aspect_ratio,
                                 max_skewness=max_skewness,
                                 min_angle_quality=min_angle_quality,
                                 min_jacobian_quality=min_jacobian_quality)
            push!(elements_to_refine, t)
        end
        if length(elements_to_refine) * 3 + nt >= max_elements
            break
        end
    end

    return apply_split_refinements!(nodeX, nodeY, nodeZ, tri, eleGma,
                                    elements_to_refine, domain, max_elements)
end

# Enhanced 1-to-4 splitting with quality preservation
function quality_split_triangle!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                tri::Array{Int,2}, ele_idx::Int, domain::DomainSpec)
    v1, v2, v3 = tri[ele_idx,1], tri[ele_idx,2], tri[ele_idx,3]
    mid12 = push_midpoint!(nodeX, nodeY, nodeZ, v1, v2, domain)
    mid23 = push_midpoint!(nodeX, nodeY, nodeZ, v2, v3, domain)
    mid31 = push_midpoint!(nodeX, nodeY, nodeZ, v3, v1, domain)
    
    # Replace original triangle and add three new ones
    tri[ele_idx, :] .= (v1, mid12, mid31)
    tri_new = Array{Int}(undef, 3, 3)
    tri_new[1,:] .= (mid12, v2, mid23)
    tri_new[2,:] .= (mid31, mid23, v3)
    tri_new[3,:] .= (mid12, mid23, mid31)  # Central triangle
    
    return vcat(tri, tri_new)
end

# Anisotropic remeshing based on flow gradients
function anisotropic_remesh!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                           tri::Array{Int,2}, eleGma::AbstractMatrix,
                           velocity_field, domain::DomainSpec;
                           refinement_threshold::Float64=0.1, max_elements::Int=50000)
    nt = size(tri, 1)
    elements_to_refine = Int[]

    @inbounds for t in 1:nt
        v1, v2, v3 = tri[t,1], tri[t,2], tri[t,3]
        cx = (nodeX[v1] + nodeX[v2] + nodeX[v3]) / 3
        cy = (nodeY[v1] + nodeY[v2] + nodeY[v3]) / 3
        cz = (nodeZ[v1] + nodeZ[v2] + nodeZ[v3]) / 3

        h = 1e-6
        u_center = velocity_field(cx, cy, cz)
        u_dx = velocity_field(cx+h, cy, cz)
        u_dy = velocity_field(cx, cy+h, cz)
        u_dz = velocity_field(cx, cy, cz+h)
        du_dx = (u_dx[1] - u_center[1]) / h
        du_dy = (u_dy[1] - u_center[1]) / h
        du_dz = (u_dz[1] - u_center[1]) / h
        grad_mag = sqrt(du_dx^2 + du_dy^2 + du_dz^2)

        if grad_mag > refinement_threshold && nt < max_elements
            push!(elements_to_refine, t)
        end
    end

    return apply_split_refinements!(nodeX, nodeY, nodeZ, tri, eleGma,
                                    elements_to_refine, domain, max_elements)
end

# Curvature-based refinement for vortex sheet tracking
function curvature_based_remesh!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                                tri::Array{Int,2}, eleGma::AbstractMatrix, domain::DomainSpec;
                                curvature_threshold::Float64=1.0, max_elements::Int=50000)
    nt = size(tri, 1)
    elements_to_refine = Int[]

    edge_map = Dict{Tuple{Int,Int}, Vector{Int}}()
    @inbounds for t in 1:nt
        v1, v2, v3 = tri[t,1], tri[t,2], tri[t,3]
        for (a,b) in ((v1,v2), (v2,v3), (v3,v1))
            edge = a < b ? (a,b) : (b,a)
            if haskey(edge_map, edge)
                push!(edge_map[edge], t)
            else
                edge_map[edge] = [t]
            end
        end
    end

    @inbounds for t in 1:nt
        v1, v2, v3 = tri[t,1], tri[t,2], tri[t,3]
        p1 = (nodeX[v1], nodeY[v1], nodeZ[v1])
        p2 = (nodeX[v2], nodeY[v2], nodeZ[v2])
        p3 = (nodeX[v3], nodeY[v3], nodeZ[v3])

        e1 = (p2[1]-p1[1], p2[2]-p1[2], p2[3]-p1[3])
        e2 = (p3[1]-p1[1], p3[2]-p1[2], p3[3]-p1[3])
        normal = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
        normal_mag = sqrt(normal[1]^2 + normal[2]^2 + normal[3]^2)
        if normal_mag > 0
            normal = (normal[1]/normal_mag, normal[2]/normal_mag, normal[3]/normal_mag)
        end

        max_angle_diff = 0.0
        for (a,b) in ((v1,v2), (v2,v3), (v3,v1))
            edge = a < b ? (a,b) : (b,a)
            if haskey(edge_map, edge) && length(edge_map[edge]) == 2
                neighbor_t = edge_map[edge][1] == t ? edge_map[edge][2] : edge_map[edge][1]
                nv1, nv2, nv3 = tri[neighbor_t,1], tri[neighbor_t,2], tri[neighbor_t,3]
                np1 = (nodeX[nv1], nodeY[nv1], nodeZ[nv1])
                np2 = (nodeX[nv2], nodeY[nv2], nodeZ[nv2])
                np3 = (nodeX[nv3], nodeY[nv3], nodeZ[nv3])
                ne1 = (np2[1]-np1[1], np2[2]-np1[2], np2[3]-np1[3])
                ne2 = (np3[1]-np1[1], np3[2]-np1[2], np3[3]-np1[3])
                nnormal = (ne1[2]*ne2[3] - ne1[3]*ne2[2], ne1[3]*ne2[1] - ne1[1]*ne2[3], ne1[1]*ne2[2] - ne1[2]*ne2[1])
                nnormal_mag = sqrt(nnormal[1]^2 + nnormal[2]^2 + nnormal[3]^2)
                if nnormal_mag > 0
                    nnormal = (nnormal[1]/nnormal_mag, nnormal[2]/nnormal_mag, nnormal[3]/nnormal_mag)
                    dot_product = normal[1]*nnormal[1] + normal[2]*nnormal[2] + normal[3]*nnormal[3]
                    angle_diff = acos(clamp(abs(dot_product), 0.0, 1.0))
                    max_angle_diff = max(max_angle_diff, angle_diff)
                end
            end
        end

        if max_angle_diff > curvature_threshold && nt < max_elements
            push!(elements_to_refine, t)
        end
    end

    return apply_split_refinements!(nodeX, nodeY, nodeZ, tri, eleGma,
                                    elements_to_refine, domain, max_elements)
end

# Flow-adaptive remeshing combining multiple criteria
function flow_adaptive_remesh!(nodeX::Vector{Float64}, nodeY::Vector{Float64}, nodeZ::Vector{Float64},
                              tri::Array{Int,2}, eleGma::AbstractMatrix,
                              velocity_field, domain::DomainSpec;
                              quality_weight::Float64=0.3, gradient_weight::Float64=0.4,
                              curvature_weight::Float64=0.3, refinement_threshold::Float64=0.5,
                              max_aspect_ratio::Float64=3.0,
                              max_skewness::Float64=0.8,
                              min_angle_quality::Float64=0.4,
                              min_jacobian_quality::Float64=0.4,
                              grad_threshold::Float64=0.2,
                              curvature_threshold::Float64=0.6,
                              max_elements::Int=50000)
    nt = size(tri, 1)
    triXC = Array{Float64}(undef, nt, 3)
    triYC = Array{Float64}(undef, nt, 3)
    triZC = Array{Float64}(undef, nt, 3)
    @inbounds for k in 1:3, t in 1:nt
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end

    qualities = compute_mesh_quality(triXC, triYC, triZC, domain)

    edge_map = Dict{Tuple{Int,Int}, Vector{Int}}()
    @inbounds for t in 1:nt
        v1, v2, v3 = tri[t,1], tri[t,2], tri[t,3]
        for (a,b) in ((v1,v2), (v2,v3), (v3,v1))
            e = a < b ? (a,b) : (b,a)
            if haskey(edge_map, e)
                push!(edge_map[e], t)
            else
                edge_map[e] = [t]
            end
        end
    end

    elements_to_refine = Int[]
    @inbounds for t in 1:nt
        v1, v2, v3 = tri[t,1], tri[t,2], tri[t,3]
        q = qualities[t]

        refine_quality = (q.aspect_ratio > max_aspect_ratio ||
                          q.skewness > max_skewness ||
                          q.angle_quality < min_angle_quality ||
                          q.jacobian_quality < min_jacobian_quality)

        cx = (nodeX[v1] + nodeX[v2] + nodeX[v3]) / 3
        cy = (nodeY[v1] + nodeY[v2] + nodeY[v3]) / 3
        cz = (nodeZ[v1] + nodeZ[v2] + nodeZ[v3]) / 3
        h = 1e-6
        u_center = velocity_field(cx, cy, cz)
        u_dx = velocity_field(cx+h, cy, cz)
        u_dy = velocity_field(cx, cy+h, cz)
        u_dz = velocity_field(cx, cy, cz+h)
        ux = (u_dx[1]-u_center[1])/h; vx = (u_dx[2]-u_center[2])/h; wx = (u_dx[3]-u_center[3])/h
        uy = (u_dy[1]-u_center[1])/h; vy = (u_dy[2]-u_center[2])/h; wy = (u_dy[3]-u_center[3])/h
        uz = (u_dz[1]-u_center[1])/h; vz = (u_dz[2]-u_center[2])/h; wz = (u_dz[3]-u_center[3])/h
        grad_mag = sqrt(ux^2 + vx^2 + wx^2 + uy^2 + vy^2 + wy^2 + uz^2 + vz^2 + wz^2)
        refine_gradient = grad_mag > grad_threshold

        p1 = (nodeX[v1], nodeY[v1], nodeZ[v1])
        p2 = (nodeX[v2], nodeY[v2], nodeZ[v2])
        p3 = (nodeX[v3], nodeY[v3], nodeZ[v3])
        e1 = (_minimg(p2[1]-p1[1], domain.Lx), _minimg(p2[2]-p1[2], domain.Ly), _minimg(p2[3]-p1[3], 2*domain.Lz))
        e2 = (_minimg(p3[1]-p1[1], domain.Lx), _minimg(p3[2]-p1[2], domain.Ly), _minimg(p3[3]-p1[3], 2*domain.Lz))
        n  = (e1[2]*e2[3] - e1[3]*e2[2], e1[3]*e2[1] - e1[1]*e2[3], e1[1]*e2[2] - e1[2]*e2[1])
        nmag = sqrt(n[1]^2 + n[2]^2 + n[3]^2)
        nx,ny,nz = nmag>0 ? (n[1]/nmag, n[2]/nmag, n[3]/nmag) : (0.0,0.0,1.0)
        max_angle_diff = 0.0
        for (a,b) in ((v1,v2), (v2,v3), (v3,v1))
            e = a < b ? (a,b) : (b,a)
            if haskey(edge_map, e) && length(edge_map[e]) == 2
                tnb = edge_map[e][1] == t ? edge_map[e][2] : edge_map[e][1]
                w1, w2, w3 = tri[tnb,1], tri[tnb,2], tri[tnb,3]
                q1 = (nodeX[w1], nodeY[w1], nodeZ[w1])
                q2 = (nodeX[w2], nodeY[w2], nodeZ[w2])
                q3 = (nodeX[w3], nodeY[w3], nodeZ[w3])
                f1 = (_minimg(q2[1]-q1[1], domain.Lx), _minimg(q2[2]-q1[2], domain.Ly), _minimg(q2[3]-q1[3], 2*domain.Lz))
                f2 = (_minimg(q3[1]-q1[1], domain.Lx), _minimg(q3[2]-q1[2], domain.Ly), _minimg(q3[3]-q1[3], 2*domain.Lz))
                nn = (f1[2]*f2[3] - f1[3]*f2[2], f1[3]*f2[1] - f1[1]*f2[3], f1[1]*f2[2] - f1[2]*f2[1])
                nnmag = sqrt(nn[1]^2 + nn[2]^2 + nn[3]^2)
                if nnmag > 0
                    n2x, n2y, n2z = (nn[1]/nnmag, nn[2]/nnmag, nn[3]/nnmag)
                    dotp = clamp(nx*n2x + ny*n2y + nz*n2z, -1.0, 1.0)
                    angle = acos(abs(dotp))
                    max_angle_diff = max(max_angle_diff, angle)
                end
            end
        end
        refine_curvature = max_angle_diff > curvature_threshold

        if (refine_quality || refine_gradient || refine_curvature)
            push!(elements_to_refine, t)
        end

        if length(elements_to_refine) + nt >= max_elements
            break
        end
    end

    return apply_split_refinements!(nodeX, nodeY, nodeZ, tri, eleGma,
                                    elements_to_refine, domain, max_elements)
end

end # module

using .RemeshAdvanced: MeshQuality, compute_mesh_quality, quality_based_remesh!,
                       element_quality_metrics, element_quality_metrics_periodic, anisotropic_remesh!,
                       curvature_based_remesh!, flow_adaptive_remesh!
