module TimeStepper

using ..DomainImpl
using ..Poisson3D
using ..Peskin3D
using ..Circulation

export node_velocities, rk2_step!

function node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom::DomainSpec, gr::GridSpec)
    VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)
    dx,dy,dz = grid_spacing(dom, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
    Ux, Uy, Uz = poisson_velocity_fft(u_rhs, v_rhs, w_rhs, dom)
    u, v, w = interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, dom, gr)
    return u, v, w
end

function rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom::DomainSpec, gr::GridSpec, dt::Float64)
    # velocities at t^n
    triXC = similar(eleGma, size(tri,1), 3); triYC = similar(triXC); triZC = similar(triXC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    u1, v1, w1 = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom, gr)

    # half-step positions
    xh = nodeX .+ 0.5 .* dt .* u1
    yh = nodeY .+ 0.5 .* dt .* v1
    zh = nodeZ .+ 0.5 .* dt .* w1
    # periodic wrap
    xh .= mod.(xh, dom.Lx)
    yh .= mod.(yh, dom.Ly)
    zh .= mod.(zh .+ dom.Lz, 2*dom.Lz) .- dom.Lz

    # velocities at half step
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = xh[v]
        triYC[t,k] = yh[v]
        triZC[t,k] = zh[v]
    end
    u2, v2, w2 = node_velocities(eleGma, triXC, triYC, triZC, xh, yh, zh, dom, gr)

    # full-step update
    nodeX .+= dt .* u2
    nodeY .+= dt .* v2
    nodeZ .+= dt .* w2
    nodeX .= mod.(nodeX, dom.Lx)
    nodeY .= mod.(nodeY, dom.Ly)
    nodeZ .= mod.(nodeZ .+ dom.Lz, 2*dom.Lz) .- dom.Lz

    # rebuild triangle coords at t^{n+1}
    triXC_new = similar(triXC); triYC_new = similar(triYC); triZC_new = similar(triZC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC_new[t,k] = nodeX[v]
        triYC_new[t,k] = nodeY[v]
        triZC_new[t,k] = nodeZ[v]
    end
    # transport element gamma by preserving node circulation across geometry change
    eleGma_new = transport_ele_gamma(eleGma, triXC, triYC, triZC, triXC_new, triYC_new, triZC_new)
    eleGma .= eleGma_new

    return nothing
end

end # module

using .TimeStepper: node_velocities, rk2_step!
