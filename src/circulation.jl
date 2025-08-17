module Circulation

export node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma

# Compute per-triangle node circulations (3 per triangle) from element vorticity vectors.
# Mirrors python: _MatrixInversion_2_cal_nodeCircu_frm_eleVor_
function node_circulation_from_ele_gamma(triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                                         eleGma::AbstractMatrix)
    nt = size(triXC,1)
    tau = Array{Float64}(undef, nt, 3)
    # triangle areas
    A = zeros(Float64, nt)
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        a = hypot(hypot(p1[1]-p2[1], p1[2]-p2[2]), p1[3]-p2[3])
        b = hypot(hypot(p2[1]-p3[1], p2[2]-p3[2]), p2[3]-p3[3])
        c = hypot(hypot(p3[1]-p1[1], p3[2]-p1[2]), p3[3]-p1[3])
        s = (a+b+c)/2
        A[t] = sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
        # edge vectors
        X12 = triXC[t,2] - triXC[t,1]; Y12 = triYC[t,2] - triYC[t,1]; Z12 = triZC[t,2] - triZC[t,1]
        X23 = triXC[t,3] - triXC[t,2]; Y23 = triYC[t,3] - triYC[t,2]; Z23 = triZC[t,3] - triZC[t,2]
        X31 = triXC[t,1] - triXC[t,3]; Y31 = triYC[t,1] - triYC[t,3]; Z31 = triZC[t,1] - triZC[t,3]
        M = [X12 X23 X31;
             Y12 Y23 Y31;
             Z12 Z23 Z31;
             1.0 1.0 1.0]
        rhs = [A[t]*eleGma[t,1]; A[t]*eleGma[t,2]; A[t]*eleGma[t,3]; 0.0]
        aτ = M \ rhs
        tau[t,1] = aτ[1]; tau[t,2] = aτ[2]; tau[t,3] = aτ[3]
    end
    return tau
end

# Compute element vorticity vectors from per-triangle node circulations.
# Mirrors python: _cal_eleVorStrgth_frm_nodeCirculation
function ele_gamma_from_node_circ(nodeTau::AbstractMatrix,
                                  triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix)
    nt = size(triXC,1)
    A = zeros(Float64, nt)
    eleGma = zeros(Float64, nt, 3)
    @inbounds for t in 1:nt
        p1 = (triXC[t,1], triYC[t,1], triZC[t,1])
        p2 = (triXC[t,2], triYC[t,2], triZC[t,2])
        p3 = (triXC[t,3], triYC[t,3], triZC[t,3])
        a = hypot(hypot(p1[1]-p2[1], p1[2]-p2[2]), p1[3]-p2[3])
        b = hypot(hypot(p2[1]-p3[1], p2[2]-p3[2]), p2[3]-p3[3])
        c = hypot(hypot(p3[1]-p1[1], p3[2]-p1[2]), p3[3]-p1[3])
        s = (a+b+c)/2
        A[t] = sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
        # vectors as in python formula: p1p2, p2p3, p1p3
        X12 = triXC[t,2] - triXC[t,1]; Y12 = triYC[t,2] - triYC[t,1]; Z12 = triZC[t,2] - triZC[t,1]
        X23 = triXC[t,3] - triXC[t,2]; Y23 = triYC[t,3] - triYC[t,2]; Z23 = triZC[t,3] - triZC[t,2]
        X13 = triXC[t,1] - triXC[t,3]; Y13 = triYC[t,1] - triYC[t,3]; Z13 = triZC[t,1] - triZC[t,3]
        τ1, τ2, τ3 = nodeTau[t,1], nodeTau[t,2], nodeTau[t,3]
        eleGma[t,1] = (τ1*X12 + τ2*X23 + τ3*X13)/A[t]
        eleGma[t,2] = (τ1*Y12 + τ2*Y23 + τ3*Y13)/A[t]
        eleGma[t,3] = (τ1*Z12 + τ2*Z23 + τ3*Z13)/A[t]
    end
    return eleGma
end

# Transport element gamma from old triangles to new triangles by preserving node circulation
function transport_ele_gamma(eleGma_old::AbstractMatrix,
                             triXC_old::AbstractMatrix, triYC_old::AbstractMatrix, triZC_old::AbstractMatrix,
                             triXC_new::AbstractMatrix, triYC_new::AbstractMatrix, triZC_new::AbstractMatrix)
    tau = node_circulation_from_ele_gamma(triXC_old, triYC_old, triZC_old, eleGma_old)
    eleGma_new = ele_gamma_from_node_circ(tau, triXC_new, triYC_new, triZC_new)
    return eleGma_new
end

end # module

