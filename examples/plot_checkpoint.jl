using VortexMethod
using Plots

# Load latest checkpoint and plot nodes colored by |gamma| averaged per node
ckpt_dir = get(ENV, "CKPT_DIR", "checkpoints")
nodeX, nodeY, nodeZ, tri, eleGma = load_latest_checkpoint(ckpt_dir)

# Map element gamma magnitude to nodes by averaging contributions
nt = size(tri,1)
nv = length(nodeX)
gmag = sqrt.(eleGma[:,1].^2 .+ eleGma[:,2].^2 .+ eleGma[:,3].^2)
acc = zeros(Float64, nv)
cnt = zeros(Int, nv)
@inbounds for t in 1:nt
    val = gmag[t]
    v1,v2,v3 = tri[t,1], tri[t,2], tri[t,3]
    acc[v1] += val; acc[v2] += val; acc[v3] += val
    cnt[v1] += 1;   cnt[v2] += 1;   cnt[v3] += 1
end
node_gamma = similar(acc)
@inbounds for i in 1:nv
    node_gamma[i] = cnt[i] > 0 ? acc[i]/cnt[i] : 0.0
end

plt = scatter3d(nodeX, nodeY, nodeZ; markersize=2, marker_z=node_gamma, colorbar=true,
                xlabel="x", ylabel="y", zlabel="z", title="Nodes colored by |gamma|")
display(plt)

