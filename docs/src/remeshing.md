# Remeshing

The sheet mesh must maintain resolution and element quality as it deforms. We provide two layers of remeshing.

## Baseline pass (`VortexMethod.Remesh.remesh_pass!`)

- Detect long edges and perform 1→4 splits using periodic midpoints on all three edges of any marked triangle (and its neighbor sharing an edge).
- Perform edge flips to remove very short edges and reduce anisotropy.
- Optionally collapse edges persistently below `ds_min` by inserting a periodic midpoint and replacing both endpoints.
- Compact node indices (optional) and finally wrap all nodes to the periodic domain using `wrap_nodes!`.

Inputs are derived from grid spacing: `ds_max ≈ O(max(dx,dy))`, `ds_min ≈ O(0.05 max(dx,dy))`.

## Advanced, thesis-style refinement (`VortexMethod.RemeshAdvanced.flow_adaptive_remesh!`)

Uses strict thresholds designed to match the thesis:

- Periodic quality metrics (minimum image)
  - Max aspect ratio: `aspect_ratio ≤ max_aspect_ratio` (default 3.0)
  - Max skewness: `skewness ≤ max_skewness` (default 0.8)
  - Min angle quality: `angle_quality ≥ min_angle_quality` (default 0.4), where 1 = perfect 60° minimum angle
  - Min Jacobian (shape) quality: `jacobian_quality ≥ min_jacobian_quality` (default 0.4)

- Flow indicators
  - Velocity-gradient magnitude: `||∇U||_F ≥ grad_threshold` (default 0.2) ⇒ refine
  - Curvature from dihedral angles between adjacent triangle normals: `angle ≥ curvature_threshold` (default 0.6 rad) ⇒ refine

Any single criterion marks an element for refinement. Refinement uses 1→4 splitting with periodic midpoints. After refinement the node coordinates are wrapped.

## Curvature-/Anisotropy-based passes

- `curvature_based_remesh!`: targets high-dihedral-angle regions to preserve sharp roll-up fronts.
- `anisotropic_remesh!`: probes a supplied `velocity_field(x,y,z)` and refines where gradients are strong.

## Conservation of circulation across remesh

We compute node circulations from `eleGma` on the old mesh, then reconstruct `eleGma` on the new geometry:

1. `nodeTau = node_circulation_from_ele_gamma(tri_old, eleGma_old)`
2. `eleGma_new = ele_gamma_from_node_circ(nodeTau, tri_new)`

This preserves the discrete circulation across re-meshing operations.

## Figures from the thesis

Add quality metrics illustrations and sheet roll-up examples from the thesis PDF into `docs/src/assets/` and reference them here.

- Figure 3.26 (remeshing/quality illustration; suggested filename: `fig_3_26.png`):

![](assets/fig_3_26.png)

- Figure 3.52 (curvature/flow-adaptive example; suggested filename: `fig_3_52.png`):

![](assets/fig_3_52.png)

Add brief captions describing what each figure illustrates once extracted.
