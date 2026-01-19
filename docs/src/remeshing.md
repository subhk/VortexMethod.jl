# Remeshing

The sheet mesh must maintain resolution and element quality as it deforms under Lagrangian motion. Without remeshing, vortex sheets become bent and contorted, losing accuracy. This page describes the edge-splitting and node-merging algorithms based on [Stock (2006), §3.4–3.5](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837).

## Why Remeshing is Essential

The proposed discretization method has ideal conservation properties under planar strain—even if triangle nodes become separated beyond the grid scale. However, in real flows:

1. Vortex sheets fold and roll up
2. Elements experience non-uniform strain
3. Triangle aspect ratios degrade
4. Resolution becomes insufficient in high-curvature regions

Two complementary operations address these issues:
- **Edge Splitting:** Inserts nodes in long edges, maintaining resolution
- **Node Merging:** Removes close nodes, simplifying the mesh

## Edge Splitting Algorithm

When triangle edges become too long (> ``\Delta_{\text{split}} \approx \delta``), they are split to maintain resolution.

### Splitting Procedure

At every time step:

1. Reset all node and element flags, recompute surface normal vectors
2. Flag all elements whose longest edge exceeds ``\Delta_{\text{split}}``
3. Sort flagged edges by length (longest first)
4. For each flagged edge:
   - Find all elements sharing that edge
   - Create a new node at the edge midpoint
   - Split each sharing element into two new elements
   - Assign vortex sheet strength to new elements (see below)
   - Remove original elements

This produces a valid connected mesh with no under-resolved regions.

### Midpoint Selection Methods

Three methods for positioning new nodes:

#### 1. Geometric Midpoint (Default)

Simply average the endpoint positions:
```math
\mathbf{x}_3 = \frac{\mathbf{x}_1 + \mathbf{x}_2}{2}
```
- **Pros:** Exactly conserves enclosed volume
- **Cons:** Poor at capturing curvature

#### 2. Cubic Spline Fit

Fits a cubic spline between endpoints using surface normal information:

1. Calculate surface normal at each endpoint ``\hat{\mathbf{n}}_j`` as weighted average of adjacent element normals:
```math
\hat{\mathbf{n}}_j = \text{norm}\left(\sum_{e=1}^{N_j} \hat{\mathbf{n}}_e \phi_{j,e}\right)
```
where ``\phi_{j,e}`` is the angle subtended at node ``j`` by element ``e``.

2. Compute tangential projection ``\mathbf{P}_j = \mathbf{I} - \hat{\mathbf{n}}_j\hat{\mathbf{n}}_j^T``

3. Build spline coefficients and evaluate at midpoint

- **Pros:** Better approximation of curved surfaces
- **Cons:** More computation

#### 3. Cylindrical Projection

Fits a cylinder to the edge using endpoint normals:

```math
\mathbf{x}_3 = \frac{\mathbf{x}_1 + \mathbf{x}_2}{2} + d\,\frac{\hat{\mathbf{n}}_1 + \hat{\mathbf{n}}_2}{\|\hat{\mathbf{n}}_1 + \hat{\mathbf{n}}_2\|}
```

where ``d`` is computed from the angle between normals and the projected edge length:

```math
\theta = \frac{1}{2}\cos^{-1}(\hat{\mathbf{n}}_1 \cdot \hat{\mathbf{n}}_2), \quad d = \frac{l}{2}\left(\frac{1}{\sin\theta} - \frac{1}{\tan\theta}\right)
```

- **Pros:** Excellent for roll-up regions
- **Cons:** Cannot be used across disconnected sheets

### Circulation Assignment to Split Elements

When splitting an edge, the child elements must receive appropriate vortex sheet strength. For **coplanar** children:

```math
\boldsymbol{\gamma}_{\text{parent}} = \boldsymbol{\gamma}_{\text{child1}} = \boldsymbol{\gamma}_{\text{child2}}
```

For **non-coplanar** children (spline/cylindrical midpoints):

1. Create new node at geometric midpoint first
2. Apply circulation assignment to coplanar children
3. Move node to perturbed position (stretches vortex lines without changing circulation)

## Baseline pass (`VortexMethod.Remesh.remesh_pass!`)

- Detect long edges and perform 1→4 splits using periodic midpoints on all three edges of any marked triangle (and its neighbor sharing an edge).
- Perform edge flips to remove very short edges and reduce anisotropy.
- Optionally collapse edges persistently below `ds_min` by inserting a periodic midpoint and replacing both endpoints.
- Compact node indices (optional) and finally wrap all nodes to the periodic domain using `wrap_nodes!`.

Inputs are derived from grid spacing: `ds_max ≈ O(max(dx,dy))`, `ds_min ≈ O(0.05 max(dx,dy))`.

## Node Merging Algorithm

Edge splitting alone can produce numerous thin, low-quality triangles. Node merging simplifies the mesh by combining close node pairs.

### Types of Merging

1. **In-Sheet Merge:** Only merges nodes that share at least one element (same connected sheet)
2. **Full Merge:** Merges any close node pairs, regardless of connectivity (can connect separate sheets)

### Merging Procedure

Node merging operates in conjunction with edge splitting:

1. Reset all node and element flags, recompute surface normals
2. Build list of node pairs within distance ``\Delta_{\text{merge}}``
3. Sort by distance (closest first)
4. For each node pair (node 1 and node 2):
   - Skip if either node was recently modified
   - Compute new node position
   - Create lists of:
     - Elements containing only node 1
     - Elements containing only node 2
     - Elements containing both (will be collapsed and removed)
     - Element pairs that will become coincident
   - Save vortex sheet strengths from affected elements
   - Remove collapsed elements
   - Replace node 2 with node 1 in all elements
   - Delete node 2
   - Relocate node 1 to computed position
   - Redistribute saved vorticity to remaining elements

### Effect on Element Count

| Remeshing Mode | Element Count Scaling | Notes |
|----------------|----------------------|-------|
| No remeshing | ``N \sim t^3`` | Unsustainable |
| Split only | ``N \sim t^3`` (thin triangles) | Many poor-quality elements |
| In-sheet merge | ``N \sim A_{\text{sheet}} \sim t^2`` | Good quality maintenance |
| Full merge | ``N \sim A_{\text{sheet}}`` | Can reconnect sheets |

The in-sheet merging case shows that element count increases proportionally to surface area—indicating consistent element sizes are maintained.

## Advanced, thesis-style refinement (`VortexMethod.RemeshAdvanced.flow_adaptive_remesh!`)

Uses strict thresholds designed to match the thesis:

### Periodic Quality Metrics (Minimum Image)

- **Max aspect ratio:** `aspect_ratio ≤ max_aspect_ratio` (default 3.0)
- **Max skewness:** `skewness ≤ max_skewness` (default 0.8)
- **Min angle quality:** `angle_quality ≥ min_angle_quality` (default 0.4), where 1 = perfect 60° minimum angle
- **Min Jacobian quality:** `jacobian_quality ≥ min_jacobian_quality` (default 0.4)

### Flow-Based Indicators

- **Velocity gradient magnitude:** ``\|\nabla\mathbf{U}\|_F \geq`` `grad_threshold` (default 0.2) ⇒ refine
- **Curvature (dihedral angle):** angle between adjacent triangle normals ≥ `curvature_threshold` (default 0.6 rad) ⇒ refine

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

_Figure 3.26: Remeshing and mesh-quality metrics (aspect ratio, skewness, angle and Jacobian quality) used to trigger refinement._

- Figure 3.52 (curvature/flow-adaptive example; suggested filename: `fig_3_52.png`):

![](assets/fig_3_52.png)

_Figure 3.52: Curvature- and flow-adaptive refinement highlighting high-dihedral-angle regions and strong velocity-gradient zones._

Add brief captions describing what each figure illustrates once extracted.
