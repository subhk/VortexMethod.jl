```@meta
CurrentModule = VortexMethod
```

# VortexMethod

Documentation for [VortexMethod](https://github.com/subhk/VortexMethod.jl).

```@index
```

```@autodocs
Modules = [VortexMethod]
```

## Notes on Remeshing and Metrics

- Periodic metrics: Advanced remeshing and sheet quality now use periodic, minimum-image geometry in x/y and symmetric in z ([-Lz,Lz]).
- Thresholds (defaults):
  - max_aspect_ratio=3.0, max_skewness=0.8
  - min_angle_quality=0.4, min_jacobian_quality=0.4
  - grad_threshold=0.2 (Frobenius norm of ∇U), curvature_threshold=0.6 rad
- Kernel-based interpolation options are available via `Kernels`.
- New helper: `grid_velocity(eleGma, triXC, triYC, triZC, dom, gr)` computes `(Ux,Uy,Uz)` for reuse.

### Tuning Thresholds

- Coarser meshes typically require looser thresholds (e.g., AR≤4.0) to avoid over-refinement; finer meshes can tolerate tighter thresholds (e.g., AR≤3.0).
- Set `grad_threshold` proportional to expected shear; use a smaller value for gentle flows and larger for strongly sheared layers.
- `curvature_threshold` is a dihedral angle; values around 0.5–0.8 rad (≈29–46°) work well for sheet roll-up detection.
- You can pass these per-call via `flow_adaptive_remesh!` or rely on the defaults.

### Velocity Reuse

- Use `make_velocity_sampler(eleGma, triXC, triYC, triZC, dom, gr)` to build a closure `(x,y,z)->(u,v,w)` backed by a single spread/Poisson solve.
- This reduces repeated Poisson solves during remeshing and sheet tracking in the same time step.
