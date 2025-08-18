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
