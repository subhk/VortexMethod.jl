module Remeshing

using LinearAlgebra
using StaticArrays
using ..DomainImpl
using ..Peskin3D
using ..Circulation

include("basic.jl")
include("quality.jl")

export detect_max_edge_length, detect_min_edge_length,
       element_splitting!, edge_flip_small_edge!, remesh_pass!,
       MeshQuality, compute_mesh_quality, quality_based_remesh!,
       element_quality_metrics, element_quality_metrics_periodic,
       anisotropic_remesh!, curvature_based_remesh!, flow_adaptive_remesh!,
       quality_split_triangle!

end

using .Remeshing: detect_max_edge_length, detect_min_edge_length,
                  element_splitting!, edge_flip_small_edge!, remesh_pass!,
                  MeshQuality, compute_mesh_quality, quality_based_remesh!,
                  element_quality_metrics, element_quality_metrics_periodic,
                  anisotropic_remesh!, curvature_based_remesh!, flow_adaptive_remesh!,
                  quality_split_triangle!
