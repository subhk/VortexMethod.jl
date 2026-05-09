using Test
using VortexMethod

repo_root = dirname(@__DIR__)
src_root = joinpath(repo_root, "src")

contains_pattern(file, pattern) = occursin(pattern, read(file, String))

const POISSON_INTERNAL_TYPES = [
    :FFTSolver,
    :HybridSolver,
    :IterativeSolver,
    :PeriodicBC,
    :DirichletBC,
    :NeumannBC,
]

const ROOT_QUALIFIED_POISSON_INTERNALS = [
    :curl_rhs_centered,
    :FFTSolver,
    :HybridSolver,
    :IterativeSolver,
    :PeriodicBC,
    :DirichletBC,
    :NeumannBC,
    :PoissonWorkspace,
]

function unqualified_poisson_type_refs(file)
    text = read(file, String)
    pattern = Regex("(?<![.])\\b(" * join(string.(POISSON_INTERNAL_TYPES), "|") * ")\\b")
    return collect(eachmatch(pattern, text))
end

function root_qualified_poisson_internal_refs(file)
    pattern = Regex("\\bVortexMethod\\.(" * join(string.(ROOT_QUALIFIED_POISSON_INTERNALS), "|") * ")\\b")
    refs = String[]
    for (line_number, line) in enumerate(eachline(file))
        m = match(pattern, line)
        m === nothing && continue
        push!(refs, "$(relpath(file, repo_root)):$(line_number):$(m.match)")
    end
    return refs
end

@testset "code structure" begin
    @testset "source tree is organized by subsystem" begin
        expected_files = [
            "core/domain.jl",
            "core/mesh.jl",
            "core/workspace.jl",
            "core/layout.jl",
            "kernels/kernels.jl",
            "kernels/grid_transfer.jl",
            "poisson/Poisson.jl",
            "poisson/fft.jl",
            "poisson/solvers.jl",
            "integration/timestep.jl",
            "integration/interface.jl",
            "physics/circulation.jl",
            "physics/dissipation.jl",
            "physics/energy.jl",
            "remeshing/Remeshing.jl",
            "remeshing/basic.jl",
            "remeshing/quality.jl",
            "sheets/sheets.jl",
            "io/checkpoint.jl",
            "diagnostics/performance.jl",
            "diagnostics/fast_linalg.jl",
        ]

        for relpath in expected_files
            @test isfile(joinpath(src_root, relpath))
        end
    end

    @testset "legacy duplicate subsystem files are removed" begin
        removed_files = [
            "peskin3d.jl",
            "poisson3d.jl",
            "poisson_advanced.jl",
            "remesh.jl",
            "remesh_advanced.jl",
            "vortex_sheets.jl",
            "soa_layout.jl",
            "cache_optimization.jl",
        ]

        for relpath in removed_files
            @test !isfile(joinpath(src_root, relpath))
        end
    end

    @testset "legacy module names are gone from source, tests, examples, and docs" begin
        roots = ["src", "test", "examples", joinpath("docs", "src")]
        stale_patterns = [
            "PoissonAdvanced",
            "RemeshAdvanced",
            "Poisson3D",
            "Peskin3D",
            "VortexSheets",
            "SoALayout",
            "CacheOptimization",
            "poisson3d",
            "poisson_advanced",
            "remesh_advanced",
            "peskin3d",
            "vortex_sheets",
            "cache_optimization",
        ]

        for root in roots
            for (dir, _, filenames) in walkdir(joinpath(repo_root, root))
                for filename in filenames
                    file = joinpath(dir, filename)
                    file == (@__FILE__) && continue
                    if endswith(file, ".jl") || endswith(file, ".md")
                        for pattern in stale_patterns
                            @test !contains_pattern(file, pattern)
                        end
                    end
                end
            end
        end
    end

    @testset "root exports expose public API only" begin
        root_exports = Set(names(VortexMethod))

        public_exports = [
            :DomainSpec,
            :GridSpec,
            :VortexWorkspace,
            :Simulation,
            :Clock,
            :VortexSheetModel,
            :structured_mesh,
            :time_step!,
            :run!,
            :set!,
            :rk2_step!,
            :rk2_step_with_dissipation!,
            :poisson_velocity_fft,
            :poisson_velocity_fft!,
            :node_velocities,
            :grid_velocity,
        ]

        for name in public_exports
            @test name in root_exports
        end

        internal_exports = [
            :triangle_areas,
            :triangle_centroids,
            :PoissonWorkspace,
            :curl_rhs_centered,
            :find_elements_nearby!,
            :element_splitting!,
            :quality_based_remesh!,
            :MeshQuality,
            :solve_3x3!,
            :fast_det_3x3,
            :TriangleSoA,
            :create_soa_layout,
            :PerformanceCounters,
            :reset_counters!,
            POISSON_INTERNAL_TYPES...,
        ]

        for name in internal_exports
            @test !(name in root_exports)
        end

        @test isdefined(VortexMethod.GridTransfer, :triangle_areas)
        @test isdefined(VortexMethod.Poisson, :PoissonWorkspace)
        for name in POISSON_INTERNAL_TYPES
            @test isdefined(VortexMethod.Poisson, name)
        end
        @test isdefined(VortexMethod.Remeshing, :quality_based_remesh!)
    end

    @testset "examples qualify Poisson solver internals" begin
        for example in ["advanced_kh3d.jl", "validation_tests.jl"]
            refs = unqualified_poisson_type_refs(joinpath(repo_root, "examples", example))
            @test isempty(refs)
        end
    end

    @testset "tests and examples do not use root-qualified Poisson internals" begin
        refs = String[]

        for root in ["test", "examples"]
            for (dir, _, filenames) in walkdir(joinpath(repo_root, root))
                for filename in filenames
                    endswith(filename, ".jl") || continue

                    file = joinpath(dir, filename)
                    file == (@__FILE__) && continue
                    append!(refs, root_qualified_poisson_internal_refs(file))
                end
            end
        end

        @test isempty(refs)
    end
end
