# Performance monitoring and profiling infrastructure for VortexMethod.jl

module Performance

using Printf
using LinearAlgebra

export @vortex_time, PerformanceCounters, reset_counters!, print_performance_report,
       @profile_function, enable_profiling!, disable_profiling!

# Global performance counters
mutable struct PerformanceCounters
    # Core computational kernels
    poisson_solve_time::Float64
    poisson_solve_count::Int64
    
    # Spreading and interpolation
    spreading_time::Float64
    spreading_count::Int64
    interpolation_time::Float64 
    interpolation_count::Int64
    
    # Circulation calculations
    circulation_time::Float64
    circulation_count::Int64
    
    # Memory allocations
    total_allocations::Int64
    peak_memory::Int64
    
    # Function call counters for hotspots
    kernel_evaluations::Int64
    matrix_inversions::Int64
    triangle_area_computations::Int64
    
    # Cache performance
    cache_hits::Int64
    cache_misses::Int64
    
    PerformanceCounters() = new(0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0)
end

const GLOBAL_COUNTERS = PerformanceCounters()

# Global profiling state
const PROFILING_ENABLED = Ref(false)

function enable_profiling!()
    PROFILING_ENABLED[] = true
    reset_counters!()
    @info "VortexMethod performance profiling enabled"
end

function disable_profiling!()
    PROFILING_ENABLED[] = false
    @info "VortexMethod performance profiling disabled"
end

function reset_counters!()
    GLOBAL_COUNTERS.poisson_solve_time = 0.0
    GLOBAL_COUNTERS.poisson_solve_count = 0
    GLOBAL_COUNTERS.spreading_time = 0.0
    GLOBAL_COUNTERS.spreading_count = 0
    GLOBAL_COUNTERS.interpolation_time = 0.0
    GLOBAL_COUNTERS.interpolation_count = 0
    GLOBAL_COUNTERS.circulation_time = 0.0
    GLOBAL_COUNTERS.circulation_count = 0
    GLOBAL_COUNTERS.total_allocations = 0
    GLOBAL_COUNTERS.peak_memory = 0
    GLOBAL_COUNTERS.kernel_evaluations = 0
    GLOBAL_COUNTERS.matrix_inversions = 0
    GLOBAL_COUNTERS.triangle_area_computations = 0
    GLOBAL_COUNTERS.cache_hits = 0
    GLOBAL_COUNTERS.cache_misses = 0
end

# High-performance timing macro
macro vortex_time(category, expr)
    quote
        if PROFILING_ENABLED[]
            local start_time = time_ns()
            local start_memory = Base.gc_num().total_allocd_bytes
            local result = $(esc(expr))
            local end_time = time_ns()
            local end_memory = Base.gc_num().total_allocd_bytes
            
            local elapsed_time = (end_time - start_time) / 1e9
            local allocated_memory = end_memory - start_memory
            
            # Update counters based on category
            if $(QuoteNode(category)) == :poisson
                GLOBAL_COUNTERS.poisson_solve_time += elapsed_time
                GLOBAL_COUNTERS.poisson_solve_count += 1
            elseif $(QuoteNode(category)) == :spreading
                GLOBAL_COUNTERS.spreading_time += elapsed_time
                GLOBAL_COUNTERS.spreading_count += 1
            elseif $(QuoteNode(category)) == :interpolation
                GLOBAL_COUNTERS.interpolation_time += elapsed_time
                GLOBAL_COUNTERS.interpolation_count += 1
            elseif $(QuoteNode(category)) == :circulation
                GLOBAL_COUNTERS.circulation_time += elapsed_time
                GLOBAL_COUNTERS.circulation_count += 1
            end
            
            GLOBAL_COUNTERS.total_allocations += allocated_memory
            GLOBAL_COUNTERS.peak_memory = max(GLOBAL_COUNTERS.peak_memory, allocated_memory)
            
            result
        else
            $(esc(expr))
        end
    end
end

# Function-level profiling macro
macro profile_function(func_name, expr)
    quote
        function $(esc(func_name))(args...)
            @vortex_time :general $(esc(expr))(args...)
        end
    end
end

function print_performance_report()
    if !PROFILING_ENABLED[]
        @info "Profiling not enabled. Call enable_profiling!() first."
        return
    end
    
    println("="^80)
    println("VortexMethod.jl Performance Report")
    println("="^80)
    
    # Timing breakdown
    total_compute_time = GLOBAL_COUNTERS.poisson_solve_time + 
                        GLOBAL_COUNTERS.spreading_time + 
                        GLOBAL_COUNTERS.interpolation_time + 
                        GLOBAL_COUNTERS.circulation_time
    
    if total_compute_time > 0
        println("Computational Breakdown:")
        @printf "  Poisson Solver:    %8.3f s (%5.1f%%) - %d calls\n" GLOBAL_COUNTERS.poisson_solve_time (GLOBAL_COUNTERS.poisson_solve_time/total_compute_time*100) GLOBAL_COUNTERS.poisson_solve_count
        @printf "  Spreading:         %8.3f s (%5.1f%%) - %d calls\n" GLOBAL_COUNTERS.spreading_time (GLOBAL_COUNTERS.spreading_time/total_compute_time*100) GLOBAL_COUNTERS.spreading_count
        @printf "  Interpolation:     %8.3f s (%5.1f%%) - %d calls\n" GLOBAL_COUNTERS.interpolation_time (GLOBAL_COUNTERS.interpolation_time/total_compute_time*100) GLOBAL_COUNTERS.interpolation_count
        @printf "  Circulation:       %8.3f s (%5.1f%%) - %d calls\n" GLOBAL_COUNTERS.circulation_time (GLOBAL_COUNTERS.circulation_time/total_compute_time*100) GLOBAL_COUNTERS.circulation_count
        @printf "  Total:             %8.3f s\n" total_compute_time
    end
    
    println()
    println("Operation Counts:")
    @printf "  Kernel Evaluations:        %12d\n" GLOBAL_COUNTERS.kernel_evaluations
    @printf "  Matrix Inversions:         %12d\n" GLOBAL_COUNTERS.matrix_inversions
    @printf "  Triangle Area Computations:%12d\n" GLOBAL_COUNTERS.triangle_area_computations
    
    println()
    println("Memory Usage:")
    @printf "  Total Allocations:         %12d bytes (%.2f MB)\n" GLOBAL_COUNTERS.total_allocations (GLOBAL_COUNTERS.total_allocations/1e6)
    @printf "  Peak Single Allocation:    %12d bytes (%.2f MB)\n" GLOBAL_COUNTERS.peak_memory (GLOBAL_COUNTERS.peak_memory/1e6)
    
    if GLOBAL_COUNTERS.cache_hits + GLOBAL_COUNTERS.cache_misses > 0
        println()
        println("Cache Performance:")
        cache_total = GLOBAL_COUNTERS.cache_hits + GLOBAL_COUNTERS.cache_misses
        @printf "  Cache Hits:                %12d (%5.1f%%)\n" GLOBAL_COUNTERS.cache_hits (GLOBAL_COUNTERS.cache_hits/cache_total*100)
        @printf "  Cache Misses:              %12d (%5.1f%%)\n" GLOBAL_COUNTERS.cache_misses (GLOBAL_COUNTERS.cache_misses/cache_total*100)
    end
    
    println("="^80)
    
    # Performance recommendations
    if GLOBAL_COUNTERS.poisson_solve_time > 0.5 * total_compute_time
        println("RECOMMENDATION: Poisson solver dominates runtime. Consider:")
        println("  - Using iterative methods for ill-conditioned problems")
        println("  - Pre-computing FFT plans")
        println("  - Using in-place operations with PoissonWorkspace")
    end
    
    if GLOBAL_COUNTERS.matrix_inversions > 1000
        println("RECOMMENDATION: Many small matrix inversions detected. Consider:")
        println("  - Batching matrix operations")
        println("  - Using specialized 3x3 or 4x4 matrix routines")
        println("  - Caching triangle geometry")
    end
    
    if GLOBAL_COUNTERS.total_allocations > 1e8
        println("RECOMMENDATION: High memory allocation detected. Consider:")
        println("  - Using pre-allocated workspaces")
        println("  - In-place operations where possible")
        println("  - Memory pooling for frequent allocations")
    end
end

# Lightweight performance counters for critical sections
@inline function count_kernel_evaluation!()
    PROFILING_ENABLED[] && (GLOBAL_COUNTERS.kernel_evaluations += 1)
end

@inline function count_matrix_inversion!()
    PROFILING_ENABLED[] && (GLOBAL_COUNTERS.matrix_inversions += 1)
end

@inline function count_triangle_area!()
    PROFILING_ENABLED[] && (GLOBAL_COUNTERS.triangle_area_computations += 1)
end

@inline function count_cache_hit!()
    PROFILING_ENABLED[] && (GLOBAL_COUNTERS.cache_hits += 1)
end

@inline function count_cache_miss!()
    PROFILING_ENABLED[] && (GLOBAL_COUNTERS.cache_misses += 1)
end

end # module

using .Performance: @vortex_time, PerformanceCounters, reset_counters!, print_performance_report,
                    @profile_function, enable_profiling!, disable_profiling!,
                    count_kernel_evaluation!, count_matrix_inversion!, count_triangle_area!,
                    count_cache_hit!, count_cache_miss!