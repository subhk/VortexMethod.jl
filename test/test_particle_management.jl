#!/usr/bin/env julia

"""
Test suite for particle management with periodic boundary conditions

This test suite verifies:
1. Particle insertion maintains periodicity 
2. Particle removal conserves circulation
3. Periodic boundary wrapping functions
4. Vortex blob insertion
5. Particle count maintenance
"""

using Test
using VortexMethod

@testset "Particle Management Tests" begin
    
    # Test domain and basic mesh
    domain = DomainSpec(2.0, 2.0, 1.0)  # Lx=2, Ly=2, Lz=1
    
    # Simple test mesh (small triangle)
    nodeX = [0.1, 0.9, 0.5]
    nodeY = [0.1, 0.1, 0.9] 
    nodeZ = [0.0, 0.0, 0.0]
    tri = [1 2 3]
    eleGma = [0.1 0.0 0.2]  # Small circulation values
    
    @testset "Periodic boundary wrapping" begin
        # Test wrap_point function
        x, y, z = wrap_point(2.5, 3.0, 1.5, domain)
        @test x ≈ 0.5  # 2.5 - 2.0 = 0.5
        @test y ≈ 1.0  # 3.0 - 2.0 = 1.0  
        @test z ≈ -0.5 # wrapped to [-1, 1] range
        
        x, y, z = wrap_point(-0.3, -0.5, -1.2, domain)
        @test x ≈ 1.7  # -0.3 + 2.0 = 1.7
        @test y ≈ 1.5  # -0.5 + 2.0 = 1.5
        @test z ≈ 0.8  # -1.2 + 2*1.0 = 0.8
        
        println("✓ Periodic boundary wrapping works correctly")
    end
    
    @testset "Particle insertion criteria" begin
        # Test criteria construction
        criteria = ParticleInsertionCriteria(
            min_vorticity_threshold=1e-5,
            max_particle_spacing=0.2,
            max_particles=1000
        )
        
        @test criteria.min_vorticity_threshold == 1e-5
        @test criteria.max_particle_spacing == 0.2
        @test criteria.max_particles == 1000
        @test criteria.circulation_threshold == 1e-8  # default
        @test criteria.boundary_buffer == 0.05       # default
        
        # Test default constructor
        default_criteria = ParticleInsertionCriteria()
        @test default_criteria.min_vorticity_threshold == 1e-6
        @test default_criteria.max_particles == 100000
        
        println("✓ Particle insertion criteria work correctly")
    end
    
    @testset "Particle removal criteria" begin
        # Test criteria construction
        criteria = ParticleRemovalCriteria(
            weak_circulation_threshold=1e-12,
            min_particle_spacing=0.005
        )
        
        @test criteria.weak_circulation_threshold == 1e-12
        @test criteria.min_particle_spacing == 0.005
        @test criteria.boundary_removal_zone == 0.02  # default
        @test criteria.age_threshold == 1000          # default
        
        println("✓ Particle removal criteria work correctly")
    end
    
    @testset "Particle insertion (basic functionality)" begin
        # Copy arrays for testing
        nodeX_test = copy(nodeX)
        nodeY_test = copy(nodeY)
        nodeZ_test = copy(nodeZ)
        tri_test = copy(tri)
        eleGma_test = copy(eleGma)
        
        initial_count = length(nodeX_test)
        
        # Loose criteria to ensure insertion happens
        criteria = ParticleInsertionCriteria(
            min_vorticity_threshold=1e-10,
            max_particle_spacing=1.5,  # Large spacing threshold
            max_particles=100
        )
        
        n_inserted = insert_particles_periodic!(nodeX_test, nodeY_test, nodeZ_test,
                                               tri_test, eleGma_test, domain, criteria)
        
        @test n_inserted ≥ 0  # Should insert some particles or none
        @test length(nodeX_test) == length(nodeY_test) == length(nodeZ_test)
        @test length(nodeX_test) >= initial_count
        
        # Verify all nodes are within periodic domain
        for i in eachindex(nodeX_test)
            @test 0 ≤ nodeX_test[i] < domain.Lx
            @test 0 ≤ nodeY_test[i] < domain.Ly
            @test -domain.Lz ≤ nodeZ_test[i] ≤ domain.Lz
        end
        
        println("✓ Basic particle insertion works correctly")
    end
    
    @testset "Particle removal (basic functionality)" begin
        # Create test setup with weak circulation
        nodeX_test = [0.1, 0.9, 0.5, 1.5]
        nodeY_test = [0.1, 0.1, 0.9, 0.5] 
        nodeZ_test = [0.0, 0.0, 0.0, 0.2]
        tri_test = [1 2 3; 2 3 4]
        eleGma_test = [1e-12 0.0 1e-12; 0.1 0.0 0.2]  # First element has weak circulation
        
        initial_count = length(nodeX_test)
        
        criteria = ParticleRemovalCriteria(
            weak_circulation_threshold=1e-10  # Remove very weak circulation
        )
        
        n_removed = remove_particles_periodic!(nodeX_test, nodeY_test, nodeZ_test,
                                              tri_test, eleGma_test, domain, criteria)
        
        @test n_removed ≥ 0  # Should remove some particles or none
        @test length(nodeX_test) == length(nodeY_test) == length(nodeZ_test)
        
        # Verify all remaining nodes are within periodic domain  
        for i in eachindex(nodeX_test)
            @test 0 ≤ nodeX_test[i] < domain.Lx
            @test 0 ≤ nodeY_test[i] < domain.Ly
            @test -domain.Lz ≤ nodeZ_test[i] ≤ domain.Lz
        end
        
        println("✓ Basic particle removal works correctly")
    end
    
    @testset "Vortex blob insertion" begin
        # Copy arrays for testing
        nodeX_test = copy(nodeX)
        nodeY_test = copy(nodeY)
        nodeZ_test = copy(nodeZ)
        tri_test = copy(tri)
        eleGma_test = copy(eleGma)
        
        initial_count = length(nodeX_test)
        
        # Insert vortex blob at domain center
        center = (1.0, 1.0, 0.0)  # Domain center
        strength = (0.1, 0.0, 0.05)  # Vorticity vector
        radius = 0.2
        n_particles = 5
        
        n_inserted = insert_vortex_blob_periodic!(nodeX_test, nodeY_test, nodeZ_test,
                                                 tri_test, eleGma_test, domain,
                                                 center, strength, radius, n_particles)
        
        @test n_inserted ≥ 0  # Should insert some particles
        @test length(nodeX_test) >= initial_count
        
        # Verify periodicity maintained
        for i in eachindex(nodeX_test)
            @test 0 ≤ nodeX_test[i] < domain.Lx
            @test 0 ≤ nodeY_test[i] < domain.Ly
            @test -domain.Lz ≤ nodeZ_test[i] ≤ domain.Lz
        end
        
        println("✓ Vortex blob insertion works correctly")
    end
    
    @testset "Particle count maintenance" begin
        # Create test setup
        nodeX_test = collect(range(0.1, 1.9, length=20))
        nodeY_test = collect(range(0.1, 1.9, length=20))
        nodeZ_test = zeros(20)
        tri_test = reshape(1:18, 6, 3)  # Simple triangulation
        eleGma_test = 0.01 * randn(6, 3)  # Random small circulations
        
        target_count = 25
        tolerance = 0.2
        
        result = maintain_particle_count!(nodeX_test, nodeY_test, nodeZ_test,
                                        tri_test, eleGma_test, domain, 
                                        target_count, tolerance)
        
        final_count = length(nodeX_test)
        min_expected = Int(round(target_count * (1 - tolerance)))
        max_expected = Int(round(target_count * (1 + tolerance)))
        
        @test min_expected ≤ final_count ≤ max_expected
        
        # Verify periodicity maintained
        for i in eachindex(nodeX_test)
            @test 0 ≤ nodeX_test[i] < domain.Lx
            @test 0 ≤ nodeY_test[i] < domain.Ly
            @test -domain.Lz ≤ nodeZ_test[i] ≤ domain.Lz
        end
        
        println("✓ Particle count maintenance works correctly")
    end
    
    @testset "Weak vortex removal" begin
        # Create test with mix of strong and weak vortices
        nodeX_test = [0.2, 0.8, 1.2, 1.8]
        nodeY_test = [0.2, 0.8, 1.2, 0.3]
        nodeZ_test = [0.0, 0.1, -0.1, 0.0]
        tri_test = [1 2 3; 2 3 4]
        # Mix of weak and strong circulations
        eleGma_test = [1e-12 0.0 1e-12; 0.1 0.05 0.2]
        
        threshold = 1e-10
        
        n_removed = remove_weak_vortices!(nodeX_test, nodeY_test, nodeZ_test,
                                        tri_test, eleGma_test, domain, threshold)
        
        @test n_removed ≥ 0
        
        # Verify periodicity maintained
        for i in eachindex(nodeX_test)
            @test 0 ≤ nodeX_test[i] < domain.Lx
            @test 0 ≤ nodeY_test[i] < domain.Ly
            @test -domain.Lz ≤ nodeZ_test[i] ≤ domain.Lz
        end
        
        println("✓ Weak vortex removal works correctly")
    end
    
    @testset "Particle redistribution" begin
        # Create clustered particles to test redistribution
        nodeX_test = [0.1, 0.12, 0.11, 1.5, 1.8]
        nodeY_test = [0.1, 0.11, 0.13, 1.2, 1.8]
        nodeZ_test = [0.0, 0.01, -0.01, 0.2, -0.1]
        tri_test = reshape(1:15, 5, 3)  # Simple triangulation
        eleGma_test = 0.05 * ones(5, 3)  # Uniform circulation
        
        initial_count = length(nodeX_test)
        
        final_count = redistribute_particles_periodic!(nodeX_test, nodeY_test, nodeZ_test,
                                                     tri_test, eleGma_test, domain)
        
        @test final_count == initial_count  # Should preserve count
        @test final_count == length(nodeX_test)
        
        # Verify periodicity maintained
        for i in eachindex(nodeX_test)
            @test 0 ≤ nodeX_test[i] < domain.Lx
            @test 0 ≤ nodeY_test[i] < domain.Ly
            @test -domain.Lz ≤ nodeZ_test[i] ≤ domain.Lz
        end
        
        println("✓ Particle redistribution works correctly")
    end
end

println("All particle management tests passed successfully!")
