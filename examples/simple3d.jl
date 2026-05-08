# Minimal high-level interface example.

using VortexMethod
using Printf

grid = RectilinearGrid(size=(8, 8, 7),
                       x=(0.0, 1.0),
                       y=(0.0, 1.0),
                       z=(-1.0, 1.0),
                       topology=(Periodic, Periodic, Periodic))

model = VortexSheetModel(; grid,
                         sheet_size=(5, 5),
                         circulation=(0.0, 1.0, 0.0),
                         amp=0.0)

simulation = Simulation(model; Δt=0.0, stop_iteration=1)
run!(simulation)

println("Grid: ", grid)
println("Model iteration: ", model.clock.iteration)
println("Model time: ", @sprintf("%.4e", model.clock.time))
println("Elements: ", size(model.tri, 1))

