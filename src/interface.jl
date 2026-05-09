module UserInterface

using ..DomainImpl
using ..Mesh
using ..TimeStepper
using ..Dissipation
using ..Kernels
using ..Workspace: VortexWorkspace

export Periodic, Bounded, Flat, RectilinearGrid, Clock, VortexSheetModel,
       Simulation, set!, time_step!, run!

abstract type Topology end
struct Periodic <: Topology end
struct Bounded  <: Topology end
struct Flat     <: Topology end

_topology_type(::Type{T}) where {T<:Topology} = T
_topology_type(t::Topology) = typeof(t)

function _validate_periodic_topology(topology)
    length(topology) == 3 ||
        throw(ArgumentError("topology must have three entries, for example (Periodic, Periodic, Periodic)"))

    topo = (_topology_type(topology[1]), _topology_type(topology[2]), _topology_type(topology[3]))
    all(t -> t === Periodic, topo) ||
        throw(ArgumentError("VortexMethod currently supports periodic FFT topology only; got topology=$topology"))
    return topo
end

function _interval_length(name::Symbol, interval)
    length(interval) == 2 || throw(ArgumentError("$name must be a two-entry interval"))
    a, b = Float64(interval[1]), Float64(interval[2])
    b > a || throw(ArgumentError("$name upper bound must exceed lower bound"))
    return a, b, b - a
end

"""
    RectilinearGrid(; size, x=(0, 1), y=(0, 1), z=(-1, 1), extent=nothing,
                      topology=(Periodic, Periodic, Periodic))

High-level wrapper around `DomainSpec` and `GridSpec`.

The current vortex solver assumes `x` and `y` start at zero and `z` is centered
around zero, so this constructor validates those restrictions instead of hiding
an origin shift.
"""
struct RectilinearGrid{T<:AbstractFloat}
    size::NTuple{3,Int}
    x::NTuple{2,T}
    y::NTuple{2,T}
    z::NTuple{2,T}
    topology::NTuple{3,DataType}
    domain::DomainSpec
    grid::GridSpec
end

function RectilinearGrid(; size,
                         x=nothing,
                         y=nothing,
                         z=nothing,
                         extent=nothing,
                         topology=(Periodic, Periodic, Periodic))
    length(size) == 3 ||
        throw(ArgumentError("VortexMethod.RectilinearGrid currently expects size=(nx, ny, nz)"))
    grid_size = (Int(size[1]), Int(size[2]), Int(size[3]))
    all(>(0), grid_size) || throw(ArgumentError("grid size entries must be positive"))

    if extent !== nothing
        length(extent) == 3 || throw(ArgumentError("extent must have three entries"))
        x = x === nothing ? (0.0, Float64(extent[1])) : x
        y = y === nothing ? (0.0, Float64(extent[2])) : y
        z = z === nothing ? (-Float64(extent[3]) / 2, Float64(extent[3]) / 2) : z
    else
        x = x === nothing ? (0.0, 1.0) : x
        y = y === nothing ? (0.0, 1.0) : y
        z = z === nothing ? (-1.0, 1.0) : z
    end

    x₁, x₂, Lx = _interval_length(:x, x)
    y₁, y₂, Ly = _interval_length(:y, y)
    z₁, z₂, Lz_total = _interval_length(:z, z)
    iszero(x₁) || throw(ArgumentError("VortexMethod currently requires x to start at 0"))
    iszero(y₁) || throw(ArgumentError("VortexMethod currently requires y to start at 0"))
    isapprox(z₁, -z₂; atol=eps(Float64)) ||
        throw(ArgumentError("VortexMethod currently requires z to be centered around 0"))

    topo = _validate_periodic_topology(topology)
    domain = DomainSpec(Lx, Ly, Lz_total / 2)
    gr = GridSpec(grid_size...)
    return RectilinearGrid{Float64}(grid_size, (x₁, x₂), (y₁, y₂), (z₁, z₂), topo, domain, gr)
end

Base.show(io::IO, grid::RectilinearGrid{T}) where T =
    print(io, "$(grid.size[1])×$(grid.size[2])×$(grid.size[3]) RectilinearGrid{$T, Periodic, Periodic, Periodic}")

mutable struct Clock
    time::Float64
    iteration::Int
    last_Δt::Float64
end

Clock() = Clock(0.0, 0, 0.0)

mutable struct VortexSheetModel{T<:AbstractFloat,A}
    grid::RectilinearGrid{T}
    clock::Clock
    nodeX::Vector{T}
    nodeY::Vector{T}
    nodeZ::Vector{T}
    tri::Matrix{Int}
    eleGma::Matrix{T}
    At::A
    adaptive::Bool
    CFL::T
    poisson_mode::Symbol
    parallel_fft::Bool
    dissipation::DissipationModel
    kernel::KernelType
    workspace::VortexWorkspace{T}
end

function VortexSheetModel(; grid::RectilinearGrid{T},
                          sheet_size::Tuple{Int,Int}=(16, 16),
                          Γ=nothing,
                          gamma=nothing,
                          circulation=nothing,
                          amp::Float64=1e-2,
                          At=nothing,
                          adaptive::Bool=false,
                          CFL::Float64=0.5,
                          poisson_mode::Symbol=:spectral,
                          parallel_fft::Bool=false,
                          dissipation::DissipationModel=NoDissipation(),
                          kernel::KernelType=PeskinStandard()) where T
    nodeX, nodeY, nodeZ, tri, _, _, _ =
        structured_mesh(Int(sheet_size[1]), Int(sheet_size[2]); domain=grid.domain, amp=amp)
    eleGma = zeros(T, size(tri, 1), 3)
    nx, ny, nz = grid.grid.nx, grid.grid.ny, grid.grid.nz
    ws = VortexWorkspace(T, length(nodeX), size(tri, 1), nx, ny, nz, grid.domain)
    At_value = At === nothing ? zero(T) : At
    A = typeof(At_value)
    model = VortexSheetModel{T,A}(grid, Clock(), Vector{T}(nodeX), Vector{T}(nodeY),
                                  Vector{T}(nodeZ), tri, eleGma, At_value, adaptive,
                                  T(CFL), poisson_mode, parallel_fft, dissipation,
                                  kernel, ws)
    specified = count(!isnothing, (Γ, gamma, circulation))
    specified <= 1 ||
        throw(ArgumentError("provide only one of Γ, gamma, or circulation"))
    Γ_value = Γ !== nothing ? Γ : gamma !== nothing ? gamma :
              circulation !== nothing ? circulation : (0.0, 1.0, 0.0)
    _assign_Γ!(model.eleGma, Γ_value)
    return model
end

function _assign_Γ!(eleGma::AbstractMatrix, Γ::AbstractMatrix)
    size(eleGma) == size(Γ) ||
        throw(DimensionMismatch("Γ matrix size $(size(Γ)) does not match model eleGma size $(size(eleGma))"))
    eleGma .= Γ
    return nothing
end

function _assign_Γ!(eleGma::AbstractMatrix, Γ)
    length(Γ) == 3 || throw(ArgumentError("Γ must have three components"))
    T = eltype(eleGma)
    γ₁, γ₂, γ₃ = T(Γ[1]), T(Γ[2]), T(Γ[3])
    @inbounds for t in axes(eleGma, 1)
        eleGma[t, 1] = γ₁
        eleGma[t, 2] = γ₂
        eleGma[t, 3] = γ₃
    end
    return nothing
end

function set!(model::VortexSheetModel; Γ=nothing, gamma=nothing, circulation=nothing)
    specified = count(!isnothing, (Γ, gamma, circulation))
    specified <= 1 ||
        throw(ArgumentError("provide only one of Γ, gamma, or circulation"))
    Γ_value = Γ !== nothing ? Γ : gamma !== nothing ? gamma : circulation
    Γ_value === nothing && return model
    _assign_Γ!(model.eleGma, Γ_value)
    return model
end

function time_step!(model::VortexSheetModel{T,A}, Δt::Real; kwargs...) where {T,A}
    dt = Float64(Δt)
    if model.dissipation isa NoDissipation
        dt_used = rk2_step!(model.workspace,
                            model.nodeX, model.nodeY, model.nodeZ, model.tri, model.eleGma,
                            model.grid.domain, model.grid.grid, dt;
                            At=model.At, adaptive=model.adaptive, CFL=model.CFL,
                            poisson_mode=model.poisson_mode, parallel_fft=model.parallel_fft,
                            kernel=model.kernel,
                            kwargs...)
    elseif model.dissipation isa SmagorinskyModel
        dt_used = rk2_step_with_dissipation!(model.workspace,
                                             model.nodeX, model.nodeY, model.nodeZ,
                                             model.tri, model.eleGma,
                                             model.grid.domain, model.grid.grid, dt,
                                             model.dissipation;
                                             At=model.At, adaptive=model.adaptive,
                                             CFL=model.CFL,
                                             poisson_mode=model.poisson_mode,
                                             parallel_fft=model.parallel_fft,
                                             kernel=model.kernel, kwargs...)
    else
        dt_used = rk2_step_with_dissipation!(model.nodeX, model.nodeY, model.nodeZ,
                                             model.tri, model.eleGma,
                                             model.grid.domain, model.grid.grid, dt,
                                             model.dissipation;
                                             At=model.At, adaptive=model.adaptive, CFL=model.CFL,
                                             poisson_mode=model.poisson_mode,
                                             parallel_fft=model.parallel_fft,
                                             kernel=model.kernel, kwargs...)
    end

    model.clock.time += dt_used
    model.clock.iteration += 1
    model.clock.last_Δt = dt_used
    return dt_used
end

mutable struct Simulation
    model::VortexSheetModel
    Δt::Float64
    stop_iteration::Union{Nothing,Int}
    stop_time::Float64
end

function Simulation(model::VortexSheetModel; Δt, stop_iteration=nothing, stop_time=Inf)
    stop_iteration === nothing && !isfinite(Float64(stop_time)) &&
        throw(ArgumentError("provide stop_iteration or finite stop_time"))
    iter = stop_iteration === nothing ? nothing : Int(stop_iteration)
    return Simulation(model, Float64(Δt), iter, Float64(stop_time))
end

function _finished(simulation::Simulation)
    clock = simulation.model.clock
    if simulation.stop_iteration !== nothing && clock.iteration >= simulation.stop_iteration
        return true
    end
    return clock.time >= simulation.stop_time
end

function run!(simulation::Simulation)
    while !_finished(simulation)
        remaining = simulation.stop_time - simulation.model.clock.time
        Δt = isfinite(remaining) ? min(simulation.Δt, remaining) : simulation.Δt
        time_step!(simulation.model, Δt)
    end
    return simulation
end

Base.show(io::IO, model::VortexSheetModel) =
    print(io, "VortexSheetModel(time=$(model.clock.time), iteration=$(model.clock.iteration), elements=$(size(model.tri, 1)))")

Base.show(io::IO, simulation::Simulation) =
    print(io, "Simulation of $(simulation.model)")

end # module

using .UserInterface: Periodic, Bounded, Flat, RectilinearGrid, Clock, VortexSheetModel,
                      Simulation, set!, time_step!, run!
