# Stock dissertation KH parameters, Chapter 4, Section 4.2.1.
#
# The medium-resolution doubly-periodic shear layer uses nondimensional
# coordinates with x,y in [0,1], z in [-2,2], grid spacing dx = 1/29, a
# unit y-directed vortex sheet strength, and the perturbation used by
# structured_mesh with amplitude 0.01.

const STOCK_KH_DOMAIN = VortexMethod.DomainSpec(1.0, 1.0, 2.0)
const STOCK_KH_GRID = VortexMethod.GridSpec(29, 29, 116)
const STOCK_KH_MESH_NX = 30
const STOCK_KH_MESH_NY = 30
const STOCK_KH_PERTURBATION_AMPLITUDE = 0.01
const STOCK_KH_INITIAL_GAMMA = (0.0, 1.0, 0.0)
const STOCK_KH_CFL = 0.5
const STOCK_KH_SPLIT_FACTOR = 0.8
const STOCK_KH_MERGE_FACTOR = 0.2

function stock_kh_remesh_thresholds(domain::VortexMethod.DomainSpec=STOCK_KH_DOMAIN,
                                    gr::VortexMethod.GridSpec=STOCK_KH_GRID)
    dx, dy, _ = VortexMethod.grid_spacing(domain, gr)
    spacing = max(dx, dy)
    return STOCK_KH_SPLIT_FACTOR * spacing, STOCK_KH_MERGE_FACTOR * spacing
end

function initialize_stock_kh_gamma!(eleGma)
    eleGma[:, 1] .= STOCK_KH_INITIAL_GAMMA[1]
    eleGma[:, 2] .= STOCK_KH_INITIAL_GAMMA[2]
    eleGma[:, 3] .= STOCK_KH_INITIAL_GAMMA[3]
    return eleGma
end
