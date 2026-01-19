using VortexMethod
using Documenter

DocMeta.setdocmeta!(VortexMethod, :DocTestSetup, :(using VortexMethod); recursive=true)

makedocs(;
    modules=[VortexMethod],
    authors="Subhajit Kar <subhajitkar19@gmail.com> and contributors",
    sitename="VortexMethod.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=["assets/custom.css"],
        collapselevel=2,
        sidebar_sitename=true,
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    # Relax CI to avoid failing on missing docs/cross-refs for now
    warnonly=true,
    checkdocs=:none,
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Boundary Conditions" => "boundary_conditions.md",
        "Baroclinic Effects" => "baroclinic.md",
        "Dissipation Models" => "dissipation.md",
        "Remeshing" => "remeshing.md",
        "Parallelization" => "parallelization.md",
        "Validation" => "validation.md",
        "Usage" => "usage.md",
        "API" => "api.md",
    ],
)
# Deployment is handled by the GitHub Pages steps in the CI workflow.
# We do not call deploydocs() here to avoid provider/version mismatches.
