using VortexMethod
using Documenter

DocMeta.setdocmeta!(VortexMethod, :DocTestSetup, :(using VortexMethod); recursive=true)

makedocs(;
    modules=[VortexMethod],
    authors="Subhajit Kar <subhajitkar19@gmail.com> and contributors",
    sitename="VortexMethod.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Remeshing" => "remeshing.md",
        "Parallelization" => "parallelization.md",
        "Validation" => "validation.md",
        "Usage" => "usage.md",
        "API" => "api.md",
    ],
)

# Deploy only when explicitly requested by CI.
# The GitHub Actions workflow uploads `docs/build` as a Pages artifact,
# so we skip Documenter's git push by default to avoid permission errors.
if get(ENV, "DOCS_DEPLOY", "false") == "true"
    deploydocs(
        repo="github.com/subhk/VortexMethod.jl",
        devbranch="main",
    )
end
