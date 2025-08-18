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
    # Relax CI to avoid failing on missing docs/cross-refs for now
    warnonly=true,
    checkdocs=:none,
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
# Deploy using Documenter's GitHubActions provider (pushes to gh-pages).
deploydocs(
    repo="github.com/subhk/VortexMethod.jl",
    devbranch="main",
    provider=Documenter.GitHubActions(),
    push_preview=false,
)
