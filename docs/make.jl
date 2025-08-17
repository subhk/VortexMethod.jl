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
    ],
)
