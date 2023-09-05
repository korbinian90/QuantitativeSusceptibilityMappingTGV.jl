using QuantitativeSusceptibilityMappingTGV
using Documenter

DocMeta.setdocmeta!(QuantitativeSusceptibilityMappingTGV, :DocTestSetup, :(using QuantitativeSusceptibilityMappingTGV); recursive=true)

makedocs(;
    modules=[QuantitativeSusceptibilityMappingTGV],
    authors="Korbinian Eckstein korbinian90@gmail.com",
    repo="https://github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl/blob/{commit}{path}#{line}",
    sitename="QuantitativeSusceptibilityMappingTGV.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://korbinian90.github.io/QuantitativeSusceptibilityMappingTGV.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/korbinian90/QuantitativeSusceptibilityMappingTGV.jl",
    devbranch="main",
)
