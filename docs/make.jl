using TGV_QSM
using Documenter

DocMeta.setdocmeta!(TGV_QSM, :DocTestSetup, :(using TGV_QSM); recursive=true)

makedocs(;
    modules=[TGV_QSM],
    authors="Korbinian Eckstein korbinian90@gmail.com",
    repo="https://github.com/korbinian90/TGV_QSM.jl/blob/{commit}{path}#{line}",
    sitename="TGV_QSM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://korbinian90.github.io/TGV_QSM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/korbinian90/TGV_QSM.jl",
    devbranch="main",
)
